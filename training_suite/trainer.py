import logging
import wandb

from omegaconf import OmegaConf, DictConfig
from pathlib import Path
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from wandb.integration.sb3 import WandbCallback

from .factory import make_robot, make_env, make_model
from .checkpoint import save_checkpoint, load_checkpoint
from .callbacks import CheckpointCallback, UploadModelCallback, TrainingMetricsCallback

logger = logging.getLogger(__name__)


def train(cfg: DictConfig, run_id=None, checkpoint: Path | None = None):
    """
    Runs a complete training given the provided config.

    Args:
        cfg:            Experiment config
        run_id:         ID of a previous run to resume.
                        If None a new run is created.
                        Default = None
    """
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # --- W&B run initliazation ---
    run = wandb.init(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True), # type: ignore
        id=run_id,
        group=cfg.wandb.get("group"),
        resume="allow",
        monitor_gym=True,
        sync_tensorboard=True
    )
    run.define_metric("*", step_metric="global_step") # fix for using global_steps in timeline
    run_dir = Path(run.dir)


    # --- Env & model creation ---
    robot = make_robot(cfg.robot)
    env = make_vec_env(
        lambda: make_env(cfg, robot, seed=cfg.seed),
        n_envs=cfg.training.n_envs,
        seed=cfg.seed,
        vec_env_cls=DummyVecEnv
    )
    model = make_model(cfg.model, env, cfg.seed, tensorboard_log=str(run_dir))


    # --- Checkpoint and branching handling ---
    start_step = 0
    if checkpoint:
        logger.info("🚩 Loading checkpoint '%s'", checkpoint)
        model, start_step = load_checkpoint(checkpoint, model, cfg)
    elif cfg.training.get("init_from"):
        logger.info("💫 Initialize model from artifact '%s'", cfg.init_from)
        artifact = run.use_artifact(cfg.init_from)
        artifact_path = artifact.download()
        model.set_parameters(artifact_path)


    total_timesteps = cfg.training.total_timesteps
    checkpoint_dir = Path(f"./checkpoints/{run.id}")
    # --- Callbacks ---
    wandb_cb = WandbCallback()
    metrics_cb = TrainingMetricsCallback(run=run)
    checkpoint_cb = CheckpointCallback(
        cfg=cfg,
        checkpoint_dir=checkpoint_dir,
        save_freq=cfg.training.save_checkpoint_freq,
        total_steps=total_timesteps
    )
    eval_cb = EvalCallback(
        eval_env=make_env(cfg, robot, seed=1234),
        eval_freq=cfg.training.eval_freq // cfg.training.n_envs,
        callback_on_new_best=UploadModelCallback(
            run=run,
            cfg=cfg,
            checkpoint_dir=checkpoint_dir,
            total_steps=total_timesteps
        )
    )
    callbacks = [wandb_cb, metrics_cb, checkpoint_cb, eval_cb]


    # --- Train loop ---
    logging.info("❇️💪 Start training!")
    model.learn( # type: ignore
        total_timesteps=total_timesteps,
        reset_num_timesteps=(start_step == 0),
        callback=CallbackList(callbacks)
    )


    # --- Finished training, saving model details ---
    save_checkpoint(model, cfg, run_dir, total_timesteps, total_timesteps, label="model_final")
    final_model_path = run_dir / "model_final.zip"
    logger.info("💾 Saved final model to '%s'", final_model_path)
    wandb.log_model(final_model_path, "model", aliases=["final"])

    wandb.finish()
