"""Training entry point for the robo_gym RL pipeline.

Usage::

    uv run python train.py                                       # fresh start
    uv run python train.py --resume                              # crash resume
    uv run python train.py wrappers=minimal                      # override group
    uv run python train.py training.init_from=runs/abc/ckpt_best.zip  # stage branch
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path

import hydra
import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import CallbackList
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv
from wandb.integration.sb3 import WandbCallback

from training_suite import LastActionWrapper, make_env, make_model, make_robot  # noqa: F401
from training_suite.checkpoint import (
    CheckpointCallback,
    compute_cfg_hash,
    find_latest_checkpoint,
    load_checkpoint,
)
from training_suite.config_schema import register_schemas

register_schemas()

# ---------------------------------------------------------------------------
# --resume detection: strip the flag before Hydra sees sys.argv
# ---------------------------------------------------------------------------
_RESUME = "--resume" in sys.argv
if _RESUME:
    sys.argv.remove("--resume")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _find_run_dir_by_hash(cfg_hash: str) -> Path | None:
    """Scan runs/ for the most recent run whose config_hash.txt matches cfg_hash."""
    runs_root = Path("runs")
    if not runs_root.exists():
        return None
    candidates = sorted(
        runs_root.iterdir(),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for run_dir in candidates:
        hash_file = run_dir / "config_hash.txt"
        if hash_file.exists() and hash_file.read_text().strip() == cfg_hash:
            return run_dir
    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

@hydra.main(version_base=None, config_path="./config", config_name="config")
def train(cfg: DictConfig) -> None:
    """Train a PPO agent with full checkpoint and resume support."""
    log = logging.getLogger(__name__)
    log.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # --- Resume detection ---
    start_step = 0
    resume_run_id: str | None = None
    ckpt_path: Path | None = None

    if _RESUME:
        cfg_hash = compute_cfg_hash(cfg)
        resume_run_dir = _find_run_dir_by_hash(cfg_hash)
        if resume_run_dir is None:
            log.warning("--resume requested but no matching run found; starting fresh.")
        else:
            ckpt_path = find_latest_checkpoint(resume_run_dir)
            if ckpt_path is None:
                log.warning(
                    "Matching run %s found but no numbered checkpoint; starting fresh.",
                    resume_run_dir,
                )
            else:
                resume_run_id = resume_run_dir.name
                log.info("Resuming run %s from %s", resume_run_id, ckpt_path)

    # Also support stage branching via config (init_from is in training group)
    init_from = cfg.training.get("init_from")

    # --- W&B init ---
    wandb_kwargs: dict = dict(
        project=cfg.wandb.project,
        config=OmegaConf.to_container(cfg, resolve=True),
        group=cfg.wandb.get("group"),
    )
    if resume_run_id is not None:
        wandb_kwargs["id"] = resume_run_id
        wandb_kwargs["resume"] = "must"

    wandb.init(**wandb_kwargs)
    assert wandb.run is not None, "wandb.init() failed to create a run"

    run_dir = Path("runs") / wandb.run.id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Write config hash + full config on first run.
    hash_file = run_dir / "config_hash.txt"
    if not hash_file.exists():
        hash_file.write_text(compute_cfg_hash(cfg))
        (run_dir / "config.yaml").write_text(OmegaConf.to_yaml(cfg))

    # --- Env & model creation ---
    robot = make_robot(cfg.robot)
    n_envs = cfg.training.n_envs

    env = make_vec_env(
        lambda: make_env(cfg, robot, seed=cfg.seed),
        n_envs=n_envs,
        seed=cfg.seed,
        vec_env_cls=SubprocVecEnv,
    )

    model = make_model(cfg.model, env, cfg.seed, tensorboard_log=str(run_dir))

    # --- Load checkpoint if resuming or branching ---
    if ckpt_path is not None:
        # Crash resume: full state restore, hash must match
        start_step = load_checkpoint(ckpt_path, model, cfg, weights_only=False)
        log.info("Crash-resumed from step %d", start_step)
    elif init_from:
        # Stage branch: weights only, optimizer + step counter reset
        branch_path = Path(init_from)
        start_step = load_checkpoint(branch_path, model, cfg, weights_only=True)
        log.info("Stage branch: loaded weights from %s", branch_path)

    # --- Callbacks ---
    total_steps = cfg.training.total_timesteps
    ckpt_cb = CheckpointCallback(
        cfg=cfg,
        run_dir=run_dir,
        save_freq=cfg.training.save_checkpoint_freq,
        keep_last=cfg.training.checkpoint_keep_last,
        total_steps=total_steps,
    )
    wandb_cb = WandbCallback(
        model_save_path=str(run_dir / "wandb_model"),
        model_save_freq=cfg.training.save_checkpoint_freq,
        gradient_save_freq=100,
        verbose=0,
    )

    # --- Train ---
    remaining = total_steps - start_step
    model.learn(
        total_timesteps=remaining,
        reset_num_timesteps=(start_step == 0),
        callback=CallbackList([ckpt_cb, wandb_cb]),
    )

    # --- Save final model ---
    final_path = run_dir / "model_final.zip"
    model.save(str(final_path))
    log.info("Saved final model to %s", final_path)

    artifact = wandb.Artifact("model", type="model")
    artifact.add_file(str(final_path))
    wandb.log_artifact(artifact)
    wandb.finish()


if __name__ == "__main__":
    train()
