"""Evaluation entry point for the robo_gym RL pipeline.

Usage::

    uv run python eval.py                                        # evaluate latest W&B artifact
    uv run python eval.py eval=full                              # full eval suite
    uv run python eval.py training.init_from=runs/abc/model_final.zip  # local model
"""
from __future__ import annotations

import logging

import hydra
import wandb
from omegaconf import DictConfig
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy

from training_suite import make_env, make_robot
from training_suite.config_schema import register_schemas

register_schemas()

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def evaluate(cfg: DictConfig) -> None:
    """Evaluate a trained PPO agent."""
    robot = make_robot(cfg.robot)

    # Resolve model path: prefer local init_from, fall back to latest W&B artifact.
    init_from = cfg.training.get("init_from")
    if init_from:
        model_file = str(init_from)
        log.info("Loading model from local path: %s", model_file)
    else:
        run = wandb.init(
            project=cfg.wandb.project,
            job_type="eval",
            config={"eval": cfg.eval},
        )
        assert run is not None
        artifact_path = f"{cfg.wandb.project}/model:latest"
        log.info("Downloading model artifact: %s", artifact_path)
        artifact = run.use_artifact(artifact_path)
        artifact_dir = artifact.download()
        model_file = f"{artifact_dir}/model_final"

    env = make_env(cfg, robot, seed=cfg.seed, render_mode="human" if cfg.eval.render else None)
    model = PPO.load(model_file, env=env)

    n_episodes = cfg.eval.episodes
    log.info("Starting evaluation (%d episodes)...", n_episodes)
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_episodes,
        deterministic=True,
        render=cfg.eval.render,
    )
    log.info("Result: mean_reward=%.2f +/- %.2f", mean_reward, std_reward)

    if init_from is None:
        wandb.finish()


if __name__ == "__main__":
    evaluate()
