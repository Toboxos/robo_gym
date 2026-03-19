import logging
import wandb

from omegaconf import DictConfig
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import cast

from .checkpoint import save_checkpoint


logger = logging.getLogger(__name__)


class CheckpointCallback(BaseCallback):
    def __init__(
        self,
        cfg: DictConfig,
        checkpoint_dir: Path,
        save_freq: int,
        total_steps: int,
        verbose: int = 0,
    ) -> None:
        """Initialise the checkpoint callback.

        Args:
            cfg:                Full resolved Hydra config root.
            checkpoint_dir:    ``checkpoints/{wandb_run_id}/`` directory.
            save_freq:          Save every *save_freq* training steps.
            keep_last:          Maximum number of numbered checkpoints to retain.
            total_steps:        Total planned training steps (for schedule position).
            verbose:            SB3 verbosity level.
        """
        super().__init__(verbose)
        self._cfg = cfg
        self._checkpoint_dir = checkpoint_dir
        self._save_freq = save_freq
        self._total_steps = total_steps

    def _on_step(self) -> bool:
        if self.num_timesteps % self._save_freq == 0:
            label = f"ckpt_{self.num_timesteps // 1000:03d}k"
            save_checkpoint(
                self.model,
                self._cfg,
                self._checkpoint_dir,
                self.num_timesteps,
                self._total_steps,
                label,
            )
        return True


class UploadModelCallback(BaseCallback):
    """Saves and uploads a new best model to W&B when EvalCallback finds one."""

    def __init__(
            self, 
            run: wandb.Run,
            cfg: DictConfig, 
            checkpoint_dir: Path, 
            total_steps: int
    ) -> None:
        super().__init__()
        self.run = run
        self.cfg = cfg
        self.checkpoint_dir = checkpoint_dir
        self.total_steps = total_steps

    def _on_step(self) -> bool:
        zip_path = save_checkpoint(
            self.model, self.cfg, self.checkpoint_dir,
            step=self.num_timesteps,
            total_steps=self.total_steps,
            label="ckpt_best",
        )

        parent: EvalCallback | None = cast(EvalCallback | None, self.parent)
        assert parent is not None

        artifact = wandb.Artifact(
            name=f"model-{self.run.id}-best",
            type="model",
            metadata={"step": self.num_timesteps, "mean_reward": parent.best_mean_reward},
        )
        artifact.add_file(str(zip_path))
        self.run.log_artifact(artifact)
        
        return True


class TrainingMetricsCallback(BaseCallback):
    """
    Logs per-episode training metrics from the info dict to W&B.

    SB3's WandbCallback only tracks internal algorithm metrics (loss, entropy).
    This callback forwards the Level-0 env metrics — ``cells_visited_mean`` and
    ``collision_rate`` — to W&B at the end of each training episode.
    """

    def __init__(self, run: wandb.Run, verbose = 0):
        super().__init__(verbose)

        self.run = run

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])

        metrics: dict[str, list[float]] = {}
        for done, info in zip(dones, infos):
            if not done:
                continue
            for key in ("cells_visited_count", "cells_visited_mean",
                        "collision_count", "collision_rate"):
                metrics.setdefault(f"train/{key}", []).append(info.get(key, 0.0))

        if metrics:
            self.run.log(
                {k: sum(v) / len(v) for k, v in metrics.items()},
                step=self.num_timesteps,
            )
        return True


