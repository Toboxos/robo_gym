import logging
import wandb

from omegaconf import DictConfig
from pathlib import Path
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from typing import cast

from .checkpoint import save_checkpoint
from .metrics import EPISODE_KEYS


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
    """Logs per-episode maze metrics to the SB3 logger (tensorboard + wandb).

    Reads from ``info["episode"]`` — populated by Monitor via ``info_keywords``
    — and calls ``self.logger.record_mean`` so values are averaged across all
    episodes that complete within one SB3 log interval.
    """

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        infos = self.locals.get("infos", [])
        for done, info in zip(dones, infos):
            if not done:
                continue
            for key in EPISODE_KEYS:
                if key in info:
                    self.logger.record_mean(f"train/{key}", info[key])
        return True


class MazeEvalCallback(EvalCallback):
    """EvalCallback extended to log per-episode maze metrics after each eval.

    Intercepts ``info["episode"]`` during evaluation via the step-level
    callback hook, then logs the per-key means through the SB3 logger once
    evaluation completes.
    """

    def __init__(self, *args, **kwargs) -> None:
        """Initialise; all arguments forwarded to EvalCallback."""
        super().__init__(*args, **kwargs)
        self._eval_episode_infos: list[dict] = []

    def _log_success_callback(self, locals_: dict, globals_: dict) -> None:
        """Capture episode infos during evaluate_policy iteration."""
        super()._log_success_callback(locals_, globals_)
        for done, info in zip(locals_.get("dones", []), locals_.get("infos", [])):
            if done:
                ep = info.get("episode")
                if ep is not None:
                    self._eval_episode_infos.append(ep)

    def _on_step(self) -> bool:
        self._eval_episode_infos.clear()
        result = super()._on_step()
        for ep in self._eval_episode_infos:
            for key in EPISODE_KEYS:
                if key in ep:
                    self.logger.record_mean(f"eval/{key}", ep[key])
        return result
