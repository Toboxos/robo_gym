"""Checkpoint save/load helpers and SB3 callback for the training pipeline.

Run-directory layout::

    runs/{wandb_run_id}/
    ├── config_hash.txt
    ├── ckpt_050k.zip          ← SB3 native save (policy + optimizer + hyperparams)
    ├── ckpt_050k.meta.json    ← step count, cfg_hash, wandb_run_id
    ├── ckpt_100k.zip
    ├── ckpt_100k.meta.json
    ├── ckpt_best.zip
    ├── ckpt_best.meta.json
    └── model_final.zip
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import wandb
from omegaconf import DictConfig, OmegaConf
from stable_baselines3.common.callbacks import BaseCallback

log = logging.getLogger(__name__)


class ConfigHashMismatch(Exception):
    """Raised when a checkpoint's config hash does not match the current config."""


# ---------------------------------------------------------------------------
# Config hashing
# ---------------------------------------------------------------------------

def compute_cfg_hash(cfg: DictConfig) -> str:
    """Return the SHA-256 hex digest of the canonicalised YAML representation of cfg."""
    yaml_str = OmegaConf.to_yaml(cfg)
    return hashlib.sha256(yaml_str.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_checkpoint(
    model,
    cfg: DictConfig,
    run_dir: Path,
    step: int,
    total_steps: int,
    label: str,
) -> Path:
    """Save a checkpoint to *run_dir* using SB3's native format and return its path.

    Uses ``model.save()`` so that all SB3 internals (policy weights, optimizer
    state, hyperparameters) are serialised correctly.  A companion
    ``<label>.meta.json`` stores the step count and config hash needed for
    crash-resume and config-identity verification.

    Also writes ``config_hash.txt`` on the first call for this run_dir.

    Args:
        model:       SB3 algorithm instance (e.g. PPO).
        cfg:         Full resolved Hydra config root.
        run_dir:     ``runs/{wandb_run_id}/`` directory (must exist).
        step:        Current training step count.
        total_steps: Total planned steps (used to record schedule position).
        label:       File stem, e.g. ``"ckpt_050k"`` or ``"ckpt_best"``.

    Returns:
        Path to the saved ``.zip`` file.
    """
    run_dir.mkdir(parents=True, exist_ok=True)

    hash_file = run_dir / "config_hash.txt"
    if not hash_file.exists():
        hash_file.write_text(compute_cfg_hash(cfg))

    # SB3 native save — handles policy, optimizer, and all hyperparams.
    zip_path = run_dir / f"{label}.zip"
    model.save(str(zip_path))

    # Sidecar with metadata that SB3's save format doesn't include.
    meta = {
        "step": step,
        "lr_schedule_pos": step / total_steps if total_steps > 0 else 0.0,
        "cfg_hash": compute_cfg_hash(cfg),
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        "stage_name": cfg.get("stage_name", "default"),
    }
    (run_dir / f"{label}.meta.json").write_text(json.dumps(meta, indent=2))

    log.debug("Saved checkpoint %s at step %d", zip_path, step)
    return zip_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_checkpoint(
    checkpoint_path: Path,
    model,
    cfg: DictConfig,
    *,
    weights_only: bool = False,
) -> int:
    """Load a checkpoint into *model* and return the saved step count.

    Uses SB3's ``model.set_parameters()`` to restore policy weights and
    optimizer state so that SB3's internal counters stay consistent.  When
    ``weights_only=False`` (crash-resume), also restores ``num_timesteps`` on
    the model so that ``model.learn(reset_num_timesteps=False)`` continues
    from the right step.

    Args:
        checkpoint_path: Path to a ``.zip`` saved by :func:`save_checkpoint`.
        model:           SB3 algorithm instance to load weights into.
        cfg:             Current resolved Hydra config (used for hash check).
        weights_only:    When ``True``, only policy weights are restored (no
                         optimizer or step counter) — use for stage branching.
                         When ``False`` (crash-resume), also restores optimizer
                         state and raises :exc:`ConfigHashMismatch` if the
                         config has diverged.

    Returns:
        The ``step`` value from the checkpoint metadata (0 if weights_only).

    Raises:
        ConfigHashMismatch: If ``weights_only=False`` and the checkpoint's
            ``cfg_hash`` does not match ``compute_cfg_hash(cfg)``.
    """
    meta_path = checkpoint_path.parent / (checkpoint_path.stem + ".meta.json")
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    if not weights_only:
        current_hash = compute_cfg_hash(cfg)
        saved_hash = meta.get("cfg_hash", "")
        if saved_hash and saved_hash != current_hash:
            raise ConfigHashMismatch(
                f"Config hash mismatch when resuming from {checkpoint_path}.\n"
                f"  Saved:   {saved_hash}\n"
                f"  Current: {current_hash}\n"
                "The config has changed since this checkpoint was saved. "
                "Use init_from + init_weights_only=true for intentional branching, "
                "or restore the original config for crash-resume."
            )

    # set_parameters restores policy weights and (by default) optimizer state.
    model.set_parameters(str(checkpoint_path), exact_match=True)
    log.info("Loaded parameters from %s", checkpoint_path)

    if not weights_only:
        step = int(meta.get("step", 0))
        # Restore SB3's internal counter so learn(reset_num_timesteps=False) works.
        model.num_timesteps = step
        log.info("Restored num_timesteps=%d", step)
        return step

    return 0


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------

def find_latest_checkpoint(run_dir: Path) -> Path | None:
    """Scan *run_dir* for ``ckpt_*.zip`` files and return the one with the highest step.

    Ignores ``ckpt_best.zip`` and ``model_final.zip`` — only numbered checkpoints
    are considered for resume (best/final may be from a different point).

    Returns:
        Path to the latest numbered checkpoint, or ``None`` if none exist.
    """
    pattern = re.compile(r"ckpt_(\d+)k\.zip$")
    best: tuple[int, Path] | None = None

    for p in run_dir.glob("ckpt_*.zip"):
        m = pattern.match(p.name)
        if m:
            step = int(m.group(1)) * 1000
            if best is None or step > best[0]:
                best = (step, p)

    return best[1] if best is not None else None


# ---------------------------------------------------------------------------
# SB3 callback
# ---------------------------------------------------------------------------

class CheckpointCallback(BaseCallback):
    """SB3 callback that saves checkpoints at a fixed step frequency.

    Enforces the ``keep_last`` retention policy by pruning the oldest numbered
    checkpoints once the limit is exceeded. ``ckpt_best`` and ``model_final``
    are never pruned.
    """

    def __init__(
        self,
        cfg: DictConfig,
        run_dir: Path,
        save_freq: int,
        keep_last: int,
        total_steps: int,
        verbose: int = 0,
    ) -> None:
        """Initialise the checkpoint callback.

        Args:
            cfg:         Full resolved Hydra config root.
            run_dir:     ``runs/{wandb_run_id}/`` directory.
            save_freq:   Save every *save_freq* training steps.
            keep_last:   Maximum number of numbered checkpoints to retain.
            total_steps: Total planned training steps (for schedule position).
            verbose:     SB3 verbosity level.
        """
        super().__init__(verbose)
        self._cfg = cfg
        self._run_dir = run_dir
        self._save_freq = save_freq
        self._keep_last = keep_last
        self._total_steps = total_steps
        self._saved_paths: list[Path] = []

    def _on_step(self) -> bool:
        """Save a checkpoint every *save_freq* steps and prune old ones."""
        if self.num_timesteps % self._save_freq == 0:
            label = f"ckpt_{self.num_timesteps // 1000:03d}k"
            path = save_checkpoint(
                self.model,
                self._cfg,
                self._run_dir,
                self.num_timesteps,
                self._total_steps,
                label,
            )
            self._saved_paths.append(path)
            self._prune_old_checkpoints()
        return True

    def _prune_old_checkpoints(self) -> None:
        """Delete the oldest numbered checkpoints beyond the keep_last limit."""
        pattern = re.compile(r"ckpt_\d+k\.zip$")
        regular = [p for p in self._saved_paths if pattern.match(p.name)]
        while len(regular) > self._keep_last:
            oldest = regular.pop(0)
            if oldest.exists():
                oldest.unlink()
                log.debug("Pruned checkpoint %s", oldest)
            meta = oldest.parent / (oldest.stem + ".meta.json")
            if meta.exists():
                meta.unlink()
                log.debug("Pruned metadata %s", meta)
            if oldest in self._saved_paths:
                self._saved_paths.remove(oldest)
