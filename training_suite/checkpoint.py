from __future__ import annotations

import io
import json
import logging
import re
import pickle
from pathlib import Path
import zipfile

import wandb
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecNormalize
from omegaconf import DictConfig, OmegaConf


log = logging.getLogger(__name__)


class ConfigMismatch(Exception):
    """Raised when a checkpoint's embedded config does not match the current config."""


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------


def save_checkpoint(
    model: BaseAlgorithm,
    cfg: DictConfig,
    checkpoint_dir: Path,
    step: int,
    total_steps: int,
    label: str,
) -> Path:
    """Save a checkpoint to *checkpoint_dir* and return its path.

    Uses ``model.save()`` so that all SB3 internals (policy weights, optimizer
    state, hyperparameters) are serialised correctly.  The resolved config and
    run metadata are injected directly into the zip — no sidecar files, making
    each checkpoint fully self-contained.

    Zip contents:
        - SB3 native entries (``policy.pth``, ``pytorch_variables.pth``, …)
        - ``config.yaml``     — full resolved config for standalone loading
        - ``_metadata.json``  — step count, schedule position, W&B run ID

    Args:
        model:          SB3 algorithm instance (e.g. PPO).
        cfg:            Full resolved config root.
        checkpoint_dir: Directory to write the checkpoint into (created if absent).
        step:           Current training step count.
        total_steps:    Total planned steps (used to record schedule position).
        label:          File stem, e.g. ``"ckpt_050k"`` or ``"ckpt_best"``.

    Returns:
        Path to the saved ``.zip`` file.
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "step": step,
        "lr_schedule_pos": step / total_steps if total_steps > 0 else 0.0,
        "wandb_run_id": wandb.run.id if wandb.run is not None else None,
        "stage_name": cfg.get("stage_name", "default"),
    }

    buf = io.BytesIO()
    model.save(buf)
    buf.seek(0)

    with zipfile.ZipFile(buf, "a") as zf:
        zf.writestr("config.yaml", OmegaConf.to_yaml(cfg))
        zf.writestr("_metadata.json", json.dumps(meta, indent=2))

        if vec_norm := model.get_vec_normalize_env():
            zf.writestr("vec_norm.pkl", pickle.dumps(vec_norm))

    zip_path = checkpoint_dir / f"{label}.zip"
    zip_path.write_bytes(buf.getvalue())

    log.debug("Saved checkpoint %s at step %d", zip_path, step)
    return zip_path


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------


def read_checkpoint_config(checkpoint_path: Path) -> DictConfig:
    """Extract the embedded config from a checkpoint zip.

    Useful for standalone scripts that need to reconstruct the model without
    an external config file.

    Args:
        checkpoint_path: Path to a ``.zip`` saved by :func:`save_checkpoint`.

    Returns:
        The resolved config that was active when the checkpoint was saved.

    Raises:
        KeyError: If the checkpoint does not contain an embedded ``config.yaml``.
    """
    with zipfile.ZipFile(checkpoint_path) as zf:
        return OmegaConf.create(zf.read("config.yaml").decode()) # type: ignore


def load_checkpoint(
    checkpoint_path: Path,
    model,
    cfg: DictConfig,
) -> tuple[object, int]:
    """Fully restore a checkpoint and return the loaded model and step count.

    Uses SB3's ``model.load()`` so that all internal state (policy weights,
    optimizer state, hyperparameters, algorithm internals) is restored
    completely.  Validates that the embedded config matches *cfg* to catch
    accidental config drift before loading.

    Because ``model.load()`` constructs a new SB3 instance, the original
    *model* argument is not mutated — use the returned model instead.

    Args:
        checkpoint_path: Path to a ``.zip`` saved by :func:`save_checkpoint`.
        model:           Existing SB3 instance — used for its class and env only.
        cfg:             Current resolved config (used for drift check).

    Returns:
        Tuple of ``(restored_model, step)`` where *step* is the training step
        count at which the checkpoint was saved.

    Raises:
        ConfigMismatch: If the checkpoint's embedded config differs from *cfg*.
    """
    with zipfile.ZipFile(checkpoint_path) as zf:
        names = zf.namelist()
        meta: dict = json.loads(zf.read("_metadata.json")) if "_metadata.json" in names else {}
        saved_cfg = OmegaConf.create(zf.read("config.yaml").decode()) if "config.yaml" in names else None
        saved_vec_norm: VecNormalize | None = pickle.loads(zf.read("vec_norm.pkl")) if "vec_norm.pkl" in names else None

    if saved_cfg is not None and saved_cfg != cfg:
        raise ConfigMismatch(
            f"Config mismatch when resuming from {checkpoint_path}.\n"
            f"The config has changed since this checkpoint was saved.\n"
            f"Diff (saved ↔ current):\n{_cfg_diff(saved_cfg, cfg)}\n" # type: ignore
            "Use branch for intentional config changes, or restore the original config for crash-resume."
        )

    restored: BaseAlgorithm = type(model).load(str(checkpoint_path), env=model.get_env())
    step = int(meta.get("step", 0))
    restored.num_timesteps = step

    if saved_vec_norm:
        vec_norm = restored.get_vec_normalize_env()
        if vec_norm is None:
            raise RuntimeError("Model environment expected to be wrapped in VecNormalization")
        
        saved_vec_norm.set_venv(vec_norm.venv)

    log.info("Restored checkpoint %s at step %d", checkpoint_path, step)
    return restored, step


def _cfg_diff(saved: DictConfig, current: DictConfig) -> str:
    """Return a human-readable diff of two configs."""
    saved_lines = set(OmegaConf.to_yaml(saved).splitlines())
    current_lines = set(OmegaConf.to_yaml(current).splitlines())
    removed = "\n".join(f"  - {l}" for l in sorted(saved_lines - current_lines))
    added = "\n".join(f"  + {l}" for l in sorted(current_lines - saved_lines))
    return "\n".join(filter(None, [removed, added])) or "  (no textual diff)"


# ---------------------------------------------------------------------------
# Discovery
# ---------------------------------------------------------------------------


def find_latest_checkpoint(checkpoint_dir: Path) -> Path | None:
    """Scan *checkpoint_dir* for ``ckpt_*.zip`` files and return the one with the highest step.

    Ignores ``ckpt_best.zip`` and ``model_final.zip`` — only numbered checkpoints
    are considered for resume (best/final may be from a different point).

    Returns:
        Path to the latest numbered checkpoint, or ``None`` if none exist.
    """
    pattern = re.compile(r"ckpt_(\d+)k\.zip$")
    best: tuple[int, Path] | None = None

    for p in checkpoint_dir.glob("ckpt_*.zip"):
        m = pattern.match(p.name)
        if m:
            step = int(m.group(1)) * 1000
            if best is None or step > best[0]:
                best = (step, p)

    return best[1] if best is not None else None
