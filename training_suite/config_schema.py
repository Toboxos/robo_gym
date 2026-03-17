"""Structured config schemas for Hydra validation.

These dataclasses mirror the YAML config groups and are registered with
Hydra's ConfigStore so that type errors and missing mandatory values are
caught at startup rather than mid-training.

Call :func:`register_schemas` once before ``@hydra.main`` runs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING


@dataclass
class TrainingConfig:
    """Schema for config/training/*.yaml files."""

    total_timesteps: int = MISSING  # MISSING → MissingMandatoryValue at startup
    n_envs: int = 4
    save_checkpoint_freq: int = 50_000
    checkpoint_keep_last: int = 3
    eval_freq: int = 25_000
    eval_episodes: int = 10
    eval_profile: str = "quick"
    record_trajectories: bool = True
    trajectory_flush_every: int = 100
    init_from: str | None = None
    init_weights_only: bool = True


@dataclass
class WrapperSpec:
    """Single wrapper entry in a wrappers config list."""

    type: str = MISSING
    params: dict = field(default_factory=dict)


@dataclass
class WrappersConfig:
    """Schema for config/wrappers/*.yaml files."""

    wrappers: list[WrapperSpec] = field(default_factory=list)
    temporal_mode: str = "none"


@dataclass
class EvalConfig:
    """Schema for config/eval/*.yaml files."""

    episodes: int = 10
    mazes: list[str] = field(default_factory=lambda: ["training"])
    render: bool = False
    report: bool = False


def register_schemas() -> None:
    """Register config schemas with Hydra's ConfigStore.

    Stores schemas under ``_base_`` names so that YAML configs can explicitly
    inherit from them via their own ``defaults`` list — the correct Hydra 1.1+
    pattern (avoids the deprecated automatic name-matching).

    Must be called before ``@hydra.main`` is invoked.
    """
    cs = ConfigStore.instance()
    cs.store(group="training", name="_base_training_", node=TrainingConfig)
    cs.store(group="eval",     name="_base_eval_",     node=EvalConfig)
