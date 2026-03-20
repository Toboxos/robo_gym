"""Tests for the SLURM deploy config resolution."""
from training_suite.deploy.slurm import resolve_experiment_config, resolve_sweep_configs


def test_resolve_experiment_config_returns_valid_config() -> None:
    """resolve_experiment_config should return a fully composed DictConfig."""
    cfg = resolve_experiment_config("ppo_example")

    assert cfg.training.total_timesteps > 0
    assert cfg.model.policy == "MlpPolicy"
    assert cfg.wandb.project == "robo_gym"
    assert cfg.seed is not None


def test_resolve_experiment_config_accepts_overrides() -> None:
    """Hydra overrides should be applied on top of the experiment config."""
    cfg = resolve_experiment_config("ppo_example", overrides=["training.total_timesteps=1234"])

    assert cfg.training.total_timesteps == 1234


def test_resolve_sweep_configs_produces_one_config_per_combination() -> None:
    """Sweep overrides with N values should yield N distinct configs."""
    configs = resolve_sweep_configs(
        "ppo_example",
        sweep_overrides=["model.learning_rate=1e-3,1e-4,3e-4"],
    )

    assert len(configs) == 3
    learning_rates = {cfg.model.learning_rate for cfg in configs}
    assert learning_rates == {1e-3, 1e-4, 3e-4}
