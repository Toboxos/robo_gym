"""Training lifecycle CLI for robo_gym.

Usage::

    uv run python train.py start ppo_example
    uv run python train.py start ppo_example -- training.total_timesteps=500000
    uv run python train.py resume <run_id>
    uv run python train.py branch <source_run_id> ppo_example
    uv run python train.py branch <source_run_id> ppo_example -- training.total_timesteps=2000000
    uv run python train.py promote-checkpoint runs/abc123/ckpt_050k.zip
    uv run python train.py promote-checkpoint runs/abc123/ckpt_050k.zip --alias best
"""
from __future__ import annotations

import click
import hydra
import logging
import sys

from pathlib import Path
from omegaconf import DictConfig, OmegaConf
from training_suite.trainer import train
from training_suite.checkpoint import find_latest_checkpoint, read_checkpoint_config


logger = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="./config", config_name="config")
def _hydra_main(cfg: DictConfig) -> None:
    """
    Wrapper function to use hydra for config creations
    """
    train(cfg)


@click.group()
@click.option("--verbose", "-v", is_flag=True, help="Enable debug logging.")
def cli(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(name)s: %(message)s")


@cli.command(context_settings={"allow_extra_args": True})
@click.argument("experiment")
@click.pass_context
def start(ctx: click.Context, experiment: str) -> None:
    """
    Start a fresh training run from EXPERIMENT config.

    Hydra overrides can be passed after '--':

        train.py start ppo_example -- training.total_timesteps=1000000
    """
    hydra_overrides: list[str] = list(ctx.args)
    click.echo(f"Starting fresh run: experiment={experiment}, overrides={hydra_overrides}")

    sys.argv = [sys.argv[0], f"+experiment={experiment}", *ctx.args]
    _hydra_main()


@cli.command()
@click.argument("run_id")
@click.option("--label", default=None, help="Label of the checkpoint without .zip extension")
def resume(run_id: str, label: str) -> None:
    """Resume training from a local checkpoint.

    RUN_ID is the W&B run ID (the name of the directory under runs/).
    Config is loaded from the saved checkpoint — no Hydra
    overrides are accepted to keep the resumed run config-identical.
    """
    checkpoint_dir = Path("checkpoints") / run_id
    if not checkpoint_dir.exists():
        raise click.ClickException(f"Checkpoint directory not found: {checkpoint_dir}")

    if label:
        checkpoint = (checkpoint_dir / f"{label}.zip").resolve()
    else:
        last_checkpoint = find_latest_checkpoint(checkpoint_dir)
        if not last_checkpoint:
            raise click.ClickException(f"Could not locate last checkpoint")
        
        checkpoint = last_checkpoint.resolve()

    cfg = read_checkpoint_config(checkpoint)
    click.echo(f"Resuming run {run_id} from {checkpoint}")
    train(cfg, run_id, checkpoint)


@cli.command("promote-checkpoint")
@click.argument("checkpoint_path", type=click.Path(exists=True, path_type=Path))
@click.option("--run-id", default=None, help="W&B run ID (inferred from checkpoint metadata if omitted).")
@click.option("--alias", default=None, help="Artifact alias to attach, e.g. 'best' or 'staging'.")
def promote_checkpoint(checkpoint_path: Path, run_id: str | None, alias: str | None) -> None:
    """Upload CHECKPOINT_PATH to W&B as a model artifact on the producing run.

    The W&B run ID is read from the checkpoint's .meta.json sidecar file.
    Pass --run-id to override if the sidecar is missing.
    """
    click.echo(f"Promoting checkpoint: {checkpoint_path}, run_id={run_id}, alias={alias}")
    raise NotImplementedError


if __name__ == "__main__":
    cli()
