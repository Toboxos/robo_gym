"""SLURM deployment bundle assembly for robo_gym training jobs."""
from __future__ import annotations

import base64
import io
import logging
import shutil
import stat
import subprocess
import tarfile
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from .templates import BOOTSTRAP_SH, JOB_SH, RUN_PY, SELF_EXTRACTING_WRAPPER

log = logging.getLogger(__name__)

_UV_LINUX_URL = (
    "https://github.com/astral-sh/uv/releases/latest/download/"
    "uv-x86_64-unknown-linux-gnu.tar.gz"
)

# Project root is two levels up from this file (training_suite/deploy/slurm.py)
_PROJECT_ROOT = Path(__file__).parent.parent.parent


@dataclass
class SlurmOptions:
    """SLURM job resource parameters."""

    time: str = "24:00:00"
    mem: str = "16G"
    gpus: int = 1
    partition: str = "gpu"


@dataclass
class DeployConfig:
    """Full deploy invocation parameters."""

    experiment: str
    out_dir: Path
    slurm: SlurmOptions = field(default_factory=SlurmOptions)
    single_file: bool = False


def resolve_experiment_config(
    experiment: str,
    overrides: list[str] | None = None,
) -> DictConfig:
    """Resolve a Hydra experiment config without running training.

    Uses the Hydra compose API so that the config is fully resolved
    (interpolations expanded, defaults merged) and can be serialised with
    OmegaConf.to_yaml(resolve=True) to a plain static YAML file.

    Args:
        experiment: Experiment name matching config/experiment/<name>.yaml.
        overrides:  Optional Hydra override strings, e.g. ["training.total_timesteps=1000000"].

    Returns:
        Fully resolved DictConfig.
    """
    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(_PROJECT_ROOT / "config")
    all_overrides = [f"+experiment={experiment}", *(overrides or [])]

    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="config", overrides=all_overrides)
    finally:
        GlobalHydra.instance().clear()

    return cfg


def resolve_sweep_configs(
    experiment: str,
    sweep_overrides: list[str],
) -> list[DictConfig]:
    """Resolve all configs produced by a Hydra sweep.

    Uses Hydra's BasicSweeper to enumerate the cartesian product of sweep
    override values, then resolves each combination into a fully merged DictConfig.

    Example::

        configs = resolve_sweep_configs(
            "ppo_example",
            sweep_overrides=["model.learning_rate=1e-3,1e-4", "training.n_envs=4,8"],
        )
        # returns 4 configs (2 × 2 cartesian product)

    Args:
        experiment:      Experiment name matching config/experiment/<name>.yaml.
        sweep_overrides: Hydra override strings with comma-separated sweep values.

    Returns:
        List of fully resolved DictConfigs, one per sweep combination.
    """
    from hydra._internal.core_plugins.basic_sweeper import BasicSweeper
    from hydra.core.override_parser.overrides_parser import OverridesParser

    parser = OverridesParser.create()
    parsed = parser.parse_overrides(sweep_overrides)
    # split_arguments returns List[List[List[str]]]: outer = batches, inner = jobs per batch.
    # With max_batch_size=None all jobs land in a single batch, so we flatten one level.
    batches = BasicSweeper.split_arguments(parsed, max_batch_size=None)
    jobs: list[list[str]] = [job for batch in batches for job in batch]

    return [resolve_experiment_config(experiment, overrides=job) for job in jobs]


def download_uv_binary(dest: Path) -> None:
    """Download the Linux x86_64 uv binary to *dest*.

    Args:
        dest: Target file path (e.g. bundle_dir / "uv").
    """
    log.info("Downloading uv binary from %s", _UV_LINUX_URL)
    buf = io.BytesIO()
    with urllib.request.urlopen(_UV_LINUX_URL) as resp:
        buf.write(resp.read())
    buf.seek(0)

    with tarfile.open(fileobj=buf, mode="r:gz") as tar:
        for member in tar.getmembers():
            if member.name.endswith("/uv") or member.name == "uv":
                f = tar.extractfile(member)
                if f is None:
                    raise RuntimeError("Could not extract uv binary from tarball")
                dest.write_bytes(f.read())
                dest.chmod(dest.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
                log.info("uv binary written to %s", dest)
                return

    raise RuntimeError("uv binary not found in downloaded tarball")


def build_wheel(project_root: Path, dist_dir: Path) -> Path:
    """Run `uv build` and return the path to the produced .whl file.

    Args:
        project_root: Root of the robo_gym project (contains pyproject.toml).
        dist_dir:     Directory where the wheel should land.

    Returns:
        Path to the .whl file.

    Raises:
        RuntimeError: If uv build fails or no .whl is produced.
    """
    log.info("Building wheel...")
    result = subprocess.run(
        ["uv", "build", "--wheel", "--out-dir", str(dist_dir)],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"uv build failed:\n{result.stderr}")

    wheels = list(dist_dir.glob("robo_gym-*.whl"))
    if not wheels:
        raise RuntimeError(f"No robo_gym wheel found in {dist_dir} after uv build")

    log.info("Wheel built: %s", wheels[0])
    return wheels[0]


def assemble_bundle(cfg: DeployConfig, resolved_cfgs: list[DictConfig]) -> Path:
    """Assemble all bundle files into cfg.out_dir.

    Creates the output directory and writes:
      - robo_gym-*.whl        (built via uv build)
      - uv                    (downloaded Linux x86_64 binary)
      - uv.lock               (copied from project root)
      - pyproject.toml        (copied — needed by uv sync --project)
      - configs/0.yaml        (one file per resolved config; N files for a sweep)
      - run.py                (minimal entry point)
      - bootstrap.sh          (chmod 755)
      - job.sh                (chmod 755, --array set automatically for sweeps)

    Args:
        cfg:           Deploy configuration.
        resolved_cfgs: Pre-resolved OmegaConf DictConfigs (one per sweep combination).

    Returns:
        Path to the output directory.
    """
    out = cfg.out_dir
    out.mkdir(parents=True, exist_ok=True)
    log.info("Assembling bundle in %s", out)

    # --- wheel ---
    build_wheel(_PROJECT_ROOT, out)

    # --- uv binary ---
    download_uv_binary(out / "uv")

    # --- lock + pyproject ---
    shutil.copy(_PROJECT_ROOT / "uv.lock", out / "uv.lock")
    shutil.copy(_PROJECT_ROOT / "pyproject.toml", out / "pyproject.toml")

    # --- resolved configs ---
    configs_dir = out / "configs"
    configs_dir.mkdir()
    for i, resolved_cfg in enumerate(resolved_cfgs):
        yaml_str = OmegaConf.to_yaml(resolved_cfg, resolve=True)
        _write_unix(configs_dir / f"{i}.yaml", yaml_str)
    log.info("Wrote %d config(s) to %s", len(resolved_cfgs), configs_dir)

    # --- run.py ---
    _write_unix(out / "run.py", RUN_PY)

    # --- bootstrap.sh ---
    bootstrap = out / "bootstrap.sh"
    _write_unix(bootstrap, BOOTSTRAP_SH)
    _make_executable(bootstrap)

    # --- job.sh ---
    # Array directive is derived from the number of configs; omitted for single runs.
    n = len(resolved_cfgs)
    array_line = f"#SBATCH --array=0-{n - 1}" if n > 1 else ""
    job = out / "job.sh"
    _write_unix(job, JOB_SH.format(
        experiment=cfg.experiment,
        time=cfg.slurm.time,
        mem=cfg.slurm.mem,
        gpus=cfg.slurm.gpus,
        partition=cfg.slurm.partition,
        array_line=array_line,
    ))
    _make_executable(job)

    log.info("Bundle assembled: %s", out)
    return out


def pack_single_file(bundle_dir: Path, cfg: DeployConfig) -> Path:
    """Pack bundle_dir into a single self-extracting bash script.

    All files except job.sh are tar-gzipped in memory, base64-encoded,
    and appended after an exit 0 / __ARCHIVE_BELOW__ marker so that bash
    never interprets the binary payload. The result is a valid sbatch script.

    Args:
        bundle_dir: Directory produced by assemble_bundle().
        cfg:        Deploy configuration (experiment name, SLURM params).

    Returns:
        Path to the produced job_<experiment>.sh file (sibling of bundle_dir).
    """
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for item in sorted(bundle_dir.iterdir()):
            if item.name == "job.sh":
                continue
            tar.add(item, arcname=item.name, recursive=True)
    buf.seek(0)

    payload = base64.b64encode(buf.read()).decode("ascii")
    configs = list((bundle_dir / "configs").iterdir())
    n = len(configs)
    array_line = f"#SBATCH --array=0-{n - 1}" if n > 1 else ""

    script = SELF_EXTRACTING_WRAPPER.format(
        experiment=cfg.experiment,
        timestamp=datetime.now(timezone.utc).isoformat(),
        time=cfg.slurm.time,
        mem=cfg.slurm.mem,
        gpus=cfg.slurm.gpus,
        partition=cfg.slurm.partition,
        array_line=array_line,
        base64_payload=payload,
    )

    out_path = bundle_dir.parent / f"job_{cfg.experiment}.sh"
    _write_unix(out_path, script)
    _make_executable(out_path)
    log.info("Single-file script written to %s", out_path)
    return out_path


def deploy_slurm(cfg: DeployConfig) -> None:
    """Orchestrate the full SLURM deploy flow.

    Resolves the experiment config on the dev machine. If the experiment config
    contains a ``sweeps`` key, all combinations are enumerated and stored as
    configs/0.yaml … configs/N.yaml with an automatic --array directive in
    job.sh. The ``sweeps`` key is stripped from each serialised config.

    Sweep format in the experiment YAML::

        sweeps:
          model.learning_rate: "1e-3,1e-4,3e-4"
          training.n_envs: "4,8"

    Args:
        cfg: DeployConfig with all parameters.
    """
    log.info("Resolving config for experiment '%s'", cfg.experiment)
    base_cfg = resolve_experiment_config(cfg.experiment)

    if "sweeps" in base_cfg:
        sweep_overrides = [
            f"{key}={val}" for key, val in base_cfg.sweeps.items()
        ]
        log.info("Sweep overrides from config: %s", sweep_overrides)
        resolved_cfgs = resolve_sweep_configs(cfg.experiment, sweep_overrides)
        log.info("Sweep produced %d configs", len(resolved_cfgs))
    else:
        resolved_cfgs = [base_cfg]

    # Strip the deploy-only `sweeps` key before writing configs to disk.
    clean_cfgs = [_strip_sweeps(c) for c in resolved_cfgs]

    bundle_dir = assemble_bundle(cfg, clean_cfgs)

    if cfg.single_file:
        out = pack_single_file(bundle_dir, cfg)
        log.info("Single-file bundle: %s", out)
    else:
        log.info("Bundle directory: %s", bundle_dir)


def _strip_sweeps(cfg: DictConfig) -> DictConfig:
    """Return a copy of cfg with the deploy-only ``sweeps`` key removed."""
    container = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(container, dict)
    container.pop("sweeps", None)
    return OmegaConf.create(container)


def _write_unix(path: Path, text: str) -> None:
    """Write text to path with Unix line endings regardless of host platform."""
    path.write_bytes(text.replace("\r\n", "\n").encode("utf-8"))


def _make_executable(path: Path) -> None:
    """Add execute permission bits to a file."""
    path.chmod(path.stat().st_mode | stat.S_IEXEC | stat.S_IXGRP | stat.S_IXOTH)
