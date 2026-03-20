"""Bash and Python script templates for SLURM deployment bundles."""

BOOTSTRAP_SH = """\
#!/usr/bin/env bash
# bootstrap.sh — idempotent shared venv setup using bundled uv.
# Prints the path to the venv directory on stdout.
#
# ROBO_GYM_BASE controls where the shared venv and uv cache are stored.
# Set it to a fast shared filesystem directory before calling sbatch, e.g.:
#   export ROBO_GYM_BASE=/scratch/$USER
# Defaults to $HOME if not set.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Resolve uv: prefer bundled binary, then PATH, then download.
if [ -f "$SCRIPT_DIR/uv" ]; then
  UV="$SCRIPT_DIR/uv"
  chmod +x "$UV"
elif command -v uv &>/dev/null; then
  UV="$(command -v uv)"
  echo "[bootstrap] Using uv from PATH: $UV" >&2
else
  echo "[bootstrap] Downloading uv..." >&2
  ROBO_GYM_BASE="${ROBO_GYM_BASE:-$HOME}"
  UV_BIN_DIR="$ROBO_GYM_BASE/.robo_gym/bin"
  mkdir -p "$UV_BIN_DIR"
  curl -LsSf https://astral.sh/uv/install.sh | UV_INSTALL_DIR="$UV_BIN_DIR" sh
  UV="$UV_BIN_DIR/uv"
fi

ROBO_GYM_BASE="${ROBO_GYM_BASE:-$HOME}"
WHEEL=$(ls "$SCRIPT_DIR"/robo_gym-*.whl | head -1)
WHEEL_HASH=$(sha256sum "$WHEEL" | awk '{print $1}' | head -c 8)
VENV_DIR="$ROBO_GYM_BASE/.robo_gym/venvs/robo_gym-$WHEEL_HASH"
LOCK_FILE="/tmp/robo_gym_setup_$WHEEL_HASH.lock"

(
  flock -x 200
  if [ ! -d "$VENV_DIR" ]; then
    echo "[bootstrap] Creating venv at $VENV_DIR" >&2
    mkdir -p "$(dirname "$VENV_DIR")"
    "$UV" venv --python python3 "$VENV_DIR"
    # Step 1: install all dependencies from the lock file (skip the project itself)
    UV_PROJECT_ENVIRONMENT="$VENV_DIR" \\
    UV_CACHE_DIR="$ROBO_GYM_BASE/.robo_gym/uv-cache" \\
      "$UV" sync \\
        --frozen \\
        --extra training \\
        --no-install-project \\
        --project "$SCRIPT_DIR"
    # Step 2: install the pre-built wheel (deps already handled above)
    "$UV" pip install \\
      --python "$VENV_DIR" \\
      --find-links "$SCRIPT_DIR" \\
      --no-index \\
      --no-deps \\
      "robo_gym[training]"
  else
    echo "[bootstrap] Reusing venv at $VENV_DIR" >&2
  fi
) 200>"$LOCK_FILE"

echo "$VENV_DIR"
"""

JOB_SH = """\
#!/usr/bin/env bash
#SBATCH --job-name=robo_gym_{experiment}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --gpus={gpus}
#SBATCH --partition={partition}
{array_line}
#SBATCH --output=%x-%j.out

set -euo pipefail

# SLURM copies the script to its spool dir before execution, so BASH_SOURCE[0]
# would point there rather than the bundle directory. SLURM_SUBMIT_DIR is always
# the directory from which sbatch was called — i.e. the bundle directory.
SCRIPT_DIR="${{SLURM_SUBMIT_DIR:-$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)}}"
VENV_DIR=$("$SCRIPT_DIR/bootstrap.sh")

export CHECKPOINT_DIR="$SCRIPT_DIR/checkpoints/${{SLURM_JOB_ID}}_${{SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$CHECKPOINT_DIR"

"$VENV_DIR/bin/python" "$SCRIPT_DIR/run.py"
"""

RUN_PY = '''\
"""Minimal cluster entry point — loads pre-resolved config and starts training."""
import os
from pathlib import Path
from omegaconf import OmegaConf, DictConfig
from training_suite.trainer import train

_idx = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
cfg: DictConfig = OmegaConf.load(Path(__file__).parent / "configs" / f"{_idx}.yaml")
train(cfg)
'''

SELF_EXTRACTING_WRAPPER = """\
#!/usr/bin/env bash
# Self-extracting SLURM job script for robo_gym experiment: {experiment}
# Generated: {timestamp}
#
# ROBO_GYM_BASE controls where the shared venv and uv cache land.
# Defaults to $HOME. Set to a fast shared filesystem path if available:
#   export ROBO_GYM_BASE=/scratch/$USER && sbatch job_{experiment}.sh
#
#SBATCH --job-name=robo_gym_{experiment}
#SBATCH --time={time}
#SBATCH --mem={mem}
#SBATCH --gpus={gpus}
#SBATCH --partition={partition}
{array_line}
#SBATCH --output=%x-%j.out

set -euo pipefail

WORK_DIR="${{SLURM_SUBMIT_DIR:-$PWD}}/runs/${{SLURM_JOB_ID:-local_$$}}_${{SLURM_ARRAY_TASK_ID:-0}}"
mkdir -p "$WORK_DIR"

echo "[deploy] Extracting bundle to $WORK_DIR" >&2
awk '/^__ARCHIVE_BELOW__/{{found=1; next}} found' "$0" | base64 -d | tar -xz -C "$WORK_DIR"

cd "$WORK_DIR"
VENV_DIR=$(./bootstrap.sh)

export CHECKPOINT_DIR="$WORK_DIR/checkpoints"
mkdir -p "$CHECKPOINT_DIR"

"$VENV_DIR/bin/python" ./run.py
exit 0

__ARCHIVE_BELOW__
{base64_payload}
"""
