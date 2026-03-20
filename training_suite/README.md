# Training Suite

CLI for running, resuming, and deploying RL experiments on top of the robo_gym environment.
Uses [Stable Baselines3](https://stable-baselines3.readthedocs.io/) for training and [Hydra](https://hydra.cc/) for config composition.

## Setup

```bash
uv sync --extra training
```

---

## Config System

Experiments are composed from config groups under `config/`:

```
config/
├── config.yaml      # root defaults
├── experiment/      # named experiment presets  ← start here
├── model/           # algorithm + hyperparameters (ppo.yaml, ...)
├── robot/           # robot platform geometry and sensor layout
├── maze/            # maze source (static file or procedural generation)
├── reward/          # reward function definitions
├── wrappers/        # observation wrapper stacks
└── training/        # timesteps, eval freq, checkpoint freq
```

An experiment file selects config groups and applies overrides:

```yaml
# config/experiment/ppo_example.yaml
# @package _global_
defaults:
  - override /robot: basic_v1
  - override /model: ppo
  - override /maze: generated_6x6

model:
  policy_kwargs:
    net_arch: [256, 64, 256]

training:
  total_timesteps: 20_000_000
```

---

## CLI Reference

```bash
# Start a fresh run
uv run python train.py start ppo_example

# With ad-hoc Hydra overrides
uv run python train.py start ppo_example -- training.total_timesteps=500000

# Resume from the latest checkpoint
uv run python train.py resume <run_id>

# Resume from a specific checkpoint
uv run python train.py resume <run_id> --label ckpt_500k

# Branch: new run from an existing checkpoint with a different config
uv run python train.py branch <run_id> ppo_example

# Upload a checkpoint to W&B as a model artifact
uv run python train.py promote-checkpoint checkpoints/<run_id>/ckpt_500k.zip
uv run python train.py promote-checkpoint checkpoints/<run_id>/ckpt_500k.zip --alias best
```

---

## SLURM Deployment

The `deploy slurm` command packages an experiment into a self-contained bundle — no project checkout or Hydra installation needed on the cluster.

### Build

```bash
# Multi-file bundle directory
uv run python train.py deploy slurm ppo_example --out dist/ppo_example/

# Single self-extracting bash script
uv run python train.py deploy slurm ppo_example --single-file
```

Resource options (all have sensible defaults):

| Option | Default | Description |
|---|---|---|
| `--time` | `24:00:00` | Wall time |
| `--mem` | `16G` | Memory per node |
| `--gpus` | `1` | Number of GPUs |
| `--partition` | `gpu` | SLURM partition |

### Deploy

```bash
scp -r dist/ppo_example/ cluster:/scratch/$USER/runs/ppo_example/
ssh cluster
cd /scratch/$USER/runs/ppo_example/
sbatch job.sh
```

On first run `bootstrap.sh` installs `uv` (if not on PATH) and creates a shared venv under `$ROBO_GYM_BASE/.robo_gym/venvs/`. Subsequent jobs reuse it.

```bash
# Optional: point to a fast shared filesystem
export ROBO_GYM_BASE=/scratch/$USER
sbatch job.sh
```

### Sweeps

Add a `sweeps` key to your experiment config. Configs are resolved as a cartesian product at deploy time — `job.sh` gets `--array=0-N` automatically:

```yaml
# config/experiment/lr_sweep.yaml
sweeps:
  model.learning_rate: "1e-4,5e-4,1e-3"
  training.n_envs: "4,8"
# → 6 array jobs (3 × 2)
```

```bash
uv run python train.py deploy slurm lr_sweep --out dist/lr_sweep/
sbatch dist/lr_sweep/job.sh   # submits all 6 jobs
```
