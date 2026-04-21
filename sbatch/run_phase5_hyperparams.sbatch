#!/bin/bash
#SBATCH --job-name=cla_phase5_hyperparams
#SBATCH --partition=gpu_30d_p
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=5-00:00:00
#SBATCH --output=/scratch/%u/cla_runs/phase5_hyperparams_%j.out
#SBATCH --error=/scratch/%u/cla_runs/phase5_hyperparams_%j.err

# HEADS UP: hyperparams sweeps lambda {0.1, 0.5, 1.0, 5.0} x tau {0.05, 0.1, 0.5} = 12 full training+eval runs.
# Total wall is large (~50-100 GPU-hours). Only submit after Phase 4 results look right.

set -euo pipefail

source /home/hh89313/miniconda/etc/profile.d/conda.sh
conda activate cla
cd /home/hh89313/continous-latent-anchoring
set -a; source .env; set +a

export BASE_MODEL="Qwen/Qwen2.5-1.5B-Instruct"
export TRAIN_ANCHORS="/scratch/$USER/cla_anchors/gsm8k/train"
export EVAL_ANCHORS="/scratch/$USER/cla_anchors/gsm8k/test"
export RUN_ROOT="/scratch/$USER/cla_ablations"
export PYTHON_BIN="python"

mkdir -p "$RUN_ROOT"

echo "=== Phase 5: hyperparams ablation (12 runs) ==="
nvidia-smi | head -15
time bash scripts/run_ablations.sh hyperparams

echo "=== Phase 5 hyperparams complete ==="
ls -la "$RUN_ROOT/hparams"
