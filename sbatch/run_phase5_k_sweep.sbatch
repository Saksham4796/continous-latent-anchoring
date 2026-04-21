#!/bin/bash
#SBATCH --job-name=cla_phase5_k_sweep
#SBATCH --partition=gpu_30d_p
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=4-00:00:00
#SBATCH --output=/scratch/%u/cla_runs/phase5_k_sweep_%j.out
#SBATCH --error=/scratch/%u/cla_runs/phase5_k_sweep_%j.err

# HEADS UP: k_sweep runs 10 full training+eval cycles (k in {1,5,10,15,20} x {CLA, baseline}).
# Total wall is large (~40-80 GPU-hours). Only submit after Phase 4 results confirm the pipeline
# gives expected trends. Uses gpu_30d_p with 4-day walltime.

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

echo "=== Phase 5: k_sweep ablation (10 runs) ==="
nvidia-smi | head -15
time bash scripts/run_ablations.sh k_sweep

echo "=== Phase 5 k_sweep complete ==="
ls -la "$RUN_ROOT/k_sweep"
