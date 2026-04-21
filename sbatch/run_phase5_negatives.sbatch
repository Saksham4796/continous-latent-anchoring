#!/bin/bash
#SBATCH --job-name=cla_phase5_negatives
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=24:00:00
#SBATCH --output=/scratch/%u/cla_runs/phase5_negatives_%j.out
#SBATCH --error=/scratch/%u/cla_runs/phase5_negatives_%j.err

set -eu

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

echo "=== Phase 5: negatives ablation (hard_semantic vs random_noise) ==="
nvidia-smi
time bash scripts/run_ablations.sh negatives

echo "=== Phase 5 negatives ablation complete ==="
ls -la "$RUN_ROOT/negatives"
