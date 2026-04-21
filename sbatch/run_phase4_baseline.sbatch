#!/bin/bash
#SBATCH --job-name=cla_phase4_baseline
#SBATCH --partition=gpu_p
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/cla_runs/phase4_baseline_%j.out
#SBATCH --error=/scratch/%u/cla_runs/phase4_baseline_%j.err

set -euo pipefail

source /home/hh89313/miniconda/etc/profile.d/conda.sh
conda activate cla
cd /home/hh89313/continous-latent-anchoring
set -a; source .env; set +a

mkdir -p /scratch/$USER/cla_runs

echo "=== Phase 4: Baseline evaluation (accuracy + drift + hardware) ==="
nvidia-smi | head -15
time python scripts/evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint /scratch/$USER/cla_checkpoints/qwen15b_cocot_baseline/final_projection.pt \
  --ponder-steps 5 \
  --dataset gsm8k \
  --split test \
  --anchor-path /scratch/$USER/cla_anchors/gsm8k/test \
  --env-file .env \
  --log-file /scratch/$USER/cla_runs/phase4_baseline.log

echo "=== Phase 4 baseline eval complete ==="
