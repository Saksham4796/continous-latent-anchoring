#!/bin/bash
#SBATCH --job-name=cla_phase4_baseline_gpt4o
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=40gb
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/%u/cla_runs/phase4_baseline_gpt4o_%j.out
#SBATCH --error=/scratch/%u/cla_runs/phase4_baseline_gpt4o_%j.err

set -eu

source /home/hh89313/miniconda/etc/profile.d/conda.sh
conda activate cla
cd /home/hh89313/continous-latent-anchoring
set -a; source .env; set +a

mkdir -p /scratch/$USER/cla_runs

echo "=== Phase 4 (GPT-4o arm): Baseline evaluation ==="
nvidia-smi
time python scripts/evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint /scratch/$USER/cla_checkpoints/qwen15b_cocot_baseline_gpt4o/final_projection.pt \
  --ponder-steps 5 \
  --dataset gsm8k \
  --split test \
  --anchor-path /scratch/$USER/cla_anchors/gsm8k_gpt4o/test \
  --env-file .env \
  --log-file /scratch/$USER/cla_runs/phase4_baseline_gpt4o.log

echo "=== Phase 4 baseline eval (GPT-4o arm) complete ==="
