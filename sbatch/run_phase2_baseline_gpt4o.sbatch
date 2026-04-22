#!/bin/bash
#SBATCH --job-name=cla_phase2_baseline_gpt4o
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/cla_runs/phase2_baseline_gpt4o_%j.out
#SBATCH --error=/scratch/%u/cla_runs/phase2_baseline_gpt4o_%j.err

set -eu

source /home/hh89313/miniconda/etc/profile.d/conda.sh
conda activate cla
cd /home/hh89313/continous-latent-anchoring
set -a; source .env; set +a

mkdir -p /scratch/$USER/cla_runs
mkdir -p /scratch/$USER/cla_checkpoints/qwen15b_cocot_baseline_gpt4o

echo "=== Phase 3 (GPT-4o anchors): CoCoT baseline (lambda=0.0, k=5) ==="
nvidia-smi
time python scripts/train.py \
  --train-anchors /scratch/$USER/cla_anchors/gsm8k_gpt4o/train \
  --eval-anchors /scratch/$USER/cla_anchors/gsm8k_gpt4o/test \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --ponder-steps 5 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --learning-rate 5e-4 \
  --lambda-cla 0.0 \
  --output-dir /scratch/$USER/cla_checkpoints/qwen15b_cocot_baseline_gpt4o \
  --env-file .env \
  --log-file /scratch/$USER/cla_runs/phase2_baseline_gpt4o.log

echo "=== Phase 3 baseline (GPT-4o) complete ==="
ls -la /scratch/$USER/cla_checkpoints/qwen15b_cocot_baseline_gpt4o/
