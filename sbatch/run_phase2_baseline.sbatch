#!/bin/bash
#SBATCH --job-name=cla_phase2_baseline
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/%u/cla_runs/phase2_baseline_%j.out
#SBATCH --error=/scratch/%u/cla_runs/phase2_baseline_%j.err

set -eu

source /home/hh89313/miniconda/etc/profile.d/conda.sh
conda activate cla
cd /home/hh89313/continous-latent-anchoring
set -a; source .env; set +a

mkdir -p /scratch/$USER/cla_runs
mkdir -p /scratch/$USER/cla_checkpoints/qwen15b_cocot_baseline

echo "=== Phase 3: Continuous-CoT baseline (lambda=0.0, k=5) ==="
nvidia-smi
time python scripts/train.py \
  --train-anchors /scratch/$USER/cla_anchors/gsm8k/train \
  --eval-anchors /scratch/$USER/cla_anchors/gsm8k/test \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --ponder-steps 5 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --learning-rate 5e-4 \
  --lambda-cla 0.0 \
  --output-dir /scratch/$USER/cla_checkpoints/qwen15b_cocot_baseline \
  --env-file .env \
  --log-file /scratch/$USER/cla_runs/phase2_baseline.log

echo "=== Phase 3 baseline complete ==="
ls -la /scratch/$USER/cla_checkpoints/qwen15b_cocot_baseline/
