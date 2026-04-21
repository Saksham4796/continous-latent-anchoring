#!/bin/bash
#SBATCH --job-name=cla_phase1_gsm8k
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:A100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80gb
#SBATCH --time=06:00:00
#SBATCH --output=/scratch/%u/cla_anchors/gsm8k/phase1_%j.out
#SBATCH --error=/scratch/%u/cla_anchors/gsm8k/phase1_%j.err

set -euo pipefail

source /home/hh89313/miniconda/etc/profile.d/conda.sh
conda activate cla
cd /home/hh89313/continous-latent-anchoring
set -a; source .env; set +a

mkdir -p /scratch/$USER/cla_anchors/gsm8k/train
mkdir -p /scratch/$USER/cla_anchors/gsm8k/test

echo "=== Phase 1: GSM8K train (1000 samples) ==="
nvidia-smi | head -15
time python scripts/generate_anchors.py \
  --dataset gsm8k \
  --split train \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /scratch/$USER/cla_anchors/gsm8k/train \
  --max-samples 1000 \
  --negative-strategy hard_semantic \
  --env-file .env \
  --log-file /scratch/$USER/cla_anchors/gsm8k/train/phase1_train.log

echo "=== Phase 1: GSM8K test (250 samples) ==="
time python scripts/generate_anchors.py \
  --dataset gsm8k \
  --split test \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /scratch/$USER/cla_anchors/gsm8k/test \
  --max-samples 250 \
  --negative-strategy hard_semantic \
  --env-file .env \
  --log-file /scratch/$USER/cla_anchors/gsm8k/test/phase1_test.log

echo "=== Phase 1 complete ==="
ls -la /scratch/$USER/cla_anchors/gsm8k/train
ls -la /scratch/$USER/cla_anchors/gsm8k/test
