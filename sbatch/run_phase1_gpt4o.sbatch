#!/bin/bash
#SBATCH --job-name=cla_phase1_gpt4o
#SBATCH --partition=gpu_p
#SBATCH --ntasks=1
#SBATCH --gres=gpu:L4:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24gb
#SBATCH --time=08:00:00
#SBATCH --output=/scratch/%u/cla_anchors/gsm8k_gpt4o/phase1_%j.out
#SBATCH --error=/scratch/%u/cla_anchors/gsm8k_gpt4o/phase1_%j.err

set -eu

source /home/hh89313/miniconda/etc/profile.d/conda.sh
conda activate cla
cd /home/hh89313/continous-latent-anchoring
set -a; source .env; set +a

mkdir -p /scratch/$USER/cla_anchors/gsm8k_gpt4o/train
mkdir -p /scratch/$USER/cla_anchors/gsm8k_gpt4o/test

echo "=== Phase 1 (GPT-4o teacher): GSM8K train (3000 samples) ==="
nvidia-smi
time python scripts/generate_anchors.py \
  --dataset gsm8k \
  --split train \
  --teacher-backend openai \
  --teacher-model gpt-4o \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /scratch/$USER/cla_anchors/gsm8k_gpt4o/train \
  --max-samples 3000 \
  --negative-strategy hard_semantic \
  --env-file .env \
  --log-file /scratch/$USER/cla_anchors/gsm8k_gpt4o/train/phase1_train.log

echo "=== Phase 1 (GPT-4o teacher): GSM8K test (500 samples) ==="
time python scripts/generate_anchors.py \
  --dataset gsm8k \
  --split test \
  --teacher-backend openai \
  --teacher-model gpt-4o \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir /scratch/$USER/cla_anchors/gsm8k_gpt4o/test \
  --max-samples 500 \
  --negative-strategy hard_semantic \
  --env-file .env \
  --log-file /scratch/$USER/cla_anchors/gsm8k_gpt4o/test/phase1_test.log

echo "=== Phase 1 (GPT-4o) complete ==="
ls -la /scratch/$USER/cla_anchors/gsm8k_gpt4o/train
ls -la /scratch/$USER/cla_anchors/gsm8k_gpt4o/test
