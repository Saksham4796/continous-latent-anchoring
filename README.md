# Contrastive Latent Anchoring (CLA)

PyTorch and Hugging Face codebase for **Contrastive Latent Anchoring (CLA): Stabilizing Continuous Chain-of-Thought in Small Language Models**.

The repository implements:

- Offline teacher-guided positive and negative anchor generation for `gsm8k` and `strategyqa`
- Continuous latent pondering on frozen small language models with a trainable `W_ponder`
- Dual-objective training with token-level distillation and stepwise InfoNCE latent anchoring
- Drift, alignment, instability, and edge-efficiency profiling utilities
- Reproducible CLI scripts for training, evaluation, and reviewer-requested ablations

## Repository Layout

```text
cla_project/
├── data/
├── evaluation/
├── models/
├── scripts/
├── training/
├── utils/
├── requirements.txt
└── README.md
```

## Design Notes

The teacher model can be larger than the student model, so hidden sizes may not match. To keep the contrastive objective well-defined, this implementation uses the teacher to generate positive and negative **reasoning trajectories**, then re-encodes those trajectories with the target student backbone during offline anchor generation. The stored `z+` and `z-` anchors therefore live in the student's latent space and can be compared directly to pondering states.

The pondering update is implemented as a frozen-backbone latent recurrence:

```text
h(0) = FrozenBackbone(x)
h(t) = FrozenBackbone(W_ponder · h(t-1) + h(t-1))
```

This keeps the code architecture-agnostic across Qwen, Llama, and Phi families while preserving the continuous latent-loop behavior required by CLA.

## Setup

```bash
cd cla_project
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env
```

## `.env` Configuration

The scripts automatically load `cla_project/.env` before any Hugging Face model, dataset, or Weights & Biases call.

Minimal example:

```dotenv
HF_TOKEN=hf_your_token_here
WANDB_API_KEY=your_wandb_key_here
OPENAI_API_KEY=your_openai_compatible_key_here
OPENAI_API_URL=http://localhost:3000/v1
```

If you prefer a different file name, every CLI also supports `--env-file /path/to/file.env`.

Important: a token is necessary for gated repositories, but it is not sufficient by itself. For `meta-llama/Meta-Llama-3-8B-Instruct`, your Hugging Face account must also have explicit access to that model repository.

## Phase 1: Offline Anchor Generation

Generate teacher-guided positive and negative anchors for GSM8K with a local Hugging Face teacher:

```bash
python3 scripts/generate_anchors.py \
  --dataset gsm8k \
  --split train \
  --teacher-model meta-llama/Meta-Llama-3-8B-Instruct \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir artifacts/anchors/gsm8k_train_qwen15b \
  --max-samples 1000 \
  --negative-strategy hard_semantic
```

If you do not have access to the Meta Llama repository, switch the teacher to an ungated model, for example:

```bash
python3 scripts/generate_anchors.py \
  --dataset gsm8k \
  --split train \
  --teacher-model Qwen/Qwen2.5-7B-Instruct \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir artifacts/anchors/gsm8k_train_qwen15b \
  --max-samples 1000 \
  --negative-strategy hard_semantic
```

Use a remote OpenAI-compatible teacher such as `gpt-4o` while keeping the anchor encoder local:

```bash
python3 scripts/generate_anchors.py \
  --dataset gsm8k \
  --split train \
  --teacher-backend openai \
  --teacher-model gpt-4o \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir artifacts/anchors/gsm8k_train_qwen15b \
  --max-samples 1000 \
  --negative-strategy hard_semantic
```

In this setup, only the teacher generation is remote. The anchor encoder, latent anchor extraction, student training, and all evaluation remain local.

Generate evaluation anchors:

```bash
python3 scripts/generate_anchors.py \
  --dataset gsm8k \
  --split test \
  --teacher-model meta-llama/Meta-Llama-3-8B-Instruct \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir artifacts/anchors/gsm8k_test_qwen15b \
  --max-samples 250 \
  --negative-strategy hard_semantic
```

Generate StrategyQA anchors:

```bash
python3 scripts/generate_anchors.py \
  --dataset strategyqa \
  --split train \
  --teacher-model meta-llama/Meta-Llama-3-8B-Instruct \
  --anchor-encoder-model Qwen/Qwen2.5-1.5B-Instruct \
  --output-dir artifacts/anchors/strategyqa_train_qwen15b \
  --negative-strategy hard_semantic
```

## Phase 2 and Phase 3: CLA Training

Train a CLA-enabled Qwen student:

```bash
python3 scripts/train.py \
  --train-anchors artifacts/anchors/gsm8k_train_qwen15b \
  --eval-anchors artifacts/anchors/gsm8k_test_qwen15b \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --ponder-steps 5 \
  --batch-size 2 \
  --grad-accum-steps 8 \
  --learning-rate 5e-4 \
  --lambda-cla 1.0 \
  --cla-temperature 0.1 \
  --output-dir artifacts/checkpoints/qwen15b_cla
```

Train a continuous-CoT baseline without CLA:

```bash
python3 scripts/train.py \
  --train-anchors artifacts/anchors/gsm8k_train_qwen15b \
  --eval-anchors artifacts/anchors/gsm8k_test_qwen15b \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --ponder-steps 5 \
  --lambda-cla 0.0 \
  --output-dir artifacts/checkpoints/qwen15b_cocot_baseline
```

Enable Weights & Biases logging:

```bash
python3 scripts/train.py \
  --train-anchors artifacts/anchors/gsm8k_train_qwen15b \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --ponder-steps 5 \
  --use-wandb \
  --wandb-project cla-project \
  --wandb-run-name qwen15b_cla_gsm8k
```

## Phase 4: Evaluation, Drift Metrics, and Hardware Profiling

Evaluate exact-match accuracy and anchor-based drift metrics:

```bash
python3 scripts/evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint artifacts/checkpoints/qwen15b_cla/final_projection.pt \
  --ponder-steps 5 \
  --dataset gsm8k \
  --split test \
  --anchor-path artifacts/anchors/gsm8k_test_qwen15b
```

Run only drift and hardware metrics:

```bash
python3 scripts/evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint artifacts/checkpoints/qwen15b_cla/final_projection.pt \
  --ponder-steps 5 \
  --anchor-path artifacts/anchors/gsm8k_test_qwen15b \
  --skip-generation
```

Run only final-answer accuracy:

```bash
python3 scripts/evaluate.py \
  --base-model Qwen/Qwen2.5-1.5B-Instruct \
  --checkpoint artifacts/checkpoints/qwen15b_cla/final_projection.pt \
  --ponder-steps 5 \
  --dataset gsm8k \
  --split test \
  --skip-anchor-metrics
```

## Phase 5: Reviewer Ablations

The ablation runner supports:

- `k_sweep`: compare baseline continuous CoT vs CLA over `k in [1, 5, 10, 15, 20]`
- `hyperparams`: sweep `lambda in [0.1, 0.5, 1.0, 5.0]` and `tau in [0.05, 0.1, 0.5]`
- `negatives`: compare `hard_semantic` vs `random_noise` negatives
- `cross_domain`: train on GSM8K and evaluate zero-shot on StrategyQA
- `all`: run the full suite

Example:

```bash
bash scripts/run_ablations.sh k_sweep
bash scripts/run_ablations.sh hyperparams
bash scripts/run_ablations.sh negatives
bash scripts/run_ablations.sh cross_domain
```

The script is controlled via environment variables when you need to override paths or models:

```bash
BASE_MODEL=Qwen/Qwen2.5-1.5B-Instruct \
TRAIN_ANCHORS=artifacts/anchors/gsm8k_train_qwen15b \
EVAL_ANCHORS=artifacts/anchors/gsm8k_test_qwen15b \
bash scripts/run_ablations.sh all
```

## Supported Student Models

- `Qwen/Qwen2.5-1.5B-Instruct`
- `meta-llama/Llama-3.2-1B-Instruct`
- `microsoft/Phi-3.5-mini-instruct`

You can swap the student model by changing `--base-model` during training and using the same model as `--anchor-encoder-model` during Phase 1.

## Key Implementation Files

- `data/anchor_generation.py`: teacher reasoning, hard negatives, latent anchor extraction, shard writing
- `models/continuous_cot.py`: frozen-backbone latent recurrence and autoregressive decoding
- `training/losses.py`: distillation and InfoNCE CLA losses
- `training/trainer.py`: mixed-precision trainer for `W_ponder`
- `evaluation/metrics.py`: norm, dispersion, alignment, and instability metrics
- `utils/profiler.py`: latency, memory, and NVML power sampling

## Outputs

Training checkpoints save only the learned projection module and metadata:

```text
artifacts/checkpoints/<run_name>/final_projection.pt
```

Anchor shards are stored as `.pt` files containing tokenized supervision and latent vectors:

```text
artifacts/anchors/<dataset>_<split>_anchors_0000.pt
```

## Practical Notes

- Meta Llama teacher models may require Hugging Face access approval.
- OpenAI-compatible teacher generation currently uses the `/chat/completions` API shape.
- `pynvml`-based energy profiling is optional. If NVML is unavailable, latency and memory are still reported.
- The default prompts ask the model to finish with `Final answer: ...`, which is what the answer parser expects during evaluation.
