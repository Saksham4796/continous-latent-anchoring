#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen2.5-1.5B-Instruct}"
TEACHER_MODEL="${TEACHER_MODEL:-meta-llama/Meta-Llama-3-8B-Instruct}"
DEVICE="${DEVICE:-auto}"
TRAIN_ANCHORS="${TRAIN_ANCHORS:-$ROOT_DIR/artifacts/anchors/gsm8k_train}"
EVAL_ANCHORS="${EVAL_ANCHORS:-$ROOT_DIR/artifacts/anchors/gsm8k_test}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/artifacts/ablations}"

mkdir -p "$RUN_ROOT"

run_train() {
  local run_dir="$1"
  shift
  "$PYTHON_BIN" "$ROOT_DIR/scripts/train.py" \
    --train-anchors "$TRAIN_ANCHORS" \
    --eval-anchors "$EVAL_ANCHORS" \
    --base-model "$BASE_MODEL" \
    --device "$DEVICE" \
    --output-dir "$run_dir" \
    "$@"
}

run_eval() {
  local checkpoint="$1"
  shift
  "$PYTHON_BIN" "$ROOT_DIR/scripts/evaluate.py" \
    --base-model "$BASE_MODEL" \
    --device "$DEVICE" \
    --checkpoint "$checkpoint" \
    "$@"
}

mode="${1:-all}"

if [[ "$mode" == "all" || "$mode" == "k_sweep" ]]; then
  for k in 1 5 10 15 20; do
    run_dir="$RUN_ROOT/k_sweep/cla_k${k}"
    run_train "$run_dir" --ponder-steps "$k" --lambda-cla 1.0
    run_eval "$run_dir/final_projection.pt" --dataset gsm8k --split test --anchor-path "$EVAL_ANCHORS"

    baseline_dir="$RUN_ROOT/k_sweep/baseline_k${k}"
    run_train "$baseline_dir" --ponder-steps "$k" --lambda-cla 0.0
    run_eval "$baseline_dir/final_projection.pt" --dataset gsm8k --split test --anchor-path "$EVAL_ANCHORS"
  done
fi

if [[ "$mode" == "all" || "$mode" == "hyperparams" ]]; then
  for lambda_ in 0.1 0.5 1.0 5.0; do
    for tau in 0.05 0.1 0.5; do
      run_dir="$RUN_ROOT/hparams/lambda_${lambda_}_tau_${tau}"
      run_train "$run_dir" --ponder-steps 5 --lambda-cla "$lambda_" --cla-temperature "$tau"
      run_eval "$run_dir/final_projection.pt" --dataset gsm8k --split test --anchor-path "$EVAL_ANCHORS"
    done
  done
fi

if [[ "$mode" == "all" || "$mode" == "negatives" ]]; then
  for strategy in hard_semantic random_noise; do
    run_dir="$RUN_ROOT/negatives/${strategy}"
    run_train "$run_dir" --ponder-steps 5 --negative-strategy "$strategy"
    run_eval "$run_dir/final_projection.pt" --dataset gsm8k --split test --anchor-path "$EVAL_ANCHORS"
  done
fi

if [[ "$mode" == "all" || "$mode" == "cross_domain" ]]; then
  cross_dir="$RUN_ROOT/cross_domain/gsm8k_to_strategyqa"
  run_train "$cross_dir" --ponder-steps 5 --lambda-cla 1.0
  run_eval \
    "$cross_dir/final_projection.pt" \
    --dataset strategyqa \
    --split test \
    --anchor-path "$ROOT_DIR/artifacts/anchors/strategyqa_test"
fi

printf 'Ablation run complete. Outputs are under %s\n' "$RUN_ROOT"
