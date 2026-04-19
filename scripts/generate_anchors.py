#!/usr/bin/env python3
"""CLI entry point for offline CLA anchor generation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.anchor_generation import AnchorGenerationConfig, ContrastiveAnchorGenerator
from utils.env import load_environment
from utils.logging import configure_logging
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the anchor generation pipeline."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "strategyqa"])
    parser.add_argument("--split", default="train")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--teacher-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    parser.add_argument(
        "--teacher-backend",
        default="local",
        choices=["local", "openai"],
    )
    parser.add_argument("--anchor-encoder-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-length", type=int, default=768)
    parser.add_argument("--max-new-tokens", type=int, default=192)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--shard-size", type=int, default=256)
    parser.add_argument(
        "--negative-strategy",
        default="hard_semantic",
        choices=["hard_semantic", "random_noise"],
    )
    parser.add_argument("--teacher-dtype", default="bfloat16")
    parser.add_argument("--encoder-dtype", default="bfloat16")
    parser.add_argument("--teacher-device", default="auto")
    parser.add_argument("--encoder-device", default="auto")
    parser.add_argument("--teacher-timeout-seconds", type=float, default=120.0)
    parser.add_argument("--teacher-max-retries", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    """Run offline teacher-trajectory generation and anchor extraction."""

    args = parse_args()
    env_path = load_environment(args.env_file)
    logger = configure_logging(log_file=args.log_file)
    seed_everything(args.seed)
    if env_path is not None:
        logger.info("Loaded environment variables from %s", env_path)

    config = AnchorGenerationConfig(
        dataset_name=args.dataset,
        split=args.split,
        output_dir=args.output_dir,
        teacher_model_name=args.teacher_model,
        teacher_backend=args.teacher_backend,
        anchor_encoder_model_name=args.anchor_encoder_model,
        cache_dir=args.cache_dir,
        dataset_path=args.dataset_path,
        dataset_subset=args.dataset_subset,
        max_samples=args.max_samples,
        max_length=args.max_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        shard_size=args.shard_size,
        negative_strategy=args.negative_strategy,
        teacher_dtype=args.teacher_dtype,
        encoder_dtype=args.encoder_dtype,
        teacher_device=args.teacher_device,
        encoder_device=args.encoder_device,
        teacher_timeout_seconds=args.teacher_timeout_seconds,
        teacher_max_retries=args.teacher_max_retries,
        seed=args.seed,
    )

    generator = ContrastiveAnchorGenerator(config)
    saved_paths = generator.generate()
    logger.info("Saved %s anchor shard(s) under %s", len(saved_paths), args.output_dir)
    for path in saved_paths:
        logger.info("anchor_shard=%s", path)


if __name__ == "__main__":
    main()
