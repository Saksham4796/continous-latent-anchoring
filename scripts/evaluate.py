#!/usr/bin/env python3
"""CLI entry point for CLA accuracy, drift, and hardware evaluation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import AnchorTensorDataset, anchor_collate_fn, resolve_anchor_shards
from evaluation import evaluate_anchor_metrics, evaluate_generation_accuracy
from models import ContinuousCoTModel, load_model_adapter
from utils.env import load_environment
from utils.logging import configure_logging


def parse_args() -> argparse.Namespace:
    """Parse evaluation arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--ponder-steps", type=int, default=5)
    parser.add_argument("--dataset", default="gsm8k", choices=["gsm8k", "strategyqa"])
    parser.add_argument("--split", default="test")
    parser.add_argument("--dataset-path", default=None)
    parser.add_argument("--dataset-subset", default=None)
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--anchor-path", default=None)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--profile-repeats", type=int, default=10)
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-anchor-metrics", action="store_true")
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    """Load a trained CLA projection and run evaluation suites."""

    args = parse_args()
    env_path = load_environment(args.env_file)
    logger = configure_logging(log_file=args.log_file)
    if env_path is not None:
        logger.info("Loaded environment variables from %s", env_path)

    device = _resolve_device(args.device)
    adapter = load_model_adapter(
        model_name_or_alias=args.base_model,
        torch_dtype=args.dtype,
        device=device,
    )
    model = ContinuousCoTModel(adapter=adapter, ponder_steps=args.ponder_steps)
    model.to(device)

    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.ponder_projection.load_state_dict(checkpoint["trainable_state_dict"]["ponder_projection"])
        backbone_dtype = next(model.adapter.model.parameters()).dtype
        model.ponder_projection.to(dtype=backbone_dtype)
        if "ponder_steps" in checkpoint:
            model.ponder_steps = int(checkpoint["ponder_steps"])

    results = {}
    if not args.skip_generation:
        results.update(
            evaluate_generation_accuracy(
                model=model,
                dataset_name=args.dataset,
                split=args.split,
                max_samples=args.max_samples,
                cache_dir=args.cache_dir,
                dataset_path=args.dataset_path,
                dataset_subset=args.dataset_subset,
                max_new_tokens=args.max_new_tokens,
            )
        )

    if args.anchor_path and not args.skip_anchor_metrics:
        anchor_dataset = AnchorTensorDataset([str(path) for path in resolve_anchor_shards(args.anchor_path)])
        anchor_loader = DataLoader(
            anchor_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=anchor_collate_fn,
        )
        results.update(
            evaluate_anchor_metrics(
                model=model,
                dataloader=anchor_loader,
                profile_repeats=args.profile_repeats,
            )
        )

    logger.info("Evaluation results:\n%s", json.dumps(results, indent=2))


def _resolve_device(device_arg: str) -> str:
    """Resolve auto device selection for local execution."""

    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
