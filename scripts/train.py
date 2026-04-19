#!/usr/bin/env python3
"""CLI entry point for CLA training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from data.datasets import AnchorTensorDataset, anchor_collate_fn, resolve_anchor_shards
from models import ContinuousCoTModel, load_model_adapter
from training import CLATrainer
from utils.config import TrainingConfig
from utils.env import load_environment
from utils.logging import configure_logging
from utils.seed import seed_everything


def parse_args() -> argparse.Namespace:
    """Parse training arguments."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-anchors", required=True)
    parser.add_argument("--eval-anchors", default=None)
    parser.add_argument("--base-model", default="Qwen/Qwen2.5-1.5B-Instruct")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument("--ponder-steps", type=int, default=5)
    parser.add_argument("--projection-dropout", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--warmup-ratio", type=float, default=0.06)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--lambda-cla", type=float, default=1.0)
    parser.add_argument("--cla-temperature", type=float, default=0.1)
    parser.add_argument(
        "--negative-strategy",
        default="hard_semantic",
        choices=["hard_semantic", "random_noise"],
    )
    parser.add_argument("--output-dir", default="artifacts/checkpoints")
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--save-every", type=int, default=250)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-wandb", action="store_true")
    parser.add_argument("--wandb-project", default="cla-project")
    parser.add_argument("--wandb-run-name", default=None)
    parser.add_argument("--env-file", default=None)
    parser.add_argument("--log-file", default=None)
    return parser.parse_args()


def main() -> None:
    """Load offline anchors, train CLA, and save projection checkpoints."""

    args = parse_args()
    env_path = load_environment(args.env_file)
    logger = configure_logging(log_file=args.log_file)
    seed_everything(args.seed)
    if env_path is not None:
        logger.info("Loaded environment variables from %s", env_path)

    device = _resolve_device(args.device)
    adapter = load_model_adapter(
        model_name_or_alias=args.base_model,
        torch_dtype=args.dtype,
        device=device,
    )
    model = ContinuousCoTModel(
        adapter=adapter,
        ponder_steps=args.ponder_steps,
        projection_dropout=args.projection_dropout,
    )
    model.to(device)

    train_dataset = AnchorTensorDataset([str(path) for path in resolve_anchor_shards(args.train_anchors)])
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=anchor_collate_fn,
    )

    eval_loader = None
    if args.eval_anchors:
        eval_dataset = AnchorTensorDataset([str(path) for path in resolve_anchor_shards(args.eval_anchors)])
        eval_loader = DataLoader(
            eval_dataset,
            batch_size=args.eval_batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=anchor_collate_fn,
        )

    training_config = TrainingConfig(
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        grad_accum_steps=args.grad_accum_steps,
        max_grad_norm=args.max_grad_norm,
        lambda_cla=args.lambda_cla,
        cla_temperature=args.cla_temperature,
        amp_dtype=args.dtype,
        log_every_n_steps=args.log_every,
        eval_every_n_steps=args.eval_every,
        save_every_n_steps=args.save_every,
        seed=args.seed,
        num_workers=args.num_workers,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_run_name=args.wandb_run_name,
    )
    trainer = CLATrainer(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        config=training_config,
        negative_strategy=args.negative_strategy,
        logger=logger,
    )
    final_metrics = trainer.train()
    logger.info("Training complete. Final metrics: %s", final_metrics)


def _resolve_device(device_arg: str) -> str:
    """Resolve auto device selection for local execution."""

    if device_arg != "auto":
        return device_arg
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":
    main()
