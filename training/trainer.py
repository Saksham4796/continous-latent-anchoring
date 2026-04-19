"""PyTorch trainer for Contrastive Latent Anchoring."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup

from models import ContinuousCoTModel
from models.adapters import resolve_torch_dtype
from training.losses import LossBreakdown, compute_total_loss
from utils.config import TrainingConfig


class CLATrainer:
    """Trainer that optimizes only the latent pondering projection."""

    def __init__(
        self,
        model: ContinuousCoTModel,
        train_loader: DataLoader,
        config: TrainingConfig,
        eval_loader: Optional[DataLoader] = None,
        negative_strategy: str = "hard_semantic",
        logger: Optional[Any] = None,
    ) -> None:
        self.model = model
        self.train_loader = train_loader
        self.eval_loader = eval_loader
        self.config = config
        self.negative_strategy = negative_strategy
        self.logger = logger

        trainable_parameters = list(self.model.trainable_parameters())
        self.optimizer = AdamW(
            trainable_parameters,
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay,
        )

        update_steps_per_epoch = max(1, math.ceil(len(train_loader) / self.config.grad_accum_steps))
        total_training_steps = max(1, update_steps_per_epoch * self.config.num_epochs)
        warmup_steps = int(total_training_steps * self.config.warmup_ratio)
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_training_steps,
        )

        self.autocast_dtype = resolve_torch_dtype(self.config.amp_dtype)
        self.autocast_enabled = self.autocast_dtype is not None
        scaler_enabled = (
            self.model.device.type == "cuda"
            and self.autocast_dtype == torch.float16
        )
        self.scaler = torch.cuda.amp.GradScaler(enabled=scaler_enabled)
        self.global_step = 0

        self._wandb = None
        if self.config.use_wandb:
            import wandb

            self._wandb = wandb
            self._wandb.init(
                project=self.config.wandb_project,
                name=self.config.wandb_run_name,
                config=asdict(self.config),
            )

    def train(self) -> Dict[str, float]:
        """Run the full training loop."""

        history: Dict[str, float] = {}
        self.optimizer.zero_grad(set_to_none=True)

        for epoch in range(self.config.num_epochs):
            self.model.train()
            for step, batch in enumerate(self.train_loader, start=1):
                batch = self._move_batch(batch)
                losses = self._forward_loss(batch)
                scaled_loss = losses.total / self.config.grad_accum_steps

                if self.scaler.is_enabled():
                    self.scaler.scale(scaled_loss).backward()
                else:
                    scaled_loss.backward()

                should_step = step % self.config.grad_accum_steps == 0 or step == len(self.train_loader)
                if should_step:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.trainable_parameters()),
                        self.config.max_grad_norm,
                    )

                    if self.scaler.is_enabled():
                        self.scaler.step(self.optimizer)
                        self.scaler.update()
                    else:
                        self.optimizer.step()

                    self.optimizer.zero_grad(set_to_none=True)
                    self.scheduler.step()
                    self.global_step += 1

                    history = {
                        "train/total_loss": float(losses.total.detach().cpu()),
                        "train/distill_loss": float(losses.distill.detach().cpu()),
                        "train/cla_loss": float(losses.cla.detach().cpu()),
                        "train/lr": float(self.scheduler.get_last_lr()[0]),
                    }
                    if self.global_step == 1 or self.global_step % self.config.log_every_n_steps == 0:
                        self._log_metrics(history)

                    if self.eval_loader and self.global_step % self.config.eval_every_n_steps == 0:
                        eval_metrics = self.evaluate()
                        history.update(eval_metrics)
                        self._log_metrics(eval_metrics)

                    if self.global_step % self.config.save_every_n_steps == 0:
                        self.save_checkpoint(Path(self.config.output_dir) / f"step_{self.global_step:06d}.pt")

            if self.eval_loader:
                eval_metrics = self.evaluate()
                history.update(eval_metrics)
                self._log_metrics(eval_metrics)

        self.save_checkpoint(Path(self.config.output_dir) / "final_projection.pt")
        return history

    @torch.no_grad()
    def evaluate(self) -> Dict[str, float]:
        """Evaluate mean total, distillation, and CLA losses."""

        self.model.eval()
        total_loss = 0.0
        total_distill = 0.0
        total_cla = 0.0
        total_batches = 0

        for batch in self.eval_loader or []:
            batch = self._move_batch(batch)
            losses = self._forward_loss(batch)
            total_loss += float(losses.total.detach().cpu())
            total_distill += float(losses.distill.detach().cpu())
            total_cla += float(losses.cla.detach().cpu())
            total_batches += 1

        self.model.train()
        if total_batches == 0:
            return {"eval/total_loss": 0.0, "eval/distill_loss": 0.0, "eval/cla_loss": 0.0}

        return {
            "eval/total_loss": total_loss / total_batches,
            "eval/distill_loss": total_distill / total_batches,
            "eval/cla_loss": total_cla / total_batches,
        }

    def save_checkpoint(self, path: Path) -> None:
        """Save only the learned CLA module and minimal metadata."""

        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "base_model_name": self.model.adapter.model_name,
            "ponder_steps": self.model.ponder_steps,
            "trainable_state_dict": {
                "ponder_projection": self.model.ponder_projection.state_dict(),
            },
            "training_config": asdict(self.config),
            "global_step": self.global_step,
        }
        torch.save(payload, path)

    def _forward_loss(self, batch: Mapping[str, Any]) -> LossBreakdown:
        """Run the model forward pass and compute the full CLA objective."""

        autocast_context = self._autocast_context()
        with autocast_context:
            model_output = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                response_mask=batch["response_mask"],
                output_step_logits=False,
            )
            negative_anchor = self._select_negative_anchor(batch)
            return compute_total_loss(
                model_output=model_output,
                labels=batch["labels"],
                positive_anchors=batch["positive_anchor"],
                negative_anchors=negative_anchor,
                lambda_cla=self.config.lambda_cla,
                cla_temperature=self.config.cla_temperature,
                teacher_logits=batch.get("teacher_logits"),
            )

    def _select_negative_anchor(self, batch: Mapping[str, Any]) -> torch.Tensor:
        """Optionally replace semantic negatives with random noise anchors."""

        if self.negative_strategy == "random_noise":
            noise = torch.randn_like(batch["positive_anchor"])
            return F.normalize(noise, dim=-1)
        return batch["negative_anchor"]

    def _autocast_context(self):
        """Return a device-appropriate autocast context manager."""

        if not self.autocast_enabled:
            return nullcontext()
        return torch.autocast(device_type=self.model.device.type, dtype=self.autocast_dtype)

    def _move_batch(self, batch: Mapping[str, Any]) -> Dict[str, Any]:
        """Move tensor entries of a batch onto the model device."""

        moved: Dict[str, Any] = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                moved[key] = value.to(self.model.device)
            else:
                moved[key] = value
        return moved

    def _log_metrics(self, metrics: Mapping[str, float]) -> None:
        """Emit metrics to the logger and to Weights & Biases."""

        if self.logger:
            metrics_str = ", ".join(f"{key}={value:.4f}" for key, value in metrics.items())
            self.logger.info("step=%s | %s", self.global_step, metrics_str)
        if self._wandb is not None:
            self._wandb.log(dict(metrics), step=self.global_step)
