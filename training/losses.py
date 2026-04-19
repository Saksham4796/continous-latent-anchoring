"""Loss functions for CLA training."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from models.continuous_cot import ContinuousCoTOutput


@dataclass
class LossBreakdown:
    """Structured loss outputs for logging and checkpoint selection."""

    total: torch.Tensor
    distill: torch.Tensor
    cla: torch.Tensor


def distillation_loss(
    student_logits: torch.Tensor,
    labels: torch.Tensor,
    teacher_logits: Optional[torch.Tensor] = None,
    temperature: float = 1.0,
) -> torch.Tensor:
    """Compute distillation loss with optional teacher distributions.

    If `teacher_logits` are provided, the loss is the masked KL divergence:

        L_distill = T^2 * sum_i m_i * KL(p_teacher^T(i) || p_student^T(i)) / sum_i m_i

    where `m_i = 1` for supervised tokens (`labels != -100`) and `T` is the
    distillation temperature. If teacher logits are unavailable, the function
    falls back to standard masked token-level cross-entropy.
    """

    supervision_mask = labels.ne(-100)
    if teacher_logits is not None:
        scaled_student = F.log_softmax(student_logits / temperature, dim=-1)
        scaled_teacher = F.softmax(teacher_logits / temperature, dim=-1)
        kl_terms = F.kl_div(scaled_student, scaled_teacher, reduction="none").sum(dim=-1)
        masked_kl = kl_terms * supervision_mask.float()
        normalizer = supervision_mask.sum().clamp_min(1)
        return masked_kl.sum() * (temperature ** 2) / normalizer

    vocab_size = student_logits.size(-1)
    return F.cross_entropy(
        student_logits.reshape(-1, vocab_size),
        labels.reshape(-1),
        ignore_index=-100,
    )


def info_nce_loss(
    student_states: torch.Tensor,
    positive_anchors: torch.Tensor,
    negative_anchors: torch.Tensor,
    temperature: float = 0.1,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute the CLA InfoNCE objective.

    For normalized student state `h_t`, positive anchor `z+`, and negative
    anchor `z-`, the stepwise loss is:

        L_CLA_t = -log(
            exp(cos(h_t, z+) / tau) /
            (exp(cos(h_t, z+) / tau) + exp(cos(h_t, z-) / tau))
        )

    The function accepts `student_states` in shape `(B, K, D)` and anchor tensors
    in shape `(B, D)`, returning either the mean scalar loss or unreduced
    per-sample/per-step losses depending on `reduction`.
    """

    normalized_student = F.normalize(student_states, dim=-1)
    normalized_positive = F.normalize(positive_anchors, dim=-1).unsqueeze(1)
    normalized_negative = F.normalize(negative_anchors, dim=-1).unsqueeze(1)

    positive_logits = (normalized_student * normalized_positive).sum(dim=-1) / temperature
    negative_logits = (normalized_student * normalized_negative).sum(dim=-1) / temperature

    stacked_logits = torch.stack([positive_logits, negative_logits], dim=-1)
    losses = torch.logsumexp(stacked_logits, dim=-1) - positive_logits

    if reduction == "none":
        return losses
    if reduction == "sum":
        return losses.sum()
    return losses.mean()


def compute_total_loss(
    model_output: ContinuousCoTOutput,
    labels: torch.Tensor,
    positive_anchors: torch.Tensor,
    negative_anchors: torch.Tensor,
    lambda_cla: float,
    cla_temperature: float,
    teacher_logits: Optional[torch.Tensor] = None,
    distill_temperature: float = 1.0,
) -> LossBreakdown:
    """Combine the final-state distillation loss with multi-step CLA loss."""

    distill = distillation_loss(
        student_logits=model_output.final_logits,
        labels=labels,
        teacher_logits=teacher_logits,
        temperature=distill_temperature,
    )
    cla_per_step = info_nce_loss(
        student_states=model_output.pooled_step_states,
        positive_anchors=positive_anchors,
        negative_anchors=negative_anchors,
        temperature=cla_temperature,
        reduction="none",
    )
    cla = cla_per_step.sum(dim=1).mean()
    total = distill + lambda_cla * cla
    return LossBreakdown(total=total, distill=distill, cla=cla)
