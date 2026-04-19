"""Explicit latent drift and stability metrics for CLA."""

from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F

from models.continuous_cot import masked_mean


def latent_norm_over_time(step_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Measure the mean hidden-state L2 norm at each pondering step."""

    token_norms = torch.linalg.norm(step_hidden_states, dim=-1)
    expanded_mask = attention_mask.unsqueeze(1).float()
    numerator = (token_norms * expanded_mask).sum(dim=(0, 2))
    denominator = expanded_mask.sum(dim=(0, 2)).clamp_min(1.0)
    return numerator / denominator


def latent_dispersion_over_time(step_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Measure batch dispersion of hidden states as pondering depth increases."""

    pooled = masked_mean(step_hidden_states, attention_mask.bool())
    centered = pooled - pooled.mean(dim=0, keepdim=True)
    return centered.pow(2).mean(dim=(0, 2))


def teacher_alignment_over_time(
    pooled_step_states: torch.Tensor,
    positive_anchors: torch.Tensor,
) -> torch.Tensor:
    """Compute cosine alignment between student latent states and positive anchors."""

    normalized_steps = F.normalize(pooled_step_states, dim=-1)
    normalized_anchors = F.normalize(positive_anchors, dim=-1).unsqueeze(1)
    return (normalized_steps * normalized_anchors).sum(dim=-1).mean(dim=0)


def answer_instability(step_logits: torch.Tensor, answer_mask: torch.Tensor) -> torch.Tensor:
    """Measure answer flips between adjacent pondering steps.

    The metric computes greedy answer predictions at each step over the supervised
    answer span and returns the fraction of samples whose answer tokens change
    between step `t` and `t + 1`.
    """

    predicted_tokens = step_logits.argmax(dim=-1)
    pairwise_changes = predicted_tokens[:, 1:, :] != predicted_tokens[:, :-1, :]
    masked_changes = pairwise_changes & answer_mask.unsqueeze(1)
    return masked_changes.any(dim=-1).float().mean(dim=0)


def summarize_drift_metrics(
    step_hidden_states: torch.Tensor,
    pooled_step_states: torch.Tensor,
    step_logits: torch.Tensor,
    attention_mask: torch.Tensor,
    answer_mask: torch.Tensor,
    positive_anchors: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    """Bundle the full set of drift metrics used in CLA experiments."""

    return {
        "latent_norm": latent_norm_over_time(step_hidden_states, attention_mask),
        "dispersion": latent_dispersion_over_time(step_hidden_states, attention_mask),
        "teacher_alignment": teacher_alignment_over_time(pooled_step_states, positive_anchors),
        "answer_instability": answer_instability(step_logits, answer_mask),
    }
