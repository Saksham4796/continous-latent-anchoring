"""Continuous chain-of-thought latent recurrence for frozen SLM backbones."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from models.adapters import BaseModelAdapter


@dataclass
class ContinuousCoTOutput:
    """Outputs produced by the continuous pondering loop."""

    base_hidden_state: torch.Tensor
    step_hidden_states: torch.Tensor
    pooled_step_states: torch.Tensor
    final_logits: torch.Tensor
    step_logits: Optional[torch.Tensor] = None


class ContinuousCoTModel(nn.Module):
    """Frozen-backbone continuous CoT model with a trainable latent projector."""

    def __init__(
        self,
        adapter: BaseModelAdapter,
        ponder_steps: int,
        projection_dropout: float = 0.0,
        freeze_backbone: bool = True,
        use_step_residual: bool = True,
    ) -> None:
        super().__init__()
        self.adapter = adapter
        self.ponder_steps = int(ponder_steps)
        self.use_step_residual = use_step_residual

        if freeze_backbone:
            self.adapter.freeze_backbone()

        hidden_size = self.adapter.hidden_size
        self.ponder_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.projection_dropout = nn.Dropout(projection_dropout)
        # Near-zero init keeps step 1 close to h(0) when use_step_residual=True
        # (identity init doubled the hidden state into the next backbone call,
        # which saturates bfloat16 at larger ponder_steps).
        nn.init.normal_(self.ponder_projection.weight, mean=0.0, std=0.02)
        backbone_dtype = next(self.adapter.model.parameters()).dtype
        self.ponder_projection.to(dtype=backbone_dtype)

    @property
    def device(self) -> torch.device:
        """Return the device of the trainable projection."""

        return self.ponder_projection.weight.device

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        response_mask: Optional[torch.Tensor] = None,
        output_step_logits: bool = False,
    ) -> ContinuousCoTOutput:
        """Run the frozen backbone followed by k latent pondering recurrences."""

        base_outputs = self.adapter.forward_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        hidden_state = base_outputs.hidden_states[-1]

        step_hidden_states = []
        step_logits = []
        for _ in range(self.ponder_steps):
            projected_state = self.projection_dropout(self.ponder_projection(hidden_state))
            if self.use_step_residual:
                projected_state = projected_state + hidden_state
            recurrent_outputs = self.adapter.forward_backbone(
                inputs_embeds=projected_state,
                attention_mask=attention_mask,
            )
            hidden_state = recurrent_outputs.hidden_states[-1]
            step_hidden_states.append(hidden_state)
            if output_step_logits:
                step_logits.append(self.adapter.logits_from_hidden(hidden_state))

        stacked_hidden_states = torch.stack(step_hidden_states, dim=1)
        pooled_mask = response_mask if response_mask is not None else attention_mask.bool()
        pooled_step_states = masked_mean(stacked_hidden_states, pooled_mask)
        final_logits = self.adapter.logits_from_hidden(hidden_state)
        stacked_logits = torch.stack(step_logits, dim=1) if step_logits else None

        return ContinuousCoTOutput(
            base_hidden_state=base_outputs.hidden_states[-1],
            step_hidden_states=stacked_hidden_states,
            pooled_step_states=pooled_step_states,
            final_logits=final_logits,
            step_logits=stacked_logits,
        )

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 64,
        eos_token_id: Optional[int] = None,
        temperature: float = 0.0,
    ) -> torch.Tensor:
        """Greedy or temperature-based autoregressive decoding with latent pondering."""

        generated_ids = input_ids
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        for _ in range(max_new_tokens):
            outputs = self.forward(
                input_ids=generated_ids,
                attention_mask=attention_mask,
                response_mask=None,
                output_step_logits=False,
            )
            next_token_logits = outputs.final_logits[:, -1, :]
            if temperature > 0.0:
                probabilities = torch.softmax(next_token_logits / temperature, dim=-1)
                next_token = torch.multinomial(probabilities, num_samples=1)
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)

            generated_ids = torch.cat([generated_ids, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)

            if eos_token_id is not None and bool(torch.all(next_token.squeeze(-1) == eos_token_id)):
                break
        return generated_ids

    def trainable_parameters(self):
        """Expose only the parameters optimized by CLA training."""

        return self.ponder_projection.parameters()


def masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Pool hidden states with a boolean mask.

    For a batch of stepwise hidden states `H in R^{B x K x S x D}` and a token mask
    `M in {0,1}^{B x S}`, this function computes:

        pooled[b, k, d] = sum_s M[b, s] * H[b, k, s, d] / max(sum_s M[b, s], 1)

    The same formula is used for `H in R^{B x S x D}` by omitting the `K` axis.
    """

    if hidden_states.dim() == 4:
        expanded_mask = mask.unsqueeze(1).unsqueeze(-1).float()
        summed = (hidden_states * expanded_mask).sum(dim=2)
        counts = expanded_mask.sum(dim=2).clamp_min(1.0)
        return summed / counts

    expanded_mask = mask.unsqueeze(-1).float()
    summed = (hidden_states * expanded_mask).sum(dim=1)
    counts = expanded_mask.sum(dim=1).clamp_min(1.0)
    return summed / counts
