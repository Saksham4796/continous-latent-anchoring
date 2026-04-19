"""Evaluation helpers for CLA training runs and ablations."""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional

import torch
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from data.datasets import (
    build_teacher_prompt,
    canonicalize_final_answer,
    extract_final_answer_from_response,
    load_reasoning_dataset,
)
from evaluation.metrics import summarize_drift_metrics
from models import ContinuousCoTModel
from utils.profiler import profile_callable


@torch.no_grad()
def evaluate_generation_accuracy(
    model: ContinuousCoTModel,
    dataset_name: str,
    split: str,
    max_samples: Optional[int] = None,
    cache_dir: Optional[str] = None,
    dataset_path: Optional[str] = None,
    dataset_subset: Optional[str] = None,
    max_new_tokens: int = 64,
) -> Dict[str, float]:
    """Evaluate exact-match final-answer accuracy with autoregressive decoding."""

    dataset = load_reasoning_dataset(
        dataset_name=dataset_name,
        split=split,
        cache_dir=cache_dir,
        max_samples=max_samples,
        dataset_path=dataset_path,
        dataset_subset=dataset_subset,
    )
    tokenizer = model.adapter.tokenizer
    model.eval()

    exact_match_total = 0.0
    for example in tqdm(dataset, desc=f"Evaluating {dataset_name}:{split}"):
        prompt_text = build_teacher_prompt(example)
        encoded = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
        )
        encoded = {key: value.to(model.device) for key, value in encoded.items()}
        generated = model.generate(
            input_ids=encoded["input_ids"],
            attention_mask=encoded["attention_mask"],
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
        )
        response_ids = generated[:, encoded["input_ids"].shape[1] :]
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)
        prediction = extract_final_answer_from_response(response_text, dataset_name)
        target = canonicalize_final_answer(example["target_text"], dataset_name)
        exact_match_total += float(prediction == target)

    if len(dataset) == 0:
        return {"accuracy/exact_match": 0.0}
    return {"accuracy/exact_match": exact_match_total / len(dataset)}


@torch.no_grad()
def evaluate_anchor_metrics(
    model: ContinuousCoTModel,
    dataloader: DataLoader,
    profile_repeats: int = 10,
    run_hardware_profile: bool = True,
) -> Dict[str, Any]:
    """Evaluate drift metrics over a tokenized anchor dataset."""

    model.eval()
    latent_norm = []
    dispersion = []
    teacher_alignment = []
    instability = []
    profile_result = None

    for batch in dataloader:
        batch = _move_batch(model, batch)
        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            response_mask=batch["response_mask"],
            output_step_logits=True,
        )
        drift = summarize_drift_metrics(
            step_hidden_states=outputs.step_hidden_states,
            pooled_step_states=outputs.pooled_step_states,
            step_logits=outputs.step_logits,
            attention_mask=batch["attention_mask"],
            answer_mask=batch["labels"].ne(-100),
            positive_anchors=batch["positive_anchor"],
        )
        latent_norm.append(drift["latent_norm"].detach().cpu())
        dispersion.append(drift["dispersion"].detach().cpu())
        teacher_alignment.append(drift["teacher_alignment"].detach().cpu())
        instability.append(drift["answer_instability"].detach().cpu())

        if profile_result is None and run_hardware_profile:
            profile_result = profile_callable(
                lambda: model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    response_mask=batch["response_mask"],
                    output_step_logits=False,
                ),
                device=model.device,
                repeats=profile_repeats,
            )

    metrics: Dict[str, Any] = {
        "drift/latent_norm": torch.stack(latent_norm).mean(dim=0).tolist() if latent_norm else [],
        "drift/dispersion": torch.stack(dispersion).mean(dim=0).tolist() if dispersion else [],
        "drift/teacher_alignment": (
            torch.stack(teacher_alignment).mean(dim=0).tolist() if teacher_alignment else []
        ),
        "drift/answer_instability": (
            torch.stack(instability).mean(dim=0).tolist() if instability else []
        ),
    }
    if profile_result is not None:
        metrics.update(
            {
                "hardware/latency_ms": profile_result.latency_ms,
                "hardware/peak_memory_mb": profile_result.peak_memory_mb,
                "hardware/average_power_w": profile_result.average_power_w,
                "hardware/energy_joules": profile_result.energy_joules,
            }
        )
    return metrics


def _move_batch(model: ContinuousCoTModel, batch: Mapping[str, Any]) -> Dict[str, Any]:
    """Move all tensor values in a batch to the model device."""

    moved: Dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            moved[key] = value.to(model.device)
        else:
            moved[key] = value
    return moved
