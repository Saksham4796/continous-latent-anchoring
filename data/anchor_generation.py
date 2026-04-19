"""Offline generation of positive and negative latent anchors for CLA."""

from __future__ import annotations

import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import torch
from tqdm.auto import tqdm

from data.datasets import (
    build_teacher_prompt,
    build_training_target,
    canonicalize_final_answer,
    extract_final_answer_from_response,
    load_reasoning_dataset,
)
from models import load_model_adapter
from data.teacher_clients import LocalTeacherClient, TeacherClient, build_teacher_client


@dataclass
class AnchorGenerationConfig:
    """Configuration for offline positive and negative anchor extraction."""

    dataset_name: str = "gsm8k"
    split: str = "train"
    output_dir: str = "artifacts/anchors/gsm8k_train_qwen15b"
    teacher_model_name: str = "meta-llama/Meta-Llama-3-8B-Instruct"
    teacher_backend: str = "local"
    anchor_encoder_model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    cache_dir: Optional[str] = None
    dataset_path: Optional[str] = None
    dataset_subset: Optional[str] = None
    max_samples: Optional[int] = None
    max_length: int = 768
    max_new_tokens: int = 192
    temperature: float = 0.2
    top_p: float = 0.9
    shard_size: int = 256
    negative_strategy: str = "hard_semantic"
    teacher_dtype: str = "bfloat16"
    encoder_dtype: str = "bfloat16"
    teacher_device: str = "auto"
    encoder_device: str = "auto"
    trust_remote_code: bool = True
    teacher_timeout_seconds: float = 120.0
    teacher_max_retries: int = 4
    use_dataset_rationale_fallback: bool = True
    seed: int = 42


class ContrastiveAnchorGenerator:
    """Generate teacher-guided latent anchors encoded in the student space."""

    def __init__(self, config: AnchorGenerationConfig) -> None:
        self.config = config
        self.rng = random.Random(config.seed)

        self.teacher_client = build_teacher_client(
            backend=config.teacher_backend,
            model_name=config.teacher_model_name,
            temperature=config.temperature,
            top_p=config.top_p,
            max_length=config.max_length,
            torch_dtype=config.teacher_dtype,
            device=None if config.teacher_device == "auto" else config.teacher_device,
            device_map="auto" if config.teacher_device == "auto" else None,
            trust_remote_code=config.trust_remote_code,
            timeout_seconds=config.teacher_timeout_seconds,
            max_retries=config.teacher_max_retries,
        )

        if (
            config.teacher_backend == "local"
            and isinstance(self.teacher_client, LocalTeacherClient)
            and config.anchor_encoder_model_name == config.teacher_model_name
        ):
            self.anchor_encoder = self.teacher_client.adapter
        else:
            self.anchor_encoder = load_model_adapter(
                model_name_or_alias=config.anchor_encoder_model_name,
                torch_dtype=config.encoder_dtype,
                device=None if config.encoder_device == "auto" else config.encoder_device,
                device_map="auto" if config.encoder_device == "auto" else None,
                trust_remote_code=config.trust_remote_code,
            )

    @torch.inference_mode()
    def generate(self) -> List[Path]:
        """Generate and save anchor shards for a dataset split."""

        dataset = load_reasoning_dataset(
            dataset_name=self.config.dataset_name,
            split=self.config.split,
            cache_dir=self.config.cache_dir,
            max_samples=self.config.max_samples,
            dataset_path=self.config.dataset_path,
            dataset_subset=self.config.dataset_subset,
        )
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        saved_paths: List[Path] = []
        pending_records: List[Dict[str, Any]] = []
        for index, example in enumerate(tqdm(dataset, desc="Generating latent anchors")):
            record = self._build_anchor_record(example)
            pending_records.append(record)
            if len(pending_records) >= self.config.shard_size:
                saved_paths.append(self._save_shard(pending_records, output_dir, len(saved_paths)))
                pending_records = []

        if pending_records:
            saved_paths.append(self._save_shard(pending_records, output_dir, len(saved_paths)))
        return saved_paths

    def _build_anchor_record(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Build a single positive/negative anchor record."""

        prompt_text = build_teacher_prompt(example)
        teacher_response = self._generate_teacher_response(prompt_text)
        positive_text = self._select_positive_response(example, teacher_response)
        negative_text = self._build_negative_response(example, positive_text)

        positive_encoding = self._encode_trajectory(prompt_text, positive_text)
        negative_encoding = self._encode_trajectory(prompt_text, negative_text)

        return {
            "id": example["id"],
            "dataset_name": example["dataset_name"],
            "task_type": example["task_type"],
            "question": example["question"],
            "target_text": example["target_text"],
            "prompt_text": prompt_text,
            "positive_text": positive_text,
            "negative_text": negative_text,
            "input_ids": positive_encoding["input_ids"],
            "attention_mask": positive_encoding["attention_mask"],
            "labels": positive_encoding["labels"],
            "response_mask": positive_encoding["response_mask"],
            "positive_anchor": positive_encoding["anchor_vector"],
            "negative_anchor": negative_encoding["anchor_vector"],
            "pad_token_id": int(self.anchor_encoder.tokenizer.pad_token_id),
        }

    def _generate_teacher_response(self, prompt_text: str) -> str:
        """Sample or greedily decode a teacher reasoning trajectory."""

        return self.teacher_client.generate(
            prompt_text=prompt_text,
            max_new_tokens=self.config.max_new_tokens,
        )

    def _select_positive_response(self, example: Dict[str, Any], teacher_response: str) -> str:
        """Choose the positive trajectory, preferring correct teacher reasoning."""

        generated_answer = extract_final_answer_from_response(
            teacher_response,
            dataset_name=example["dataset_name"],
        )
        gold_answer = canonicalize_final_answer(example["target_text"], example["dataset_name"])

        if generated_answer == gold_answer and teacher_response.strip():
            return _ensure_final_answer_line(
                teacher_response,
                gold_answer,
                example["dataset_name"],
            )

        if self.config.use_dataset_rationale_fallback and example.get("rationale"):
            return build_training_target(
                rationale=example["rationale"],
                target_text=gold_answer,
                dataset_name=example["dataset_name"],
            )

        return build_training_target(
            rationale=teacher_response,
            target_text=gold_answer,
            dataset_name=example["dataset_name"],
        )

    def _build_negative_response(self, example: Dict[str, Any], positive_text: str) -> str:
        """Construct an explicit negative trajectory."""

        if self.config.negative_strategy == "hard_semantic":
            if example["task_type"] == "math":
                return _perturb_math_reasoning(positive_text, example["target_text"], self.rng)
            return _perturb_logic_reasoning(positive_text, self.rng)

        if self.config.negative_strategy == "random_noise":
            return _random_noise_reasoning(positive_text, example["target_text"], self.rng)

        raise ValueError(
            f"Unsupported negative strategy '{self.config.negative_strategy}'."
        )

    def _encode_trajectory(self, prompt_text: str, response_text: str) -> Dict[str, torch.Tensor]:
        """Encode a prompt/response trajectory into latent anchors and labels."""

        tokenizer = self.anchor_encoder.tokenizer
        full_text = f"{prompt_text}{response_text}"
        prompt_encoding = tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )
        full_encoding = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length,
        )

        prompt_length = int(prompt_encoding["input_ids"].shape[1])
        full_encoding = {key: value.to(self.anchor_encoder.device) for key, value in full_encoding.items()}
        outputs = self.anchor_encoder.forward_backbone(
            input_ids=full_encoding["input_ids"],
            attention_mask=full_encoding["attention_mask"],
        )
        hidden_states = outputs.hidden_states[-1][0]

        response_mask = torch.zeros(hidden_states.size(0), dtype=torch.bool, device=hidden_states.device)
        response_mask[prompt_length:] = True
        anchor_vector = _masked_mean(hidden_states, response_mask).cpu()

        labels = full_encoding["input_ids"][0].detach().cpu().clone()
        labels[:prompt_length] = -100

        return {
            "input_ids": full_encoding["input_ids"][0].detach().cpu(),
            "attention_mask": full_encoding["attention_mask"][0].detach().cpu(),
            "labels": labels,
            "response_mask": response_mask.detach().cpu(),
            "anchor_vector": anchor_vector,
        }

    def _save_shard(self, records: Sequence[Dict[str, Any]], output_dir: Path, shard_index: int) -> Path:
        """Persist a shard of latent anchor records."""

        path = output_dir / f"{self.config.dataset_name}_{self.config.split}_anchors_{shard_index:04d}.pt"
        payload = {
            "metadata": asdict(self.config),
            "records": list(records),
        }
        torch.save(payload, path)
        return path


def _ensure_final_answer_line(response_text: str, target_text: str, dataset_name: str) -> str:
    """Ensure a response ends with a canonical final answer line."""

    canonical_target = canonicalize_final_answer(target_text, dataset_name)
    stripped = response_text.strip()
    if re.search(r"final answer\s*:", stripped, flags=re.IGNORECASE):
        stripped = re.sub(
            r"final answer\s*:\s*.+",
            f"Final answer: {canonical_target}",
            stripped,
            flags=re.IGNORECASE,
        )
        return f"{stripped}\n"
    return f"{stripped}\nFinal answer: {canonical_target}\n"


def _perturb_math_reasoning(response_text: str, target_text: str, rng: random.Random) -> str:
    """Create a hard math negative by corrupting numerals and the final answer."""

    corrupted = response_text
    numeric_spans = list(re.finditer(r"-?\d[\d,]*(?:\.\d+)?", response_text))
    if numeric_spans:
        span = rng.choice(numeric_spans)
        original_text = span.group(0).replace(",", "")
        replacement = _offset_number(original_text, rng)
        corrupted = f"{corrupted[: span.start()]}{replacement}{corrupted[span.end() :]}"

    operator_swaps = [
        (r"\bplus\b", "minus"),
        (r"\bminus\b", "plus"),
        (r"\btimes\b", "divided by"),
        (r"\bdivided by\b", "times"),
        (r"\badd\b", "subtract"),
        (r"\bsubtract\b", "add"),
    ]
    pattern, substitute = rng.choice(operator_swaps)
    corrupted = re.sub(pattern, substitute, corrupted, count=1, flags=re.IGNORECASE)

    wrong_answer = _offset_number(str(target_text), rng)
    return _ensure_final_answer_line(corrupted, wrong_answer, "gsm8k")


def _perturb_logic_reasoning(response_text: str, rng: random.Random) -> str:
    """Create a hard semantic negative by flipping logical cues."""

    corrupted = response_text
    replacements = [
        (r"\btrue\b", "false"),
        (r"\bfalse\b", "true"),
        (r"\band\b", "or"),
        (r"\bor\b", "and"),
        (r"\bcan\b", "cannot"),
        (r"\bis\b", "is not"),
        (r"\bwould\b", "would not"),
        (r"\bdoes\b", "does not"),
    ]
    pattern, substitute = rng.choice(replacements)
    corrupted = re.sub(pattern, substitute, corrupted, count=1, flags=re.IGNORECASE)

    current_answer = extract_final_answer_from_response(corrupted, "strategyqa")
    flipped_answer = "false" if current_answer == "true" else "true"
    return _ensure_final_answer_line(corrupted, flipped_answer, "strategyqa")


def _random_noise_reasoning(response_text: str, target_text: str, rng: random.Random) -> str:
    """Create a random-noise negative response for ablation baselines."""

    tokens = response_text.split()
    rng.shuffle(tokens)
    sample = tokens[: max(8, min(len(tokens), 32))]
    noisy_rationale = " ".join(sample) if sample else "corrupted reasoning signal"
    wrong_answer = "false" if canonicalize_final_answer(target_text, "strategyqa") == "true" else "true"
    if re.search(r"-?\d[\d,]*(?:\.\d+)?", str(target_text)):
        wrong_answer = _offset_number(str(target_text), rng)
    return f"{noisy_rationale}\nFinal answer: {wrong_answer}\n"


def _offset_number(number_text: str, rng: random.Random) -> str:
    """Shift a numeric string by a small random offset."""

    if "." in number_text:
        number = float(number_text)
        shifted = number + rng.choice([-1.0, -0.5, 0.5, 1.0, 2.0])
        return f"{shifted:.2f}".rstrip("0").rstrip(".")

    number = int(float(number_text))
    shifted = number + rng.choice([-3, -2, -1, 1, 2, 3, 5])
    return str(shifted)


def _masked_mean(hidden_states: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token states over a boolean response mask."""

    expanded_mask = mask.unsqueeze(-1).float()
    denominator = expanded_mask.sum(dim=0).clamp_min(1.0)
    return (hidden_states * expanded_mask).sum(dim=0) / denominator
