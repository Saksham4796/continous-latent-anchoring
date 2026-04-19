"""Dataset loading, prompt formatting, and anchor shard collation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import torch
from datasets import Dataset, load_dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset as TorchDataset

from utils.env import get_hf_token


@dataclass(frozen=True)
class DatasetSpec:
    """Declarative description of a supported reasoning dataset."""

    path: str
    subset: Optional[str]
    task_type: str
    question_fields: Sequence[str]
    answer_fields: Sequence[str]
    rationale_fields: Sequence[str]
    id_fields: Sequence[str]


DATASET_SPECS: Dict[str, DatasetSpec] = {
    "gsm8k": DatasetSpec(
        path="openai/gsm8k",
        subset="main",
        task_type="math",
        question_fields=("question",),
        answer_fields=("answer",),
        rationale_fields=("answer",),
        id_fields=("id", "question"),
    ),
    "strategyqa": DatasetSpec(
        path="ChilleD/StrategyQA",
        subset=None,
        task_type="logic",
        question_fields=("question",),
        answer_fields=("answer", "label"),
        rationale_fields=("facts", "rationale", "explanation", "description"),
        id_fields=("id", "qid", "question"),
    ),
}


def load_reasoning_dataset(
    dataset_name: str,
    split: str,
    cache_dir: Optional[str] = None,
    max_samples: Optional[int] = None,
    dataset_path: Optional[str] = None,
    dataset_subset: Optional[str] = None,
) -> Dataset:
    """Load and normalize a supported reasoning dataset."""

    key = dataset_name.lower()
    if key not in DATASET_SPECS:
        raise ValueError(f"Unsupported dataset '{dataset_name}'. Expected one of {sorted(DATASET_SPECS)}.")

    spec = DATASET_SPECS[key]
    path = dataset_path or spec.path
    subset = dataset_subset if dataset_subset is not None else spec.subset
    dataset_kwargs = {"split": split, "cache_dir": cache_dir}
    hf_token = get_hf_token()
    if hf_token is not None:
        dataset_kwargs["token"] = hf_token

    try:
        raw_dataset = load_dataset(path, subset, **dataset_kwargs)
    except Exception as exc:
        raise _rewrite_dataset_auth_error(exc, path) from exc

    if max_samples is not None:
        limit = min(max_samples, len(raw_dataset))
        raw_dataset = raw_dataset.select(range(limit))

    original_columns = list(raw_dataset.column_names)
    normalized = raw_dataset.map(
        lambda row: normalize_reasoning_example(key, row),
        remove_columns=original_columns,
        desc=f"Normalizing {dataset_name}:{split}",
    )
    return normalized


def normalize_reasoning_example(dataset_name: str, example: Mapping[str, Any]) -> Dict[str, Any]:
    """Convert raw dataset rows into a shared schema."""

    spec = DATASET_SPECS[dataset_name]
    question = _first_present(example, spec.question_fields, default="")
    rationale_raw = _first_present(example, spec.rationale_fields)
    answer_raw = _first_present(example, spec.answer_fields)
    source_id = str(_first_present(example, spec.id_fields, default=question))

    if dataset_name == "gsm8k":
        rationale, target_text = _parse_gsm8k_answer(str(answer_raw or rationale_raw or ""))
    else:
        rationale = _stringify_rationale(rationale_raw)
        target_text = _normalize_strategyqa_answer(answer_raw)

    prompt_question = _augment_question(example, question, dataset_name)
    return {
        "id": source_id,
        "dataset_name": dataset_name,
        "task_type": spec.task_type,
        "question": prompt_question.strip(),
        "rationale": rationale.strip(),
        "target_text": canonicalize_final_answer(target_text, dataset_name),
    }


def build_teacher_prompt(example: Mapping[str, Any]) -> str:
    """Format a teacher generation prompt for step-by-step reasoning."""

    task_type = example["task_type"]
    answer_format = "a single number" if task_type == "math" else "'true' or 'false'"
    return (
        "You are a careful reasoning teacher.\n"
        "Solve the task with concise but correct reasoning.\n"
        f"Finish with a final line formatted exactly as 'Final answer: {answer_format}'.\n\n"
        f"Question: {example['question']}\n"
        "Reasoning:\n"
    )


def build_training_target(rationale: str, target_text: str, dataset_name: str) -> str:
    """Construct the supervised response used for teacher forcing."""

    cleaned_rationale = rationale.strip()
    cleaned_target = canonicalize_final_answer(target_text, dataset_name)
    final_line = f"Final answer: {cleaned_target}"

    if cleaned_rationale:
        if re.search(r"final answer\s*:", cleaned_rationale, flags=re.IGNORECASE):
            return cleaned_rationale
        return f"{cleaned_rationale}\n{final_line}\n"
    return f"{final_line}\n"


def extract_final_answer_from_response(response_text: str, dataset_name: str) -> str:
    """Extract the final answer string from a generated response."""

    final_answer_match = re.search(
        r"final answer\s*:\s*(.+)",
        response_text,
        flags=re.IGNORECASE,
    )
    if final_answer_match:
        return canonicalize_final_answer(final_answer_match.group(1), dataset_name)

    lines = [line.strip() for line in response_text.splitlines() if line.strip()]
    if not lines:
        return ""
    return canonicalize_final_answer(lines[-1], dataset_name)


def canonicalize_final_answer(answer_text: Any, dataset_name: str) -> str:
    """Map answers into a canonical exact-match format."""

    text = str(answer_text).strip()
    if not text:
        return ""

    if dataset_name == "gsm8k":
        numeric_matches = re.findall(r"-?\d[\d,]*(?:\.\d+)?", text)
        if numeric_matches:
            return numeric_matches[-1].replace(",", "")
        return text.lower()

    lowered = text.lower()
    if lowered in {"yes", "true", "1"}:
        return "true"
    if lowered in {"no", "false", "0"}:
        return "false"
    return lowered


def resolve_anchor_shards(path_like: str) -> List[Path]:
    """Resolve a file or directory into a sorted list of anchor shard files."""

    path = Path(path_like)
    if path.is_file():
        return [path]
    if not path.exists():
        raise FileNotFoundError(f"Anchor path does not exist: {path}")

    shards = sorted(item for item in path.glob("*.pt") if item.is_file())
    if not shards:
        raise FileNotFoundError(f"No .pt anchor shards found under {path}")
    return shards


class AnchorTensorDataset(TorchDataset):
    """PyTorch dataset over one or more saved anchor shards."""

    def __init__(self, shards: Sequence[str]) -> None:
        self.records: List[Dict[str, Any]] = []
        for shard in shards:
            payload = torch.load(shard, map_location="cpu")
            if isinstance(payload, dict) and "records" in payload:
                self.records.extend(payload["records"])
            elif isinstance(payload, list):
                self.records.extend(payload)
            else:
                raise ValueError(f"Unsupported anchor shard format in {shard}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> Dict[str, Any]:
        return self.records[index]


def anchor_collate_fn(batch: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    """Pad variable-length tensors and stack latent anchors."""

    input_ids = pad_sequence(
        [item["input_ids"].long() for item in batch],
        batch_first=True,
        padding_value=int(batch[0].get("pad_token_id", 0)),
    )
    attention_mask = pad_sequence(
        [item["attention_mask"].long() for item in batch],
        batch_first=True,
        padding_value=0,
    )
    labels = pad_sequence(
        [item["labels"].long() for item in batch],
        batch_first=True,
        padding_value=-100,
    )
    response_mask = pad_sequence(
        [item["response_mask"].bool() for item in batch],
        batch_first=True,
        padding_value=False,
    )
    positive_anchor = torch.stack([item["positive_anchor"].float() for item in batch], dim=0)
    negative_anchor = torch.stack([item["negative_anchor"].float() for item in batch], dim=0)

    collated = {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "response_mask": response_mask,
        "positive_anchor": positive_anchor,
        "negative_anchor": negative_anchor,
        "prompt_text": [item["prompt_text"] for item in batch],
        "positive_text": [item["positive_text"] for item in batch],
        "negative_text": [item["negative_text"] for item in batch],
        "target_text": [item["target_text"] for item in batch],
        "question": [item["question"] for item in batch],
        "task_type": [item["task_type"] for item in batch],
        "dataset_name": [item["dataset_name"] for item in batch],
    }

    if "teacher_logits" in batch[0]:
        teacher_logits = [item.get("teacher_logits") for item in batch]
        if all(tensor is not None for tensor in teacher_logits):
            collated["teacher_logits"] = pad_sequence(
                [tensor.float() for tensor in teacher_logits if tensor is not None],
                batch_first=True,
                padding_value=0.0,
            )

    return collated


def _parse_gsm8k_answer(answer_text: str) -> List[str]:
    """Split GSM8K answer strings into rationale and numeric final answer."""

    if "####" in answer_text:
        rationale, final_answer = answer_text.split("####", maxsplit=1)
        return [rationale.strip(), canonicalize_final_answer(final_answer, "gsm8k")]
    return [answer_text.strip(), canonicalize_final_answer(answer_text, "gsm8k")]


def _normalize_strategyqa_answer(answer_value: Any) -> str:
    """Normalize StrategyQA labels to 'true'/'false'."""

    if isinstance(answer_value, bool):
        return "true" if answer_value else "false"
    if isinstance(answer_value, int):
        return "true" if answer_value == 1 else "false"
    return canonicalize_final_answer(answer_value, "strategyqa")


def _augment_question(example: Mapping[str, Any], question: str, dataset_name: str) -> str:
    """Add lightweight context when the dataset exposes it."""

    if dataset_name != "strategyqa":
        return question

    entity = _first_present(example, ("entity", "subject", "term", "title"))
    description = _first_present(example, ("description",))
    context_lines = []
    if entity:
        context_lines.append(f"Entity: {entity}")
    if description:
        context_lines.append(f"Context: {description}")
    context_lines.append(question)
    return "\n".join(context_lines)


def _stringify_rationale(value: Any) -> str:
    """Convert rationale-like payloads into a compact string."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence):
        return " ".join(str(item) for item in value)
    return str(value)


def _first_present(
    mapping: Mapping[str, Any],
    keys: Iterable[str],
    default: Optional[Any] = None,
) -> Optional[Any]:
    """Return the first key present in a dictionary-like object."""

    for key in keys:
        if key in mapping and mapping[key] not in (None, ""):
            return mapping[key]
    return default


def _rewrite_dataset_auth_error(error: Exception, dataset_path: str) -> Exception:
    """Attach `.env` guidance when a dataset load fails due to auth."""

    message = str(error)
    auth_markers = ("401", "gated repo", "restricted", "Please log in", "token")
    if any(marker.lower() in message.lower() for marker in auth_markers):
        return RuntimeError(
            f"{message}\n\n"
            f"If '{dataset_path}' is gated or private, add your Hugging Face token "
            "to .env as 'HF_TOKEN=hf_...'."
        )
    return error
