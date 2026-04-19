"""Supported base model registry for CLA experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict


@dataclass(frozen=True)
class SupportedModel:
    """Metadata for a supported Hugging Face causal language model."""

    alias: str
    model_name: str
    family: str


SUPPORTED_MODELS: Dict[str, SupportedModel] = {
    "qwen2.5-1.5b": SupportedModel(
        alias="qwen2.5-1.5b",
        model_name="Qwen/Qwen2.5-1.5B-Instruct",
        family="qwen",
    ),
    "llama-3.2-1b": SupportedModel(
        alias="llama-3.2-1b",
        model_name="meta-llama/Llama-3.2-1B-Instruct",
        family="llama",
    ),
    "phi-3.5-mini": SupportedModel(
        alias="phi-3.5-mini",
        model_name="microsoft/Phi-3.5-mini-instruct",
        family="phi",
    ),
}


def resolve_model_name(model_name_or_alias: str) -> str:
    """Resolve a short alias to a fully qualified Hugging Face model id."""

    key = model_name_or_alias.lower()
    if key in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[key].model_name
    return model_name_or_alias


def resolve_model_family(model_name_or_alias: str) -> str:
    """Infer the model family used to select the correct adapter class."""

    key = model_name_or_alias.lower()
    if key in SUPPORTED_MODELS:
        return SUPPORTED_MODELS[key].family

    lowered = model_name_or_alias.lower()
    if "qwen" in lowered:
        return "qwen"
    if "phi" in lowered:
        return "phi"
    if "llama" in lowered:
        return "llama"
    return "generic"
