"""Thin wrappers around supported Hugging Face causal LMs."""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from models.registry import resolve_model_family, resolve_model_name
from utils.env import get_hf_token


def resolve_torch_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    """Map a string name to a torch dtype."""

    if dtype_name is None:
        return None
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    key = dtype_name.lower()
    if key not in mapping:
        raise ValueError(f"Unsupported dtype '{dtype_name}'.")
    return mapping[key]


class BaseModelAdapter(nn.Module):
    """Architecture-agnostic wrapper over a frozen causal language model."""

    def __init__(
        self,
        model_name_or_path: Optional[str] = None,
        model_name_or_alias: Optional[str] = None,
        torch_dtype: Optional[str] = None,
        device: Optional[str] = None,
        device_map: Optional[str] = None,
        trust_remote_code: bool = True,
    ) -> None:
        super().__init__()
        resolved_name = model_name_or_alias or model_name_or_path
        if resolved_name is None:
            raise ValueError("Expected either 'model_name_or_path' or 'model_name_or_alias'.")

        self.model_name = resolve_model_name(resolved_name)
        self.tokenizer = self._load_tokenizer(self.model_name, trust_remote_code=trust_remote_code)
        model_kwargs = {"trust_remote_code": trust_remote_code}
        resolved_dtype = resolve_torch_dtype(torch_dtype)
        hf_token = get_hf_token()
        if resolved_dtype is not None:
            model_kwargs["torch_dtype"] = resolved_dtype
        if device_map == "auto":
            model_kwargs["device_map"] = "auto"
        if hf_token is not None:
            model_kwargs["token"] = hf_token

        try:
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name, **model_kwargs)
        except OSError as exc:
            raise _rewrite_hf_auth_error(exc, self.model_name) from exc
        if device and device_map != "auto":
            self.model.to(device)
        self.model.eval()

    @property
    def hidden_size(self) -> int:
        """Hidden size of the wrapped backbone."""

        for attribute in ("hidden_size", "n_embd", "d_model"):
            if hasattr(self.model.config, attribute):
                return int(getattr(self.model.config, attribute))
        raise AttributeError(f"Could not infer hidden size for {self.model_name}.")

    @property
    def device(self) -> torch.device:
        """Device of the first model parameter."""

        return next(self.model.parameters()).device

    def freeze_backbone(self) -> None:
        """Freeze all parameters of the base LM."""

        for parameter in self.model.parameters():
            parameter.requires_grad = False

    def forward_backbone(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
    ):
        """Run the frozen LM and return hidden states."""

        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )

    def logits_from_hidden(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Project hidden states back into vocabulary logits."""

        output_head = self.model.get_output_embeddings()
        if output_head is None:
            raise ValueError(f"Model {self.model_name} does not expose output embeddings.")
        return output_head(hidden_states)

    def embed_tokens(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Embed token ids with the model's input embedding table."""

        embeddings = self.model.get_input_embeddings()
        if embeddings is None:
            raise ValueError(f"Model {self.model_name} does not expose input embeddings.")
        return embeddings(input_ids)

    @staticmethod
    def _load_tokenizer(
        model_name_or_path: str,
        trust_remote_code: bool = True,
    ) -> PreTrainedTokenizerBase:
        """Load a tokenizer and ensure a padding token exists."""

        tokenizer_kwargs = {"trust_remote_code": trust_remote_code}
        hf_token = get_hf_token()
        if hf_token is not None:
            tokenizer_kwargs["token"] = hf_token

        try:
            tokenizer = AutoTokenizer.from_pretrained(
                model_name_or_path,
                **tokenizer_kwargs,
            )
        except OSError as exc:
            raise _rewrite_hf_auth_error(exc, model_name_or_path) from exc
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        return tokenizer


class QwenAdapter(BaseModelAdapter):
    """Qwen-specific adapter placeholder for future family overrides."""


class LlamaAdapter(BaseModelAdapter):
    """Llama-specific adapter placeholder for future family overrides."""


class PhiAdapter(BaseModelAdapter):
    """Phi-specific adapter placeholder for future family overrides."""


def load_model_adapter(
    model_name_or_alias: str,
    torch_dtype: Optional[str] = None,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = True,
) -> BaseModelAdapter:
    """Instantiate the correct adapter class for a supported model family."""

    family = resolve_model_family(model_name_or_alias)
    adapter_class = {
        "qwen": QwenAdapter,
        "llama": LlamaAdapter,
        "phi": PhiAdapter,
        "generic": BaseModelAdapter,
    }[family]
    return adapter_class(
        model_name_or_alias=model_name_or_alias,
        torch_dtype=torch_dtype,
        device=device,
        device_map=device_map,
        trust_remote_code=trust_remote_code,
    )


def _rewrite_hf_auth_error(error: OSError, model_name: str) -> OSError:
    """Attach actionable `.env` guidance to gated/private Hugging Face failures."""

    message = str(error)
    auth_markers = ("401", "gated repo", "restricted", "Please log in", "access to model")
    if any(marker.lower() in message.lower() for marker in auth_markers):
        return OSError(
            f"{message}\n\n"
            f"To use '{model_name}', add your Hugging Face token to "
            ".env as 'HF_TOKEN=hf_...' and make sure your account has access "
            "to the repository."
        )
    return error
