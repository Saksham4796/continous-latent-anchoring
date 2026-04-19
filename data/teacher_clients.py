"""Teacher backends for offline reasoning-trajectory generation."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

import httpx

from models import BaseModelAdapter, load_model_adapter
from utils.env import get_openai_api_key, get_openai_api_url


class TeacherClient:
    """Interface for teacher backends used during anchor generation."""

    def generate(self, prompt_text: str, max_new_tokens: int) -> str:
        """Generate a reasoning trajectory for a single prompt."""

        raise NotImplementedError


@dataclass
class LocalTeacherClient(TeacherClient):
    """Local Hugging Face teacher backend."""

    adapter: BaseModelAdapter
    temperature: float
    top_p: float
    max_length: int

    def generate(self, prompt_text: str, max_new_tokens: int) -> str:
        """Generate a reasoning trajectory with a local causal LM."""

        encoded = self.adapter.tokenizer(
            prompt_text,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_length,
        )
        encoded = {key: value.to(self.adapter.device) for key, value in encoded.items()}
        generated = self.adapter.model.generate(
            **encoded,
            do_sample=self.temperature > 0.0,
            temperature=max(self.temperature, 1e-5),
            top_p=self.top_p,
            max_new_tokens=max_new_tokens,
            pad_token_id=self.adapter.tokenizer.pad_token_id,
            eos_token_id=self.adapter.tokenizer.eos_token_id,
        )
        new_tokens = generated[0, encoded["input_ids"].shape[1] :]
        return self.adapter.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


@dataclass
class OpenAICompatibleTeacherClient(TeacherClient):
    """Remote teacher backend using an OpenAI-compatible chat-completions API."""

    model_name: str
    temperature: float
    top_p: float
    timeout_seconds: float = 120.0
    max_retries: int = 4

    def __post_init__(self) -> None:
        api_key = get_openai_api_key()
        base_url = get_openai_api_url()
        if not api_key:
            raise ValueError(
                "OPENAI_API_KEY is not set. Add it to .env before using "
                "--teacher-backend openai."
            )
        if not base_url:
            raise ValueError(
                "OPENAI_API_URL is not set. Add it to .env before using "
                "--teacher-backend openai."
            )

        self._base_url = base_url.rstrip("/")
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        self._client = httpx.Client(timeout=self.timeout_seconds)

    def generate(self, prompt_text: str, max_new_tokens: int) -> str:
        """Generate a reasoning trajectory from a remote teacher model."""

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt_text}],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": max_new_tokens,
        }

        last_error: Optional[Exception] = None
        for attempt in range(1, self.max_retries + 1):
            try:
                response = self._client.post(
                    f"{self._base_url}/chat/completions",
                    headers=self._headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()
                return _extract_chat_completion_text(body)
            except Exception as exc:
                last_error = exc
                if attempt == self.max_retries:
                    break
                time.sleep(min(2 ** (attempt - 1), 8))

        raise RuntimeError(
            f"Remote teacher request failed after {self.max_retries} attempts "
            f"for model '{self.model_name}'."
        ) from last_error


def build_teacher_client(
    backend: str,
    model_name: str,
    *,
    temperature: float,
    top_p: float,
    max_length: int,
    torch_dtype: Optional[str] = None,
    device: Optional[str] = None,
    device_map: Optional[str] = None,
    trust_remote_code: bool = True,
    timeout_seconds: float = 120.0,
    max_retries: int = 4,
) -> TeacherClient:
    """Instantiate the configured teacher backend."""

    if backend == "openai":
        return OpenAICompatibleTeacherClient(
            model_name=model_name,
            temperature=temperature,
            top_p=top_p,
            timeout_seconds=timeout_seconds,
            max_retries=max_retries,
        )

    if backend == "local":
        adapter = load_model_adapter(
            model_name_or_alias=model_name,
            torch_dtype=torch_dtype,
            device=device,
            device_map=device_map,
            trust_remote_code=trust_remote_code,
        )
        return LocalTeacherClient(
            adapter=adapter,
            temperature=temperature,
            top_p=top_p,
            max_length=max_length,
        )

    raise ValueError(f"Unsupported teacher backend '{backend}'.")


def _extract_chat_completion_text(response_body: Dict[str, Any]) -> str:
    """Extract the first assistant message from a chat-completions response."""

    choices = response_body.get("choices", [])
    if not choices:
        raise ValueError("OpenAI-compatible response did not contain any choices.")

    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        text_chunks = []
        for item in content:
            if isinstance(item, dict) and item.get("type") in {"text", "output_text"}:
                text_value = item.get("text") or item.get("content") or ""
                if text_value:
                    text_chunks.append(str(text_value))
        if text_chunks:
            return "\n".join(text_chunks).strip()

    raise ValueError("Could not parse assistant content from chat-completions response.")
