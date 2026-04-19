"""Environment loading helpers for local development and CLI scripts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

try:
    from dotenv import load_dotenv as _load_dotenv
except ImportError:
    _load_dotenv = None


def load_environment(env_file: Optional[str] = None, override: bool = False) -> Optional[Path]:
    """Load environment variables from a `.env` file if one is present.

    The loader normalizes common Hugging Face token aliases so downstream code
    can consistently read `HF_TOKEN`.
    """

    if env_file:
        candidate = Path(env_file).expanduser().resolve()
    else:
        candidate = Path(__file__).resolve().parents[1] / ".env"

    if candidate.exists():
        if _load_dotenv is not None:
            _load_dotenv(candidate, override=override)
        else:
            _load_dotenv_fallback(candidate, override=override)
        _normalize_hf_token_aliases()
        return candidate

    _normalize_hf_token_aliases()
    return None


def get_hf_token() -> Optional[str]:
    """Return the active Hugging Face token, if configured."""

    token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
    return token if token else None


def get_openai_api_key() -> Optional[str]:
    """Return the configured OpenAI-compatible API key, if available."""

    token = os.getenv("OPENAI_API_KEY")
    return token if token else None


def get_openai_api_url() -> Optional[str]:
    """Return the configured OpenAI-compatible base URL, if available."""

    base_url = os.getenv("OPENAI_API_URL")
    return base_url if base_url else None


def _normalize_hf_token_aliases() -> None:
    """Mirror known Hugging Face token variable names for compatibility."""

    hf_token = os.getenv("HF_TOKEN")
    hub_token = os.getenv("HUGGINGFACE_HUB_TOKEN")

    if hf_token and not hub_token:
        os.environ["HUGGINGFACE_HUB_TOKEN"] = hf_token
    if hub_token and not hf_token:
        os.environ["HF_TOKEN"] = hub_token


def _load_dotenv_fallback(path: Path, override: bool = False) -> None:
    """Minimal `.env` parser used when `python-dotenv` is unavailable."""

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", maxsplit=1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if override or key not in os.environ:
            os.environ[key] = value
