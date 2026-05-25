"""LLM factory for the Frontier Wiki agent system.

All LLMs route through OpenRouter using LangChain's ChatOpenAI interface.
This gives access to any model on OpenRouter without changing any node code.

Model selection philosophy (verified on OpenRouter 2026-05-25):
- WRITER (Editorial, MVB):   anthropic/claude-opus-4-7 — $5/M, 1M ctx — maximum prose intelligence
- REVIEWER (Schema checks):  google/gemini-3.1-pro-preview — $2/M, 1M ctx — best structured reasoning
- RESEARCH (fast search):    google/gemini-3.5-flash — $1.50/M, 1M ctx — high-quality, low latency
- FALLBACK:                  anthropic/claude-sonnet-4.6 — $3/M, 1M ctx — still high quality
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Default model IDs — override via .env
_DEFAULTS = {
    "WRITER_MODEL":   "anthropic/claude-opus-4.7",
    "MVB_MODEL":      "anthropic/claude-opus-4.7",
    "REVIEWER_MODEL": "google/gemini-3.1-pro-preview",
    "RESEARCH_MODEL": "google/gemini-3.5-flash",
    "FALLBACK_MODEL": "anthropic/claude-sonnet-4.6",
}


def _get_key() -> str:
    key = os.getenv("OPENROUTER_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
    if not key:
        raise ValueError(
            "Set OPENROUTER_API_KEY in .env (preferred) or ANTHROPIC_API_KEY as fallback"
        )
    return key


def _get_base_url() -> str:
    # If only ANTHROPIC_API_KEY is set (not OPENROUTER_API_KEY), use Anthropic direct via langchain-anthropic
    if os.getenv("OPENROUTER_API_KEY"):
        return OPENROUTER_BASE_URL
    # Fallback: use Anthropic direct — caller must use ChatAnthropic separately
    return OPENROUTER_BASE_URL


@lru_cache(maxsize=8)
def get_llm(role: str = "writer", temperature: float = 0.3) -> ChatOpenAI:
    """Return a configured LangChain LLM for the given role.

    Args:
        role: "writer" | "reviewer" | "mvb" | "fallback"
        temperature: sampling temperature (lower = more deterministic)

    Returns:
        ChatOpenAI instance configured for OpenRouter
    """
    model_env_key = {
        "writer":   "WRITER_MODEL",
        "reviewer": "REVIEWER_MODEL",
        "mvb":      "MVB_MODEL",
        "research": "RESEARCH_MODEL",
        "fallback": "FALLBACK_MODEL",
    }.get(role, "WRITER_MODEL")

    model = os.getenv(model_env_key, _DEFAULTS[model_env_key])

    # OpenRouter requires these headers for routing and billing
    extra_headers = {
        "HTTP-Referer": "https://prabakaranc98.github.io/FAIRE",
        "X-Title": "Frontier Wiki Agent",
    }

    # Full wiki pages need 4000-8000 tokens of output — set explicitly per role
    max_tokens_by_role = {
        "writer":   8192,   # full schema page: ~3000-6000 tokens
        "mvb":      4096,   # MVB section only
        "reviewer": 2048,   # structured JSON feedback
        "research": 4096,   # plan + summaries
        "fallback": 8192,
    }
    max_tokens = max_tokens_by_role.get(role, 8192)

    if os.getenv("OPENROUTER_API_KEY"):
        return ChatOpenAI(
            model=model,
            openai_api_key=os.environ["OPENROUTER_API_KEY"],
            openai_api_base=OPENROUTER_BASE_URL,
            temperature=temperature,
            max_tokens=max_tokens,
            model_kwargs={"extra_headers": extra_headers},
        )

    # Fallback: direct Anthropic via langchain-anthropic (when no OpenRouter key)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(  # type: ignore[return-value]
        model=model.replace("anthropic/", "") or "claude-opus-4.7",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
    )
