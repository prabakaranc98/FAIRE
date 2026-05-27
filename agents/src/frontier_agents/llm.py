"""LLM factory for the Frontier Wiki agent system.

All LLM calls go through LangChain's ChatOpenAI, which speaks the standard
OpenAI Chat Completions API. By default we talk to OpenRouter (cloud). To
swap in a local model, just point `OPENAI_API_BASE` at any OpenAI-compatible
endpoint (MLX, Ollama, vLLM, LMStudio, …) and set the role MODELs to whatever
that server exposes.

Cloud (default):
  # .env
  OPENROUTER_API_KEY=sk-or-...
  WRITER_MODEL=openai/gpt-5.1-codex-mini

Local (MLX server on Apple Silicon):
  # .env
  OPENAI_API_BASE=http://127.0.0.1:8080/v1
  OPENROUTER_API_KEY=local-not-needed     # ChatOpenAI requires *some* string
  WRITER_MODEL=qwen2.5-coder-32b-instruct
  REVIEWER_MODEL=qwen2.5-7b-instruct
  # ...etc

See docs/system/local-mode.md for the setup script and model recommendations.
"""

from __future__ import annotations

import os
from functools import lru_cache

from langchain_openai import ChatOpenAI

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

# Cloud model defaults (verified on OpenRouter 2026-05-26). Override via .env.
_DEFAULTS = {
    "WRITER_MODEL":   "openai/gpt-5.1-codex-mini",
    "MVB_MODEL":      "openai/gpt-5.1-codex-mini",
    "REVIEWER_MODEL": "openai/gpt-5-mini",
    "CRITIC_MODEL":   "google/gemini-3.1-flash-lite",
    "RESEARCH_MODEL": "google/gemini-3.5-flash",
    "FALLBACK_MODEL": "google/gemini-3.1-flash-lite",
}


@lru_cache(maxsize=8)
def get_llm(role: str = "writer", temperature: float = 0.3) -> ChatOpenAI:
    """Return a ChatOpenAI configured for the given role.

    Routes to whatever `OPENAI_API_BASE` points at; defaults to OpenRouter.
    The same agent code therefore runs against cloud or a local server with
    no code changes — only env vars.

    Args:
        role: writer | reviewer | critic | mvb | research | fallback
        temperature: sampling temperature (lower = more deterministic)
    """
    model_env_key = {
        "writer":   "WRITER_MODEL",
        "reviewer": "REVIEWER_MODEL",
        "critic":   "CRITIC_MODEL",
        "mvb":      "MVB_MODEL",
        "research": "RESEARCH_MODEL",
        "fallback": "FALLBACK_MODEL",
    }.get(role, "WRITER_MODEL")

    model = os.getenv(model_env_key, _DEFAULTS[model_env_key])

    # The single switch that swings cloud↔local: where do we POST?
    base_url = os.getenv("OPENAI_API_BASE", OPENROUTER_BASE_URL)

    max_tokens_by_role = {
        "writer":   16000,
        "mvb":      4096,
        "reviewer": 8192,
        "critic":   8192,
        "research": 4096,
        "fallback": 16000,
    }
    max_tokens = max_tokens_by_role.get(role, 8192)

    # Cloud (OpenRouter) needs the routing headers; local servers ignore them.
    extra_kwargs = {}
    if base_url == OPENROUTER_BASE_URL:
        extra_kwargs["model_kwargs"] = {"extra_headers": {
            "HTTP-Referer": "https://prabakaranc98.github.io/FAIRE",
            "X-Title": "Frontier Wiki Agent",
        }}

    # ChatOpenAI requires *some* api_key string. Cloud uses the real
    # OPENROUTER_API_KEY; local servers accept any placeholder.
    api_key = os.getenv("OPENROUTER_API_KEY") or "local-not-needed"

    if os.getenv("OPENROUTER_API_KEY") or base_url != OPENROUTER_BASE_URL:
        return ChatOpenAI(
            model=model,
            openai_api_key=api_key,
            openai_api_base=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            **extra_kwargs,
        )

    # Last-resort: direct Anthropic (only when no OpenRouter key AND no local URL)
    from langchain_anthropic import ChatAnthropic
    return ChatAnthropic(  # type: ignore[return-value]
        model=model.replace("anthropic/", "") or "claude-opus-4.7",
        anthropic_api_key=os.environ["ANTHROPIC_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
    )
