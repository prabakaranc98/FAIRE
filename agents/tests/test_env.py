"""Validate environment configuration — .env.example completeness, model ID format, API key detection."""

import os
from pathlib import Path
from unittest.mock import patch

import pytest


ENV_EXAMPLE = Path(__file__).parent.parent / ".env.example"
REQUIRED_KEYS = [
    "OPENROUTER_API_KEY",
    "ANTHROPIC_API_KEY",
    "EXA_API_KEY",
    "WRITER_MODEL",
    "REVIEWER_MODEL",
    "MVB_MODEL",
    "WIKI_DOCS_DIR",
]
OPENROUTER_MODELS = [
    "anthropic/claude-opus-4-7",
    "google/gemini-2.5-flash-preview",
    "meta-llama/llama-3.3-70b-instruct",
]


def _parse_env_example() -> dict[str, str]:
    result = {}
    for line in ENV_EXAMPLE.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            result[k.strip()] = v.strip()
    return result


class TestEnvExample:
    def test_file_exists(self):
        assert ENV_EXAMPLE.exists(), ".env.example not found at agents/.env.example"

    @pytest.mark.parametrize("key", REQUIRED_KEYS)
    def test_required_key_present(self, key):
        env = _parse_env_example()
        assert key in env, f"Missing required key in .env.example: {key}"

    def test_writer_model_is_openrouter_format(self):
        env = _parse_env_example()
        model = env.get("WRITER_MODEL", "")
        assert "/" in model, f"WRITER_MODEL should be 'provider/model' format, got: {model}"

    def test_reviewer_model_is_gemini_flash(self):
        env = _parse_env_example()
        model = env.get("REVIEWER_MODEL", "")
        assert "gemini" in model.lower(), (
            f"REVIEWER_MODEL should be Gemini Flash (not Haiku), got: {model}"
        )

    def test_no_real_api_keys_committed(self):
        env = _parse_env_example()
        for key, val in env.items():
            if "API_KEY" in key:
                assert not val.startswith("sk-ant-api"), (
                    f"Real Anthropic API key found in .env.example for {key}"
                )
                assert len(val) < 20 or "..." in val, (
                    f"Possible real key committed for {key}: {val[:10]}..."
                )


class TestLlmFactory:
    def test_get_llm_raises_without_key(self):
        from frontier_agents.llm import get_llm
        get_llm.cache_clear()
        with patch.dict(os.environ, {}, clear=True):
            # Remove both key vars
            env = {k: v for k, v in os.environ.items()
                   if k not in ("OPENROUTER_API_KEY", "ANTHROPIC_API_KEY")}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises((ValueError, KeyError)):
                    get_llm("writer")
        get_llm.cache_clear()

    def test_get_llm_returns_with_openrouter_key(self):
        from frontier_agents.llm import get_llm
        get_llm.cache_clear()
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "test-key-12345"}):
            llm = get_llm("writer", temperature=0.0)
            assert llm is not None
        get_llm.cache_clear()

    def test_default_models_are_set(self):
        from frontier_agents.llm import _DEFAULTS
        assert _DEFAULTS["WRITER_MODEL"] == "anthropic/claude-opus-4-7"
        assert "gemini" in _DEFAULTS["REVIEWER_MODEL"].lower()

    def test_all_roles_resolve(self):
        from frontier_agents.llm import _DEFAULTS
        for role in ("writer", "reviewer", "mvb", "fallback"):
            key = {"writer": "WRITER_MODEL", "reviewer": "REVIEWER_MODEL",
                   "mvb": "MVB_MODEL", "fallback": "FALLBACK_MODEL"}[role]
            assert key in _DEFAULTS
