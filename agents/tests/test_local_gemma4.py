"""Unit + integration tests against a local Gemma 4 MLX server.

Designed to be run AFTER `scripts/local-setup.sh` has started an MLX server on
http://127.0.0.1:8081 with a Gemma 4 model loaded.

Skips automatically when no local server is reachable, so CI doesn't break.

Run with:
    cd agents && uv run pytest tests/test_local_gemma4.py -v

Test plan:
  Unit:
    1. /v1/models endpoint reachable
    2. Single-token completion within reasonable latency
    3. The writer prompt (small variant) returns markdown with the v2 sections
    4. The reviewer prompt returns parseable JSON-ish structured output
    5. The critic prompt returns a numeric score in expected range
    6. Tool/function-calling smoke (Gemma 4's native agentic feature)

  Integration:
    7. Full graph.invoke() on a tiny topic produces a non-empty draft and a
       review_confidence > 0.0 (i.e. pipeline doesn't crash mid-flow)
"""

from __future__ import annotations

import os
import time

import httpx
import pytest

LOCAL_BASE_URL = os.getenv("LOCAL_LLM_BASE_URL", "http://127.0.0.1:8081/v1")
LOCAL_MODEL = os.getenv(
    "WRITER_MODEL_LOCAL_TEST",
    os.getenv("WRITER_MODEL", "mlx-community/gemma-4-e4b-it-8bit"),
)


def _server_up() -> bool:
    try:
        r = httpx.get(f"{LOCAL_BASE_URL}/models", timeout=2.0)
        return r.status_code == 200
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _server_up(),
    reason=f"Local MLX server not reachable at {LOCAL_BASE_URL} — start one first.",
)


# ────────────────────────────────────────────────────────────────────────────
# Unit tests — each one ~5-30s on Gemma 4 E4B 8bit
# ────────────────────────────────────────────────────────────────────────────

def _chat(messages, max_tokens=200, temperature=0.0, timeout=120.0):
    """Tiny helper around POST /v1/chat/completions."""
    r = httpx.post(
        f"{LOCAL_BASE_URL}/chat/completions",
        json={
            "model": LOCAL_MODEL,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        },
        timeout=timeout,
    )
    r.raise_for_status()
    data = r.json()
    return data["choices"][0]["message"]["content"]


def test_01_models_endpoint():
    """/v1/models returns the loaded model."""
    r = httpx.get(f"{LOCAL_BASE_URL}/models", timeout=3.0)
    assert r.status_code == 200
    data = r.json()
    assert data.get("object") == "list"
    assert len(data.get("data", [])) >= 1
    model_ids = [m["id"] for m in data["data"]]
    print(f"\n  loaded models: {model_ids}")


def test_02_single_token_completion():
    """Sanity: the model returns SOMETHING within 20s for a trivial prompt."""
    t0 = time.time()
    reply = _chat(
        [{"role": "user", "content": "Reply with exactly one word: ready"}],
        max_tokens=10,
        timeout=20.0,
    )
    elapsed = time.time() - t0
    assert reply.strip(), f"empty reply (elapsed={elapsed:.1f}s)"
    print(f"\n  reply={reply!r}  elapsed={elapsed:.1f}s")


def test_03_writer_micro_prompt():
    """The writer can produce a hook + one v2 heading for a known topic."""
    system = (
        "You are a Frontier Wiki editorial agent. Produce a 100-word hook "
        "(no heading) followed by a '## The territory' section (~120 words) "
        "for the topic 'gradient descent'. Output markdown only."
    )
    t0 = time.time()
    out = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": "Write the hook + The territory."},
        ],
        max_tokens=800,
        timeout=120.0,
    )
    elapsed = time.time() - t0
    print(f"\n  elapsed={elapsed:.1f}s  len={len(out)}chars")
    print(f"  preview: {out[:200]!r}")
    assert len(out) > 200, "writer output too short"
    assert "## The territory" in out, "expected '## The territory' heading"


def test_04_reviewer_structured_output():
    """The reviewer returns parseable JSON-ish output for a one-paragraph review."""
    sample_draft = (
        "# Attention\n\n"
        "Attention lets each token query a global memory of all other tokens. "
        "The mechanism is a softmax over QK^T then V multiplication. "
        "Originally introduced in Vaswani et al. (2017)."
    )
    system = (
        "You review wiki page drafts. Return ONLY a JSON object with keys "
        "passed (bool), confidence (0.0-1.0), one_issue (str). No prose."
    )
    out = _chat(
        [
            {"role": "system", "content": system},
            {"role": "user", "content": f"Review this draft:\n\n{sample_draft}"},
        ],
        max_tokens=300,
        timeout=90.0,
    )
    import json, re
    print(f"\n  raw: {out!r}")
    # Try to extract JSON from response
    m = re.search(r"\{.*\}", out, re.DOTALL)
    assert m, f"no JSON object in reviewer output: {out[:200]!r}"
    parsed = json.loads(m.group(0))
    assert "passed" in parsed or "confidence" in parsed, f"missing keys: {parsed}"
    print(f"  parsed: {parsed}")


def test_05_critic_numeric_score():
    """A critic returns a number in 0.0-1.0 range."""
    sample = "Diffusion models train a neural network to reverse a noise-adding process."
    system = (
        "You are a critic scoring the cohesiveness of a wiki paragraph from 0.0 to 1.0. "
        "Reply with ONLY the score, nothing else (e.g. '0.7')."
    )
    out = _chat(
        [{"role": "system", "content": system}, {"role": "user", "content": sample}],
        max_tokens=10,
        timeout=30.0,
    )
    import re
    m = re.search(r"\b(0\.\d+|1\.0|0|1)\b", out)
    assert m, f"no numeric score in: {out!r}"
    score = float(m.group(0))
    assert 0.0 <= score <= 1.0, f"score out of range: {score}"
    print(f"\n  score={score}")


def test_06_tool_calling_smoke():
    """Gemma 4 supports native function calling per the model card. Verify."""
    # OpenAI-compatible tool schema
    tools = [{
        "type": "function",
        "function": {
            "name": "get_paper_title",
            "description": "Look up a paper by arXiv ID",
            "parameters": {
                "type": "object",
                "properties": {"arxiv_id": {"type": "string"}},
                "required": ["arxiv_id"],
            },
        },
    }]
    r = httpx.post(
        f"{LOCAL_BASE_URL}/chat/completions",
        json={
            "model": LOCAL_MODEL,
            "messages": [{
                "role": "user",
                "content": "Look up arXiv 2006.11239",
            }],
            "tools": tools,
            "max_tokens": 200,
            "temperature": 0.0,
        },
        timeout=60.0,
    )
    # MLX server may or may not implement tools yet — soft assertion
    if r.status_code != 200:
        pytest.skip(f"local server doesn't support tools (HTTP {r.status_code})")
    data = r.json()
    msg = data["choices"][0]["message"]
    print(f"\n  message: {msg}")
    # Either a tool_call OR plain content is acceptable here — we just verify
    # the endpoint doesn't crash on a tools-augmented request.
    assert "content" in msg or "tool_calls" in msg


# ────────────────────────────────────────────────────────────────────────────
# Integration — exercise FAIRE's pipeline against the local server
# ────────────────────────────────────────────────────────────────────────────

@pytest.mark.integration
def test_07_full_pipeline_against_local():
    """Run graph.invoke() end-to-end on a small topic. Pipeline must complete
    (no exception) and produce a non-empty draft + a numeric review_confidence."""
    # Force the agent to route to the local server for this test
    os.environ["OPENAI_API_BASE"] = LOCAL_BASE_URL
    os.environ["WRITER_MODEL"] = LOCAL_MODEL
    os.environ["REVIEWER_MODEL"] = LOCAL_MODEL
    os.environ["CRITIC_MODEL"] = LOCAL_MODEL
    os.environ["RESEARCH_MODEL"] = LOCAL_MODEL
    os.environ["MVB_MODEL"] = LOCAL_MODEL
    os.environ["FALLBACK_MODEL"] = LOCAL_MODEL
    # Disable git auto-commit/push for tests
    os.environ["GIT_AUTO_COMMIT"] = "false"
    os.environ["GIT_AUTO_PUSH"] = "false"

    from frontier_agents.graph import compile_wiki_graph
    from frontier_agents.llm import get_llm
    get_llm.cache_clear()  # bust LRU so env changes take effect

    graph = compile_wiki_graph()
    t0 = time.time()
    result = graph.invoke({
        "topic": "gradient-descent",
        "track": "04-neural-networks-deep-learning",
        "depth": "all",
        "mode": "full",
        "page_type": "core-concept",
        "depth_emphasis": ["applied"],
        "arc_context": {},
    })
    elapsed = time.time() - t0
    print(f"\n  pipeline elapsed: {elapsed:.1f}s")
    print(f"  draft length: {len(result.get('draft','') or '')}chars")
    print(f"  review_confidence: {result.get('review_confidence')}")
    print(f"  approved: {result.get('approved')}")

    assert result.get("draft"), "draft must be non-empty"
    assert isinstance(result.get("review_confidence"), (int, float)), "review_confidence missing"
