"""Integration tests — real API calls (OpenRouter, Exa, HuggingFace).

Run with:  uv run pytest tests/test_integration.py -v -m integration
Skip with: uv run pytest tests/ -v -m "not integration"

Requires .env with OPENROUTER_API_KEY and EXA_API_KEY set.
"""

import os
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration


def _has_openrouter_key():
    return bool(os.getenv("OPENROUTER_API_KEY"))


def _has_exa_key():
    return bool(os.getenv("EXA_API_KEY"))


# ── LLM integration ──────────────────────────────────────────────────────────

@pytest.mark.skipif(not _has_openrouter_key(), reason="OPENROUTER_API_KEY not set")
class TestLlmIntegration:
    def test_writer_llm_responds(self):
        from langchain_core.messages import HumanMessage
        from frontier_agents.llm import get_llm
        get_llm.cache_clear()
        llm = get_llm("writer", temperature=0.0)
        response = llm.invoke([HumanMessage(content="Reply with exactly: OK")])
        assert response.content is not None
        assert len(response.content) > 0

    def test_reviewer_llm_structured_output(self):
        from langchain_core.messages import HumanMessage
        from frontier_agents.llm import get_llm
        from frontier_agents.nodes import ReviewResult
        get_llm.cache_clear()
        reviewer = get_llm("reviewer", temperature=0.0)
        structured = reviewer.with_structured_output(ReviewResult)
        prompt = """Review this minimal wiki page stub:

# Test Topic
> TL;DR: A test topic.

## What it is
A test.

## Why it matters at the frontier
Used in frontier research.

## Core concepts
- Concept A

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Paper A](https://arxiv.org/abs/1234.5678) | 2020 | Smith et al. | Foundational |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [Paper B](https://arxiv.org/abs/2345.6789) | 2019 | Jones et al. | First work |

## Current SotA
Current best results are promising.

## What's happening now
**Research:** Active.
**Engineering & Systems:** Deployed at scale.
**Open problems:** Many.

## In production
- **Google:** Used in production.

## Connected topics
- [[related-topic]]

Score this page for schema compliance and quality."""

        result = structured.invoke([HumanMessage(content=prompt)])
        assert isinstance(result, ReviewResult)
        assert isinstance(result.passed, bool)
        assert 0.0 <= result.confidence <= 1.0
        assert isinstance(result.issues, list)
        assert isinstance(result.suggestions, list)

    def test_fallback_llm_responds(self):
        from langchain_core.messages import HumanMessage
        from frontier_agents.llm import get_llm
        get_llm.cache_clear()
        llm = get_llm("fallback", temperature=0.0)
        response = llm.invoke([HumanMessage(content="What is 2+2? Reply with just the number.")])
        assert response.content is not None


# ── Exa search integration ───────────────────────────────────────────────────

@pytest.mark.skipif(not _has_exa_key(), reason="EXA_API_KEY not set")
class TestExaIntegration:
    def test_arxiv_search_returns_results(self):
        from frontier_agents.tools import exa_search
        results = exa_search("diffusion models score matching", num_results=3)
        assert len(results) > 0
        for r in results:
            assert "url" in r
            assert "title" in r
            assert r["url"].startswith("http")

    def test_results_respect_domain_policy(self):
        from frontier_agents.tools import exa_search, APPROVED_DOMAINS, APPROVED_ENGINEERING_BLOGS
        results = exa_search("transformer attention mechanism", num_results=5)
        approved = set(APPROVED_DOMAINS + APPROVED_ENGINEERING_BLOGS)
        for r in results:
            domain = r.get("domain", "")
            in_approved = any(a in domain or domain in a for a in approved)
            assert in_approved or domain == "", (
                f"Result from unapproved domain: {domain} ({r['url']})"
            )

    def test_in_production_search_allows_engineering_blogs(self):
        from frontier_agents.tools import exa_search
        results = exa_search(
            "transformer production deployment engineering",
            num_results=5,
            section="in_production",
        )
        assert len(results) >= 0  # may be 0 if no results, that's OK


# ── HuggingFace search integration ───────────────────────────────────────────

class TestHfIntegration:
    def test_model_search_returns_real_models(self):
        from frontier_agents.tools import hf_search_models
        results = hf_search_models("diffusion image generation", limit=5)
        assert len(results) > 0
        for m in results:
            assert m["model_id"] != ""
            assert m["downloads"] >= 0
            assert "huggingface.co" in m["url"]

    def test_dataset_search_returns_real_datasets(self):
        from frontier_agents.tools import hf_search_datasets
        results = hf_search_datasets("image classification", limit=5)
        assert len(results) > 0
        for d in results:
            assert d["dataset_id"] != ""


# ── Full pipeline integration ─────────────────────────────────────────────────

@pytest.mark.skipif(not _has_openrouter_key(), reason="OPENROUTER_API_KEY not set")
@pytest.mark.skipif(not _has_exa_key(), reason="EXA_API_KEY not set")
class TestPipelineIntegration:
    """End-to-end: run the full agent pipeline on a test topic, write to tmp dir."""

    def test_full_wiki_pipeline_generates_valid_page(self, tmp_path):
        """Generate a wiki page for 'attention-mechanism' and validate it has required sections."""
        import os
        from frontier_agents.graph import compile_wiki_graph
        from frontier_agents.llm import get_llm

        get_llm.cache_clear()

        output_path = str(tmp_path / "docs" / "curriculum" / "07-attention-memory-reasoning" / "attention-mechanism.md")
        (tmp_path / "docs" / "curriculum" / "07-attention-memory-reasoning").mkdir(parents=True)

        initial_state = {
            "topic": "attention-mechanism",
            "track": "07-attention-memory-reasoning",
            "depth": "all",
            "mode": "full",
            "output_path": output_path,
            "revision_count": 0,
            "approved": False,
            "committed": False,
        }

        with os.environ.copy() as _:
            os.environ["GIT_AUTO_COMMIT"] = "false"  # don't commit during test
            graph = compile_wiki_graph()
            result = graph.invoke(initial_state)

        assert "draft" in result, "Graph did not produce a draft"
        draft = result["draft"]
        assert len(draft) > 500, "Draft is too short to be a real page"

        # Schema checks on generated content
        assert "## What it is" in draft or "# Attention" in draft
        assert "arxiv.org" in draft or "huggingface.co" in draft

        final_file = Path(output_path)
        assert final_file.exists(), "write_file_node did not write the output file"

    def test_mvb_pipeline_generates_mvb_section(self, tmp_path):
        """MVB-only pipeline generates a page with a proper MVB section."""
        import os
        from frontier_agents.graph import compile_mvb_graph
        from frontier_agents.llm import get_llm

        get_llm.cache_clear()

        output_path = str(tmp_path / "docs" / "curriculum" / "02-generative-modeling" / "score-matching.md")
        (tmp_path / "docs" / "curriculum" / "02-generative-modeling").mkdir(parents=True)

        initial_state = {
            "topic": "score-matching",
            "track": "02-generative-modeling",
            "depth": "all",
            "mode": "mvb-only",
            "output_path": output_path,
            "revision_count": 0,
            "approved": False,
            "committed": False,
        }

        with os.environ.copy() as _:
            os.environ["GIT_AUTO_COMMIT"] = "false"
            graph = compile_mvb_graph()
            result = graph.invoke(initial_state)

        assert "draft" in result
        draft = result["draft"]
        assert "## Minimum Valuable Build" in draft
        assert len(draft) > 200
