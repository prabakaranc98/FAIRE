"""Unit tests for LangGraph nodes — all LLM calls mocked."""

import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frontier_agents.nodes import (
    flag_human_review_node,
    link_node,
    load_persona_node,
    merge_mvb_node,
    read_stub_node,
    research_node,
    review_node,
    revise_draft_node,
    route_after_review,
    write_draft_node,
    write_file_node,
)

PERSONAS_DIR = Path(__file__).parent.parent / "src" / "frontier_agents" / "personas"


def _make_mock_llm(content: str = "Mock LLM output"):
    llm = MagicMock()
    response = MagicMock()
    response.content = content
    llm.invoke.return_value = response
    return llm


class TestLoadPersonaNode:
    def test_sets_persona_and_output_path(self, base_state, tmp_path):
        with patch("frontier_agents.nodes._PERSONAS_DIR", PERSONAS_DIR):
            with patch.dict(os.environ, {"WIKI_DOCS_DIR": str(tmp_path)}):
                result = load_persona_node(base_state)
        assert "persona" in result
        assert isinstance(result["persona"], dict)
        assert "diffusion-models.md" in result["output_path"]
        assert "02-generative-modeling" in result["output_path"]

    def test_output_path_uses_env_var(self, base_state, tmp_path):
        custom_docs = str(tmp_path / "custom_docs")
        with patch("frontier_agents.nodes.DOCS_DIR", custom_docs):
            with patch("frontier_agents.nodes._PERSONAS_DIR", PERSONAS_DIR):
                result = load_persona_node(base_state)
        assert custom_docs in result["output_path"]


class TestReadStubNode:
    def test_sets_empty_stub_for_new_topic(self, base_state, tmp_path):
        state = {**base_state, "output_path": str(tmp_path / "nonexistent.md")}
        result = read_stub_node(state)
        assert result["existing_stub"] == ""
        assert result["revision_count"] == 0

    def test_reads_existing_stub(self, base_state, tmp_path):
        stub_file = tmp_path / "diffusion-models.md"
        stub_file.write_text("# Diffusion Models\n> 🚧 Pending")
        state = {**base_state, "output_path": str(stub_file)}
        result = read_stub_node(state)
        assert "Diffusion Models" in result["existing_stub"]
        assert result["revision_count"] == 0


class TestResearchNode:
    def test_handles_exa_failure_gracefully(self, base_state):
        with patch("frontier_agents.nodes.exa_search_papers", side_effect=Exception("no key")):
            with patch("frontier_agents.nodes.exa_search_sota", return_value=[]):
                with patch("frontier_agents.nodes.exa_search_production", return_value=[]):
                    with patch("frontier_agents.nodes.hf_search_models", return_value=[]):
                        with patch("frontier_agents.nodes.hf_search_datasets", return_value=[]):
                            result = research_node(base_state)
        assert "research_results" in result
        assert isinstance(result["research_results"], list)

    def test_deduplicates_by_url(self, base_state):
        dup = {"url": "https://arxiv.org/abs/123", "title": "T", "text": "t", "source_type": "paper"}
        with patch("frontier_agents.nodes.exa_search_papers", return_value=[dup, dup]):
            with patch("frontier_agents.nodes.exa_search_sota", return_value=[]):
                with patch("frontier_agents.nodes.exa_search_production", return_value=[]):
                    with patch("frontier_agents.nodes.hf_search_models", return_value=[]):
                        with patch("frontier_agents.nodes.hf_search_datasets", return_value=[]):
                            result = research_node(base_state)
        urls = [r["url"] for r in result["research_results"]]
        assert len(urls) == len(set(urls))


class TestWriteDraftNode:
    def test_sets_draft_in_state(self, base_state):
        # write_draft_node makes a single LLM call (single-pass, not chunked)
        mock_llm = _make_mock_llm("# Diffusion Models\nFull page content here.")
        with patch("frontier_agents.nodes.get_llm", return_value=mock_llm):
            result = write_draft_node(base_state)
        assert "# Diffusion Models" in result["draft"]
        assert mock_llm.invoke.call_count == 1  # single pass

    def test_includes_existing_stub_in_prompt(self, base_state):
        state = {
            **base_state,
            "existing_stub": "# Diffusion Models\nThis is existing content without the stub marker.",
        }
        mock_llm = _make_mock_llm("Revised content.")
        with patch("frontier_agents.nodes.get_llm", return_value=mock_llm):
            write_draft_node(state)
        # IMPROVE note goes into the first chunk call (index 0)
        first_call_args = mock_llm.invoke.call_args_list[0][0][0]
        prompt_text = " ".join(str(m.content) for m in first_call_args)
        assert "IMPROVE" in prompt_text


class TestReviewNode:
    def test_structured_output_path(self, draft_state):
        from frontier_agents.nodes import ReviewResult
        mock_result = ReviewResult(
            passed=True,
            confidence=0.92,
            issues=[],
            suggestions=["Add more LaTeX detail"],
        )
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("frontier_agents.nodes.get_llm", return_value=mock_llm):
            result = review_node(draft_state)

        assert result["review_pass"] is True
        assert result["review_confidence"] == 0.92
        assert result["approved"] is True

    def test_approved_requires_confidence_above_threshold(self, draft_state):
        """Confidence below GIT_COMMIT_THRESHOLD means not approved."""
        from frontier_agents.nodes import ReviewResult
        # Use 0.5 — well below any reasonable threshold (0.7 or 0.8)
        mock_result = ReviewResult(passed=True, confidence=0.5, issues=["Missing sources"], suggestions=[])
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.return_value = mock_result

        with patch("frontier_agents.nodes.get_llm", return_value=mock_llm):
            result = review_node(draft_state)

        assert result["review_pass"] is True
        assert result["approved"] is False  # confidence below threshold

    def test_fallback_text_parsing_on_structured_failure(self, draft_state):
        mock_llm = MagicMock()
        mock_llm.with_structured_output.return_value.invoke.side_effect = Exception("structured failed")
        fallback_response = MagicMock()
        fallback_response.content = "PASS confidence 0.8 looks good"
        mock_llm.invoke.return_value = fallback_response

        with patch("frontier_agents.nodes.get_llm", return_value=mock_llm):
            result = review_node(draft_state)

        assert "review_feedback" in result
        assert result["review_pass"] is True


class TestReviseDraftNode:
    def test_increments_revision_count(self, draft_state):
        state = {**draft_state, "revision_count": 0, "review_feedback": "Fix section headers"}
        mock_llm = _make_mock_llm("Revised page content.")
        with patch("frontier_agents.nodes.get_llm", return_value=mock_llm):
            result = revise_draft_node(state)
        assert result["revision_count"] == 1
        assert result["draft"] == "Revised page content."

    def test_increments_from_existing_count(self, draft_state):
        state = {**draft_state, "revision_count": 1, "review_feedback": "Still issues"}
        mock_llm = _make_mock_llm("Second revision.")
        with patch("frontier_agents.nodes.get_llm", return_value=mock_llm):
            result = revise_draft_node(state)
        assert result["revision_count"] == 2


class TestMergeMvbNode:
    def test_wraps_mvb_in_skeleton_for_empty_stub(self, base_state):
        state = {
            **base_state,
            "existing_stub": "",
            "mvb_section": "## Minimum Valuable Build\n**What:** Build a diffusion model.",
        }
        result = merge_mvb_node(state)
        assert "## Minimum Valuable Build" in result["draft"]
        assert "has_mvb: true" in result["draft"]

    def test_replaces_existing_mvb_section(self, base_state):
        existing = """# Diffusion Models

## Minimum Valuable Build
Old MVB content here.

## Code & implementations
Links here.
"""
        new_mvb = "## Minimum Valuable Build\nNew and improved MVB."
        state = {**base_state, "existing_stub": existing, "mvb_section": new_mvb}
        result = merge_mvb_node(state)
        assert "New and improved MVB" in result["draft"]
        assert "Old MVB content here" not in result["draft"]
        assert "## Code & implementations" in result["draft"]

    def test_inserts_before_code_section(self, base_state):
        existing = "# Diffusion Models\n\n## Code & implementations\nLinks.\n"
        new_mvb = "## Minimum Valuable Build\nBrand new MVB."
        state = {**base_state, "existing_stub": existing, "mvb_section": new_mvb}
        result = merge_mvb_node(state)
        draft = result["draft"]
        assert draft.index("Minimum Valuable Build") < draft.index("Code & implementations")


class TestRouteAfterReview:
    def test_approved_routes_to_write_file(self, base_state):
        state = {**base_state, "approved": True, "revision_count": 0}
        assert route_after_review(state) == "write_file"

    def test_rejected_first_attempt_routes_to_revise(self, base_state):
        state = {**base_state, "approved": False, "revision_count": 0}
        assert route_after_review(state) == "revise_draft"

    def test_rejected_second_attempt_routes_to_revise(self, base_state):
        state = {**base_state, "approved": False, "revision_count": 1}
        assert route_after_review(state) == "revise_draft"

    def test_rejected_after_max_revisions_flags_human(self, base_state):
        state = {**base_state, "approved": False, "revision_count": 2}
        assert route_after_review(state) == "flag_human_review"


class TestWriteFileNode:
    def test_writes_draft_to_output_path(self, draft_state, tmp_path):
        state = {**draft_state, "output_path": str(tmp_path / "out" / "page.md")}
        with patch("frontier_agents.nodes.write_file") as mock_write:
            result = write_file_node(state)
        mock_write.assert_called_once()
        assert result["final_content"] == draft_state["draft"]


class TestFlagHumanReviewNode:
    def test_sets_approved_false_and_committed_false(self, base_state):
        state = {
            **base_state,
            "output_path": "/tmp/page.md",
            "review_confidence": 0.45,
            "review_feedback": "Major issues found.",
        }
        result = flag_human_review_node(state)
        assert result["approved"] is False
        assert result["committed"] is False


class TestLinkNode:
    def _draft_with_connected_section(self):
        return """---
title: Score Matching
track: 02-generative-modeling
---
# Score Matching
> TL;DR: Learns the score function of a data distribution.

## Core concepts
Score matching learns to estimate the gradient of the log-density.

## Connected topics
- [[diffusion-models]]
- [[flow-matching]]

## Further reading
- [Some paper](https://arxiv.org/abs/xxxx)
"""

    def test_returns_state_unchanged_when_no_docs(self, base_state, tmp_path):
        """With no existing pages on disk, link_node is a no-op."""
        state = {**base_state, "draft": self._draft_with_connected_section()}
        with patch("frontier_agents.nodes.DOCS_DIR", str(tmp_path)):
            result = link_node(state)
        assert result["draft"] == state["draft"]

    def test_returns_state_unchanged_when_no_draft(self, base_state):
        """Empty draft → no-op."""
        state = {**base_state, "draft": ""}
        result = link_node(state)
        assert result["draft"] == ""

    def test_injects_real_links_for_existing_pages(self, base_state, tmp_path):
        """When existing pages are found, real links replace placeholders."""
        # Create a real page on disk
        curriculum = tmp_path / "curriculum" / "02-generative-modeling"
        curriculum.mkdir(parents=True)
        (curriculum / "diffusion-models.md").write_text(
            "---\ntitle: Diffusion Models\ntrack: 02-generative-modeling\n---\n"
            "# Diffusion Models\n## Core concepts\nDiffusion models reverse a noise process.\n"
        )

        draft = self._draft_with_connected_section()
        state = {
            **base_state,
            "topic": "score-matching",
            "track": "02-generative-modeling",
            "draft": draft,
        }

        # Mock the LLM call to return a deterministic link list
        mock_response = MagicMock()
        mock_response.content = (
            '[{"slug": "02-generative-modeling/diffusion-models", '
            '"title": "Diffusion Models", '
            '"relationship": "Score matching is the theoretical foundation of diffusion models"}]'
        )

        with patch("frontier_agents.nodes.DOCS_DIR", str(tmp_path)):
            with patch("frontier_agents.nodes.get_llm") as mock_get_llm:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = mock_response
                mock_get_llm.return_value = mock_llm
                result = link_node(state)

        # The draft should now contain a real link to diffusion-models.md
        assert "[Diffusion Models](./diffusion-models.md)" in result["draft"]

    def test_never_links_to_nonexistent_pages(self, base_state, tmp_path):
        """LLM may hallucinate slugs — link_node must drop them."""
        curriculum = tmp_path / "curriculum" / "02-generative-modeling"
        curriculum.mkdir(parents=True)
        # No files written — catalog is empty

        draft = self._draft_with_connected_section()
        state = {**base_state, "draft": draft}

        # LLM tries to link to a page that doesn't exist
        mock_response = MagicMock()
        mock_response.content = (
            '[{"slug": "02-generative-modeling/nonexistent-page", '
            '"title": "Nonexistent Page", "relationship": "made up"}]'
        )

        with patch("frontier_agents.nodes.DOCS_DIR", str(tmp_path)):
            with patch("frontier_agents.nodes.get_llm") as mock_get_llm:
                mock_llm = MagicMock()
                mock_llm.invoke.return_value = mock_response
                mock_get_llm.return_value = mock_llm
                result = link_node(state)

        # No link to nonexistent page should appear
        assert "nonexistent-page" not in result["draft"]

    def test_never_links_to_self(self, base_state, tmp_path):
        """Current page must be excluded from link candidates."""
        curriculum = tmp_path / "curriculum" / "02-generative-modeling"
        curriculum.mkdir(parents=True)
        (curriculum / "diffusion-models.md").write_text(
            "---\ntitle: Diffusion Models\ntrack: 02-generative-modeling\n---\n"
            "# Diffusion Models\nReal content here.\n"
        )

        # State represents the diffusion-models page itself
        state = {
            **base_state,
            "topic": "diffusion-models",
            "track": "02-generative-modeling",
            "draft": self._draft_with_connected_section(),
        }

        from frontier_agents.nodes import _index_existing_pages
        docs_path = tmp_path
        pages = _index_existing_pages(docs_path)

        # Self-slug must be excluded before LLM call
        self_slug = "02-generative-modeling/diffusion-models"
        assert self_slug not in pages or pages.pop(self_slug, None) is not None

    def test_does_not_crash_on_llm_failure(self, base_state, tmp_path):
        """LLM error → returns original draft unchanged (publishing never blocked)."""
        curriculum = tmp_path / "curriculum" / "02-generative-modeling"
        curriculum.mkdir(parents=True)
        (curriculum / "flow-matching.md").write_text(
            "---\ntitle: Flow Matching\ntrack: 02-generative-modeling\n---\nReal content.\n"
        )

        draft = self._draft_with_connected_section()
        state = {**base_state, "draft": draft}

        with patch("frontier_agents.nodes.DOCS_DIR", str(tmp_path)):
            with patch("frontier_agents.nodes.get_llm") as mock_get_llm:
                mock_llm = MagicMock()
                mock_llm.invoke.side_effect = RuntimeError("LLM API down")
                mock_get_llm.return_value = mock_llm
                result = link_node(state)

        # Draft must be returned as-is, pipeline must not crash
        assert result["draft"] == draft
