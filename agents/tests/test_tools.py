"""Unit tests for tools.py — no real API calls."""

import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frontier_agents.tools import (
    _extract_domain,
    git_commit,
    hf_search_datasets,
    hf_search_models,
    load_persona,
    read_stub,
    update_arc_json,
    write_file,
)

PERSONAS_DIR = Path(__file__).parent.parent / "src" / "frontier_agents" / "personas"


class TestReadStub:
    def test_returns_empty_for_nonexistent(self, tmp_path):
        result = read_stub(str(tmp_path / "nonexistent.md"))
        assert result == ""

    def test_returns_content_for_existing(self, tmp_path):
        f = tmp_path / "page.md"
        f.write_text("# Hello\n> TL;DR: test")
        assert read_stub(str(f)) == "# Hello\n> TL;DR: test"


class TestWriteFile:
    def test_creates_file_and_parents(self, tmp_path):
        path = tmp_path / "deep" / "nested" / "page.md"
        write_file(str(path), "# Content")
        assert path.exists()
        assert path.read_text() == "# Content"

    def test_overwrites_existing(self, tmp_path):
        f = tmp_path / "page.md"
        f.write_text("old content")
        write_file(str(f), "new content")
        assert f.read_text() == "new content"


class TestExtractDomain:
    @pytest.mark.parametrize("url,expected", [
        ("https://arxiv.org/abs/2006.11239", "arxiv.org"),
        ("https://huggingface.co/stabilityai/stable-diffusion-xl", "huggingface.co"),
        ("https://cs.stanford.edu/~author/paper.pdf", "cs.stanford.edu"),
        ("https://engineering.linkedin.com/blog/ai", "engineering.linkedin.com"),
    ])
    def test_extracts_correctly(self, url, expected):
        assert _extract_domain(url) == expected


class TestLoadPersona:
    def test_loads_known_track(self):
        persona = load_persona("02-generative-modeling", str(PERSONAS_DIR))
        assert isinstance(persona, dict)
        assert "domain" in persona or "track" in persona

    def test_fallback_for_unknown_track(self):
        persona = load_persona("99-nonexistent-track", str(PERSONAS_DIR))
        assert isinstance(persona, dict)
        assert "track" in persona
        assert persona["track"] == "99-nonexistent-track"

    def test_all_15_tracks_have_personas(self):
        tracks = [
            "01-ai", "02-generative-modeling", "03-representation-learning",
            "04-neural-networks-dl", "05-statistical-probabilistic-ml",
            "06-reinforcement-learning", "07-attention-memory-reasoning",
            "08-causal-statistical-inference", "09-algorithms-systems-ai",
            "10-complexity-cognition", "11-robotics-embodied-ai",
            "12-physics-scientific-ai", "13-graph-relational-ai",
            "14-biology-life-sciences", "15-ml-theory-foundations",
        ]
        for track in tracks:
            p = PERSONAS_DIR / f"{track}.yaml"
            assert p.exists(), f"Missing persona YAML for track: {track}"


class TestGitCommit:
    def test_returns_false_when_auto_commit_disabled(self, tmp_path):
        with patch.dict(os.environ, {"GIT_AUTO_COMMIT": "false"}):
            result = git_commit(str(tmp_path / "page.md"), "test", str(tmp_path))
        assert result is False

    def test_returns_false_on_subprocess_error(self, tmp_path):
        import subprocess
        with patch.dict(os.environ, {"GIT_AUTO_COMMIT": "true"}):
            with patch("subprocess.run", side_effect=subprocess.CalledProcessError(1, "git")):
                result = git_commit(str(tmp_path / "page.md"), "test", str(tmp_path))
        assert result is False


class TestUpdateArcJson:
    def test_patches_node_status(self, tmp_arcs_json):
        update_arc_json(
            "arc-02-generative",
            "diffusion-models",
            {"status": "full", "doc_path": "curriculum/02-generative-modeling/diffusion-models.md"},
            str(tmp_arcs_json),
        )
        data = json.loads(tmp_arcs_json.read_text())
        node = data["arcs"][0]["nodes"][0]
        assert node["status"] == "full"
        assert "doc_path" in node

    def test_unknown_arc_is_ignored(self, tmp_arcs_json):
        original = tmp_arcs_json.read_text()
        update_arc_json("nonexistent-arc", "some-node", {"status": "full"}, str(tmp_arcs_json))
        assert tmp_arcs_json.read_text() == original


class TestHfSearch:
    def test_hf_search_models_parses_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"modelId": "google/ddpm-cifar10-32", "downloads": 50000, "likes": 100, "cardData": {}},
            {"modelId": "CompVis/stable-diffusion-v1-4", "downloads": 1000000, "likes": 5000, "cardData": {}},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            results = hf_search_models("diffusion models", limit=2)

        assert len(results) == 2
        assert results[0]["model_id"] == "google/ddpm-cifar10-32"
        assert results[0]["downloads"] == 50000
        assert "huggingface.co" in results[0]["url"]

    def test_hf_search_datasets_parses_response(self):
        mock_response = MagicMock()
        mock_response.json.return_value = [
            {"id": "cifar10", "downloads": 200000, "likes": 300},
        ]
        mock_response.raise_for_status = MagicMock()

        with patch("httpx.get", return_value=mock_response):
            results = hf_search_datasets("image classification", limit=1)

        assert len(results) == 1
        assert results[0]["dataset_id"] == "cifar10"
        assert "huggingface.co/datasets/cifar10" in results[0]["url"]


class TestExaSearch:
    def test_raises_without_api_key(self):
        from frontier_agents.tools import exa_search
        with patch.dict(os.environ, {}, clear=True):
            env = {k: v for k, v in os.environ.items() if k != "EXA_API_KEY"}
            with patch.dict(os.environ, env, clear=True):
                with pytest.raises(ValueError, match="EXA_API_KEY"):
                    exa_search("diffusion models")
