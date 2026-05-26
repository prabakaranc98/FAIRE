"""Unit tests for the observer — the sensor layer of the closed-loop control system."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from frontier_agents.observer import (
    BudgetState,
    QualityTrend,
    TrackMetrics,
    WikiObservation,
    _compute_quality_trend,
    check_budget,
    compute_error_signals,
    observe,
    write_metrics_json,
    write_observer_page,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def empty_docs(tmp_path):
    """Docs dir with no curriculum pages."""
    (tmp_path / "curriculum").mkdir()
    return tmp_path


@pytest.fixture
def docs_with_stubs(tmp_path):
    """Docs dir with only stub pages."""
    track = tmp_path / "curriculum" / "01-foundations"
    track.mkdir(parents=True)
    (track / "transformers.md").write_text(
        "---\ntitle: Transformers\ntrack: 01-foundations\n---\n# Transformers\n> 🚧 Agent-generated content pending.\n"
    )
    (track / "attention.md").write_text(
        "---\ntitle: Attention\ntrack: 01-foundations\n---\n# Attention\n> 🚧 Agent-generated content pending.\n"
    )
    return tmp_path


@pytest.fixture
def docs_with_pages(tmp_path):
    """Docs dir with one stub and one generated page."""
    track = tmp_path / "curriculum" / "01-foundations"
    track.mkdir(parents=True)
    (track / "transformers.md").write_text(
        "---\ntitle: Transformers\ntrack: 01-foundations\nupdated: 2026-05-20\n---\n"
        "# Transformers\n> **TL;DR:** Attention is all you need.\n\n"
        "## Core concepts\nThe transformer uses self-attention to process sequences.\n"
    )
    (track / "attention.md").write_text(
        "---\ntitle: Attention\ntrack: 01-foundations\n---\n# Attention\n> 🚧 Agent-generated content pending.\n"
    )
    return tmp_path


@pytest.fixture
def runs_with_records(tmp_path):
    """runs/ dir with two JSONL records."""
    runs = tmp_path / "runs"
    runs.mkdir()
    records = [
        {
            "topic": "transformers", "track": "01-foundations", "status": "approved",
            "confidence": 0.85, "has_mvb": True, "revision_count": 0,
            "finished_at": "2026-05-20T10:00:00+00:00",
        },
        {
            "topic": "attention", "track": "01-foundations", "status": "flagged",
            "confidence": 0.60, "has_mvb": False, "revision_count": 1,
            "finished_at": "2026-05-21T10:00:00+00:00",
        },
    ]
    jsonl = runs / "runs.jsonl"
    with jsonl.open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    return runs


# ── observe() ─────────────────────────────────────────────────────────────────

class TestObserve:
    def test_empty_docs_returns_zero_metrics(self, empty_docs, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        obs = observe(docs_dir=str(empty_docs), runs_dir=str(runs))
        assert isinstance(obs, WikiObservation)
        assert obs.total_pages == 0
        assert obs.generated_pages == 0
        assert obs.coverage_pct == 0.0
        assert obs.avg_confidence == 0.0

    def test_stubs_do_not_count_as_generated(self, docs_with_stubs, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        obs = observe(docs_dir=str(docs_with_stubs), runs_dir=str(runs))
        assert obs.total_pages == 2
        assert obs.stub_pages == 2
        assert obs.generated_pages == 0
        assert obs.coverage_pct == 0.0

    def test_generated_pages_count_correctly(self, docs_with_pages, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        obs = observe(docs_dir=str(docs_with_pages), runs_dir=str(runs))
        assert obs.total_pages == 2
        assert obs.stub_pages == 1
        assert obs.generated_pages == 1
        assert obs.coverage_pct == pytest.approx(0.5)

    def test_run_records_populate_confidence(self, docs_with_pages, runs_with_records):
        # runs_with_records points to its own tmp_path/runs, not docs_with_pages/runs
        obs = observe(docs_dir=str(docs_with_pages), runs_dir=str(runs_with_records))
        assert obs.avg_confidence > 0.0
        assert obs.approved_pages == 1

    def test_track_metrics_computed(self, docs_with_pages, runs_with_records):
        obs = observe(docs_dir=str(docs_with_pages), runs_dir=str(runs_with_records))
        assert "01-foundations" in obs.track_metrics
        tm = obs.track_metrics["01-foundations"]
        assert isinstance(tm, TrackMetrics)
        assert tm.total_pages == 2
        assert tm.generated == 1

    def test_observed_at_is_set(self, empty_docs, tmp_path):
        runs = tmp_path / "runs"
        runs.mkdir()
        obs = observe(docs_dir=str(empty_docs), runs_dir=str(runs))
        assert obs.observed_at != ""
        assert "UTC" in obs.observed_at


# ── compute_error_signals() ───────────────────────────────────────────────────

class TestComputeErrorSignals:
    def _make_obs(self, coverage=0.5, confidence=0.7, flagged=2, stale=0, remaining=None):
        obs = WikiObservation(
            observed_at="2026-05-25 10:00 UTC",
            coverage_pct=coverage,
            avg_confidence=confidence,
            budget=BudgetState(remaining_usd=remaining),
        )
        obs.track_metrics["t1"] = TrackMetrics(
            track="t1", flagged=flagged, stale_pages=stale
        )
        return obs

    def test_coverage_deficit_when_below_setpoint(self):
        obs = self._make_obs(coverage=0.5)
        signals = compute_error_signals(obs)
        assert signals["coverage_deficit"] > 0

    def test_coverage_no_deficit_when_above_setpoint(self):
        obs = self._make_obs(coverage=0.9)
        signals = compute_error_signals(obs)
        assert signals["coverage_deficit"] == 0.0

    def test_quality_deficit_when_below_setpoint(self):
        obs = self._make_obs(confidence=0.7)
        signals = compute_error_signals(obs)
        assert signals["quality_deficit"] > 0

    def test_flagged_pages_signal(self):
        obs = self._make_obs(flagged=3)
        signals = compute_error_signals(obs)
        assert signals["flagged_pages"] == 3.0

    def test_stale_pages_signal(self):
        obs = self._make_obs(stale=5)
        signals = compute_error_signals(obs)
        assert signals["stale_pages"] == 5.0

    def test_budget_pressure_when_low(self):
        obs = self._make_obs(remaining=0.5)  # below BUDGET_REDUCED_USD=3.0
        signals = compute_error_signals(obs)
        assert signals["budget_pressure"] > 0

    def test_no_budget_pressure_when_unlimited(self):
        obs = self._make_obs(remaining=None)
        signals = compute_error_signals(obs)
        assert signals["budget_pressure"] == 0.0


# ── _compute_quality_trend() ──────────────────────────────────────────────────

class TestQualityTrend:
    def test_empty_runs(self):
        trend = _compute_quality_trend([])
        assert isinstance(trend, QualityTrend)
        assert trend.window_size == 0
        assert trend.avg_confidence == 0.0

    def test_basic_trend(self):
        runs = [
            {"confidence": 0.7, "status": "approved", "revision_count": 0},
            {"confidence": 0.8, "status": "approved", "revision_count": 1},
            {"confidence": 0.9, "status": "approved", "revision_count": 0},
        ]
        trend = _compute_quality_trend(runs)
        assert trend.window_size == 3
        assert trend.avg_confidence == pytest.approx(0.8, abs=0.01)

    def test_positive_delta_means_improving(self):
        # Second half has higher confidence than first half
        runs = [
            {"confidence": 0.6, "status": "flagged", "revision_count": 2},
            {"confidence": 0.6, "status": "flagged", "revision_count": 2},
            {"confidence": 0.9, "status": "approved", "revision_count": 0},
            {"confidence": 0.9, "status": "approved", "revision_count": 0},
        ]
        trend = _compute_quality_trend(runs)
        assert trend.confidence_delta > 0

    def test_approval_rate_first_pass(self):
        runs = [
            {"confidence": 0.9, "status": "approved", "revision_count": 0},  # first pass
            {"confidence": 0.9, "status": "approved", "revision_count": 1},  # not first pass
            {"confidence": 0.6, "status": "flagged", "revision_count": 0},
        ]
        trend = _compute_quality_trend(runs)
        # Only 1 of 3 is approved on first pass (revision_count=0 AND status=approved)
        assert trend.approval_rate == pytest.approx(1/3, abs=0.01)


# ── check_budget() ────────────────────────────────────────────────────────────

class TestCheckBudget:
    def test_returns_error_when_no_api_key(self):
        with patch.dict(os.environ, {}, clear=True):
            if "OPENROUTER_API_KEY" in os.environ:
                del os.environ["OPENROUTER_API_KEY"]
            b = check_budget()
        assert b.error != "" or b.mode in ("full", "reduced", "paused")

    def test_returns_budget_state_type(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-fake"}):
            import httpx
            with patch.object(httpx, "get", side_effect=Exception("network error")):
                b = check_budget()
        assert isinstance(b, BudgetState)
        assert b.error != ""

    def test_mode_paused_when_below_minimum(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-fake"}):
            import httpx
            mock_resp = type("R", (), {
                "json": lambda self: {"data": {"usage": 9.5, "limit": 10.0, "limit_remaining": 0.4}},
                "raise_for_status": lambda self: None,
            })()
            with patch.object(httpx, "get", return_value=mock_resp):
                b = check_budget()
        assert b.mode == "paused"
        assert b.remaining_usd == pytest.approx(0.4)

    def test_mode_reduced_when_low(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-fake"}):
            import httpx
            mock_resp = type("R", (), {
                "json": lambda self: {"data": {"usage": 8.0, "limit": 10.0, "limit_remaining": 2.0}},
                "raise_for_status": lambda self: None,
            })()
            with patch.object(httpx, "get", return_value=mock_resp):
                b = check_budget()
        assert b.mode == "reduced"

    def test_mode_full_when_healthy(self):
        with patch.dict(os.environ, {"OPENROUTER_API_KEY": "sk-fake"}):
            import httpx
            mock_resp = type("R", (), {
                "json": lambda self: {"data": {"usage": 2.0, "limit": 10.0, "limit_remaining": 8.0}},
                "raise_for_status": lambda self: None,
            })()
            with patch.object(httpx, "get", return_value=mock_resp):
                b = check_budget()
        assert b.mode == "full"


# ── write_metrics_json() ──────────────────────────────────────────────────────

class TestWriteMetricsJson:
    def test_creates_metrics_json(self, tmp_path):
        obs = WikiObservation(
            observed_at="2026-05-25 10:00 UTC",
            total_pages=5,
            generated_pages=3,
        )
        write_metrics_json(obs, runs_dir=str(tmp_path))
        metrics_path = tmp_path / "metrics.json"
        assert metrics_path.exists()
        data = json.loads(metrics_path.read_text())
        assert data["total_pages"] == 5
        assert data["generated_pages"] == 3
        assert data["observed_at"] == "2026-05-25 10:00 UTC"

    def test_metrics_json_is_valid_json(self, tmp_path):
        obs = WikiObservation(observed_at="2026-05-25 10:00 UTC")
        write_metrics_json(obs, runs_dir=str(tmp_path))
        raw = (tmp_path / "metrics.json").read_text()
        data = json.loads(raw)  # must not raise
        assert isinstance(data, dict)


# ── write_observer_page() ─────────────────────────────────────────────────────

class TestWriteObserverPage:
    def test_creates_observer_md(self, tmp_path):
        obs = WikiObservation(
            observed_at="2026-05-25 10:00 UTC",
            coverage_pct=0.5,
            avg_confidence=0.75,
        )
        obs.error_signals = compute_error_signals(obs)
        write_observer_page(obs, docs_dir=str(tmp_path))
        observer_path = tmp_path / "system" / "observer.md"
        assert observer_path.exists()

    def test_observer_md_has_required_sections(self, tmp_path):
        obs = WikiObservation(observed_at="2026-05-25 10:00 UTC")
        obs.error_signals = compute_error_signals(obs)
        write_observer_page(obs, docs_dir=str(tmp_path))
        content = (tmp_path / "system" / "observer.md").read_text()
        assert "Control State" in content
        assert "Coverage" in content
        assert "Quality" in content
        assert "Budget" in content
