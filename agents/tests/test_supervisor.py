"""Unit tests for the supervising agent — all LLM calls mocked."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from frontier_agents.supervisor import (
    WikiHealth,
    SupervisorAction,
    SupervisorReport,
    _assess_wiki,
    _build_action_list,
    run_supervisor,
)
from frontier_agents.audit import AuditReport


def _make_mock_llm(content: str = "The wiki is in good shape."):
    llm = MagicMock()
    response = MagicMock()
    response.content = content
    llm.invoke.return_value = response
    return llm


class TestWikiHealth:
    def test_default_values(self):
        h = WikiHealth()
        assert h.total_pages == 0
        assert h.stub_pages == 0
        assert h.avg_confidence == 0.0
        assert h.tracks_total == 15

    def test_assess_wiki_with_empty_docs(self, tmp_path):
        health = _assess_wiki(tmp_path, tmp_path / "runs")
        assert health.total_pages == 0
        assert health.generated_pages == 0

    def test_assess_wiki_counts_stubs(self, tmp_path):
        curriculum = tmp_path / "curriculum"
        track = curriculum / "02-generative-modeling"
        track.mkdir(parents=True)
        (track / "diffusion-models.md").write_text("# Diffusion\n> 🚧 Agent-generated content pending")
        (track / "index.md").write_text("# Index")

        health = _assess_wiki(tmp_path, tmp_path / "runs")
        assert health.total_pages == 1
        assert health.stub_pages == 1

    def test_assess_wiki_reads_runs_jsonl(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        record = {
            "run_id": "abc", "topic": "diffusion-models", "track": "02-generative-modeling",
            "status": "approved", "confidence": 0.92, "has_mvb": True,
            "page_type": "core-concept", "depth_emphasis": ["applied"],
            "finished_at": "2026-05-25T10:00:00+00:00",
        }
        (runs_dir / "runs.jsonl").write_text(json.dumps(record) + "\n")

        health = _assess_wiki(tmp_path, runs_dir)
        assert health.generated_pages == 1
        assert health.approved_pages == 1
        assert health.avg_confidence == pytest.approx(0.92)
        assert health.pages_with_mvb == 1


class TestBuildActionList:
    def test_flagged_pages_get_priority_1(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        record = {
            "topic": "score-matching", "track": "02-generative-modeling",
            "status": "flagged", "confidence": 0.3,
            "page_type": "core-concept", "depth_emphasis": ["theoretical"],
        }
        (runs_dir / "runs.jsonl").write_text(json.dumps(record) + "\n")

        health = WikiHealth(flagged_pages=1)
        audit = AuditReport(generated_at="2026-05-25", docs_dir=str(tmp_path))
        actions = _build_action_list(health, audit, runs_dir, tmp_path)

        assert len(actions) >= 1
        flag_actions = [a for a in actions if a.topic == "score-matching"]
        assert flag_actions[0].priority == 1

    def test_stub_pages_get_priority_3(self, tmp_path):
        curriculum = tmp_path / "curriculum" / "02-generative-modeling"
        curriculum.mkdir(parents=True)
        (curriculum / "score-matching.md").write_text("# Score Matching\n> 🚧 Agent-generated content pending")

        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        health = WikiHealth()
        audit = AuditReport(generated_at="2026-05-25", docs_dir=str(tmp_path))
        actions = _build_action_list(health, audit, runs_dir, tmp_path)

        stub_actions = [a for a in actions if a.topic == "score-matching"]
        assert len(stub_actions) >= 1
        assert stub_actions[0].priority == 3
        assert stub_actions[0].action == "generate"

    def test_actions_sorted_by_priority(self, tmp_path):
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        records = [
            {"topic": "z-topic", "track": "02-generative-modeling",
             "status": "flagged", "confidence": 0.2,
             "page_type": "core-concept", "depth_emphasis": ["applied"]},
        ]
        jsonl = "\n".join(json.dumps(r) for r in records)
        (runs_dir / "runs.jsonl").write_text(jsonl)

        health = WikiHealth()
        audit = AuditReport(generated_at="2026-05-25", docs_dir=str(tmp_path))
        actions = _build_action_list(health, audit, runs_dir, tmp_path)

        priorities = [a.priority for a in actions]
        assert priorities == sorted(priorities)


class TestSupervisorReport:
    def test_to_markdown_contains_health_table(self):
        health = WikiHealth(total_pages=10, stub_pages=5, generated_pages=5, approved_pages=3)
        audit = AuditReport(generated_at="2026-05-25", docs_dir="/tmp/docs")
        report = SupervisorReport(
            generated_at="2026-05-25 10:00 UTC",
            health=health,
            audit=audit,
            actions=[],
            llm_analysis="The wiki is growing steadily.",
        )
        md = report.to_markdown()
        assert "Wiki Health" in md
        assert "10" in md  # total pages
        assert "The wiki is growing steadily." in md

    def test_to_markdown_shows_prioritised_actions(self):
        health = WikiHealth()
        audit = AuditReport(generated_at="2026-05-25", docs_dir="/tmp/docs")
        actions = [
            SupervisorAction(
                priority=1, action="improve", topic="score-matching",
                track="02-generative-modeling", page_type="core-concept",
                depth_emphasis=["theoretical"], reason="Low confidence",
            )
        ]
        report = SupervisorReport(
            generated_at="2026-05-25 10:00 UTC",
            health=health, audit=audit, actions=actions,
        )
        md = report.to_markdown()
        assert "score-matching" in md
        assert "critical" in md.lower() or "1" in md


class TestRunSupervisor:
    def test_dry_run_does_not_write_sprint(self, tmp_path):
        sprints_dir = tmp_path / "sprints"
        sprints_dir.mkdir()
        sprint_file = sprints_dir / "current.md"

        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()

        with patch("frontier_agents.supervisor.get_llm", return_value=_make_mock_llm()):
            with patch("frontier_agents.supervisor.audit_wiki") as mock_audit:
                mock_audit.return_value = AuditReport(
                    generated_at="2026-05-25", docs_dir=str(docs_dir)
                )
                report = run_supervisor(
                    docs_dir=str(docs_dir),
                    runs_dir=str(runs_dir),
                    sprints_dir=str(sprints_dir),
                    dry_run=True,
                )

        assert not sprint_file.exists()  # dry run → no sprint write
        assert report is not None
        assert isinstance(report.health, WikiHealth)

    def test_non_dry_run_writes_supervisor_report(self, tmp_path):
        docs_dir = tmp_path / "docs"
        docs_dir.mkdir()
        system_dir = docs_dir / "system"
        system_dir.mkdir()
        runs_dir = tmp_path / "runs"
        runs_dir.mkdir()
        sprints_dir = tmp_path / "sprints"
        sprints_dir.mkdir()

        with patch("frontier_agents.supervisor.get_llm", return_value=_make_mock_llm()):
            with patch("frontier_agents.supervisor.audit_wiki") as mock_audit:
                mock_audit.return_value = AuditReport(
                    generated_at="2026-05-25", docs_dir=str(docs_dir)
                )
                run_supervisor(
                    docs_dir=str(docs_dir),
                    runs_dir=str(runs_dir),
                    sprints_dir=str(sprints_dir),
                    dry_run=False,
                )

        assert (system_dir / "supervisor.md").exists()
