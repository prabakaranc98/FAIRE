"""Tests for the agent skill loader."""
from __future__ import annotations

from pathlib import Path

import pytest

from frontier_agents.skills import (
    Skill,
    context_tokens_from_state,
    format_skills_block,
    load_skills,
    select_skills,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────

def _write_skill(tmp_path: Path, name: str, applies_to: list, triggers: list, body: str) -> Path:
    content = (
        f"---\nskill: {name}\ndescription: test skill\n"
        f"applies_to: {applies_to}\ntriggers: {triggers}\n---\n\n{body}"
    )
    path = tmp_path / f"{name}.md"
    path.write_text(content)
    return path


# ── load_skills ───────────────────────────────────────────────────────────────

class TestLoadSkills:
    def test_load_skill_from_dir(self, tmp_path):
        _write_skill(tmp_path, "my-skill", ["write_draft"], ["equation"], "# Body\nsome content")
        skills = load_skills(tmp_path)
        assert len(skills) == 1
        s = skills[0]
        assert s.name == "my-skill"
        assert s.applies_to == ["write_draft"]
        assert s.triggers == ["equation"]
        assert "some content" in s.body

    def test_load_multiple_skills_sorted(self, tmp_path):
        _write_skill(tmp_path, "z-skill", ["write_draft"], [], "Z")
        _write_skill(tmp_path, "a-skill", ["write_draft"], [], "A")
        skills = load_skills(tmp_path)
        assert [s.name for s in skills] == ["a-skill", "z-skill"]

    def test_skip_file_without_frontmatter(self, tmp_path):
        (tmp_path / "no-front.md").write_text("# Just a plain file\nno frontmatter")
        skills = load_skills(tmp_path)
        assert skills == []

    def test_missing_dir_returns_empty(self, tmp_path):
        skills = load_skills(tmp_path / "nonexistent")
        assert skills == []

    def test_body_strips_whitespace(self, tmp_path):
        _write_skill(tmp_path, "s", ["write_draft"], [], "  content with spaces  ")
        skills = load_skills(tmp_path)
        assert skills[0].body == "content with spaces"


# ── select_skills ─────────────────────────────────────────────────────────────

class TestSelectSkills:
    def _make_skill(self, name, applies_to, triggers):
        return Skill(name=name, description="", applies_to=applies_to, triggers=triggers, body="")

    def test_select_by_node(self):
        skills = [
            self._make_skill("a", ["write_draft"], ["equation"]),
            self._make_skill("b", ["scratch"], ["equation"]),
        ]
        result = select_skills(skills, "write_draft", ["equation"])
        assert [s.name for s in result] == ["a"]

    def test_select_by_trigger_overlap(self):
        skills = [
            self._make_skill("match", ["write_draft"], ["equation", "latex"]),
            self._make_skill("no-match", ["write_draft"], ["arc", "capstone"]),
        ]
        result = select_skills(skills, "write_draft", ["equation", "benchmark"])
        assert [s.name for s in result] == ["match"]

    def test_empty_triggers_always_matches(self):
        skill = self._make_skill("always", ["write_draft"], [])
        result = select_skills([skill], "write_draft", [])
        assert result == [skill]

    def test_wildcard_trigger_always_matches(self):
        skill = self._make_skill("wiki-prose", ["write_draft"], ["all pages"])
        result = select_skills([skill], "write_draft", ["some", "random", "tokens"])
        assert result == [skill]

    def test_no_match_wrong_node(self):
        skill = self._make_skill("s", ["scratch"], ["equation"])
        result = select_skills([skill], "write_draft", ["equation"])
        assert result == []

    def test_case_insensitive_matching(self):
        skill = self._make_skill("s", ["write_draft"], ["EQUATION", "LaTeX"])
        result = select_skills([skill], "write_draft", ["equation"])
        assert result == [skill]


# ── format_skills_block ───────────────────────────────────────────────────────

class TestFormatSkillsBlock:
    def test_empty_list_returns_empty_string(self):
        assert format_skills_block([]) == ""

    def test_block_contains_skill_name(self):
        skill = Skill("my-skill", "", ["write_draft"], [], "skill body text")
        block = format_skills_block([skill])
        assert "my-skill" in block
        assert "skill body text" in block

    def test_block_contains_header(self):
        skill = Skill("s", "", ["write_draft"], [], "body")
        block = format_skills_block([skill])
        assert "Agent Skills" in block

    def test_multiple_skills_all_included(self):
        skills = [
            Skill("alpha", "", ["write_draft"], [], "alpha body"),
            Skill("beta", "", ["write_draft"], [], "beta body"),
        ]
        block = format_skills_block(skills)
        assert "alpha body" in block
        assert "beta body" in block


# ── context_tokens_from_state ─────────────────────────────────────────────────

class TestContextTokens:
    def test_topic_split_on_dashes(self):
        tokens = context_tokens_from_state({"topic": "score-matching", "track": ""})
        assert "score" in tokens
        assert "matching" in tokens

    def test_track_tokens_included(self):
        tokens = context_tokens_from_state({"topic": "t", "track": "02-generative-modeling"})
        assert "generative" in tokens
        assert "modeling" in tokens

    def test_depth_emphasis_included(self):
        tokens = context_tokens_from_state({
            "topic": "t", "track": "", "depth_emphasis": ["frontier", "applied"]
        })
        assert "frontier" in tokens
        assert "applied" in tokens

    def test_scratch_pad_equation_detected(self):
        tokens = context_tokens_from_state({
            "topic": "t", "track": "",
            "scratch_pad": "The loss function involves an equation with LaTeX",
        })
        assert "equation" in tokens
        assert "latex" in tokens

    def test_writing_plan_benchmark_detected(self):
        tokens = context_tokens_from_state({
            "topic": "t", "track": "",
            "writing_plan": "Include benchmark numbers from MMLU evaluation",
        })
        assert "benchmark" in tokens


# ── Integration: load real skills ─────────────────────────────────────────────

class TestRealSkills:
    """Smoke-test that the actual skills directory is valid."""

    def test_real_skills_load(self):
        from frontier_agents.skills import SKILLS_DIR
        if not SKILLS_DIR.exists():
            pytest.skip("skills dir not found")
        skills = load_skills(SKILLS_DIR)
        assert len(skills) >= 3, f"Expected ≥3 skills, found {len(skills)}"

    def test_write_draft_skills_exist(self):
        from frontier_agents.skills import SKILLS_DIR
        if not SKILLS_DIR.exists():
            pytest.skip("skills dir not found")
        skills = load_skills(SKILLS_DIR)
        write_skills = [s for s in skills if "write_draft" in s.applies_to]
        assert len(write_skills) >= 3, "Need ≥3 skills for write_draft node"

    def test_all_skills_have_body(self):
        from frontier_agents.skills import SKILLS_DIR
        if not SKILLS_DIR.exists():
            pytest.skip("skills dir not found")
        for skill in load_skills(SKILLS_DIR):
            assert skill.body, f"Skill {skill.name} has empty body"
