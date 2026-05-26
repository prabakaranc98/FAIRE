"""Load and apply agent skill files from agents/skills/.

Skills are Markdown files with YAML frontmatter specifying when they apply.
The loader reads trigger lists and injects relevant skill bodies into prompts.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import NamedTuple

import yaml

SKILLS_DIR = Path(__file__).parent.parent.parent / "skills"

# These trigger values cause the skill to match any context (wildcards).
_WILDCARD_TRIGGERS = {"all pages", "always", "*"}


class Skill(NamedTuple):
    name: str
    description: str
    applies_to: list[str]
    triggers: list[str]
    body: str


def load_skills(skills_dir: Path = SKILLS_DIR) -> list[Skill]:
    """Read all *.md skill files and parse frontmatter + body.

    Files without valid YAML frontmatter are silently skipped.
    """
    skills: list[Skill] = []
    if not skills_dir.exists():
        return skills
    for path in sorted(skills_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        match = re.match(r"^---\n(.+?)\n---\n(.+)$", text, re.DOTALL)
        if not match:
            continue
        try:
            meta = yaml.safe_load(match.group(1)) or {}
        except yaml.YAMLError:
            continue
        body = match.group(2).strip()
        skills.append(Skill(
            name=meta.get("skill", path.stem),
            description=meta.get("description", ""),
            applies_to=meta.get("applies_to") or [],
            triggers=meta.get("triggers") or [],
            body=body,
        ))
    return skills


def select_skills(
    skills: list[Skill],
    node: str,
    context_tokens: list[str],
) -> list[Skill]:
    """Return skills whose applies_to includes node and whose triggers overlap context.

    A skill with no triggers or a wildcard trigger ("all pages") always matches.
    """
    ctx = {t.lower() for t in context_tokens if t}
    selected: list[Skill] = []
    for skill in skills:
        if node not in skill.applies_to:
            continue
        triggers = {t.lower() for t in skill.triggers}
        if not triggers or triggers & _WILDCARD_TRIGGERS or triggers & ctx:
            selected.append(skill)
    return selected


def format_skills_block(skills: list[Skill]) -> str:
    """Format selected skills into a prompt appendix block."""
    if not skills:
        return ""
    parts = ["\n\n---\n## Agent Skills (active for this page)\n"]
    for skill in skills:
        parts.append(f"\n### {skill.name}\n\n{skill.body}")
    return "".join(parts)


def context_tokens_from_state(state: dict) -> list[str]:
    """Extract keyword tokens from page state for skill trigger matching."""
    tokens: list[str] = []

    # Topic and track names: split on dashes and spaces
    for field in ("topic", "track"):
        val = state.get(field, "")
        tokens.extend(re.split(r"[-_\s]+", val.lower()))

    # Page type and depth emphasis
    tokens.append(state.get("page_type", "").lower())
    tokens.extend(d.lower() for d in state.get("depth_emphasis", []))

    # Scan scratch_pad and writing_plan for high-value trigger keywords
    text = (state.get("scratch_pad", "") + " " + state.get("writing_plan", "")).lower()
    for kw in [
        "equation", "latex", "formula", "derivation",
        "mvb", "build", "huggingface",
        "benchmark", "sota", "evaluation",
        "arc", "capstone", "integration",
        "citation", "paper", "arxiv",
    ]:
        if kw in text:
            tokens.append(kw)

    return [t for t in tokens if t]
