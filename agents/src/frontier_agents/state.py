"""WikiPageState — the shared state TypedDict passed through all agent nodes."""

from __future__ import annotations

from typing import Literal, TypedDict


class WikiPageState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    topic: str                          # e.g., "diffusion-models"
    track: str                          # e.g., "02-generative-modeling"
    depth: Literal["all", "applied", "foundations", "research"]
    mode: Literal["full", "mvb-only", "review-only", "update"]

    # ── Arc context (set by CLI; tells agent where this page sits) ─────────────
    page_type: Literal["arc-entry", "core-concept", "supporting"]
    # arc-entry: opens an arc; MVB unless cost-prohibitive
    # core-concept: pivotal mid-arc; agent judges if it earns MVB
    # supporting: builds vocabulary for parent; NO standalone MVB

    depth_emphasis: list[str]           # ["applied"] | ["theoretical"] | ["frontier"] | combo
    # applied:     longer MVB, JAX/code examples, serving/inference angle
    # theoretical: derivation sketches, proof intuition, key theorems
    # frontier:    open problems as specific questions, named papers, benchmark numbers

    arc_context: dict                   # {"arc_id": "generative-stack", "position": 3,
                                        #  "prev": "vae", "next": "score-matching"}
    mvb_decision: bool                  # resolved by plan_node from page_type + depth_emphasis

    # ── Loaded at runtime ──────────────────────────────────────────────────────
    persona: dict                       # from personas/[track].yaml
    existing_stub: str                  # current file content if exists (may be empty)
    output_path: str                    # e.g., docs/curriculum/02-generative-modeling/diffusion-models.md

    # ── Research phase ─────────────────────────────────────────────────────────
    research_results: list[dict]        # Exa paper search results; [{url, title, text, domain}]
    sota_results: list[dict]            # Exa SotA search (2024+); [{url, title, highlights}]
    production_results: list[dict]      # Exa engineering blog results; [{url, title, summary}]
    hf_models: list[dict]               # HuggingFace model search results (for MVB)
    hf_datasets: list[dict]             # HuggingFace dataset search results (for MVB)

    # ── Planning phase ────────────────────────────────────────────────────────
    writing_plan: str                   # 200-300 word plan from plan_node
    scratch_pad: str                    # compiled fact sheet from scratch_node:
                                        # verified citations, equations, prod examples,
                                        # MVB stack, opening scenario, open problem

    # ── Writing phase ──────────────────────────────────────────────────────────
    draft: str                          # full markdown page draft
    mvb_section: str                    # Minimum Valuable Build section (if mode=mvb-only)

    # ── Review phase ───────────────────────────────────────────────────────────
    review_feedback: str                # structured feedback from reviewer agent
    review_confidence: float            # 0.0-1.0; <0.8 flags for human review
    review_pass: bool                   # True if reviewer approved
    revision_count: int                 # number of revision loops (max 2)

    # ── Output phase ───────────────────────────────────────────────────────────
    final_content: str                  # approved page content
    approved: bool                      # True when reviewer approved
    committed: bool                     # True when git commit succeeded
    error: str                          # error message if any step failed

    # ── Run tracking (new) ────────────────────────────────────────────────────
    run_id: str                         # UUID for this generation run
    started_at: str                     # ISO timestamp when run began
    finished_at: str                    # ISO timestamp when run ended
