"""WikiPageState — the shared state TypedDict passed through all agent nodes."""

from __future__ import annotations

from typing import Literal, TypedDict


class WikiPageState(TypedDict, total=False):
    # ── Input ──────────────────────────────────────────────────────────────────
    topic: str                          # e.g., "diffusion-models"
    track: str                          # e.g., "02-generative-modeling"
    depth: Literal["all", "applied", "foundations", "research"]
    mode: Literal["full", "mvb-only", "review-only", "update", "arc-step", "arc-index"]
    # full:        curriculum reference page (no MVB)
    # arc-step:    arc step build page (MVB always present)
    # arc-index:   arc index page (chapters + curated readings + compounding trajectory)
    # mvb-only:    inject/update only the MVB section of an existing page
    # review-only: re-review an existing page without rewriting
    # update:      improve an existing page in-place

    # ── Arc context (set by CLI; tells agent where this page sits) ─────────────
    page_type: Literal["arc-entry", "core-concept", "supporting", "arc-step", "arc-index"]
    # arc-entry:   opens a curriculum arc; sets the stage
    # core-concept: pivotal curriculum concept; no MVB (builds live in arc steps)
    # supporting:  builds vocabulary for a parent curriculum page
    # arc-step:    one build step inside an arc; has_mvb always true
    # arc-index:   top-level arc page; chapters + readings + compounding trajectory

    depth_emphasis: list[str]           # ["applied"] | ["theoretical"] | ["frontier"] | combo
    # applied:     longer recipe, exact hyperparams, production framing
    # theoretical: derivation steps, proof intuition, key theorems
    # frontier:    open problems as specific questions, named papers, benchmark numbers

    arc_context: dict                   # {
                                        #   "arc_id": "generative-stack",
                                        #   "arc_title": "The Generative Stack",
                                        #   "destination": "conditional latent diffusion",
                                        #   "position": 3,        # step number (arc-step only)
                                        #   "total": 8,           # total steps in arc
                                        #   "chapter": 2,         # chapter number
                                        #   "chapter_title": "Score-Based Denoising",
                                        #   "prev": "beta-vae",   # previous step slug
                                        #   "next": "ddim",       # next step slug
                                        #   "prev_artifact": "trained VAE encoder/decoder weights",
                                        #   "chapters": [...]     # arc-index only: list of chapter dicts
                                        # }
    mvb_decision: bool                  # resolved by plan_node from page_type + depth_emphasis

    # ── Chapter / compounding context (arc-step pages) ────────────────────────
    chapter: int                        # chapter number this step belongs to (1-based)
    chapter_title: str                  # e.g., "Score-Based Denoising"
    compounding_artifact: str           # what this step's build produces (artifact name)
    prev_artifact: str                  # artifact produced by the previous step (input to this build)

    # ── Loaded at runtime ──────────────────────────────────────────────────────
    persona: dict                       # from personas/[track].yaml
    existing_stub: str                  # current file content if exists (may be empty)
    output_path: str                    # e.g., docs/curriculum/core/02-generative-modeling/concepts/diffusion-models.md

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
    review_issues: list                 # union of structured + panel issues
    review_rubric: dict                 # per-dimension scores (structured + critic panel)
    revision_count: int                 # number of revision loops (max 2)

    # ── Critic panel (multi-critic review, runs in parallel) ──────────────────
    critic_panel: dict                  # {critic_name: {score: float, issues: list, fixes: list}}

    # ── Persona (for arc step pages with multi-persona MVB) ───────────────────
    mvb_persona: str                    # one of: curious-learner, ml-tinkerer,
                                        # applied-engineer, applied-researcher,
                                        # theory-student, frontier-researcher

    # ── Output phase ───────────────────────────────────────────────────────────
    final_content: str                  # approved page content
    approved: bool                      # True when reviewer approved
    committed: bool                     # True when git commit succeeded
    error: str                          # error message if any step failed

    # ── Run tracking (new) ────────────────────────────────────────────────────
    run_id: str                         # UUID for this generation run
    started_at: str                     # ISO timestamp when run began
    finished_at: str                    # ISO timestamp when run ended
