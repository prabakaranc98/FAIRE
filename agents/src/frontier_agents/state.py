"""WikiPageState — the shared state TypedDict passed through all agent nodes."""

from __future__ import annotations

from typing import Literal, TypedDict


class WikiPageState(TypedDict, total=False):
    # Input
    topic: str                          # e.g., "diffusion-models"
    track: str                          # e.g., "02-generative-modeling"
    depth: Literal["all", "applied", "foundations", "research"]
    mode: Literal["full", "mvb-only", "review-only", "update"]

    # Loaded at runtime
    persona: dict                       # from personas/[track].yaml
    existing_stub: str                  # current file content if exists (may be empty)
    output_path: str                    # e.g., docs/curriculum/02-generative-modeling/diffusion-models.md

    # Research phase
    research_results: list[dict]        # Exa search results; [{url, title, text, domain}]
    hf_models: list[dict]               # HuggingFace model search results (for MVB)
    hf_datasets: list[dict]             # HuggingFace dataset search results (for MVB)

    # Writing phase
    draft: str                          # full markdown page draft
    mvb_section: str                    # Minimum Valuable Build section (if mode=mvb-only)

    # Review phase
    review_feedback: str                # structured feedback from reviewer agent
    review_confidence: float            # 0.0-1.0; <0.8 flags for human review
    review_pass: bool                   # True if reviewer approved
    revision_count: int                 # number of revision loops (max 2)

    # Output phase
    final_content: str                  # approved page content
    approved: bool                      # True when reviewer approved
    committed: bool                     # True when git commit succeeded
    error: str                          # error message if any step failed
