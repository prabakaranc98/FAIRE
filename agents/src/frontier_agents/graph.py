"""LangGraph StateGraph definition for the Frontier Wiki editorial agent system.

Two graphs:
1. `build_wiki_graph()` — full editorial pipeline (research → plan → write → review → commit → log)
2. `build_mvb_graph()`  — MVB-only pipeline (HF search → generate MVB → merge → review → commit → log)

The `plan` node (between research and write_draft) forces deliberate thinking before writing —
the primary fix for the vague, listy output of the previous version.

The `log_run` node fires at END regardless of outcome, writing to agents/runs/runs.jsonl
and updating agents/runs/wiki_status.md for coverage/quality tracking.
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    commit_node,
    flag_human_review_node,
    load_persona_node,
    log_run_node,
    merge_mvb_node,
    mvb_recipe_node,
    plan_node,
    read_stub_node,
    research_node,
    review_node,
    revise_draft_node,
    route_after_review,
    write_draft_node,
    write_file_node,
)
from .state import WikiPageState


def build_wiki_graph() -> StateGraph:
    """Full editorial pipeline: persona → stub → research → plan → draft → review → commit → log.

    Graph topology:
        START
          → load_persona
          → read_stub
          → research
          → plan             ← NEW: deliberate planning before writing
          → write_draft
          → review ─────────────── [approved] ──→ write_file → commit → log_run → END
                    └── [revise, count<2] ──→ revise_draft → review
                    └── [revise, count≥2] ──→ flag_human_review → log_run → END
    """
    graph = StateGraph(WikiPageState)

    # Nodes
    graph.add_node("load_persona", load_persona_node)
    graph.add_node("read_stub", read_stub_node)
    graph.add_node("research", research_node)
    graph.add_node("plan", plan_node)
    graph.add_node("write_draft", write_draft_node)
    graph.add_node("review", review_node)
    graph.add_node("revise_draft", revise_draft_node)
    graph.add_node("write_file", write_file_node)
    graph.add_node("commit", commit_node)
    graph.add_node("log_run", log_run_node)
    graph.add_node("flag_human_review", flag_human_review_node)

    # Edges
    graph.add_edge(START, "load_persona")
    graph.add_edge("load_persona", "read_stub")
    graph.add_edge("read_stub", "research")
    graph.add_edge("research", "plan")
    graph.add_edge("plan", "write_draft")
    graph.add_edge("write_draft", "review")

    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "write_file": "write_file",
            "revise_draft": "revise_draft",
            "flag_human_review": "flag_human_review",
        },
    )

    graph.add_edge("revise_draft", "review")
    graph.add_edge("write_file", "commit")
    graph.add_edge("commit", "log_run")
    graph.add_edge("log_run", END)
    graph.add_edge("flag_human_review", "log_run")

    return graph


def build_mvb_graph() -> StateGraph:
    """MVB-only pipeline: persona → stub → research → MVB generate → merge → review → commit → log.

    Graph topology:
        START
          → load_persona
          → read_stub
          → research         (HF model/dataset search + exa_search_papers)
          → mvb_recipe       (generate MVB section)
          → merge_mvb        (inject into existing page)
          → review ─────────────── [approved] ──→ write_file → commit → log_run → END
                    └── [revise, count<2] ──→ revise_draft → review
                    └── [revise, count≥2] ──→ flag_human_review → log_run → END
    """
    graph = StateGraph(WikiPageState)

    graph.add_node("load_persona", load_persona_node)
    graph.add_node("read_stub", read_stub_node)
    graph.add_node("research", research_node)
    graph.add_node("mvb_recipe", mvb_recipe_node)
    graph.add_node("merge_mvb", merge_mvb_node)
    graph.add_node("review", review_node)
    graph.add_node("revise_draft", revise_draft_node)
    graph.add_node("write_file", write_file_node)
    graph.add_node("commit", commit_node)
    graph.add_node("log_run", log_run_node)
    graph.add_node("flag_human_review", flag_human_review_node)

    graph.add_edge(START, "load_persona")
    graph.add_edge("load_persona", "read_stub")
    graph.add_edge("read_stub", "research")
    graph.add_edge("research", "mvb_recipe")
    graph.add_edge("mvb_recipe", "merge_mvb")
    graph.add_edge("merge_mvb", "review")

    graph.add_conditional_edges(
        "review",
        route_after_review,
        {
            "write_file": "write_file",
            "revise_draft": "revise_draft",
            "flag_human_review": "flag_human_review",
        },
    )

    graph.add_edge("revise_draft", "review")
    graph.add_edge("write_file", "commit")
    graph.add_edge("commit", "log_run")
    graph.add_edge("log_run", END)
    graph.add_edge("flag_human_review", "log_run")

    return graph


def compile_wiki_graph():
    """Return a compiled, runnable editorial graph."""
    return build_wiki_graph().compile()


def compile_mvb_graph():
    """Return a compiled, runnable MVB graph."""
    return build_mvb_graph().compile()
