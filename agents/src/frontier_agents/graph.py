"""LangGraph StateGraph definition for the Frontier Wiki editorial agent system.

Three graphs:
1. `build_wiki_graph()` — curriculum page pipeline (research → plan_and_scratch → write → review → commit → log)
2. `build_arc_step_graph()` — arc step build page pipeline (same flow but write_arc_step_node instead of write_draft)
3. `build_mvb_graph()`  — MVB-only pipeline (HF search → generate MVB → merge → review → commit → log)

Routing: after plan_and_scratch, state["mode"] == "arc-step" routes to write_arc_step_node;
everything else routes to write_draft_node (curriculum pages).
"""

from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from .nodes import (
    build_writing_checklist_node,
    commit_node,
    flag_human_review_node,
    keep_best_draft_node,
    link_node,
    load_persona_node,
    log_run_node,
    merge_mvb_node,
    mvb_recipe_node,
    plan_and_scratch_node,
    read_stub_node,
    research_node,
    review_node,
    revise_draft_node,
    route_after_review,
    write_arc_index_node,
    write_arc_step_node,
    write_draft_node,
    write_file_node,
)
from .state import WikiPageState


def _route_after_plan_scratch(state: WikiPageState) -> str:
    """Route to the correct writer based on mode."""
    mode = state.get("mode", "full")
    if mode == "arc-step":
        return "write_arc_step"
    if mode == "arc-index":
        return "write_arc_index"
    return "write_draft"


def build_wiki_graph() -> StateGraph:
    """Curriculum page pipeline: persona → stub → research → plan_and_scratch → draft → review → commit → log.

    Graph topology:
        START
          → load_persona → read_stub → research → plan_and_scratch
          → [mode==arc-step] write_arc_step → link → review
          → [mode==curriculum] write_draft → link → review
          review ─── [approved] ──→ write_file → commit → log_run → END
                └── [revise, count<2] ──→ revise_draft → review
                └── [revise, count≥2] ──→ flag_human_review → log_run → END
    """
    graph = StateGraph(WikiPageState)

    # Nodes
    graph.add_node("load_persona", load_persona_node)
    graph.add_node("read_stub", read_stub_node)
    graph.add_node("research", research_node)
    graph.add_node("plan_and_scratch", plan_and_scratch_node)
    graph.add_node("build_writing_checklist", build_writing_checklist_node)
    graph.add_node("write_draft", write_draft_node)
    graph.add_node("write_arc_step", write_arc_step_node)
    graph.add_node("write_arc_index", write_arc_index_node)
    graph.add_node("link", link_node)
    graph.add_node("review", review_node)
    graph.add_node("keep_best_draft", keep_best_draft_node)
    graph.add_node("revise_draft", revise_draft_node)
    graph.add_node("write_file", write_file_node)
    graph.add_node("commit", commit_node)
    graph.add_node("log_run", log_run_node)
    graph.add_node("flag_human_review", flag_human_review_node)

    # Edges
    graph.add_edge(START, "load_persona")
    graph.add_edge("load_persona", "read_stub")
    graph.add_edge("read_stub", "research")
    graph.add_edge("research", "plan_and_scratch")
    graph.add_edge("plan_and_scratch", "build_writing_checklist")

    graph.add_conditional_edges(
        "build_writing_checklist",
        _route_after_plan_scratch,
        {
            "write_arc_step": "write_arc_step",
            "write_arc_index": "write_arc_index",
            "write_draft": "write_draft",
        },
    )

    graph.add_edge("write_draft", "link")
    graph.add_edge("write_arc_step", "link")
    graph.add_edge("write_arc_index", "link")
    graph.add_edge("link", "review")
    graph.add_edge("review", "keep_best_draft")

    graph.add_conditional_edges(
        "keep_best_draft",
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
    graph.add_node("link", link_node)
    graph.add_node("review", review_node)
    graph.add_node("keep_best_draft", keep_best_draft_node)
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
    graph.add_edge("merge_mvb", "link")
    graph.add_edge("link", "review")
    graph.add_edge("review", "keep_best_draft")

    graph.add_conditional_edges(
        "keep_best_draft",
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
