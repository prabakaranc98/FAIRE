"""Smoke tests for LangGraph graph compilation and routing logic."""

import pytest

from frontier_agents.graph import build_mvb_graph, build_wiki_graph, compile_mvb_graph, compile_wiki_graph


class TestGraphCompilation:
    def test_wiki_graph_compiles(self):
        graph = compile_wiki_graph()
        assert graph is not None

    def test_mvb_graph_compiles(self):
        graph = compile_mvb_graph()
        assert graph is not None

    def test_wiki_graph_has_required_nodes(self):
        g = build_wiki_graph()
        compiled = g.compile()
        graph_nodes = set(compiled.get_graph().nodes.keys())
        for node in ("load_persona", "read_stub", "research", "write_draft", "review",
                     "revise_draft", "write_file", "commit", "flag_human_review"):
            assert node in graph_nodes, f"Missing node: {node}"

    def test_mvb_graph_has_required_nodes(self):
        g = build_mvb_graph()
        compiled = g.compile()
        graph_nodes = set(compiled.get_graph().nodes.keys())
        for node in ("load_persona", "read_stub", "mvb_recipe", "merge_mvb", "review",
                     "write_file", "commit"):
            assert node in graph_nodes, f"Missing node: {node}"

    def test_wiki_graph_does_not_include_mvb_recipe(self):
        g = build_wiki_graph()
        compiled = g.compile()
        graph_nodes = set(compiled.get_graph().nodes.keys())
        assert "mvb_recipe" not in graph_nodes
        assert "merge_mvb" not in graph_nodes

    def test_mvb_graph_does_not_include_write_draft_or_plan(self):
        """MVB graph generates via mvb_recipe + merge_mvb, not the full draft pipeline."""
        g = build_mvb_graph()
        compiled = g.compile()
        graph_nodes = set(compiled.get_graph().nodes.keys())
        assert "write_draft" not in graph_nodes
        assert "plan" not in graph_nodes
