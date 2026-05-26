---
skill: langgraph-patterns
description: LangGraph primitives the FAIRE agent system uses (or should use). API names + when to reach for each. Read by the planner and supervisor when reasoning about graph topology.
applies_to: [plan, supervisor]
triggers: [all pages]
---

# Skill: LangGraph patterns for FAIRE

This skill is the agents' durable knowledge of LangGraph's primitives — when to use which, by name, with the FAIRE context attached. Source of truth: the system's actual `graph.py` and `nodes.py`, plus LangGraph's public API.

## What we use today

| Primitive | Where in FAIRE | API |
|---|---|---|
| `StateGraph(WikiPageState)` | `graph.py::build_wiki_graph` | `langgraph.graph.StateGraph` |
| `add_node`, `add_edge`, `add_conditional_edges` | every node wiring | same |
| Routers as pure functions | `_route_after_plan_scratch`, `route_after_review` | return string key → match dict |
| `lru_cache`'d LLM factories | `llm.py::get_llm` | not a LangGraph primitive but the discipline of one LLM-per-role |
| In-process parallel fanout | `_run_critic_panel` uses `ThreadPoolExecutor` | not yet `Send` — see below |

## What we should use (gaps the supervisor can decide to close)

### `Send` for native parallel fanout

Today the critic panel uses Python's `ThreadPoolExecutor`. That works, but the graph engine can't *see* the fanout — it can't checkpoint or stream per critic. LangGraph's `Send` API lets a node return a list of `Send(target_node, partial_state)` calls; the engine spawns those branches and waits for them all before continuing.

When to migrate: when we add a checkpointer (below). At that point per-critic visibility matters because a crash mid-panel should resume from the unfinished critics, not restart all 8.

API: `from langgraph.constants import Send` and return `[Send("critic_node", {...}) for skill in critics]` from a router node.

### Checkpointers — `MemorySaver` / `SqliteSaver`

The graph runs to completion or crashes; there is no resume. For a long arc-step write (research + plan + write + 8 critics + revise + commit) that's 6+ minutes of API calls; a network blip currently means starting over.

API: `from langgraph.checkpoint.sqlite import SqliteSaver` (persistent) or `from langgraph.checkpoint.memory import MemorySaver` (in-process). Pass to `graph.compile(checkpointer=...)`. Each `invoke` takes a `config={"configurable": {"thread_id": run_id}}` so multiple pages can run concurrently with separate state.

When to add: as soon as we start running arcs (steps are longer than curriculum pages and benefit more from resume).

### `interrupt()` for human-in-the-loop

The arc-proposal phase writes `docs/system/arc-proposals.md` and waits for the human to manually edit the sprint queue. The proper LangGraph idiom is `interrupt(payload)` inside a node — the graph pauses, the API returns the payload, the user calls `graph.stream(Command(resume=<choice>), config=...)` to continue.

API: `from langgraph.types import interrupt, Command`. Use inside a `propose_arcs_node` to pause until the user picks. Pairs naturally with a checkpointer (state persists across the pause).

When to add: when the CLI grows a `spin-arc --interactive` flag that wants live picking.

### Sub-graphs

The critic panel could be its own `StateGraph` compiled separately and invoked as a single node in the main graph. Same with the research phase (Exa + HF + find_similar). Sub-graphs improve modularity and let each sub-pipeline have its own checkpointer and streaming.

API: build the sub-graph, `compile()` it, then `graph.add_node("review_panel", subgraph)` in the parent. The state schemas must be compatible (sub-graph's input keys must be present in the parent's state).

When to add: when the critic panel grows to ≥ 12 critics, or when we want to A/B test critic-panel configurations.

### Streaming with `.stream(stream_mode="updates")`

Today the `/status` HTTP endpoint shows opaque "running." LangGraph's `.stream()` yields per-node updates as they happen. The dashboard could show "now in research_node," "now in plan_and_scratch_node," etc.

API: `for chunk in graph.stream(state, config, stream_mode="updates"): ...`. `stream_mode` options: `"values"` (full state per step), `"updates"` (only diffs — recommended for dashboards), `"messages"` (token-level for LLM nodes — useful for live-typing UX).

When to add: when the user is sitting at the dashboard waiting for a run and wants per-second progress.

## When NOT to reach for LangGraph

- For a single-call LLM that doesn't need persistence, multi-node coordination, or branching: just use the LangChain LLM directly. Don't graph-ify everything.
- For long-running background queues unrelated to a single graph invocation (e.g., the sprint job loop): use APScheduler / FastAPI BackgroundTasks. LangGraph is for *the graph*, not for the cycle around it.

## Quick reference — adding a new node

1. Define the node as `def my_node(state: WikiPageState) -> WikiPageState`.
2. `graph.add_node("my_node", my_node)`.
3. `graph.add_edge("previous_node", "my_node")` or `add_conditional_edges(...)` if branching.
4. Add any new state keys to `state.py::WikiPageState`.
5. If the node needs an LLM, use `get_llm("role")` — don't instantiate `ChatOpenAI` directly.
6. If the node calls tools, route them through `tools.py` so source policy + retries stay centralized.

## Related skills

- `agent-runtime.md` — the loop around the graph (server, scheduler, observer)
- `openrouter-routing.md` — which model goes with which role
- `arc-anatomy.md` — the diagonal-arc shape arcs follow
- `arc-exploration.md` — the explorer playbook (survey → map → pick)
