---
skill: agent-runtime
description: The closed-loop runtime FAIRE's agents run inside — server, scheduler, observer, sprint_job, budget gate. What each loop closes and how to reason about timing. Read by the supervisor when deciding what to queue.
applies_to: [supervisor]
triggers: [all pages]
---

# Skill: Agent runtime — the loop around the graph

This skill is the agents' durable knowledge of the runtime they live inside. Source of truth: `agents/server.py`, `agents/src/frontier_agents/scheduler.py`, `agents/src/frontier_agents/observer.py`, `start.sh`.

## The control-systems mapping

| Role | Component | File |
|---|---|---|
| Sensor | Observer | `observer.py::observe()` |
| Plant state | `WikiObservation` dataclass | observer.py |
| Set points | quality 0.85, coverage 0.80, staleness 180d | observer.py constants |
| Error signals | `coverage_deficit`, `quality_deficit`, `stale_pages`, `flagged_pages`, `budget_pressure` | `compute_error_signals` |
| Controller | Supervisor | `supervisor.py::run_supervisor()` |
| Actuator | LangGraph pipeline | `graph.py::compile_wiki_graph` |
| Feedback path | `log_run_node` → `runs.jsonl` → next observer call | nodes.py + observer.py |

If you mentally swap "PID controller for a thermostat" for this stack, the analogy holds.

## The cycle — what `full_cycle_job` actually does

`scheduler.py::full_cycle_job` runs every `--interval` hours (default 48; lowered to 1 during active development). One cycle, in order:

1. **Supervisor** — assess wiki health, rewrite `sprints/current.md`, optionally run `maybe_propose_arcs()` if the three preconditions hold.
2. **Audit** — `audit.py::audit_wiki()` scans for structural issues (banned URLs, missing sections, broken compounding chains).
3. **Sprint** — `sprint_job` reads the rewritten queue, spawns up to `SPRINT_WORKERS` graph invocations in parallel.
4. **Changelog** — `write_changelog_entry()` appends a per-cycle quality delta summary.

Each step writes to disk so a crash mid-cycle doesn't lose progress. The supervisor's first action in any cycle is `_assess_wiki()` against the actual filesystem — so even if `runs.jsonl` is wrong, the supervisor sees the truth.

## The HTTP layer — endpoints the user calls

`server.py` (FastAPI) on port 8765:

| Method · Path | What it does |
|---|---|
| `GET /` | Plain-text health dashboard |
| `GET /status` | Full observer snapshot as JSON |
| `GET /metrics` | metrics.json |
| `GET /budget` | live OpenRouter budget |
| `GET /observer` | observer.md rendered |
| `GET /sprint` | current sprint queue |
| `GET /supervisor` | supervisor.md latest report |
| `GET /changelog` | quality changelog |
| `POST /trigger` | run a full cycle NOW (blocks until done) |
| `POST /generate` | generate one specific page (JSON body) |

Use `POST /trigger` to force a cycle (e.g., right after the user updates `.env`). Use `POST /generate` when the supervisor's prioritization isn't picking what the human wants right now.

## The shell wrapper — `start.sh` and crash recovery

`start.sh` is a process supervisor with infinite restart-on-crash and log rotation. Important behaviors:

- Restarts the server on any non-clean exit (10-second cooldown).
- Rotates `agents/logs/server.log` when it exceeds 10MB.
- Clean shutdown on SIGTERM/SIGINT — propagates to the python process.
- Args passthrough — `./start.sh --interval 1 --run-now --dry-run` works.

To stop the loop entirely: `pkill -TERM -f "start.sh\|server.py"`. To restart: `./start.sh --interval 1 --run-now &` from the repo root.

## The four feedback loops — what closes when

| Loop | When | What's measured | Where it closes |
|---|---|---|---|
| **Per-revision** | Inside one graph run | Reviewer + critic-panel confidence | `route_after_review` triggers `revise_draft_node`, max 2 revisions |
| **Per-cycle** | Across one `full_cycle_job` | Quality trend (last 10 runs) | Supervisor reprioritizes sprint based on `confidence_delta` |
| **Per-budget-mode** | Continuous | OpenRouter usage | `check_budget()` → mode transition → sprint_job skips actions |
| **Long-horizon voice** | NOT YET CLOSED | Failed-critic patterns | TODO: critic-attribution → persona-update — see `critic-editor.md` for sketch |

Today the long-horizon voice loop is open — pages drift in style if reviewer feedback isn't aggregated. This is on the gap list and will be closed in a follow-up.

## Concurrency model

- **`SPRINT_WORKERS`** (env var, default 4) — parallel pages per sprint job. Each is its own thread invoking the graph.
- **`RESEARCH_WORKERS`** (env var, default 7) — parallel Exa+HF calls per page (inside `research_node`).
- **Critic panel** — up to 8 parallel critic calls per page (inside `review_node`).
- **Total in-flight LLM calls during peak**: ≈ `SPRINT_WORKERS × (RESEARCH_WORKERS + 1 writer + 8 critics + 1 rubric reviewer)` = 4 × 17 ≈ 68. OpenRouter handles this; OpenAI structured-output endpoints may throttle. If you see 429s, lower SPRINT_WORKERS first.

## What the `sprints/current.md` format encodes

Each line is one work item:

```
topic | track | page-type | depth_emphasis [| arc:id pos:N ch:K ch_title:"..." prev:slug next:slug prev_artifact:"..." artifact:"..." total:M]
```

Parsed by `scheduler.py::_parse_sprint_item`. Arc steps need the full `arc:...` extension; curriculum pages don't. The supervisor rewrites this file every cycle; humans normally don't edit it (but they can — the parser is lenient).

## What `runs.jsonl` records

One JSON object per line, appended by `log_run_node`. Fields you can rely on: `run_id`, `topic`, `track`, `page_type`, `mode`, `model_writer`, `model_reviewer`, `started_at`, `finished_at`, `status`, `confidence`, `revision_count`, `output_path`, `has_mvb`, `committed`, `error`, `review_issues`, `review_feedback`, `critic_panel` (new, per-critic scores + issues).

The observer reads this file to compute quality trend. If you query this file directly, sort by `finished_at` and take the latest per `topic`.

## Pause vs full-shutdown

| Goal | How |
|---|---|
| Pause work but keep server alive | Set `BUDGET_LIMIT_USD` below current `usage_usd` → mode goes paused |
| Stop a cycle that's in progress | `POST /pause` *(not implemented yet — current path is to kill server)* |
| Full shutdown | `pkill -TERM -f "start.sh\|server.py"` then wait, then `kill -9` if needed |
| Restart fresh | shutdown + `./start.sh --interval 1 --run-now &` from repo root |

The "pause without shutdown" path is a known gap — added to the follow-up list.

## When to call this skill in a prompt

The supervisor LLM call should always have `agent-runtime.md` loaded so it knows what mode the system is in and what the budget allows when reasoning about what to queue next. The `applies_to: [supervisor]` frontmatter ensures the skill loader picks it up automatically.

## Related skills

- `langgraph-patterns.md` — what the graph itself looks like inside the loop
- `openrouter-routing.md` — which model each role uses, budget gate behavior
- `arc-selection.md` — the supervisor's scoring rubric that uses `remaining_budget × 0.5`
- `critic-editor.md` — the editor's veto authority that the supervisor honors
