---
title: System architecture
description: How the FAIRE editorial agent system is wired — values, layers, agents, skills, memory, and the closed loop. The blueprint both Claude and the running agents read before touching anything.
---

# System architecture

> A wiki built by agents. Local-only runtime. Closed-loop control. Budget-aware. Source-disciplined.
>
> This page complements [`docs/system/sense.md`](./sense.md). **Sense** = what we're building and for whom. **Architecture** = how the system that builds it is wired.

---

## 1. The mental model (one paragraph)

FAIRE is a wiki + curated reading list + nudge-to-build, written by editorial agents that read primary sources only, organized in three layers — **curriculum** (range, one page per concept), **arc index** (opinionated path to a frontier capability), **arc step** (one build per page, MVB as milestone, compounding artifact contract). Voice is reference, not pitch, not tutorial. Citations are seminal · test-of-time · current SotA, nothing else. The system is **closed-loop**: it observes its own state, decides what to write next, writes it, reviews it through a panel of critics, and learns from the feedback for the next cycle. It runs on a $10/cycle budget, locally.

---

## 2. The four contracts

Every page the system writes must honor four contracts simultaneously. A critic owns each.

| # | Contract | Owner skill | Failure mode it catches |
|---|---|---|---|
| 1 | **Sense** — page matches what FAIRE is for | `faire-sense` | Tutorial-ish, pitchy, exhaustive-survey style |
| 2 | **Human** — each of the four readers gets what they came for | `critic-human-centered`, `critic-beginner-onramp` | Wall of equations with no intuition; jargon dump |
| 3 | **Source** — citations are seminal/test-of-time/SotA, approved domains only | `source-policy`, `critic-info-architecture` | Medium links, Wikipedia citations, filler readings |
| 4 | **Nudge** — page ends with a directed, specific invitation to do something | `critic-build-nudge`, `mvb-recipe` | "Try training a model" generic CTA |

---

## 3. The layers (top to bottom)

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 5 — Human interface                                       │
│     mkdocs site (docs/)   ·   /server :8765 dashboard            │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ writes / reads
┌─────────────────────────────────────────────────────────────────┐
│  Layer 4 — Editorial pipeline (LangGraph)                        │
│     load_persona → read_stub → research → plan_and_scratch       │
│        → write_{draft|arc_step|arc_index} → link → review_PANEL  │
│        → write_file → commit → log_run                           │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ uses
┌─────────────────────────────────────────────────────────────────┐
│  Layer 3 — Skills (agents/skills/*.md)                           │
│     faire-sense · wiki-prose · math-latex · mvb-recipe           │
│     source-policy · sota-coverage · navigation-ia · arc-context  │
│     critic-human-centered · critic-beginner-onramp ·             │
│     critic-wiki-voice · critic-info-architecture ·               │
│     critic-build-nudge                                            │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ accesses
┌─────────────────────────────────────────────────────────────────┐
│  Layer 2 — Tools (agents/src/frontier_agents/tools.py)           │
│     Exa: papers · sota · production · find_similar               │
│     HF:  models · datasets                                        │
│     FS:  read_stub · write_file · ensure_track_index             │
│     Git: git_commit (auto when conf ≥ 0.7)                       │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ routed via
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1 — Models (OpenRouter via LangChain ChatOpenAI)          │
│     writer · reviewer · research · mvb · fallback                │
│     Budget gate: full → reduced → paused                         │
└─────────────────────────────────────────────────────────────────┘
                              ▲
                              │ measured by
┌─────────────────────────────────────────────────────────────────┐
│  Layer 0 — Control loop                                          │
│     observer (sensor) → supervisor (controller) →                │
│     pipeline (actuator) → runs.jsonl (feedback)                  │
│     Set points: quality 0.85 · coverage 0.80 · staleness 180d    │
└─────────────────────────────────────────────────────────────────┘
```

---

## 4. The agent roster

| Agent | What it is | Inputs | Outputs | Lives in |
|---|---|---|---|---|
| **Supervisor** | Decides what to write next | observer + audit + runs.jsonl | rewrites `sprints/current.md` | `supervisor.py` |
| **Persona loader** | Picks the track's editorial voice | track id | `persona` dict in state | `nodes.py::load_persona_node` |
| **Stub reader** | Picks up any existing draft | output_path | `existing_stub` | `nodes.py::read_stub_node` |
| **Research agent** | 3-channel Exa search (papers/SotA/production) + HF model+dataset lookup | topic + persona search_seeds | `research_results`, `sota_results`, `production_results`, `hf_models`, `hf_datasets` | `nodes.py::research_node` |
| **Planner** | 5-question planning prompt → 200-word writing plan | research results | `writing_plan` | `nodes.py::plan_and_scratch_node` |
| **Scratch compiler** | Verified fact-sheet (citations, equations, prod examples, MVB stack, opening scenario, open problem) | writing_plan + raw research | `scratch_pad` (writer never sees raw results) | same node |
| **Writer** | Produces a full schema-compliant page draft | persona + plan + scratch_pad | `draft` | `nodes.py::write_{draft,arc_step,arc_index}_node` |
| **Sanitizer** | Strips fenced YAML, preambles; ensures file starts with `---` | draft | sanitized final | `nodes.py::_sanitize_draft` |
| **Linker** | Finds related curriculum pages, injects real backlinks; updates `backlinks.json` | draft + filesystem | draft with injected links | `nodes.py::link_node` |
| **Critic panel (planned)** | 5 specialized critics, run in parallel; each scores one dimension | draft + scratch_pad | per-critic {score, issues, fixes} | `nodes.py::review_node` (refactor target) |
| **Aggregator (planned)** | Combines critic outputs into single confidence + issue list | critic outputs | `review_confidence`, `review_feedback` | same as review |
| **Reviser** | One revision pass on flagged drafts | draft + critic feedback | revised draft | `nodes.py::revise_draft_node` |
| **Committer** | git add + commit if confidence ≥ `GIT_COMMIT_THRESHOLD` | output_path + confidence | git side-effect | `nodes.py::commit_node` |
| **Logger** | Appends run record; recomputes metrics + observer page | full state | runs.jsonl + metrics.json + observer.md | `nodes.py::log_run_node` |
| **Observer** | Builds `WikiObservation` snapshot (sensor) | filesystem + runs.jsonl + OpenRouter | metrics.json + observer.md + budget state | `observer.py::observe` |
| **Audit** | Structural scan (banned URLs, missing sections, nested lists, frontmatter) | docs/ | `last_audit.json` | `audit.py::audit_wiki` |

---

## 5. The skills / memory boundary

These are different on purpose:

| Layer | Lives in | Read by | Persists |
|---|---|---|---|
| **Agent skills** | `agents/skills/*.md` | LangGraph nodes via `skills.py` loader | Across cycles; injected into writer/reviewer prompts |
| **Personas** | `agents/src/frontier_agents/personas/{track}.yaml` | `load_persona_node` | Per-track voice; rarely changes |
| **Scratch pad** | `state["scratch_pad"]` | Writer, reviser | One run only — discarded after `write_file` |
| **Run log** | `agents/runs/runs.jsonl` | observer, supervisor | All-history; quality trend computed over last 10 |
| **Metrics** | `agents/runs/metrics.json` | dashboard, supervisor | Overwritten every run |
| **Sprint queue** | `agents/sprints/current.md` | scheduler, supervisor | Rewritten by supervisor every cycle |
| **Claude memories** | `~/.claude/projects/.../memory/*.md` | future Claude conversations | Across sessions; never read by agents |

**Rule of thumb:** if the *agents* need it, it's a skill or persona. If *future Claude* needs it, it's a memory.

---

## 6. The file-system contract

Every file in the repo has one of these jobs. Anything else is cruft.

| Path | Role | Owner |
|---|---|---|
| `docs/index.md` | Public homepage | human, hand-tuned |
| `docs/curriculum/{N}/index.md` | Track scaffold = the seed for what to write | human seeds; supervisor extends |
| `docs/curriculum/{N}/{slug}.md` | One concept page | **writer agent** |
| `docs/arcs/index.md` | Arc registry / overview | human |
| `docs/arcs/{arc}/index.md` | One arc syllabus | **writer agent (mode=arc-index)** |
| `docs/arcs/{arc}/step-NN-{slug}.md` | One build page (MVB lives here) | **writer agent (mode=arc-step)** |
| `docs/system/sense.md` | What FAIRE is | human + Claude |
| `docs/system/architecture.md` | This page | human + Claude |
| `docs/system/observer.md` | Live control dashboard | observer agent, auto-overwritten |
| `docs/system/supervisor.md` | Latest supervisor report | supervisor agent |
| `docs/system/changelog.md` | Quality delta per sprint | scheduler |
| `docs/system/backlinks.json` | Forward/reverse link index | linker agent |
| `agents/sprints/current.md` | Work queue | supervisor agent |
| `agents/sprints/history/*` | Archived sprints | scheduler |
| `agents/runs/runs.jsonl` | Run record append log | logger agent |
| `agents/runs/metrics.json` | Latest observation | observer agent |
| `agents/skills/*.md` | Agent skills | human + Claude |
| `agents/.env` | Keys + model IDs + budget cap | human only |
| `agents/src/frontier_agents/personas/*.yaml` | Per-track voice | human + Claude |
| `agents/src/frontier_agents/*.py` | The system itself | human + Claude |
| `PRINCIPLES.md` | The 4 objectives + 10 rules | human |
| `README.md` | Repo intro | human |

---

## 7. The self-control mechanisms (closed loop)

This is what makes the system actually *self-control* rather than just be "automated."

### 7.1 Set points (the system's goals)
- `QUALITY_SETPOINT = 0.85` — reviewer confidence per page
- `COVERAGE_SETPOINT = 0.80` — fraction of pages with real content per track
- `STALENESS_THRESHOLD = 180` days — when SotA goes stale
- `BUDGET_LIMIT_USD` — soft cap on OpenRouter spend

### 7.2 Error signals (compute_error_signals)
Per observation:
- `coverage_deficit = max(0, COVERAGE_SETPOINT - coverage_pct)`
- `quality_deficit = max(0, QUALITY_SETPOINT - avg_confidence)`
- `stale_pages`, `flagged_pages` (counts)
- `budget_pressure = 1 - remaining / BUDGET_REDUCED`

### 7.3 Actuator modes (driven by budget)
- **full** → claude-opus or gpt-5-class writer, full panel of critics
- **reduced** → writer drops to FALLBACK_MODEL; skip low-priority improvements
- **paused** → no generation; only audit + improve for already-generated pages

### 7.4 Feedback paths (the loops that actually close)

| Loop | Where it closes |
|---|---|
| Per-run quality | review fails → revise → re-review (max 2 revisions) |
| Per-cycle quality | runs.jsonl → quality_trend → supervisor adjusts sprint priorities |
| Per-cycle coverage | filesystem stub count → supervisor queues generate actions |
| Long-horizon voice | (planned) failed-critic patterns → persona YAML diff proposal |
| Budget | OpenRouter `/auth/key` → check_budget → mode change → sprint_job behavior |

The system has feedback for everything *except* the writer's own voice over time. That last loop is the next upgrade — see `critic-attribution + persona-update` under "what's missing" below.

---

## 8. The long-running cycle (what actually happens for $10)

Concrete numbers under current model config (writer = gemini-3.1-flash-lite, reviewer = gpt-5-mini, research = gemini-2.0-flash-lite, panel of 5 critics in parallel):

| Step | Cost | Time | Cumulative |
|---|---|---|---|
| Per page: research (3 Exa calls + 2 HF) | $0.02 | 10s | |
| Per page: plan + scratch | $0.02 | 8s | |
| Per page: write draft | $0.06 | 25s | |
| Per page: link injection | $0.01 | 5s | |
| Per page: review panel (5× parallel critics) | $0.05 | 12s (longest critic) | |
| Per page: revise (if needed, avg 0.7×) | $0.04 | 15s | |
| Per page: commit + log | $0 | 2s | |
| **One full page** | **~$0.20** | **~70s** | $0.20 |
| ~25 curriculum pages | $5.00 | ~30 min @ 4 parallel workers | $5.00 |
| ~10 arc steps + 2 arc indexes | $2.40 | ~12 min | $7.40 |
| ~12 improvement passes on flagged | $1.80 | ~10 min | $9.20 |
| Audit + supervisor LLM call | $0.20 | 2 min | **~$9.40** |

That brings us to ~50 pages on a $10 cap, then `paused` mode. Maintenance after that is audit + supervisor only until the next budget anchor.

---

## 9. What's still missing (the honest list)

| # | Gap | Effort | Why it matters |
|---|---|---|---|
| 1 | **Critic panel.** Current `review_node` is one model, one call. | Medium | One model can't be expert in IA + voice + sources + nudge + reader-fit at once. |
| 2 | **Critic-attribution feedback.** Failed-critic patterns don't feed back into the writer's persona. | Medium | Without this, the system doesn't actually improve its voice over time. |
| 3 | **Arc proposal phase.** After curriculum coverage > 60%, supervisor should propose a slate of arcs to materialize. | Small | The user picks which arcs become real; matches the "2 active arcs at a time" constraint. |
| 4 | **Backlinks bidirectional.** Curriculum links forward to arcs that may not exist yet; arc steps need to add backlinks to curriculum on creation. | Small | The "Where this concept appears" section currently can be wrong. |
| 5 | **Visual sanity for math.** No automated check that LaTeX renders. | Small | Some pages have unrendered `\(...\)` because of escaping. |
| 6 | **MVB executability check.** No verification that the named model ID + dataset ID exist on HuggingFace at write time. | Small | An MVB pointing to a renamed/deleted model is a dead end. |

---

## 10. The one-line summary

**FAIRE is a closed-loop deep-agent system that writes a frontier-AI wiki under a fixed budget, in a single voice, citing only primary sources, and nudges every reader toward making something.**

When you change this system, that sentence is the test.
