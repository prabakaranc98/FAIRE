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
│  Layer 4 — Editorial pipeline (LangGraph, per-page)              │
│     load_persona → read_stub → research → plan_and_scratch       │
│        → build_writing_checklist → write_{draft|arc_step|         │
│           arc_index} → link → review (rubric + 8-critic panel)   │
│        → revise → review' → keep_best_draft (knockout) →         │
│           route_after_review → write_file (H1-fixed, arc-       │
│           breadcrumbed) → commit (if conf ≥ 0.7) → log_run        │
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
│  Layer 0 — Closed control loop (per-cycle)                       │
│     observer (sensor) → supervisor (controller) →                │
│     sprint (actuator, N pages parallel) → runs.jsonl (feedback)  │
│     → retrospective (reflector: scrum-style retro + safe         │
│         auto-applies stub-seeds) → next cycle's supervisor       │
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
| **Checklist builder** | Promotes scratch_pad facts to mandatory: must-cite papers (arxiv-id resolved), must-use HF model IDs (pre-verified by `verify_mvb_stack`), must-include equations, must-link concept slugs | scratch_pad | `writing_checklist` dict | `nodes.py::build_writing_checklist_node` |
| **Writer** | Produces a full schema-compliant page draft constrained by the checklist | persona + plan + scratch_pad + checklist | `draft` | `nodes.py::write_{draft,arc_step,arc_index}_node` |
| **Sanitizer + H1 fixer** | Strips fenced YAML and preambles; promotes the first heading after frontmatter to `# Topic` if writer drifted to `## Topic` | draft | sanitized final | `nodes.py::_sanitize_draft`, `_ensure_h1` |
| **Linker** | Finds related curriculum pages, injects real backlinks; updates `backlinks.json` | draft + filesystem | draft with injected links | `nodes.py::link_node` |
| **Critic panel (8 critics, parallel)** | Each `critic-*` skill spawns one parallel API call scoring its dimension (info-architecture · beginner-onramp · human-centered · wiki-voice · build-nudge · cohesion · coverage · prerequisites). Combined with structured rubric reviewer + deterministic checklist enforcement + future-arxiv-ID validator | draft + scratch_pad + checklist | per-critic {score, issues, fixes}, aggregated `review_confidence` | `nodes.py::review_node`, `_run_critic_panel`, `_aggregate_review` |
| **Reviser** | Up to 2 revision passes on flagged drafts | draft + critic feedback | revised draft | `nodes.py::revise_draft_node` |
| **Knockout selector** | After revise + re-review, keeps the higher-confidence of {previous, revised}; restores prior draft if revision regressed by ≥0.02 (PerFine pattern, arxiv 2510.24469) | review_confidence vs prev_review_confidence | `draft`, `review_confidence` | `nodes.py::keep_best_draft_node` |
| **Committer** | git add + commit if confidence ≥ `GIT_COMMIT_THRESHOLD`; never-throw-away routing lands ≥0.6 drafts on disk | output_path + confidence | git side-effect | `nodes.py::commit_node` |
| **Logger** | Appends run record; recomputes metrics + observer page | full state | runs.jsonl + metrics.json + observer.md | `nodes.py::log_run_node` |
| **Observer** | Builds `WikiObservation` snapshot (sensor) | filesystem + runs.jsonl + OpenRouter | metrics.json + observer.md + budget state | `observer.py::observe` |
| **Audit** | Structural scan (banned URLs, missing sections, nested lists, frontmatter) | docs/ | `last_audit.json` | `audit.py::audit_wiki` |
| **Retrospective (backlog agent)** | After every cycle: aggregates deterministic signals (per-track health, recurring critic issues, unresolved wikilinks, heading drift, citation health), runs scrum-style retro through gpt-5-mini with structured output, auto-applies safe items (stub-seeds for high-reference unresolved slugs) | runs.jsonl + supervisor.json + backlinks.json + sprint queue | `docs/system/backlog.md`, auto-seeded stub files | `retrospective.py::retrospective_job` |

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
| `docs/system/changelog.md` | Per-page generation log | logger agent |
| `docs/system/backlog.md` | Sprint retrospectives (scrum-style: went-well, went-wrong, needs-depth, new-to-add, process-improvements) | **retrospective agent** |
| `docs/system/learnings-log.md` | Human-curated cross-cycle learnings | human |
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
| Per-run quality (within page) | review fails → revise → re-review → **knockout selector** keeps higher-confidence draft (max 2 revisions) |
| Per-run hallucination guards | deterministic future-arxiv-ID validator + checklist enforcement inside `review_node` — pure regex/arithmetic, zero LLM cost |
| Per-cycle quality | runs.jsonl → quality_trend → supervisor adjusts sprint priorities |
| Per-cycle coverage | filesystem stub count + unresolved wikilinks → supervisor queues generate actions |
| **Per-cycle retrospective** (new) | runs.jsonl + critic_panel + backlinks.json → `retrospective_job` → `backlog.md` + auto-seeded stubs → supervisor reads on next cycle |
| Long-horizon voice | (planned) failed-critic patterns → persona YAML diff proposal |
| Budget | OpenRouter `/auth/key` → check_budget → mode change → sprint_job behavior |

The retrospective loop is the centerpiece — it makes the system *learn from itself*. Each cycle's scrum retro names what went well, what regressed, what needs depth, and what to add. Safe items (stub-seed for unresolved wikilinks referenced 2+ times) auto-apply; risky items (queue-priority changes, arc proposals, author pages) are queued for human review. The next cycle's supervisor sees the new stubs and the retro context, and the wiki grows in the direction the agents themselves identified as weak.

---

## 8. What a real cycle actually cost ($30 top-up, May 2026)

These are measured numbers from the production run that shipped 60 v2 pages across the 10 canonical tracks. Model stack: writer/MVB = `openai/gpt-5.1-codex-mini`, reviewer = `openai/gpt-5-mini`, critics = `google/gemini-3.1-flash-lite`, research = `google/gemini-3.5-flash`.

| Metric | Value |
|---|---|
| Pages shipped (approved) | 53 / 60 (88%) |
| First-try approval rate | 73% |
| Average confidence | 0.76 |
| Tracks with ≥4 concept pages | 10 / 10 |
| Total spent | **$7.41** |
| Per-page cost (incl. revisions) | **$0.18** |
| Budget remaining (of $30 top-up) | $22.77 |
| Wall-clock for full sprint | ~3.5 hours @ 4 parallel workers |
| Retrospective cycle (gpt-5-mini, structured output) | $0.04, 25s |
| Auto-seeded stubs from first retro | 7 |

Per-page breakdown holds at roughly: research $0.02 · plan+scratch $0.02 · checklist $0.005 · write $0.05 · link $0.005 · review (rubric + 8 critics in parallel) $0.05 · revise (~0.27× of pages) $0.03. The closed loop is significantly cheaper than the v1 budget table above because the checklist + knockout selector reduce revision count and the critic panel runs in parallel.

---

## 9. What's still missing (the honest list)

| # | Gap | Effort | Why it matters |
|---|---|---|---|
| 1 | **Critic-attribution → persona update.** Failed-critic patterns surface in `backlog.md`, but the writer's persona YAML isn't yet auto-amended by the retrospective. | Medium | Without this, the same critic flags can recur cycle after cycle even when the retro identified them. |
| 2 | **Arc proposal phase.** Supervisor should propose arc slates once curriculum coverage stabilises; retrospective already flags candidates but doesn't promote them. | Small | The user picks which arcs become real; matches the "2 active arcs at a time" canon. |
| 3 | **Author pages.** High-frequency cited researchers (Bengio, He, Vaswani, Pearl…) deserve author pages with their seminal works; retrospective flags this but it's marked moderate-risk and stays human-review. | Small | Compounds citation density and gives every concept page a place to deep-link to. |
| 4 | **Visual sanity for math.** No automated check that LaTeX renders. | Small | Some pages have unrendered `\(...\)` because of escaping. |
| 5 | **Hybrid local/cloud mode.** Critics could run locally (Gemma 3 4B) while writer/reviewer stay cloud — cuts cost ~30%. Local-mode has working scaffold but Gemma 4 MLX is broken upstream. | Medium | Cheap parallel critic fanout without burning OpenRouter budget on every dimension. |

---

## 10. The one-line summary

**FAIRE is a closed-loop deep-agent system that writes a frontier-AI wiki under a fixed budget, in a single voice, citing only primary sources, and nudges every reader toward making something.**

When you change this system, that sentence is the test.
