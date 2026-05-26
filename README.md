# FAIRE — Frontiers in AI Research and Engineering

> **A wiki that nudges you toward getting valuable builds done.**
>
> Not a tutorial chase. Not a bookmark stack. Not a pile of links. A reference
> for frontier AI/ML — curated, primary-source only — explicitly directed at the
> next thing you can build. Generated and maintained by a local multi-agent
> system, not written by hand.

Canonical statement: [pracha.me/frontier/faire](https://pracha.me/frontier/faire) · Curriculum source: [pracha.me/curriculum](https://pracha.me/curriculum).

---

## The pedagogical bet — three layers that compound

Wikipedia gives knowledge but no journey. roadmap.sh gives a path but no
compounding (random courses, each from scratch). paperswithcode gives
implementations but no pedagogy. FAIRE is built around the one thing none of
them offer: **learning that actually compounds.**

```
   CURRICULUM           ARCS                  MVBs
   (range)       →      (depth)         →     (proof)
   ─────────            ──────                ──────
   One page per         Opinionated 6–10      One build per
   concept across       step sequences        arc step, persona-
   15 tracks.           from concept to       tagged. Each
   Context that         frontier capability.  artifact is what
   carries forward.     Curated readings +    the next step
                        compounding-          loads.
                        trajectory table.
   ↑                                          ↑
   ↑                                          ↑
   ↑─────── nudges back into curriculum ──────↑
            for the prereqs each step needs
```

- **Curriculum** (`docs/curriculum/`) — 15 tracks (10 canonical + 5 extensions). Each pivotal page carries a **6-variant MVB block** (one per reader persona).
- **Arc index** (`docs/arcs/{arc}/index.md`) — opinionated paths with a **Build menu** showing every step's MVB + persona + compute + metric at a glance.
- **Arc step** (`docs/arcs/{arc}/step-NN-*.md`) — one build per page. `mvb_persona` declared in frontmatter. Step N's artifact is literally what step N+1 loads (verified by the compounding-chain audit).

---

## The 7-persona MVB model

Every pivotal curriculum page serves 7 reader personas through the same article,
routed by section and by their own MVB variant. See `docs/system/sense.md`.

| # | Persona | Comes to do | Time | Compute | Their build |
|---|---|---|---|---|---|
| 1 | Curious learner | Build a mental model | 30 min – 1 hr | Browser / Colab | A notebook that *shows* the concept |
| 2 | CS student / tinkerer | Reproduce on a laptop | 4 hr – 1 day | RTX 3060/4070 | Small training run that hits a target metric |
| 3 | Applied / production engineer | Ship at quality + latency | 3 days – 1 week | A10 / L4 / cloud | Real checkpoint served with measured latency |
| 4 | Applied researcher | Run one focused experiment | 3 days – 1 week | A100 × few | Ablation with a stated hypothesis |
| 5 | Theory student | Derive from first principles | 4 hr – 1 day | CPU | Derivation verified on toy data; one plot |
| 6 | Frontier researcher | Find an open problem to push | 1 week+ | Varies | Probe of an open question, with a falsifier named |
| 7 | PM / decision-maker | Decide whether to invest | 30 min | None | (synthesis only, no build) |

Each MVB variant must pass the **SENSIBLE · VALUABLE · FEASIBLE** quality bar
(see `agents/skills/mvb-recipe.md`). The `verify_mvb_stack` tool checks against
HuggingFace reality + rough GPU-VRAM math.

---

## The diagonal arc pattern

A good arc is **diagonal**, not vertical. It crosses domains: starts at a
specialized tool the reader can already touch, broadens to a research frame,
synthesizes a capability, lands at a frontier intersection. See
`agents/skills/arc-anatomy.md`.

| Specialized tool | Broader frame | Capability | Frontier intersection |
|---|---|---|---|
| Causal state-space models | Causal representation learning | Counterfactual reasoning | Causal RL |
| Transformer | Efficient attention | Long-context retrieval | Agentic memory systems |
| RLHF | Reward modeling | RL fine-tuning | Agent RL with tool use |
| VAE | Information bottleneck | Representation learning | World models |

Vertical arcs (same column 1 and 4) are vetoed by the `critic-editor` skill
during proposal — they're curriculum subsections, not arcs.

---

## Dual-mode arc proposal

The system can propose arcs **autonomously** OR you can drive it
**human-in-the-loop** via the CLI. Both modes share the same skills and write
to `docs/system/arc-proposals.md`. The human always picks before any arc is
materialized.

| Mode | Trigger | When to use |
|---|---|---|
| **Autonomous** | Supervisor on cycle (when canonical coverage ≥ 60%, budget=full, active_arcs<2) | Idle background discovery |
| **Human-in-the-loop** | `uv run python generate.py explore <seed>` | Named-seed exploration |

Both invoke the same 3-step explorer playbook (survey → map → pick + outline)
from `agents/skills/arc-exploration.md`. The `critic-editor` has veto authority
in both modes.

---

## The 8-critic review panel

Every page goes through a parallel panel of specialized critics, each scoring
one dimension. Composite confidence = `min(structured-rubric, min(critic scores))`
— any critic flagging a real problem can block.

| Critic skill | Lens | What it kills |
|---|---|---|
| `critic-human-centered` | 7-persona coverage | Pages that only serve one reader |
| `critic-beginner-onramp` | First 300 words | Jargon dumps, no analogy hook |
| `critic-wiki-voice` | Neutral / encyclopedic | Marketing-speak, tutorial cadence |
| `critic-info-architecture` | Backlinks + anchors + URL trust | Orphan pages, unresolved wikilinks |
| `critic-build-nudge` | Directed CTA + MVB feasibility | Generic "try training a model" |
| `critic-coverage` | Layer-specific scope | Stub-sized pages on major concepts |
| `critic-ux` | Math / code / scannability | Unrendered LaTeX, unlabeled code blocks |
| `critic-cohesion` | Through-line, synthesis | Correct-but-disconnected fact lists |

Plus one supervisor-level critic (`critic-editor`) with **veto authority** over
arc proposals when the diagonal pattern, compounding chain, or persona spread
fail.

The panel runs on `google/gemini-2.0-flash-lite-001` (non-reasoning) for
predictable parallel structured-output calls. The single structured rubric
review runs on `openai/gpt-5-mini` (reasoning) — reasoning is worth the cost
on one call but not eight.

---

## Verification tools (no LLM)

Two pure-tool checks the critics consult. See `agents/src/frontier_agents/tools.py`.

**`verify_mvb_stack(model_id, dataset_id, compute, training)`**
- Pings `huggingface.co/api/models/{id}` and `/api/datasets/{id}` — catches phantom HF IDs.
- Extracts parameter count from the model name (e.g. "Llama-3-8B" → 8e9).
- Estimates min required VRAM, compares against the named GPU.
- Returns `{feasible, model_check, dataset_check, vram_check, issues}`.

**`verify_source_trust(url, section)`**
- Multi-signal trust score on 0.0–1.0. Replaces the binary domain whitelist.
- Positives: approved-research-domain (+0.40), known-lab-domain (+0.50, e.g. transformer-circuits.pub, distill.pub, lilianweng.github.io, the major labs), `.edu` (+0.30), arxiv-paper (+0.30), HF card (+0.20), engineering blog in "In production" only (+0.50), approved-org GitHub in "Code & implementations" only (+0.50).
- Terminal negatives: medium.com, towardsdatascience, substack, wikipedia, reddit, twitter/x, youtube. Score = 0.
- Per-URL scores cached in `agents/runs/url_trust_cache.json`.

---

## The closed-loop control system

```
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Observer    │ →  │  Supervisor  │ →  │  Pipeline    │ →  │  runs.jsonl  │
│  (sensor)    │    │  (controller)│    │  (actuator)  │    │  (feedback)  │
└──────────────┘    └──────────────┘    └──────────────┘    └──────┬───────┘
       ▲                                                            │
       └────────────────────────────────────────────────────────────┘
```

- **Observer** (`observer.py::observe`) builds a `WikiObservation` snapshot — track metrics, quality trend, budget state, error signals (coverage deficit, quality deficit, stale pages, flagged pages, budget pressure).
- **Supervisor** (`supervisor.py::run_supervisor`) reads the observation, rewrites `sprints/current.md` with prioritized actions, may invoke `maybe_propose_arcs`.
- **Pipeline** is the LangGraph editorial loop (research → plan_and_scratch → write → link → review-panel → revise → write_file → commit → log_run).
- **Feedback** closes per-revision (review → revise loop), per-cycle (quality_trend → supervisor priorities), and **long-horizon** (critic patterns → `propose_persona_updates` → persona YAML diffs).

### Set points (overridable via env)
- `QUALITY_SETPOINT = 0.85` (reviewer confidence per page)
- `COVERAGE_SETPOINT = 0.80` (fraction generated per track)
- `STALENESS_THRESHOLD = 180` days
- `BUDGET_MINIMUM = $1.00` → paused mode
- `BUDGET_REDUCED = $3.00` → reduced mode (writer downgrades to fallback)

---

## CLI — what you can drive directly

```bash
# The autonomous loop (FastAPI server + APScheduler)
./start.sh --interval 1 --run-now             # 1h cycle, immediate first cycle
curl localhost:8765/                          # text dashboard
curl localhost:8765/budget                    # OpenRouter usage + mode
curl -X POST localhost:8765/trigger           # force one full cycle now

# Page generation
uv run python generate.py generate --topic diffusion-models --track 02-generative-modeling
uv run python generate.py generate --all-stubs --track 02-generative-modeling
uv run python generate.py improve --topic diffusion-models --track 02-generative-modeling

# Arc workflow (human-in-the-loop)
uv run python generate.py explore "world models"            # survey + propose arcs
uv run python generate.py spin-arc jepa-world-models        # materialize a proposal
uv run python generate.py spin-arc jepa-world-models --dry-run

# Supervisor + status
uv run python generate.py supervise                          # one supervisor cycle
uv run python generate.py status                             # coverage + quality table
uv run python generate.py persona-review                     # critic-pattern → persona-update proposals
```

---

## Quick start

```bash
# 1. Clone + install
git clone https://github.com/prabakaranc98/FAIRE.git && cd FAIRE
cd agents && uv sync

# 2. Configure API keys (gitignored)
cp .env.example .env    # then fill in OPENROUTER_API_KEY + EXA_API_KEY
# Optionally set BUDGET_LIMIT_USD=<cap> for a session spend cap

# 3. Run the autonomous loop
cd .. && ./start.sh --interval 1 --run-now &
curl localhost:8765/

# 4. Preview the wiki
uv --project agents run mkdocs serve  # http://localhost:8000
```

---

## Skills inventory (24 files in `agents/skills/`)

The agents read skills with matching `applies_to:` on every relevant LLM call.
The loader (`agents/src/frontier_agents/skills.py`) picks them up automatically.

**Editorial standards (7):**
`faire-sense` · `wiki-prose` · `math-latex` · `source-policy` · `sota-coverage` · `navigation-ia` · `mvb-recipe`

**Arc design (3):**
`arc-anatomy` (diagonal pattern + 5-bundle) · `arc-selection` (EV/$ scoring) · `arc-exploration` (survey → map → pick playbook) · `arc-context`

**Review panel — 8 page-level critics (applies_to: review):**
`critic-human-centered` · `critic-beginner-onramp` · `critic-wiki-voice` · `critic-info-architecture` · `critic-build-nudge` · `critic-coverage` · `critic-ux` · `critic-cohesion`

**Supervisor-level (1):**
`critic-editor` — veto authority over arc proposals

**Runtime knowledge (4):**
`langgraph-patterns` · `exa-search-deep` · `openrouter-routing` · `agent-runtime`

**Reasoning scaffolding (1):**
`reasoning-scaffolding` — plan-then-write (writer), step-by-step audit (critics), self-critique (arc proposer)

---

## Models (cheap-but-capable stack)

| Role | Model | Why |
|---|---|---|
| Writer | `google/gemini-3.1-flash-lite` | Long output, no hidden reasoning tokens |
| MVB | `google/gemini-3.1-flash-lite` | Same |
| Reviewer (structured rubric) | `openai/gpt-5-mini` | One call per page, reasoning worth it |
| Critic (8 parallel critics) | `google/gemini-2.0-flash-lite-001` | Non-reasoning, predictable parallel-fanout |
| Research / Planning | `google/gemini-2.0-flash-lite-001` | Cheap synthesis |
| Fallback (reduced mode) | `google/gemini-2.0-flash-lite-001` | Non-reasoning by design |

All routed via OpenRouter (`HTTP-Referer` + `X-Title` headers). Override via `.env`.

The README's previous premium stack (Claude Opus + Gemini Pro) plugs in by editing `.env`.

---

## Source policy (enforced via `verify_source_trust`)

| Section | Allowed |
|---|---|
| Default / Further reading | arxiv.org · *.edu · huggingface.co · pytorch.org · jax.readthedocs.io · openai.com/research · anthropic.com · deepmind.google · distill.pub · lilianweng.github.io · transformer-circuits.pub · the major lab research orgs |
| In production | + engineering.linkedin.com · ai.meta.com/research · developer.nvidia.com/blog · research.google · blog.google · aws.amazon.com/blogs/machine-learning · databricks.com/blog · stability.ai/research · techblog.netflix.com |
| Code & implementations | + GitHub repos under approved orgs: huggingface, openai, facebookresearch, pytorch, google-research, deepmind, NVlabs, allenai, EleutherAI, anthropics, stability-ai, lucidrains |

**Never approved:** medium.com · towardsdatascience.com · substack.com · wikipedia.org · reddit.com · twitter/x.com · youtube.com · personal `*.github.io` pages (except the curated few above).

---

## Budget control

`BUDGET_LIMIT_USD` in `agents/.env` is a soft cap on cumulative OpenRouter spend.
Observer reads OpenRouter's `/auth/key` to compute `remaining = cap - usage`.

| Remaining | Mode | Effect |
|---|---|---|
| `> $3` | `full` | All roles use their default models; arc proposal allowed |
| `$1 – $3` | `reduced` | Writer downgrades to `FALLBACK_MODEL`; supervisor refuses to propose arcs |
| `< $1` | `paused` | Sprint job skips all generation; only audit + improve-flagged remain |

To extend the budget mid-session, edit `BUDGET_LIMIT_USD = current_usage + N`
and restart the server. To re-anchor: `curl localhost:8765/budget` → take
`usage_usd` → cap = usage + your fresh allowance.

---

## Where the editorial intent lives

| File | Purpose |
|---|---|
| `docs/system/sense.md` | What FAIRE is, who it serves, what counts as right. The canonical brief readers see. |
| `docs/system/architecture.md` | System-engineering blueprint — layers, agent roster, file-system contract, control loops. |
| `agents/skills/faire-sense.md` | The agents' version of `sense.md`. Read on every writer call. |
| `agents/skills/arc-anatomy.md` | The diagonal arc pattern + 5-bundle anatomy with the pre-training arc as the worked example. |
| `PRINCIPLES.md` | The four objectives (discovery, evidence, inference, optimization) + 10 rules + Bird by Bird. The operating philosophy. |

When the system seems unsure what to do, the answer is in one of these.

---

## Security + operational constraints

- API keys in `agents/.env` only (gitignored). Never committed.
- Agents run LOCAL ONLY — not deployed online.
- Local-cap soft-budget (`BUDGET_LIMIT_USD`) plus OpenRouter's own hard cap.
- All generated content is reviewer-gated (rubric + 8-critic panel + critic-editor veto on arcs).
- `verify_mvb_stack` catches phantom HF IDs before they ship in MVB recipes.
- `verify_source_trust` catches non-approved external links before pages are committed.

See [PRINCIPLES.md](PRINCIPLES.md) for the operating philosophy and [docs/system/architecture.md](docs/system/architecture.md) for the full system blueprint.
