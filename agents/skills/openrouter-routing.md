---
skill: openrouter-routing
description: OpenRouter routing patterns FAIRE uses — header conventions, role-to-model mapping, reasoning vs non-reasoning models, budget gate. Read by the supervisor when deciding actions under different budget modes.
applies_to: [supervisor]
triggers: [all pages]
---

# Skill: OpenRouter routing for FAIRE

This skill is the agents' durable knowledge of how FAIRE routes LLM calls through OpenRouter. Source of truth: `agents/src/frontier_agents/llm.py` (factory) and `agents/.env` (model IDs).

## All LLM calls go through one factory

```python
from frontier_agents.llm import get_llm
llm = get_llm("writer", temperature=0.3)
```

`get_llm` is `@lru_cache`'d on `(role, temperature)` — same role+temperature always returns the same `ChatOpenAI` instance. Adding a new role creates a new cache entry; no manual cache management needed.

Never instantiate `ChatOpenAI` or `ChatAnthropic` directly inside a node. Doing so bypasses the role-based max_tokens routing, the OpenRouter headers, and the budget gate.

## The roles and what they're for

| Role | Used by | Model (cheap stack) | Notes |
|---|---|---|---|
| `writer` | `write_draft_node`, `write_arc_step_node`, `write_arc_index_node` | `google/gemini-3.1-flash-lite` | full pages; max_tokens=16K |
| `mvb` | `mvb_recipe_node` | `google/gemini-3.1-flash-lite` | MVB section only; max_tokens=4K |
| `reviewer` | structured rubric in `review_node` | `openai/gpt-5-mini` | reasoning model; max_tokens=8K. One call per page. |
| `critic` | `_run_critic_panel` (8 parallel calls) | `google/gemini-2.0-flash-lite-001` | **non-reasoning** model — required for parallel fanout. max_tokens=4K. |
| `research` | research summary, supervisor analysis | `google/gemini-2.0-flash-lite-001` | plan + summaries; max_tokens=4K |
| `fallback` | reduced-budget writer | `google/gemini-2.0-flash-lite-001` | downgrade target when budget enters reduced mode |

If `.env` overrides these via `*_MODEL` env vars, the override wins.

## Reasoning vs non-reasoning models — when each fails

OpenAI's o-series and `gpt-5-mini` are *reasoning models*: they spend tokens on internal chain-of-thought *before* producing visible output. Those reasoning tokens count against `max_tokens`. A typical reasoning-model failure mode is `LengthFinishReasonError` — the model ran out of token budget mid-CoT and never produced the final answer.

Implications for FAIRE:
- **Single-call slots** (the structured rubric reviewer): reasoning is worth the cost. Use `gpt-5-mini`.
- **Parallel-fanout slots** (the critic panel with 8 calls): never use a reasoning model. Use `gemini-2.0-flash-lite-001` (or any non-reasoning model). Reasoning models in fanout context multiply the LengthFinish risk by N.
- **Writer/MVB**: usually non-reasoning is fine — the writer needs LONG output, not deep thinking. Reasoning models cap the visible output indirectly via their hidden CoT budget.

If you ever add a new parallel-fanout LLM use, route it through a non-reasoning role. The `critic` role is the canonical example.

## Required OpenRouter headers

Every call must set these headers via `model_kwargs={"extra_headers": ...}`:

```python
extra_headers = {
    "HTTP-Referer": "https://prabakaranc98.github.io/FAIRE",
    "X-Title": "Frontier Wiki Agent",
}
```

OpenRouter uses these for attribution and rate-limit routing. `llm.py` already injects them — don't re-implement.

## Budget gate — what `observer.check_budget()` decides

`observer.py::check_budget()` queries `https://openrouter.ai/api/v1/auth/key` and returns a `BudgetState`. The local `BUDGET_LIMIT_USD` env var soft-caps usage when OpenRouter has no hard limit.

Three modes:

| Remaining | Mode | Effect |
|---|---|---|
| > $3 (BUDGET_REDUCED) | `full` | all roles use their default models |
| $1–$3 | `reduced` | sprint job overrides `WRITER_MODEL` to `FALLBACK_MODEL`; skips low-priority improvements |
| < $1 (BUDGET_MINIMUM) | `paused` | sprint job skips all generation; only audit + improve-flagged remain |

The supervisor's `maybe_propose_arcs()` refuses to propose new arcs unless mode is `full` — proposing arcs in reduced mode would queue work the system can't complete.

## Setting / changing models

Edit `agents/.env`:
```
WRITER_MODEL=google/gemini-3.1-flash-lite       # current cheap stack
# WRITER_MODEL=anthropic/claude-opus-4.7        # premium when budget allows
REVIEWER_MODEL=openai/gpt-5-mini
CRITIC_MODEL=google/gemini-2.0-flash-lite-001
RESEARCH_MODEL=google/gemini-2.0-flash-lite-001
FALLBACK_MODEL=google/gemini-2.0-flash-lite-001
```

Restart the server for changes to take effect — `@lru_cache` holds instances across calls but is invalidated on process restart.

Always keep `FALLBACK_MODEL` non-reasoning. Reduced mode is exactly when you don't want to spend reasoning tokens.

## `BUDGET_LIMIT_USD` — the soft cap

The user's $10-per-session budget is enforced via `BUDGET_LIMIT_USD` in `.env`. The cap is *cumulative spend at OpenRouter*, not delta-since-last-set, so the supervisor should:
- Read `usage_usd` from `/budget`
- When the user asks for "another $10", update `BUDGET_LIMIT_USD = usage_usd + 10`

Never set the cap *below* current usage — that immediately puts mode = paused.

## Anthropic fallback path

If only `ANTHROPIC_API_KEY` is set (not `OPENROUTER_API_KEY`), `get_llm` falls through to `langchain_anthropic.ChatAnthropic`. This bypasses OpenRouter entirely. Not the default path — use OpenRouter unless OpenAI/Anthropic billing differs meaningfully.

## Related skills

- `agent-runtime.md` — `observer.check_budget()` and the sprint_job budget gate
- `langgraph-patterns.md` — when to add a new role (each parallel LLM call should map to one role)
- `mvb-recipe.md` — the MVB quality bar verification could route through a dedicated role someday
