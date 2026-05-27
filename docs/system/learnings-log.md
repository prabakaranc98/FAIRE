---
title: Learnings Log
description: Honest write-ups of what the FAIRE system discovered each session — what worked, what didn't, what's been re-thought. Human-curated, session-by-session. Distinct from the auto-generated Agent Changelog (which tracks each page run).
---

# Learnings Log

> A growing record of what we actually learned about the system, the models,
> and the pedagogy as the wiki was built. **Distinct from the auto-generated
> Agent Changelog** (per-page runs) — this is a human-curated record of
> sessions where something non-obvious surfaced and is worth carrying forward.

---

## 2026-05-27 — Local-mode reality check on M4 24GB unified RAM

**Goal of the session:** validate that the FAIRE pipeline can run end-to-end
against a local LLM (MLX on Apple Silicon) — no cloud spend, no API budget
pressure, full offline.

**What worked:**

- The integration is a **one-env-var swap**. Set `OPENAI_API_BASE=http://127.0.0.1:8081/v1`,
  change the role MODEL names to local IDs, restart `start.sh`. Zero code
  branches. ChatOpenAI speaks the standard OpenAI Chat Completions protocol
  that MLX/Ollama/vLLM/LMStudio all implement.
- Full FAIRE pipeline (`research → plan → checklist → write → link → review →
  log`) runs against `mlx-community/gemma-3-4b-it-qat-4bit` in **~4 min per
  page** on a 24GB M4 base — comparable to cloud wall-clock, at $0 cost.
- The v2 structural template lands cleanly: all 6 sections + 5-persona Build it
  variants. The page-shape works regardless of which model produces it.

**What didn't work — and the four real causes:**

The 4B model produces structurally-correct but **factually unreliable** pages.
Concrete failures we observed:

| Failure mode | Concrete example |
|---|---|
| **Hallucinated arxiv IDs with future-year prefixes** | `arxiv.org/abs/2604.16324`, `arxiv.org/abs/2512.22473v4` — invented IDs the model can't actually verify |
| **Context bleed from scratch_pad** | The Bayesian-inference page's "Where the field is now" section cited FLUX.1, DDPM, Latent Diffusion — none of which are Bayesian-inference papers |
| **Missing equations on math-heavy topics** | The do-calculus page produced **zero** LaTeX equations on a topic literally defined by 3 inference rules |
| **H1 format drift** | Model consistently emits `## Topic Name` (H2) where the schema requires `# Topic Name` (H1) |
| **Lenient self-review** | Reviewer (also 4B) scores its writer-self at 0.90 confidence on pages with the above defects, doesn't catch them |
| **Build it bleed** | Backpropagation page's Build it section described training Stable Diffusion on CIFAR-10 — wrong topic entirely |

The causes, ranked by leverage:

1. **Parameter count.** A 4B model has dramatically less factual memory than a
   ~100B+ cloud model. It knows the *shape* of a topic but can't recall specific
   papers, equations, or model IDs.
2. **Training cutoff.** Gemma 3 stops in March 2025. When asked for a recent
   paper, it invents a future-year arxiv ID instead of admitting ignorance.
3. **Context confusion under 17K-token prompts.** FAIRE's writer prompt is large
   (SCHEMA.md slice + WRITE_INSTRUCTIONS + scratch_pad). The 4B model's
   attention struggles to keep "this fact is about Topic A, that one is about
   adjacent Topic B" cleanly separated and mashes them.
4. **Same-model reviewer.** The reviewer is also 4B Gemma. Same-model
   self-review rarely catches its own writer's mistakes.

**Gemma 4 on MLX is not usable yet (as of mlx-lm 0.31.3, May 2026):**

All three Gemma 4 quants we tested (`gemma-4-e2b-it-4bit`,
`gemma-4-e4b-it-4bit`, `gemma-4-e4b-it-8bit`) load partially — missing
`v_proj`/`k_proj`/`k_norm` weights for the upper transformer layers — and
completions hang silently. The model registers at `/v1/models` but inference
never returns. This is a mlx-lm framework gap, not a quant bug. Use **Gemma 3
QAT variants** (Google's own QAT weights, more reliable than community quants)
until mlx-lm ships proper Gemma 4 support.

**Memory budgeting on 24GB:**

Gemma 3 27B QAT works for a single request but OOMs once FAIRE's pipeline
drives multiple LLM calls in a cycle. Gemma 3 12B QAT survives single requests
but OOMs on the 17K-token writer prompt + parallel critics. **Gemma 3 4B QAT
is the realistic ceiling on 24GB unified memory** with FAIRE's current prompt
shape. To go larger requires either an M4 Pro 48GB+ or a meaningful prompt
trim.

**The trim knobs that made it work:**

```bash
CRITIC_PANEL_DISABLE=true      # skip 8-way critic fan-out (saves ~5GB KV cache duplication)
SPRINT_WORKERS=1               # serial pages; parallel writer calls = Metal OOM
SCHEMA_PROMPT_BYTES=4000       # was 12000; trim agents/SCHEMA.md slice
SCRATCH_PAD_BYTES=8000         # was effectively unlimited
```

All four default to current cloud values — zero impact on the cloud loop.

**The honest recommendation that came out of this:**

Local mode is **fit for prototyping and structure validation**, not for
production wiki content as it stands. The pipeline works; the 4B model output
isn't publishable because half the citations are fake and adjacent topics
bleed in. Three usable paths:

- **Hybrid mode (recommended)**: keep `WRITER_MODEL=openai/gpt-5.1-codex-mini`
  (cloud), route reviewer + critics + research to local. Cuts ~50% of cloud
  spend without giving up writer accuracy.
- **Bigger local hardware**: M4 Pro 48GB lets you run Gemma 3 12B QAT
  comfortably; M4 Max 64GB+ runs the 27B. Quality recovers significantly.
- **Deterministic post-write validator** (TODO): add a no-LLM step in
  `review_node` that pings the arxiv API for every citation in the draft and
  strips/flags ones that don't resolve. Defends against hallucination
  regardless of model size; benefits cloud-generated pages too.

**What we shipped this session:**

- `agents/src/frontier_agents/llm.py` — `OPENAI_API_BASE` env-overrideable base URL
- `agents/.env.example` — local-mode block, four trim knobs documented
- `scripts/local-setup.sh` — one-time MLX install + model download + server start
- `docs/system/local-mode.md` — runtime guide with model recommendations per M4 tier
- `agents/tests/test_local_gemma4.py` — 7-test harness (unit + integration) for any local MLX server
- `agents/src/frontier_agents/nodes.py` — four trim knobs (`CRITIC_PANEL_DISABLE`, `CRITIC_PANEL_WORKERS`, `SCHEMA_PROMPT_BYTES`, `SCRATCH_PAD_BYTES`)

See commits `31df99d`, `7ab29ce`, and the README "Local-mode" section for the
shipped artifacts.

---

*This log is human-written. The auto-generated per-page record lives in the
[Agent Changelog](changelog.md).*
