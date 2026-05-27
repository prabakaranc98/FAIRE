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

## 2026-05-27 (late) — Closing the loop: retrospective agent, hallucination defenses, knockout selection

**Goal of the session:** validate the closed-loop is actually a loop — that
the system can write pages, reflect on them, seed its own next sprint, and
keep cycling without a human in the queue.

**What worked**

- **The loop closed end-to-end.** Six retrospective cycles ran in one day,
  cycle #1 (~50 pages) → manual tests → cycle #6 (22 pages from a fresh
  sprint). Each cycle's scrum retro auto-seeded the next: 7 stubs after
  cycle #5, 10 stubs after cycle #6 — covering `policy-evaluation`,
  `representation-learning`, `quantization-aware-training`, `normalization`,
  `gradient-descent`, `mixed-precision-training`, and others. All of those
  topics surfaced because the retrospective spotted them being referenced
  across 2+ existing pages without backing files. The system is now growing
  its own knowledge graph.
- **Knockout selector earned its keep.** Of 22 sprint-2 topics, 5 improve
  passes regressed (`data-parallelism` 0.75 → 0.61, `policy-evaluation`
  0.81 → 0.60, `representation-learning` 0.81 → 0.74, `backpropagation`
  0.81 → 0.76, `flash-attention` 0.77 → 0.70). The knockout selector
  retained the higher-confidence first draft in every case — no shipped
  page got worse on the improve pass. Without it, the revision spiral
  would have actively degraded the wiki.
- **Hallucination defense at three layers works.** (1) Regex catches obvious
  fakes (future YYMM, implausibly-high suffix). (2) HTTP HEAD against
  `arxiv.org/abs/<id>` catches well-formed phantom IDs by matching the
  sentinel title `[<id>] Article identifier not recognized`. (3) The LLM
  reviewer's own issues list is post-scanned for hallucination keywords
  and applies a hard −0.10 penalty. The HEAD verifier is cached to
  `agents/runs/arxiv_cache.json` so every ID is hit at most once — current
  cost is well under 1s per page for a typical 5-citation draft.
- **Cost held at $0.18/page.** Sprint 2 was 22 topics + 22 improve passes
  + 8 critics × 30 invocations + 6 retrospective runs for **~$7**. Per-page
  economy is genuinely linear at this scale; nothing about the closed loop
  added overhead.

**What we got wrong — and corrected**

- We initially flagged `arxiv:2603.01761` as a "phantom citation" because
  the retrospective reported it appeared 6× across 4 different pages. The
  HEAD verifier confirmed it's real — "Modular Memory is the Key to
  Continual Learning Agents" (Dorovatas et al. 2026). **Citation
  concentration is not the same as fabrication.** A real seminal paper
  legitimately cited by multiple pages will cluster the same way. The
  retrospective should keep flagging concentration (it's a hint), but
  the conclusion has to come from the HEAD check, not from the count.
- The first version of the regex validator only blocked future-dated YYMM.
  That missed `2605.21058` (current month, implausible counter 21058 for
  May 27 2026), which the LLM reviewer caught as suspicious but didn't
  hard-fail. The fix: add a `suffix > 25000` check for the current and
  previous YYMM, plus the LLM-keyword post-scan. Layered defense — the
  regex is cheap, the HEAD verifier is authoritative, the LLM is the
  third opinion.

**Universal weakest signal across all 10 tracks**

`critic-info-architecture` is the lowest-scoring critic on **every** track
in cycle 6 (range 0.38 in 05-stat-prob / 08-causal up to 0.65 in 09-systems).
The recurring complaint: pages are missing a "Where this concept appears"
section linking back to arcs, and a "Connected topics" section linking
sideways to peer concepts. This is the next big lever — fixing the writer
prompt to mandate both sections should lift every track's avg confidence
by ~0.05–0.08.

**What we changed today**

- `agents/src/frontier_agents/retrospective.py` — new scrum retro agent
  (aggregator + LLM proposer + safe auto-applier), 720 LOC, wired into
  the scheduler.
- `agents/src/frontier_agents/nodes.py` — H1 fixer in `write_file_node`,
  layered arxiv validator + HEAD verifier + LLM-keyword post-scan in
  `review_node`.
- `docs/system/architecture.md` — rewritten as a closed-loop blueprint;
  the retrospective is now part of the per-cycle diagram, not a "planned"
  item in the gap list.
- `docs/overrides/home.html` — hero rewritten with Made-to-Stick framing
  around the MVB promise, plus a four-node closed-loop showcase strip.

**What's next**

- Persona/prompt update so the writer always emits "Where this concept
  appears" and "Connected topics" — closes the universal weakest-critic
  gap.
- Author pages for the most-cited researchers (`2604.15469` topped this
  cycle at 8 citations; an anchor page would let every concept page
  link back to one canonical author write-up).
- A `MAX_CYCLES` or `STOP_WHEN_COVERAGE_PCT` env var so the server has a
  declarative stop rule instead of relying on the budget envelope alone.

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
