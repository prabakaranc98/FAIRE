---
title: Phase 2 backlog
description: What the next sessions pick up after Phase 1 closed on 2026-05-27. Pre-loaded sprint state, known issues to address, and the rough week-by-week progression toward the full 30-arc roadmap.
updated: 2026-05-27
---

# Phase 2 backlog

> Phase 1 closed at 2026-05-27 16:06 EDT with 7 arc-index pages live, 110 substantive concept pages, and the full architectural loop (arc autonomy, primer-quality critic, multi-objective supervisor, frontier-grounded roadmap). This doc is what Phase 2 picks up from.
>
> See [Phase 1 retrospective](#) (in chat history) for the narrative of what shipped and what we learned.

## Pre-loaded state (no action needed — these activate on next cycle)

The sprint queue and code are already loaded for the next session. When the server restarts, the next cycle automatically:

### Sprint queue (28 arc items + 4 new content pending)

| Track | Arc | Status | Items waiting |
|---|---|---|---|
| 06-RL | `world-models-and-imagination` | mid-arc | 1 step (the last one) |
| 04-NN | `scaling-and-emergence` | not started | 1 index + 4 steps |
| 05-stat-prob | `probabilistic-programming-end-to-end` | not started | 1 index + 5 steps |
| 09-systems | `serve-an-llm-efficiently` | not started | 1 index + 5 steps |
| 09-systems | `train-at-scale` | not started | 1 index + 4 steps |
| 04-NN | `efficient-large-model-training` | not started | 1 index + 4 steps |
| + 06-RL / 08-causal / 10-complexity | `rl-for-post-training`, `causal-discovery-in-practice`, `scaling-laws-empirical`, `emergence-and-double-descent` | not started | ~20 items |

**Cost to drain the full ready queue:** approximately $10-15 at full quality, ~2 cycles.

### Code improvements active on next cycle (no opt-in needed)

- Path-style wikilinks resolve correctly — `[[curriculum/.../slug]]` form no longer renders as raw text.
- Arc-index breadcrumb correct — "Arc syllabus" instead of "Step 0 of N".
- Track index pages show arcs — the rebuilder walks subdirectories now.
- Catalog auto-rebuilds with live badges (Step 4.7 in `scheduler.py`).
- Primer-first writer voice — averaging 0.85 primer-quality on arc-step pages.
- Arc-spin from roadmap — supervisor reads `arc-roadmap.md` for canonical arcs (no more random `<track>-foundations` placeholders).

## Known issues to address in Phase 2

Ordered by impact on reader experience.

### 1. Arc-index pages struggle on primer (high priority)

**Observation:** of the 7 arc-index pages written this cycle, 3 scored primer ≤ 0.70 — including `agentic-rlvr-reasoner-index`, `world-models-and-imagination-index`, `causal-deep-learning-index`. The arc-step pages within those same arcs all landed at primer=0.85 cleanly.

**Root cause:** the writer's primer-first prompt was designed for *concept pages* (mentor walking through one mechanism). Arc-index pages are *syllabi* — they frame a journey, name a destination, set persona-fit, and pitch the reader on investing the time. Different voice. Different shape.

**Fix:** ship `agents/skills/arc-index-craft.md` — a writer skill for `mode == "arc-index"` only. Voice: "the editor of a curated 6-week course, telling you why this is worth it and what you'll have built at the end." Estimate: ~80 LOC, half a day.

### 2. `bayesian-neural-networks` chronically errors

**Observation:** the page has errored 4+ times across cycles, consistently at rev=2 with primer scores in the 0.70-0.74 band but other critic floors tripping. The writer keeps producing dense math without intuition tissue.

**Fix:** investigate at the page level — the concept may genuinely need a more careful narrative scaffold. Could be a one-off prompt amendment for this specific topic, OR could be a sign that BNNs are a topic where the model lacks training depth.

### 3. Critic-attribution → persona feedback loop

**Observation:** `critic-info-architecture` has been the universal weakest critic across every track for 8+ cycles, consistently flagging missing "Connected topics" and "Where this concept appears" sections. The system *sees* the pattern (the retrospective names it in every report) but doesn't *act* on it.

**Fix:** when a critic flags the same failure mode N cycles in a row, the retrospective should propose a persona amendment that the writer reads on subsequent runs. The persona update is a small YAML diff in `agents/src/frontier_agents/personas/<track>.yaml`. ~120 LOC plus tests.

### 4. The 13 needs-seeds arcs

13 of the 30 designed arcs in `arc-roadmap.md` reference concept slugs that don't exist yet (sparse-autoencoders, feature-circuits, cot-monitoring, video-generation, controlnet, lora-finetuning, clip-architecture, vision-language-pretraining, latent-dynamics, ring-attention, vector-search, embedding-models, self-verification, test-time-compute, causal-intervention, triton-kernels, physical-consistency, systematic-generalization).

**Fix:** the retrospective auto-seeder picks these up naturally — referenced 2+ times in any existing page → stub gets seeded → arc unlocks. Over 2-3 cycles this completes itself. No new code needed; just budget to run the cycles.

### 5. Author pages

The retrospective proposes author pages for heavily-cited researchers (Pearl, He, Vaswani, Bengio, Schölkopf, Schmidhuber). Currently marked moderate-risk and skipped. Worth promoting: author pages anchor citation discoverability and let concept pages deep-link back to one canonical write-up per major contributor.

**Fix:** extend `apply_safe_proposals` in `retrospective.py` to handle `action_type == "author-page-seed"` similarly to the arc-proposal branch — with guardrails (author cited 5+ times across pages, named in 3+ tracks). ~40 LOC.

## Rough week-by-week progression

**Week 1 — finish the ready arcs.** Top up $20, run 2 cycles. All 12 remaining ready arcs land. Live catalog shows 17 🟢 arcs out of 30. Cost: ~$15. Quality watch: do the new arc-index pages still struggle?

**Week 2 — ship arc-index-craft + author pages.** Code: ~120 LOC. Run 1 cycle to refresh the 3 weak arc-indexes from Week 1 + add 5-8 author pages. Cost: ~$5. By end of week 2, every live arc has a primer-grade syllabus + author anchors.

**Week 3 — seeds unlock more arcs.** The retrospective will have auto-seeded most of the 13 needs-seeds arc dependencies. 5-7 more arcs become ready. Run 2 cycles to spin them. Live catalog shows ~22-24 🟢 arcs. Cost: ~$10.

**Week 4 — critic-attribution + persona feedback.** Code: ~120 LOC. Run 1 cycle with persona auto-amendment active. Watch whether critic-info-architecture scores rise across the corpus. Quality watch: avg confidence should climb from 0.77 toward 0.85.

**End of Phase 2:** 25-28 of 30 arcs live, primer-quality averaging 0.85, info-architecture critic no longer the universal complaint, and 8+ author pages anchoring citations. Total Phase-2 spend: ~$40.

## The clean restart sequence

When you have a fresh top-up and want to resume:

```bash
# 1. Bump cap to match new headroom (current: 98)
sed -i.bak 's/^BUDGET_LIMIT_USD=.*/BUDGET_LIMIT_USD=120.0/' agents/.env

# 2. Restart server — DIRECT python, NOT via start.sh (that wrapper auto-respawns
#    on crash, which makes "stopping" require killing the wrapper too)
cd agents && source .venv/bin/activate
nohup python -m frontier_agents.cli serve --interval 999 > logs/server.log 2>&1 &

# 3. Trigger first cycle
curl -sX POST http://localhost:8765/trigger
```

The first cycle will drain the existing 32-item queue. After that, the supervisor auto-spins the next batch of ready arcs from `arc-roadmap.md`. Hands-off from there.

## Phase 3 horizon (not yet started — for later thinking)

- **MVB executability harness.** Every MVB recipe needs to be runnable. A nightly job that picks one MVB at random, sets up the named environment, runs the recipe, and checks the success metric. The 5-gate quality bar says the MVB should work; this would verify it.
- **Reader telemetry.** Currently zero observation of which arcs/pages readers actually engage with. Even a simple "did you scroll past 60%" signal would let the retrospective prioritize improvements on the most-read pages.
- **Hybrid local + cloud critic fanout.** Run the 8 cheap critics locally on the M-series Mac (Gemma 3 4B) while writer/reviewer stay cloud. Cuts ~$0.05/page = ~30% of per-page cost.
- **Cross-arc backlinking.** When an arc step in `causal-deep-learning` references a concept that's also a step in `causal-rl`, link the two arcs to each other. The retro can compute this from the artifact-chain graph.

---

*This doc is the canonical Phase 2 plan. Edit it as priorities shift.*
