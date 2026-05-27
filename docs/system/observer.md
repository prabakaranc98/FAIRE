---
title: Observer Dashboard
description: Closed-loop control system — real-time wiki quality signals
---

# Observer Dashboard

> Last observed: **2026-05-27 08:38 UTC** · [Source: `agents/runs/metrics.json`]

## Control State

| Signal | Set Point | Current | Error | Status |
|---|---|---|---|---|
| Coverage | 80% | 87.7% | 0.0% deficit | 🟢 |
| Quality | 0.85 | 0.77 | 0.08 deficit | 🟡 |
| Stale pages | 0 | 67 | 67 pages | 🔴 |
| Flagged pages | 0 | 4 | 4 pages | 🔴 |
| Budget | >$1 remaining | unlimited | full mode | 🟢 |

## Coverage

**71** pages generated of **81** total (87.7%) · **10** stubs remaining · **10** / 10 tracks with content

## Per-Track Status

| Track | Total | Generated | Approved | Coverage | Avg Conf | MVB | Stale |
|---|---|---|---|---|---|---|---|
| 01-ai | 7 | 7 | 7 | 🟢 100% | 🟡 0.76 | 0 | 7 |
| 02-generative-modeling | 7 | 7 | 7 | 🟢 100% | 🟡 0.81 | 0 | 7 |
| 03-representation-learning | 7 | 6 | 6 | 🟢 86% | 🟡 0.76 | 0 | 6 |
| 04-neural-networks-deep-learning | 9 | 6 | 6 | 🟡 67% | 🟡 0.78 | 0 | 5 |
| 05-statistical-probabilistic-ml | 8 | 8 | 5 | 🟢 100% | 🟡 0.78 | 0 | 7 |
| 06-reinforcement-learning | 8 | 8 | 8 | 🟢 100% | 🟡 0.78 | 0 | 8 |
| 07-attention-memory-reasoning-continual | 6 | 6 | 6 | 🟢 100% | 🟡 0.79 | 0 | 6 |
| 08-causal-statistical-inference | 9 | 8 | 6 | 🟢 89% | 🟡 0.73 | 0 | 8 |
| 09-algorithms-systems-for-ai | 15 | 10 | 8 | 🟡 67% | 🟡 0.74 | 0 | 9 |
| 10-complexity-cognition-natural-intelligence | 5 | 5 | 5 | 🟢 100% | 🟡 0.79 | 0 | 4 |

## Quality Trend

Last **10** runs · avg confidence **0.77** · first-pass approval **60%** · avg revisions **0.8** · trend **↑ improving** (+0.038)

## Budget

> Budget query failed: `OPENROUTER_API_KEY not set`

---

*This dashboard is updated automatically after every agent run and at the start of each 48h cycle.*
*Source: `agents/runs/metrics.json` · Pipeline: `observer.observe()` → `write_observer_page()`*
