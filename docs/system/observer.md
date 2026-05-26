---
title: Observer Dashboard
description: Closed-loop control system — real-time wiki quality signals
---

# Observer Dashboard

> Last observed: **2026-05-26 09:55 UTC** · [Source: `agents/runs/metrics.json`]

## Control State

| Signal | Set Point | Current | Error | Status |
|---|---|---|---|---|
| Coverage | 80% | 5.8% | 74.2% deficit | 🔴 |
| Quality | 0.85 | 0.73 | 0.12 deficit | 🔴 |
| Stale pages | 0 | 10 | 10 pages | 🔴 |
| Flagged pages | 0 | 0 | 0 pages | 🟢 |
| Budget | >$1 remaining | $3.69 | full mode | 🟢 |

## Coverage

**10** pages generated of **171** total (5.8%) · **161** stubs remaining · **8** / 15 tracks with content

## Per-Track Status

| Track | Total | Generated | Approved | Coverage | Avg Conf | MVB | Stale |
|---|---|---|---|---|---|---|---|
| 01-ai | 7 | 0 | 0 | 🔴 0% | 🟡 0.85 | 0 | 0 |
| 02-generative-modeling | 8 | 0 | 0 | 🔴 0% | ⚫ 0.00 | 0 | 0 |
| 03-representation-learning | 8 | 1 | 1 | 🟡 12% | 🟡 0.70 | 0 | 1 |
| 04-neural-networks-dl | 11 | 1 | 1 | 🟡 9% | 🟡 0.72 | 0 | 1 |
| 05-statistical-probabilistic-ml | 11 | 2 | 2 | 🟡 18% | 🟡 0.74 | 0 | 2 |
| 06-reinforcement-learning | 13 | 0 | 0 | 🔴 0% | 🟡 0.70 | 0 | 0 |
| 07-attention-memory-reasoning | 15 | 2 | 2 | 🟡 13% | 🟡 0.73 | 0 | 2 |
| 08-causal-statistical-inference | 12 | 0 | 0 | 🔴 0% | 🟡 0.70 | 0 | 0 |
| 09-algorithms-systems-ai | 14 | 1 | 1 | 🟡 7% | 🟡 0.73 | 0 | 1 |
| 10-complexity-cognition | 11 | 1 | 1 | 🟡 9% | 🟡 0.73 | 0 | 1 |
| 11-robotics-embodied-ai | 12 | 0 | 0 | 🔴 0% | ⚫ 0.00 | 0 | 0 |
| 12-physics-scientific-ai | 11 | 0 | 0 | 🔴 0% | ⚫ 0.00 | 0 | 0 |
| 13-graph-relational-ai | 12 | 0 | 0 | 🔴 0% | ⚫ 0.00 | 0 | 0 |
| 14-biology-life-sciences | 11 | 1 | 1 | 🟡 9% | 🟡 0.77 | 0 | 1 |
| 15-ml-theory-foundations | 15 | 1 | 1 | 🟡 7% | 🟡 0.70 | 0 | 1 |

## Quality Trend

Last **10** runs · avg confidence **0.78** · first-pass approval **10%** · avg revisions **1.8** · trend **↓ declining** (-0.047)

## Budget

$43.0472 used of $46.74 limit · $3.69 remaining · **full mode**

| Action | Est. Cost | Actions in Budget |
|---|---|---|
| generate | $0.20 | ~18 pages |
| improve | $0.15 | ~24 pages |
| mvb-only | $0.07 | ~52 pages |

---

*This dashboard is updated automatically after every agent run and at the start of each 48h cycle.*
*Source: `agents/runs/metrics.json` · Pipeline: `observer.observe()` → `write_observer_page()`*
