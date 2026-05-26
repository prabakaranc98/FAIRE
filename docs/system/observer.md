---
title: Observer Dashboard
description: Closed-loop control system — real-time wiki quality signals
---

# Observer Dashboard

> Last observed: **2026-05-26 10:50 UTC** · [Source: `agents/runs/metrics.json`]

## Control State

| Signal | Set Point | Current | Error | Status |
|---|---|---|---|---|
| Coverage | 80% | 14.6% | 65.4% deficit | 🔴 |
| Quality | 0.85 | 0.72 | 0.13 deficit | 🔴 |
| Stale pages | 0 | 25 | 25 pages | 🔴 |
| Flagged pages | 0 | 8 | 8 pages | 🔴 |
| Budget | >$1 remaining | $30.13 | full mode | 🟢 |

## Coverage

**25** pages generated of **171** total (14.6%) · **146** stubs remaining · **10** / 15 tracks with content

## Per-Track Status

| Track | Total | Generated | Approved | Coverage | Avg Conf | MVB | Stale |
|---|---|---|---|---|---|---|---|
| 01-ai | 7 | 0 | 0 | 🔴 0% | 🟡 0.80 | 0 | 0 |
| 02-generative-modeling | 8 | 0 | 0 | 🔴 0% | 🟡 0.62 | 0 | 0 |
| 03-representation-learning | 8 | 2 | 2 | 🟡 25% | 🟡 0.77 | 0 | 2 |
| 04-neural-networks-dl | 11 | 2 | 2 | 🟡 18% | 🟡 0.74 | 0 | 2 |
| 05-statistical-probabilistic-ml | 11 | 3 | 3 | 🟡 27% | 🟡 0.68 | 0 | 3 |
| 06-reinforcement-learning | 13 | 0 | 0 | 🔴 0% | 🟡 0.76 | 0 | 0 |
| 07-attention-memory-reasoning | 15 | 3 | 3 | 🟡 20% | 🟡 0.74 | 0 | 3 |
| 08-causal-statistical-inference | 12 | 4 | 3 | 🟡 33% | 🟡 0.70 | 0 | 4 |
| 09-algorithms-systems-ai | 14 | 1 | 1 | 🟡 7% | 🟡 0.75 | 0 | 1 |
| 10-complexity-cognition | 11 | 4 | 4 | 🟡 36% | 🟡 0.78 | 0 | 4 |
| 11-robotics-embodied-ai | 12 | 0 | 0 | 🔴 0% | 🟡 0.60 | 0 | 0 |
| 12-physics-scientific-ai | 11 | 1 | 1 | 🟡 9% | 🟡 0.65 | 0 | 1 |
| 13-graph-relational-ai | 12 | 0 | 0 | 🔴 0% | 🟡 0.60 | 0 | 0 |
| 14-biology-life-sciences | 11 | 2 | 2 | 🟡 18% | 🟡 0.75 | 0 | 2 |
| 15-ml-theory-foundations | 15 | 3 | 3 | 🟡 20% | 🟡 0.69 | 0 | 3 |

## Quality Trend

Last **10** runs · avg confidence **0.52** · first-pass approval **10%** · avg revisions **1.8** · trend **↓ declining** (-0.040)

## Budget

$46.8707 used of $77.00 limit · $30.13 remaining · **full mode**

| Action | Est. Cost | Actions in Budget |
|---|---|---|
| generate | $0.20 | ~150 pages |
| improve | $0.15 | ~200 pages |
| mvb-only | $0.07 | ~430 pages |

---

*This dashboard is updated automatically after every agent run and at the start of each 48h cycle.*
*Source: `agents/runs/metrics.json` · Pipeline: `observer.observe()` → `write_observer_page()`*
