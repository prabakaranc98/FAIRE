---
title: Observer Dashboard
description: Closed-loop control system — real-time wiki quality signals
---

# Observer Dashboard

> Last observed: **2026-05-27 15:57 EDT** · [Source: `agents/runs/metrics.json`]

## Control State

| Signal | Set Point | Current | Error | Status |
|---|---|---|---|---|
| Coverage | 80% | 96.6% | 0.0% deficit | 🟢 |
| Quality | 0.85 | 0.76 | 0.09 deficit | 🟡 |
| Stale pages | 0 | 137 | 137 pages | 🔴 |
| Flagged pages | 0 | 9 | 9 pages | 🔴 |
| Budget | >$1 remaining | $0.26 | paused mode | 🔴 |

## Coverage

**144** pages generated of **149** total (96.6%) · **5** stubs remaining · **11** / 10 tracks with content

## Per-Track Status

| Track | Total | Generated | Approved | Coverage | Avg Conf | MVB | Stale |
|---|---|---|---|---|---|---|---|
| 01-ai | 11 | 11 | 7 | 🟢 100% | 🟡 0.78 | 0 | 11 |
| 02-generative-modeling | 12 | 12 | 6 | 🟢 100% | 🟡 0.76 | 0 | 12 |
| 03-representation-learning | 12 | 12 | 6 | 🟢 100% | 🟡 0.76 | 0 | 12 |
| 04-neural-networks-deep-learning | 19 | 18 | 10 | 🟢 95% | 🟡 0.75 | 0 | 16 |
| 05-statistical-probabilistic-ml | 20 | 20 | 9 | 🟢 100% | 🟡 0.77 | 0 | 19 |
| 06-reinforcement-learning | 13 | 13 | 7 | 🟢 100% | 🟡 0.75 | 0 | 13 |
| 07-attention-memory-reasoning-continual | 6 | 6 | 6 | 🟢 100% | 🟡 0.79 | 0 | 6 |
| 08-causal-statistical-inference | 13 | 13 | 9 | 🟢 100% | 🟡 0.78 | 0 | 13 |
| 09-algorithms-systems-for-ai | 37 | 33 | 33 | 🟢 89% | 🟡 0.76 | 0 | 31 |
| 10-complexity-cognition-natural-intelligence | 5 | 5 | 5 | 🟢 100% | 🟡 0.79 | 0 | 4 |
| curriculum | 1 | 1 | 0 | 🟢 100% | ⚫ 0.00 | 0 | 0 |

## Quality Trend

Last **10** runs · avg confidence **0.75** · first-pass approval **0%** · avg revisions **1.5** · trend **↓ declining** (-0.029)

## Budget

$97.7356 used of $98.00 limit · $0.26 remaining · **paused mode**

| Action | Est. Cost | Actions in Budget |
|---|---|---|
| generate | $0.20 | ~1 pages |
| improve | $0.15 | ~1 pages |
| mvb-only | $0.07 | ~3 pages |

---

*This dashboard is updated automatically after every agent run and at the start of each 48h cycle.*
*Source: `agents/runs/metrics.json` · Pipeline: `observer.observe()` → `write_observer_page()`*
