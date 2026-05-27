---
title: Observer Dashboard
description: Closed-loop control system — real-time wiki quality signals
---

# Observer Dashboard

> Last observed: **2026-05-27 08:13 UTC** · [Source: `agents/runs/metrics.json`]

## Control State

| Signal | Set Point | Current | Error | Status |
|---|---|---|---|---|
| Coverage | 80% | 100.0% | 0.0% deficit | 🟢 |
| Quality | 0.85 | 0.77 | 0.08 deficit | 🟡 |
| Stale pages | 0 | 67 | 67 pages | 🔴 |
| Flagged pages | 0 | 0 | 0 pages | 🟢 |
| Budget | >$1 remaining | $15.31 | full mode | 🟢 |

## Coverage

**71** pages generated of **71** total (100.0%) · **0** stubs remaining · **1** / 15 tracks with content

## Per-Track Status

| Track | Total | Generated | Approved | Coverage | Avg Conf | MVB | Stale |
|---|---|---|---|---|---|---|---|
| concepts | 71 | 71 | 0 | 🟢 100% | ⚫ 0.00 | 0 | 67 |

## Quality Trend

Last **10** runs · avg confidence **0.77** · first-pass approval **60%** · avg revisions **0.8** · trend **↑ improving** (+0.038)

## Budget

$69.6868 used of $85.00 limit · $15.31 remaining · **full mode**

| Action | Est. Cost | Actions in Budget |
|---|---|---|
| generate | $0.20 | ~76 pages |
| improve | $0.15 | ~102 pages |
| mvb-only | $0.07 | ~218 pages |

---

*This dashboard is updated automatically after every agent run and at the start of each 48h cycle.*
*Source: `agents/runs/metrics.json` · Pipeline: `observer.observe()` → `write_observer_page()`*
