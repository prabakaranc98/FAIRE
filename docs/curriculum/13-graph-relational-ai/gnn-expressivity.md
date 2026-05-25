---
title: Expressive Power of GNNs
track: 13-graph-relational-ai
tags: [gnn-expressivity, weisfeiler-leman, wl-test, graph-isomorphism, k-wl]
depth: theoretical
prereqs: [message-passing]
updated: 2026-05-25
---

# Expressive Power of GNNs
> **TL;DR:** Standard message-passing GNNs are at most as expressive as the 1-WL graph isomorphism test — they cannot distinguish some non-isomorphic graphs, motivating higher-order GNNs and subgraph methods.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **Weisfeiler-Leman (WL) test** — iterative color refinement algorithm for graph isomorphism
- **1-WL** — standard WL; equivalent in power to MPNN (Xu et al. 2019)
- **GIN** — Graph Isomorphism Network; most expressive MPNN (uses SUM aggregation)
- **k-WL** — higher-order WL; considers k-tuples of nodes; strictly more expressive
- **Higher-order GNNs** — use k-tuple features to exceed 1-WL expressiveness
- **Subgraph GNNs** — achieve higher expressiveness by running GNNs on subgraphs

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [How Powerful are Graph Neural Networks?](https://arxiv.org/abs/1810.00826) | 2018 | Xu et al. | Proves MPNN ≡ 1-WL; introduces GIN |

## Connected topics
- [[message-passing]] — MPNNs are bounded by 1-WL expressiveness
- [[graph-transformers]] — attention extends expressiveness beyond 1-WL

## Further reading
- [Equivariant Subgraph Aggregation Networks](https://arxiv.org/abs/2110.02910) — Bevilacqua et al. 2022
