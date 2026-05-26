---
title: Message Passing Neural Networks
track: 13-graph-relational-ai
tags: [message-passing, mpnn, gcn, gat, graphsage, aggregation]
depth: foundations
prereqs: []
updated: 2026-05-25
---

# Message Passing Neural Networks
> **TL;DR:** The dominant GNN framework — nodes aggregate messages from neighbors, update their representations, and repeat for k layers — unifying GCN, GAT, GraphSAGE, and most practical GNN architectures.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
MPNN (Gilmer et al., 2017) frames GNNs as a two-phase process per layer: (1) message passing — compute messages from neighbors m_{ij} = M(h_i, h_j, e_{ij}), (2) aggregation and update — h_i' = U(h_i, AGG({m_{ij}})). Different instantiations of M, AGG, and U give different architectures. GCN uses mean aggregation with normalized adjacency; GAT uses learned attention weights.

## Core concepts
- **Message function M** — how to compute the message from neighbor j to i
- **Aggregation AGG** — combine all incoming messages (sum, mean, max, attention)
- **Update function U** — update node representation from aggregated messages
- **Receptive field** — after k layers, each node sees k-hop neighborhood
- **Over-smoothing** — after many layers, all representations converge to same value
- **Over-squashing** — information from exponentially many nodes squeezed through one representation

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Neural Message Passing for Quantum Chemistry](https://arxiv.org/abs/1704.01212) | 2017 | Gilmer et al. | MPNN framework — unifies all GNNs |
| [Semi-supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907) | 2016 | Kipf & Welling | GCN — the simplest practical GNN |

## Connected topics
- [Equivariant Graph Neural Networks](./equivariant-gnn.md) — MPNN + physical symmetry constraints
- [Expressive Power of GNNs](./gnn-expressivity.md) — what functions can message passing compute?
- Molecular Property — MPNNs applied to molecular graphs

## Further reading
- [A Gentle Introduction to Graph Neural Networks](https://distill.pub/2021/gnn-intro/) — Sanchez-Lengeling et al.; distill.pub
