---
title: Equivariant Neural Networks
track: 12-physics-scientific-ai
tags: [equivariance, e3-equivariance, geometric-dl, symmetry, group-theory]
depth: theoretical
prereqs: [backpropagation]
updated: 2026-05-25
---

# Equivariant Neural Networks
> **TL;DR:** Neural networks designed to respect physical symmetries (rotations, translations, reflections) by construction — transforming inputs predictably rather than learning symmetry empirically, leading to dramatic data efficiency.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
A function f is equivariant to group G if f(T_g(x)) = T_g'(f(x)) for all group elements g. For 3D molecular data: rotating the input molecule should rotate the output forces by the same rotation. E(3)-equivariant networks encode this invariance into their architecture — removing the need to learn it from data, which would require astronomically large datasets.

## Why it matters at the frontier
AlphaFold 2's structure module is E(3)-equivariant. MACE, NequIP, and Allegro are equivariant force field models that outperform classical force fields by orders of magnitude. SE(3)-Transformer and Equiformer bring equivariance to attention mechanisms.

## Core concepts
- **Group** — a set with a composition operation satisfying closure, associativity, identity, inverse
- **Equivariance** — f(g·x) = g·f(x); output transforms consistently with input
- **Invariance** — special case of equivariance where f(g·x) = f(x) (output unchanged)
- **E(3)** — the group of all rotations, reflections, and translations in 3D space
- **SE(3)** — proper rotations and translations only (no reflections)
- **Steerable features** — features that transform as irreducible representations of the symmetry group

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) | 2021 | Satorras et al. | EGNN — simple E(n) equivariant network |
| [Geometric Deep Learning: Grids, Groups, Graphs, Geodesics, and Gauges](https://arxiv.org/abs/2104.13478) | 2021 | Bronstein et al. | The unifying geometric DL blueprint |

## Current SotA
> *Updated: 2026-05-25*
MACE (Batatia et al., 2022) is the current best equivariant force field, used in AlphaFold 3's molecular dynamics. Equiformer v2 extends equivariance to attention-based architectures for materials and catalysis (Open Catalyst Project). RFDiffusion uses SE(3) equivariance for protein design.

## Connected topics
- [Equivariant Graph Neural Networks](../13-graph-relational-ai/equivariant-gnn.md) — GNNs with E(3) equivariance for molecules and materials
- [Protein Structure Prediction](../14-biology-life-sciences/protein-structure.md) — AlphaFold uses equivariance in its structure module
- Molecular Simulation — equivariant force fields for MD simulation

## Further reading
- [A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems](https://arxiv.org/abs/2312.07511) — Duval et al. 2023
