---
title: Equivariant Graph Neural Networks
track: 13-graph-relational-ai
tags: [equivariant-gnn, e3, se3, molecular, nequip, mace, alphafold]
depth: research
prereqs: [message-passing, equivariant-networks]
updated: 2026-05-25
---

# Equivariant Graph Neural Networks
> **TL;DR:** GNNs that respect 3D physical symmetries by construction — essential for molecular property prediction, force fields, and protein structure modeling, where rotation/translation should not change the answer.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **E(3)-equivariance** — output transforms predictably under 3D rotations/reflections/translations
- **Steerable message passing** — messages are equivariant vectors/tensors, not scalars
- **Spherical harmonics** — basis for decomposing equivariant features by transformation type
- **EGNN** — simple E(n)-equivariant network; updates node positions and features jointly
- **NequIP/Allegro** — high-accuracy equivariant neural network potentials for MD
- **MACE** — message-passing with equivariant many-body interactions

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [E(n) Equivariant Graph Neural Networks](https://arxiv.org/abs/2102.09844) | 2021 | Satorras et al. | EGNN — simple, effective, widely used |
| [MACE: Higher Order Equivariant Message Passing Neural Networks](https://arxiv.org/abs/2206.07697) | 2022 | Batatia et al. | MACE — current SotA for atomic simulations |

## Current SotA
> *Updated: 2026-05-25*
MACE-MP-0 (2023) is a universal foundation force field covering most of the periodic table. AlphaFold 3 uses E(3)-equivariant diffusion for structure prediction. Equiformer v2 achieves SotA on OC20 catalyst discovery benchmark with attention + equivariance.

## Connected topics
- [[equivariant-networks]] — general symmetry-preserving architectures
- [[protein-structure]] — AlphaFold's structure module is equivariant
- [[molecular-simulation]] — equivariant force fields for ab initio MD

## Further reading
- [A Hitchhiker's Guide to Geometric GNNs for 3D Atomic Systems](https://arxiv.org/abs/2312.07511) — Duval et al. 2023
