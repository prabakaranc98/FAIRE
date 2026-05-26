---
title: Fourier Neural Operator (FNO)
track: 12-physics-scientific-ai
tags: [fno, neural-operator, fourier, pde, operator-learning, discretization-invariant]
depth: research
prereqs: [pinn]
updated: 2026-05-25
---

# Fourier Neural Operator (FNO)
> **TL;DR:** A neural operator that learns mappings between function spaces by parameterizing integral operators in Fourier space — discretization-invariant, and up to three orders of magnitude faster than classical PDE solvers on fluid dynamics benchmarks.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Neural operators learn maps between infinite-dimensional function spaces rather than finite-dimensional vectors. FNO parameterizes the operator's kernel in Fourier space: apply a Fourier transform, multiply by learnable weights in frequency space, then inverse-transform. This captures global dependencies across the spatial domain efficiently and is resolution-invariant.

## Why it matters at the frontier
FNO achieves 1000× speedup over classical finite-element solvers on Navier-Stokes equations. It generalizes across discretizations — trained on one resolution, evaluated on another. GraphCast (weather prediction) and GeoFNO (irregular geometry) extend the approach to real-world scientific problems.

## Core concepts
- **Neural operator** — learns f: A → U mapping between function spaces (e.g., initial condition → solution)
- **Fourier layer** — linear integral operator parameterized in frequency space; captures long-range correlations
- **Discretization invariance** — learned operator applies to any discretization of the domain
- **Truncated Fourier modes** — only low-frequency modes are learned; high frequencies use a local linear transform
- **Universal approximation** — FNO can approximate any continuous operator (theory)

## Mathematical foundations
FNO layer (simplified):
\[
u_{l+1}(x) = \sigma\left(W u_l(x) + \mathcal{F}^{-1}\left[R_\phi \cdot \mathcal{F}[u_l]\right](x)\right)
\]

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895) | 2020 | Li, Kovachki et al. | Original FNO paper |
| [Neural Operator: Learning Maps Between Function Spaces](https://arxiv.org/abs/2108.08481) | 2021 | Kovachki et al. | Comprehensive framework |

## Current SotA
> *Updated: 2026-05-25*
Geometry-Informed Neural Operator (GINO, 2023) handles irregular geometries. UNO (U-Net-style) improves resolution handling. Aurora (Microsoft, 2024) and NeuralGCM apply operator learning to global weather prediction. The frontier: unifying neural operators with equivariant architectures for physics with symmetries.

## Connected topics
- [Physics-Informed Neural Networks (PINNs)](./pinn.md) — PINNs enforce physics as constraints; FNO learns solution operators
- [Equivariant Neural Networks](./equivariant-networks.md) — combining equivariance with operator learning
- Molecular Simulation — neural operators as fast force fields

## Further reading
- [Neural Operator: Graph Kernel Network for PDEs](https://arxiv.org/abs/2003.03485) — Li et al. 2020
