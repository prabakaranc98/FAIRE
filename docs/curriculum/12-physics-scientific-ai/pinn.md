---
title: Physics-Informed Neural Networks (PINNs)
track: 12-physics-scientific-ai
tags: [pinn, pde, physics-constrained, collocation, scientific-ml]
depth: foundations
prereqs: [backpropagation, optimization]
updated: 2026-05-25
---

# Physics-Informed Neural Networks (PINNs)
> **TL;DR:** Neural networks trained to satisfy both data and physical laws (PDEs) as soft constraints in the loss — enabling solution of differential equations without mesh generation, and learning from limited noisy data.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
PINNs solve forward and inverse PDE problems by encoding physics as additional loss terms. Given a PDE L[u(x,t)] = f, the PINN loss has three terms: (1) a data loss on known boundary/initial conditions, (2) a residual loss at interior collocation points measuring PDE violation, and (3) optionally a regularization term. The network learns a differentiable solution approximating the PDE.

## Core concepts
- **Collocation points** — spatial/temporal sample points where PDE residual is evaluated
- **PDE residual loss** — ||L[u_θ(x,t)] - f(x,t)||² at collocation points
- **Automatic differentiation** — computes partial derivatives u_x, u_t, u_xx needed for PDE evaluation
- **Forward problem** — given known PDE, find solution u(x,t)
- **Inverse problem** — given observations of u, infer PDE parameters
- **Adaptive sampling** — place more collocation points where residual is high

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear PDEs](https://arxiv.org/abs/1711.10561) | 2019 | Raissi, Perdikaris, Karniadakis | Original PINN paper |

## Current SotA
> *Updated: 2026-05-25*
PINNs are practical for small-to-medium dimensional problems but struggle with high frequencies and chaotic systems. Neural operators (FNO, DeepONet) have largely replaced PINNs for problems with many query points. PINNs remain valuable for inverse problems with sparse data and when incorporating physics regularization for generalization.

## Connected topics
- [Fourier Neural Operator (FNO)](./fno.md) — Fourier Neural Operators are the scalable alternative to PINNs for solution operators
- Neural Odes — ODE version of physics-constrained learning
- [Equivariant Neural Networks](./equivariant-networks.md) — incorporating physical symmetries directly into architecture

## Further reading
- [Scientific Machine Learning Through Physics-Informed Neural Networks](https://arxiv.org/abs/2201.05624) — Cuomo et al. 2022
