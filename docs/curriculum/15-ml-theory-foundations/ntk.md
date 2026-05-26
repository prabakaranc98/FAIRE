---
title: Neural Tangent Kernel (NTK)
track: 15-ml-theory-foundations
tags: [ntk, infinite-width, kernel-regime, feature-learning, lazy-training]
depth: theoretical
prereqs: [pac-learning, gaussian-processes]
updated: 2026-05-25
---

# Neural Tangent Kernel (NTK)
> **TL;DR:** In the infinite-width limit, neural networks trained by gradient descent behave like kernel methods — the NTK is the fixed kernel they converge to — providing theoretical tractability but revealing important limitations vs. finite-width feature learning.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
For an infinitely wide neural network, gradient descent converges to a fixed kernel method. This kernel — the Neural Tangent Kernel — is determined by the network architecture at initialization. The NTK framework gives exact convergence guarantees and connects neural networks to GP regression, but only applies in the "lazy training" regime where weights barely move.

## Why it matters at the frontier
The NTK provides the first rigorous convergence theory for neural networks. But it reveals a fundamental limitation: NTK networks don't learn features — they stay near initialization. Real neural networks leave the NTK regime and learn representations, which is why scaling works in practice. µP (maximal update parametrization, Yang & Hu) characterizes the boundary between NTK and feature learning regimes.

## Core concepts
- **NTK** — K_θ(x,x') = ⟨∂f/∂θ(x), ∂f/∂θ(x')⟩; fixed at initialization for infinite width
- **Lazy training** — weights barely move; network stays in linear regime around initialization
- **Feature learning regime** — weights change significantly; representations adapt to data
- **Infinite-width limit** — as width → ∞, network at init converges to a GP
- **µP** — maximal update parametrization; scales hyperparameters to stay in feature learning regime at large width
- **NTK convergence** — GD converges to global minimum under NTK; finite-width generalizes

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Neural Tangent Kernel: Convergence and Generalization in Neural Networks](https://arxiv.org/abs/1806.07572) | 2018 | Jacot, Gabriel, Hongler | Original NTK paper |
| [Tensor Programs IV: Feature Learning in Infinite-Width Neural Networks](https://arxiv.org/abs/2011.14522) | 2020 | Yang & Hu | µP — beyond NTK; characterizes feature learning |

## Current SotA
> *Updated: 2026-05-25*
µP (Yang & Hu) is the practical tool: it gives hyperparameter transfer across widths and distinguishes NTK from feature learning. The frontier is understanding why deep finite-width networks in the feature learning regime generalize better than the NTK predicts.

## Connected topics
- [Gaussian Processes](../05-statistical-probabilistic-ml/gaussian-processes.md) — infinite-width networks at init converge to GPs
- [Double Descent](./double-descent.md) — NTK theory predicts aspects of double descent behavior
- [Scaling Laws](../04-neural-networks-dl/scaling-laws.md) — µP provides the framework for understanding what changes with scale

## Further reading
- [Tensor Programs I: Wide Feedforward or Recurrent Neural Networks of Any Architecture are Gaussian Processes](https://arxiv.org/abs/1902.04760) — Yang 2019
