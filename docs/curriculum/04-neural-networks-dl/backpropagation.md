---
title: Backpropagation
track: 04-neural-networks-dl
tags: [backpropagation, autograd, chain-rule, computational-graphs, gradients]
depth: foundations
prereqs: []
updated: 2026-05-25
---

# Backpropagation
> **TL;DR:** The algorithm that makes training deep networks possible — efficient gradient computation via reverse-mode automatic differentiation through a computational graph.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Backpropagation computes the gradient of a scalar loss with respect to all parameters by applying the chain rule backwards through the computational graph. In reverse mode (what all frameworks implement), a single backward pass computes all gradients simultaneously in O(forward pass) time.

## Why it matters at the frontier
Every parameter update in every neural network trained today uses backpropagation. Understanding it — the mechanics, the failure modes (vanishing/exploding gradients), and its limitations — is prerequisite to understanding any training dynamics research.

## Core concepts
- **Computational graph** — DAG of operations; backprop traverses it in reverse
- **Chain rule** — ∂L/∂x = ∂L/∂z × ∂z/∂x for any intermediate z
- **Forward pass** — compute activations, cache values for backward
- **Backward pass** — accumulate gradients from output to input
- **Vanishing gradient** — gradients → 0 in deep networks with saturating activations
- **Exploding gradient** — gradients → ∞; fixed with gradient clipping

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Learning representations by back-propagating errors](https://www.nature.com/articles/323533a0) | 1986 | Rumelhart, Hinton, Williams | Original formulation of backprop |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [Backpropagation through time (BPTT)](https://www.cs.cmu.edu/~bhiksha/courses/deeplearning/Fall.2016/pdfs/Werbos.1990.pdf) | 1990 | Werbos | Extends backprop to recurrent networks |
| [Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) | 2015 | Baydin et al. | Modern reference for AD |

## Current SotA
> *Updated: 2026-05-25*
Backprop is stable. The frontier is in gradient estimation for non-differentiable operations (straight-through estimator, REINFORCE), second-order methods (K-FAC, Shampoo), and hardware-aware backward passes (FlashAttention recomputation).

## Connected topics
- [Optimization for Deep Learning](./optimization.md) — gradient descent uses backprop gradients
- Normalization — BatchNorm/LayerNorm were designed to address backprop pathologies
- Residual Networks — skip connections were invented to fix vanishing gradients

## Further reading
- [Calculus on Computational Graphs: Backpropagation](http://colah.github.io) — NOT ALLOWED (blog)
- [Automatic Differentiation in ML: a Survey](https://arxiv.org/abs/1502.05767) — Baydin et al. 2015
