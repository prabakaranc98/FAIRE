---
title: Optimization for Deep Learning
track: 04-neural-networks-dl
tags: [optimization, sgd, adam, learning-rate, loss-landscape, training-dynamics]
depth: foundations
prereqs: [backpropagation]
updated: 2026-05-25
---

# Optimization for Deep Learning
> **TL;DR:** The algorithms that update neural network parameters — from SGD to Adam to modern adaptive methods — and the loss landscape geometry that determines whether they succeed.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
[stub — agent will fill with full explanation of SGD, momentum, Adam, AdamW, learning rate schedules, and loss landscape geometry]

## Why it matters at the frontier
Optimization choices directly determine whether large models train stably, how fast they converge, and what implicit biases they develop. The choice of optimizer and learning rate schedule is one of the most consequential engineering decisions in training a frontier model.

## Core concepts
- **SGD** — stochastic gradient descent; the baseline optimizer
- **Momentum** — exponential moving average of gradients; escapes local optima
- **Adam** — adaptive learning rates per parameter; m̂_t / (√v̂_t + ε)
- **AdamW** — Adam + decoupled weight decay; the default for LLMs
- **Learning rate schedule** — warmup + cosine decay is the standard recipe
- **Gradient clipping** — cap gradient norm; prevents exploding gradients in transformers
- **Loss landscape** — the geometry of the loss function over parameter space

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | 2014 | Kingma & Ba | Adam — the default optimizer for most DL |
| [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101) | 2017 | Loshchilov & Hutter | AdamW — fixes Adam's weight decay |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [Adam](https://arxiv.org/abs/1412.6980) | 2014 | Adaptive learning rates; still dominant 10 years later |
| [Visualizing the Loss Landscape of Neural Nets](https://arxiv.org/abs/1712.09913) | 2017 | Li et al. | Skip connections flatten the loss landscape |

## Current SotA
> *Updated: 2026-05-25*
AdamW remains the standard for transformer pretraining. Lion (symbolic optimizer, Chen et al. 2023) is more memory-efficient. Muon (Kosson et al. 2024) and Shampoo (second-order) are competitive for small-to-medium runs. For very large models, the muP (maximal update parametrization) regime changes hyperparameter scaling.

## Connected topics
- [[backpropagation]] — gradients that optimization acts on
- [[scaling-laws]] — learning rate schedule design interacts with scaling
- [[normalization]] — LayerNorm stabilizes the loss landscape

## Further reading
- [An overview of gradient descent optimization algorithms](https://arxiv.org/abs/1609.04747) — Ruder 2016
