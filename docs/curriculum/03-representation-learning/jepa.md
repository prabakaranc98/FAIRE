---
title: Joint Embedding Predictive Architectures (JEPA)
track: 03-representation-learning
tags: [jepa, i-jepa, v-jepa, predictive-learning, world-models, self-supervised]
depth: research
prereqs: [self-supervised-learning, masked-autoencoders]
updated: 2026-05-25
---

# Joint Embedding Predictive Architectures (JEPA)
> **TL;DR:** A family of self-supervised architectures that predict abstract representations of missing context — rather than reconstructing pixels — learning higher-level world models without generative overhead.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
JEPAs consist of two encoders (context and target) and a predictor. Given a masked input, the context encoder produces an embedding; the predictor maps it to the expected target embedding. Training is in representation space, not pixel space — no decoder required. This avoids the model wasting capacity on unpredictable low-level details.

## Why it matters at the frontier
JEPAs (LeCun's group, Meta AI) are the current alternative to masked autoencoders and contrastive methods. They learn higher-level semantic representations by predicting *in latent space*. V-JEPA learns spatiotemporal representations from video without labels. The JEPA framework is central to LeCun's vision for autonomous AI systems.

## Core concepts
- **Abstract prediction** — predict in embedding space, not pixel space
- **Context encoder** — encodes visible (unmasked) patches
- **Target encoder** — EMA-updated encoder for masked patch targets (no gradient flow)
- **Predictor** — lightweight network mapping context embedding to predicted target embedding
- **Collapse prevention** — target encoder EMA + stop gradient avoids representation collapse
- **Multi-block masking** — mask large contiguous regions (not random pixels) to force semantic prediction

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture](https://arxiv.org/abs/2301.08243) | 2023 | Assran et al. (Meta AI) | I-JEPA — image JEPA; core formulation |
| [Revisiting Feature Prediction for Learning Visual Representations](https://arxiv.org/abs/2408.00687) | 2024 | Bardes et al. (Meta AI) | V-JEPA — video JEPA |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [A Path Towards Autonomous Machine Intelligence](https://openreview.net/forum?id=BZ5a1r-kVsf) | 2022 | LeCun | Conceptual framework for JEPAs as world models |
| [I-JEPA](https://arxiv.org/abs/2301.08243) | 2023 | Assran et al. | First instantiation; strong vision features |

## Current SotA
> *Updated: 2026-05-25*
V-JEPA achieves strong video understanding without labels or text supervision. The JEPA framework is being extended to multimodal and robot trajectory prediction domains. A3D-JEPA (audio-3D) and robotic JEPA variants are active research.

## Connected topics
- Masked Autoencoders — MAE predicts pixels; JEPA predicts representations
- [Self-Supervised Learning](./self-supervised-learning.md) — JEPA is a non-generative SSL method
- World Models Robotics — JEPA as a learned world model

## Further reading
- [Emerging Properties of Self-Predictive Architectures](https://arxiv.org/abs/2209.07399) — Garrido et al. 2022
