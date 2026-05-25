---
title: Scaling Laws
track: 04-neural-networks-dl
tags: [scaling-laws, chinchilla, compute-optimal, emergence, llms]
depth: research
prereqs: [backpropagation, optimization]
updated: 2026-05-25
---

# Scaling Laws
> **TL;DR:** Empirical power-law relationships between model performance, model size, dataset size, and compute — the scientific foundation for modern large model training decisions.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Scaling laws describe how model performance (measured as loss) scales predictably with model parameters N, dataset tokens D, and compute budget C = 6ND. Kaplan et al. (2020) established the original laws; Hoffmann et al. (Chinchilla, 2022) revised them to show that compute-optimal training requires much more data than previously used.

## Why it matters at the frontier
Scaling laws are how frontier labs decide how large to make their models and how much data to train on. They also predict emergent capabilities — abilities that appear suddenly at certain scales. Understanding them is essential for interpreting every major model release.

## Core concepts
- **Power law** — L ∝ N^{-α}: loss decreases as a power of model size
- **Compute-optimal** — for a given compute budget C, the Chinchilla scaling law says N ∝ √C, D ∝ √C
- **Chinchilla ratio** — ~20 tokens per parameter for optimal single-epoch training
- **Emergent capabilities** — abilities that appear discontinuously at certain scales (controversial)
- **IsoFLOP curve** — all model/data combinations using the same total compute

## Mathematical foundations
Kaplan scaling law (simplified):
$$L(N, D) \approx \left(\frac{N_c}{N}\right)^{\alpha_N} + \left(\frac{D_c}{D}\right)^{\alpha_D}$$

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Kaplan et al. (OpenAI) | Original scaling laws; model-size dominated regime |
| [Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) | 2022 | Hoffmann et al. (DeepMind) | Chinchilla; showed data matters as much as model size |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Power-law scaling; predict-before-train paradigm |
| [Chinchilla](https://arxiv.org/abs/2203.15556) | 2022 | Compute-optimal training; revised the field's training recipes |

## Current SotA
> *Updated: 2026-05-25*
Scaling law research has extended to multimodal models, RL post-training, and inference-time compute (test-time scaling). The "inference scaling" regime — where more compute at test time buys better answers — is the current frontier, with implications for model sizing different from training-time scaling.

## Connected topics
- [[lm-pretraining]] — scaling laws are derived from LM training runs
- [[emergent-capabilities]] — the discontinuous phenomena at scale
- [[optimization]] — learning rate schedules are calibrated against scaling predictions

## Further reading
- [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — Schaeffer et al. 2023
