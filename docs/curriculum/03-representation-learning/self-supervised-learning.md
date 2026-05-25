---
title: Self-Supervised Learning
track: 03-representation-learning
tags: [self-supervised, ssl, pretext-tasks, masked-modeling, joint-embedding]
depth: foundations
prereqs: []
updated: 2026-05-25
---

# Self-Supervised Learning
> **TL;DR:** Learning representations from unlabeled data by constructing supervised signals from the data itself — the engine behind modern foundation models, from BERT to MAE to GPT.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Self-supervised learning defines a pretext task from unlabeled data — predicting masked tokens, predicting the next frame, contrastive discrimination, rotation prediction — and uses this as the training signal. The learned representations generalize to downstream tasks without task-specific labels.

## Why it matters at the frontier
SSL is the dominant pretraining paradigm for language (GPT, BERT), vision (MAE, DINOv2), audio (wav2vec), and multimodal models (CLIP). Foundation models *are* SSL models scaled up.

## Core concepts
- **Pretext task** — a self-defined supervised objective (masked prediction, contrastive, next-frame)
- **Downstream transfer** — using SSL features for labeled classification/regression tasks
- **Joint-embedding** — two branches process different views; trained to agree in embedding space
- **Generative SSL** — reconstruct masked/corrupted inputs (BERT, MAE, GPT)
- **Contrastive SSL** — compare and contrast pairs (SimCLR, CLIP, MoCo)
- **Non-contrastive SSL** — avoid collapse without explicit negatives (BYOL, DINO)

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | 2018 | Devlin et al. | Masked language modeling — the SSL template for language |
| [Masked Autoencoders Are Scalable Vision Learners](https://arxiv.org/abs/2111.06377) | 2021 | He et al. | MAE — the SSL template for vision at scale |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Masked language modeling for transformers |
| [wav2vec 2.0](https://arxiv.org/abs/2006.11477) | 2020 | SSL for speech with contrastive objectives |

## Current SotA
> *Updated: 2026-05-25*
SSL at scale remains the core pretraining strategy. DINOv2 and MAE dominate vision. For language, GPT-style causal LM pretraining is dominant. Joint-embedding predictive architectures (I-JEPA, V-JEPA) are the current Lecun-group frontier for learning world representations without reconstruction.

## Connected topics
- [[contrastive-learning]] — one SSL objective class
- [[masked-autoencoders]] — generative SSL; MAE
- [[jepa]] — predictive SSL without pixel reconstruction
- [[lm-pretraining]] — SSL applied to language sequences

## Further reading
- [Self-Supervised Learning: Generative or Contrastive](https://arxiv.org/abs/2006.08218) — Liu et al. 2021; survey
