---
title: Contrastive Learning
track: 03-representation-learning
tags: [contrastive-learning, simclr, moco, self-supervised, nt-xent]
depth: foundations
prereqs: [self-supervised-learning]
updated: 2026-05-25
---

# Contrastive Learning
> **TL;DR:** A self-supervised objective that learns representations by pulling augmented views of the same image together and pushing different images apart — the dominant SSL paradigm that gave rise to CLIP and modern vision encoders.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Contrastive learning creates positive pairs (two views of the same sample via augmentation) and negative pairs (views from different samples). The network learns to map positives close together and negatives far apart in embedding space, developing rich representations without labels.

## Why it matters at the frontier
SimCLR and MoCo established contrastive learning as a viable path to representations competitive with supervised pretraining. CLIP extended this to vision-language pairs, producing the most widely used vision encoder in frontier AI systems.

## Core concepts
- **Positive pair** — two augmented views of the same sample
- **Negative pair** — views from different samples in the batch
- **NT-Xent loss** — normalized temperature-scaled cross entropy; the SimCLR loss
- **InfoNCE** — the loss function seen as estimating mutual information
- **Momentum encoder** — EMA-updated key encoder (MoCo) to stabilize training with a memory bank
- **Temperature** — controls the sharpness of the distribution; critical hyperparameter

## Mathematical foundations
NT-Xent loss for a positive pair (i, j):
$$\mathcal{L}_{i,j} = -\log \frac{\exp(\text{sim}(z_i, z_j)/\tau)}{\sum_{k \neq i} \exp(\text{sim}(z_i, z_k)/\tau)}$$

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709) | 2020 | Chen et al. | SimCLR — the clean, reproducible baseline |
| [Momentum Contrast for Unsupervised Visual Representation Learning](https://arxiv.org/abs/1911.05722) | 2019 | He et al. | MoCo — memory-efficient contrastive training |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [Representation Learning with Contrastive Predictive Coding](https://arxiv.org/abs/1807.03748) | 2018 | Oord et al. | InfoNCE loss; CPCv1 — original contrastive SSL |
| [A Simple Framework for Contrastive Learning](https://arxiv.org/abs/2002.05709) | 2020 | Chen et al. | SimCLR — defined the modern paradigm |

## Current SotA
> *Updated: 2026-05-25*
Pure contrastive learning has been largely superseded by non-contrastive methods (BYOL, DINOv2) and masked modeling (MAE) for image features. However, contrastive vision-language pretraining (CLIP) remains the standard for zero-shot transfer and multimodal alignment. SigLIP (sigmoid loss) and DFN-CLIP are recent improvements.

## Connected topics
- [[self-supervised-learning]] — broader family; contrastive is one objective class
- [[vision-language-models]] — CLIP as the scaling of contrastive to cross-modal pairs
- [[bootstrapping-methods]] — non-contrastive alternatives (BYOL, DINOv2)

## Further reading
- [Intriguing Properties of Contrastive Losses](https://arxiv.org/abs/2011.02803) — Chen & Luo 2021
