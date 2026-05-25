---
title: Representation Learning
tags: [representation-learning, self-supervised, contrastive, embeddings, ssl]
---

# Track 03 · Representation Learning

> Learning structure from data without labels: self-supervised objectives, contrastive methods, and the geometry of learned representations.

What makes a good representation? This is one of the most fundamental questions in deep learning. Representation learning asks how to extract structure from raw data — without relying on expensive human labels — and how to build representations that transfer, generalize, and compose.

---

## Topics

### Contrastive Methods
- [Contrastive Learning](contrastive-learning.md) — SimCLR, MoCo, NT-Xent loss, positive/negative pairs
- [Self-Supervised Learning](self-supervised-learning.md) — pretext tasks, masked modeling, joint-embedding architectures

### Non-Contrastive Methods
- [Bootstrapping Methods](bootstrapping-methods.md) — BYOL, SimSiam, momentum encoders
- [Redundancy Reduction](redundancy-reduction.md) — Barlow Twins, VICReg, feature decorrelation

### Joint-Embedding & Predictive Architectures
- [JEPA](jepa.md) — Joint Embedding Predictive Architectures, I-JEPA, V-JEPA
- [Masked Autoencoders](masked-autoencoders.md) — MAE, BEiT, masked image modeling

### Representation Analysis
- [Probing & Interpretability](probing.md) — linear probing, representational similarity analysis (RSA), CKA
- [Geometry of Representations](geometry.md) — manifold hypothesis, intrinsic dimensionality, alignment vs. uniformity

---

## Connections to frontier research

- **Foundation models** — large-scale SSL pretraining as the standard paradigm for vision and language
- **Multimodal alignment** — CLIP and successors; aligning representations across modalities
- **World models** — learning predictive latent representations for embodied agents
- **Causal representation learning** — disentanglement, identifiability, and causal structure in representations

---

## Recommended entry points

Start with [Contrastive Learning](contrastive-learning.md) and [Self-Supervised Learning](self-supervised-learning.md). Then move to [JEPA](jepa.md) for the current frontier in predictive architectures.
