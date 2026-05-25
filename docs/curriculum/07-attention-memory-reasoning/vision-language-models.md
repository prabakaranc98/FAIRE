---
title: Vision-Language Models
track: 07-attention-memory-reasoning
tags: [vision-language, clip, multimodal, zero-shot, contrastive-pretraining]
depth: research
prereqs: [transformer, contrastive-learning]
updated: 2026-05-25
---

# Vision-Language Models
> **TL;DR:** Models that jointly encode images and text, enabling zero-shot classification, image captioning, and multimodal reasoning — CLIP established the paradigm; GPT-4V and Gemini Ultra are the frontier.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Vision-language models learn aligned representations of images and text. CLIP does this via contrastive pretraining: maximize similarity of correct image-text pairs and minimize similarity of incorrect pairs. Modern VLMs go further — combining a vision encoder with a language model to enable generation, reasoning, and instruction following over visual content.

## Why it matters at the frontier
CLIP is the most widely deployed vision encoder in AI systems. VLMs (LLaVA, GPT-4V, Gemini, Claude) are central to modern AI assistants, robotics (VLAs), and multimodal generation. Visual question answering, chart understanding, and document AI are all downstream applications.

## Core concepts
- **Contrastive image-text pretraining** — align image and text embeddings via InfoNCE loss over pairs
- **Zero-shot transfer** — classify new categories by comparing image embeddings to text embeddings of class names
- **Visual instruction tuning** — connect a frozen vision encoder to an LLM; train on image+instruction data
- **SigLIP** — sigmoid loss variant; better calibration than contrastive softmax
- **Multi-modal projector** — maps vision encoder outputs into LLM token space (linear, MLP, or cross-attention)

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Learning Transferable Visual Models From Natural Language Supervision](https://arxiv.org/abs/2103.00020) | 2021 | Radford et al. (OpenAI) | CLIP — the foundational vision-language model |
| [Visual Instruction Tuning (LLaVA)](https://arxiv.org/abs/2304.08485) | 2023 | Liu et al. | LLaVA — minimal recipe for VLM fine-tuning |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [CLIP](https://arxiv.org/abs/2103.00020) | 2021 | Contrastive vision-language pretraining at scale |
| [Flamingo](https://arxiv.org/abs/2204.14198) | 2022 | Alayrac et al. (DeepMind) | Few-shot VLMs via cross-attention |

## Current SotA
> *Updated: 2026-05-25*
GPT-4o, Gemini Ultra, and Claude 3 Opus represent the frontier for multimodal understanding. For open models, InternVL 2.5 and LLaVA-OneVision achieve near-proprietary performance. SigLIP (Google, 2023) is the current best open vision encoder. DFN-CLIP scales contrastive pretraining with data filtering.

## Connected topics
- [[contrastive-learning]] — CLIP's training objective
- [[transformer]] — vision encoder + LLM both transformer-based
- [[multimodal-reasoning]] — downstream reasoning capabilities
- [[foundation-models-robotics]] — VLMs as the perception backbone in VLAs

## Further reading
- [An Introduction to Vision-Language Modeling](https://arxiv.org/abs/2405.17247) — Bordes et al. 2024
