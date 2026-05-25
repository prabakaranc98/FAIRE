---
title: Foundation Models for Robotics
track: 11-robotics-embodied-ai
tags: [vla, foundation-models, openvla, rt2, pi0, generalist-policy]
depth: research
prereqs: [imitation-learning, vision-language-models]
updated: 2026-05-25
---

# Foundation Models for Robotics
> **TL;DR:** Large pretrained models (language models, vision-language models) adapted for robotic control — Vision-Language-Action models (VLAs) that understand natural language instructions and generate robot actions.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
VLAs combine a vision encoder (e.g., DINOv2, SigLIP) with a large language model backbone to process visual observations and language instructions, then output robot actions. The paradigm: pretrain on internet-scale vision-language data, then fine-tune on robot demonstration datasets. RT-2 (Google, 2023) proved this works at scale; OpenVLA (2024) open-sourced it; π0 (Physical Intelligence, 2024) added a flow-matching action head for dexterity.

## Why it matters at the frontier
VLAs are the current frontier paradigm for generalist robots. They leverage the world knowledge and language grounding in LLMs while learning robot-specific behavior from demonstrations. The holy grail: one model that can be deployed across diverse robots and tasks with minimal fine-tuning.

## Core concepts
- **VLA** — Vision-Language-Action model; extends VLM to output robot actions
- **Action tokenization** — discretize continuous actions into tokens; predict with LM head
- **Continuous action head** — output continuous actions via regression or flow matching (π0)
- **Co-training** — combine robot data with internet vision-language data during training
- **RT-X** — Open X-Embodiment; diverse robot data from multiple labs for co-training

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control](https://arxiv.org/abs/2307.15818) | 2023 | Brohan et al. (Google) | RT-2 — established the VLA paradigm |
| [OpenVLA: An Open-Source Vision-Language-Action Model](https://arxiv.org/abs/2406.09246) | 2024 | Kim et al. | 7B open VLA; outperforms RT-2-X (55B) on 29 tasks |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [RT-2](https://arxiv.org/abs/2307.15818) | 2023 | VLA paradigm; web pretraining → robot control |
| [π0: A Vision-Language-Action Flow Model for General Robot Control](https://arxiv.org/abs/2410.24164) | 2024 | Hejna et al. (Physical Intelligence) | Flow matching action head; >200 Hz control |

## Current SotA
> *Updated: 2026-05-25*
π0 and OpenVLA-OFT represent the current open frontier. NVIDIA GR00T N1 (2025) is the first humanoid-specific foundation model. V-JEPA 2 (Meta, 2025) enables zero-shot robot planning via world model predictions without action labels. RL fine-tuning of VLAs (GRPO-style) is emerging as a post-training step.

## Connected topics
- [[vision-language-models]] — the VLM backbone of VLAs
- [[imitation-learning]] — VLAs are trained on large imitation learning datasets
- [[world-models-robotics]] — planning using learned dynamics models

## Further reading
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models](https://arxiv.org/abs/2310.08864) — Open X-Embodiment Collaboration 2023
