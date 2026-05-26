---
title: Imitation Learning
track: 11-robotics-embodied-ai
tags: [imitation-learning, behavioral-cloning, dagger, diffusion-policy, action-chunking]
depth: foundations
prereqs: [mdp]
updated: 2026-05-25
---

# Imitation Learning
> **TL;DR:** Learning to act by mimicking expert demonstrations rather than learning from scalar rewards — simpler than RL, more data-efficient, and the current dominant approach for robot learning.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Imitation learning trains a policy to replicate expert behavior from demonstration data. Behavioral cloning (BC) is the simplest form: treat it as supervised learning on (state, action) pairs. The challenge is covariate shift — small errors compound over time since BC never sees the states caused by its own mistakes. DAgger solves this with interactive data collection. Diffusion Policy represents the current state of the art.

## Core concepts
- **Behavioral cloning** — supervised learning on (state, action) pairs; off-policy, simple, brittle
- **Covariate shift** — distribution mismatch between training states and states visited by the learned policy
- **DAgger** — Dataset Aggregation; interactively collect data from learner's visited states
- **Diffusion Policy** — model action distribution as a diffusion process; multi-modal, high capacity
- **Action chunking** — predict k future actions at once; reduces compounding errors (ACT)
- **Inverse RL** — infer reward function from demonstrations rather than directly cloning actions

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning (DAgger)](https://arxiv.org/abs/1011.0686) | 2011 | Ross, Gordon, Bagnell | DAgger — the principled fix for behavioral cloning |
| [Diffusion Policy: Visuomotor Policy Learning via Action Diffusion](https://arxiv.org/abs/2303.04137) | 2023 | Chi et al. | Diffusion Policy — current SotA for robot manipulation |

## Current SotA
> *Updated: 2026-05-25*
Diffusion Policy and ACT (Action Chunking with Transformers, Zhao et al. 2023) are the dominant approaches for dexterous manipulation. π0 (Physical Intelligence, 2024) uses flow matching for the action head at 200+ Hz on diverse robot platforms. Large-scale data collection (DROID, Open X-Embodiment) is shifting the field toward data-driven generalist policies.

## Connected topics
- [Foundation Models for Robotics](./foundation-models-robotics.md) — scale imitation learning to generalist robot policies
- Rl Robotics — RL fine-tuning on top of imitation learning initialization
- [Diffusion Models](../02-generative-modeling/diffusion-models.md) — diffusion policy applies generative models to action sequences

## Further reading
- [An Algorithmic Perspective on Imitation Learning](https://arxiv.org/abs/1811.06711) — Osa et al. 2018
