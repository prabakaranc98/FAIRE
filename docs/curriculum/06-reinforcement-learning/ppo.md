---
title: Proximal Policy Optimization (PPO)
track: 06-reinforcement-learning
tags: [ppo, policy-gradient, trpo, clipping, rlhf, actor-critic]
depth: foundations
prereqs: [policy-gradient, mdp]
updated: 2026-05-25
---

# Proximal Policy Optimization (PPO)
> **TL;DR:** A policy gradient algorithm that constrains update steps to stay "proximal" to the current policy via a clipped objective — the workhorse of deep RL and RLHF.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
PPO is a policy gradient algorithm that improves stability by clipping the probability ratio between new and old policies. Rather than enforcing an explicit KL constraint (like TRPO), it uses a simple clipped surrogate objective that prevents large policy updates. This makes it easy to implement, robust, and parallelizable.

## Why it matters at the frontier
PPO is the training algorithm behind most RLHF systems (ChatGPT, Claude, Gemini). It is also the baseline for GRPO and other modern post-training algorithms. Understanding PPO is prerequisite to understanding the entire post-training alignment literature.

## Core concepts
- **Probability ratio** — r_t(θ) = π_θ(a|s) / π_θ_old(a|s)
- **Clipped objective** — L_CLIP = E[min(r_t × Â_t, clip(r_t, 1-ε, 1+ε) × Â_t)]
- **Advantage estimate** — Â_t measures how much better action a is than average
- **GAE** — Generalized Advantage Estimation; variance-reduced advantage
- **KL penalty variant** — alternative to clipping; adds adaptive KL term to objective
- **Value function loss** — critic trained jointly to estimate V(s)

## Mathematical foundations
PPO-Clip objective:
\[
L^{CLIP}(\theta) = \hat{\mathbb{E}}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]
\]

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) | 2017 | Schulman et al. (OpenAI) | The PPO paper — short, clear, essential |
| [Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) | 2015 | Schulman et al. | TRPO — the principled predecessor PPO simplified |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [PPO](https://arxiv.org/abs/1707.06347) | 2017 | Clipped surrogate; practical default for deep RL |
| [High-Dimensional Continuous Control Using GAE](https://arxiv.org/abs/1506.02438) | 2015 | Schulman et al. | GAE advantage estimation used in most PPO impls |

## Current SotA
> *Updated: 2026-05-25*
PPO remains the backbone of RLHF (InstructGPT, TRL library). GRPO (DeepSeekMath, 2024) replaces the critic with group-relative advantage estimation, reducing memory cost. DAPO (ByteDance, 2025) further improves stability for reasoning tasks. DPO (NeurIPS 2023) sidesteps PPO entirely for preference learning.

## Connected topics
- [Reinforcement Learning from Human Feedback (RLHF)](./rlhf.md) — PPO is how RLHF is implemented in practice
- Policy Gradient — PPO is a constrained policy gradient method
- Dpo — the alternative to PPO for preference-based alignment

## Further reading
- [Towards Theoretical Understanding of RLHF](https://arxiv.org/abs/2401.01672) — 2024 survey
