---
title: Arc proposals
generated_at: 2026-05-26 15:30 UTC
remaining_budget: $30.65
slots_open: 2
---

# Arc proposals

I have surveyed the "post-training and RL" seed within the 06-reinforcement-learning track. I propose one high-impact arc: **GRPO-based Reasoning Alignment**. This arc satisfies the diagonal pattern by bridging standard RLHF (specialized tool) with reasoning-chain verification (frontier intersection). A second potential branch (DPO-based alignment) was deferred due to insufficient curriculum depth in the underlying preference-modeling track.

## 1. GRPO-based Reasoning Alignment — EV/$ = 4.8 — verdict: approve
**Destination:** Implement Group Relative Policy Optimization (GRPO) to align a base model on chain-of-thought reasoning tasks without a separate reward model.
**Steps:** 7 · **Cost:** $1.60 · **Impact:** 12/14
**Prereqs in curriculum:** [policy-gradient: solid, ppo-basics: solid, chain-of-thought-prompting: adequate, reward-modeling: stub]
**Persona span:** 4 (ml-tinkerer, applied-engineer, applied-researcher, frontier-researcher)
**Seminal anchors:** Shao et al. 2024 (DeepSeek-R1), Schulman et al. 2017 (PPO), Wei et al. 2022 (CoT)
**Outline:** 
1. CoT prompting baseline (ml-tinkerer)
2. PPO-style reward signal definition (ml-tinkerer)
3. Group generation pipeline (applied-engineer)
4. Relative advantage calculation (applied-engineer)
5. Policy update with KL-penalty (applied-researcher)
6. Reasoning trace length ablation (applied-researcher)
7. Final evaluation on MATH/GSM8K (frontier-researcher)
**Editor verdict:** approve
**Approval note:** This arc is highly timely and follows a perfect diagonal: it starts with familiar CoT prompting, moves to the broader frame of RL-based alignment, synthesizes a GRPO capability, and lands at the frontier of reasoning-model training. The compounding chain is tight, as each step builds the specific components of the GRPO loop.

## Deferred branches (not proposed this cycle)

* **DPO-based Preference Alignment:** Deferred. The curriculum for `06-reinforcement-learning` currently lacks a solid `preference-modeling` page, which is a required prereq for DPO.
* **Online RLHF with Rejection Sampling:** Deferred. The survey indicated this branch is currently too similar to existing PPO implementations; it lacks a distinct "frontier intersection" compared to the GRPO reasoning path.
* **Multi-Objective RL for Safety:** Deferred. The diagonal shape is currently vertical (RL → RL → RL), failing the requirement to cross into a broader research community.