---
title: "Arc: Reinforcement Learning — Bandits to DeepSeek-R1"
arc: reinforcement-learning
super_domain: C-Decision
tracks: [06-reinforcement-learning, 07-attention-memory-reasoning, 15-ml-theory-foundations]
estimated_depth: "8-10 weeks, ~30 papers"
prereqs: [backpropagation, optimization, mdp]
---

# Arc: Reinforcement Learning
> **What this arc builds:** A full understanding of how agents learn to act from rewards — from the mathematics of Bellman equations to the post-training algorithms that align frontier language models.

## Why this arc exists

Reinforcement learning is the only framework in ML where an agent actively interacts with its environment to improve. That simple idea produces a remarkably general theory — covering game playing, robotics, and the post-training of language models with human feedback.

This arc traces the field from its mathematical foundations through the deep RL revolution to the current frontier: RL for verifiable rewards, GRPO, DAPO, and DeepSeek-R1's demonstration that reasoning ability can be incentivized through pure RL. The sequence is causal: each method exists because the previous one had a specific limitation.

## Prerequisites

Be comfortable with: backpropagation, gradient descent, the basics of probability (expectations, Markov chains). You don't need to know neural networks deeply — the arc builds up the RL-specific machinery from scratch.

## The sequence

**Foundations**

1. **Markov Decision Processes** (foundational) — states, actions, rewards, transitions, discount factor. The formalism that all RL sits inside. [→](../../curriculum/06-reinforcement-learning/mdp.md)
2. **Bellman equations** (foundational) — recursive definition of value; value iteration; policy iteration. The first exact solutions.
3. **Temporal Difference learning** (foundational) — learn value functions from experience without a model. TD(0), Q-learning, SARSA.
4. **Deep Q-Networks** (applied) — approximate Q with a neural network; replay buffer; target network. First deep RL success (Atari, 2015).

**Policy Gradient Methods**

5. **Policy gradient theorem** (theoretical) — the score function gradient; REINFORCE; variance reduction via baselines.
6. **Advantage estimation** (applied) — actor-critic; Generalized Advantage Estimation (GAE); A2C, A3C.
7. **TRPO** (theoretical) — trust region constraint on policy updates; monotonic improvement guarantee. [→](https://arxiv.org/abs/1502.05477)
8. **PPO** (applied) — clipped surrogate objective; simpler than TRPO; the workhorse of modern RLHF. [→](../../curriculum/06-reinforcement-learning/ppo.md)

**Alignment & LLM Post-Training**

9. **RLHF** (applied) — reward model from human preferences; KL-penalized PPO; InstructGPT. [→](../../curriculum/06-reinforcement-learning/rlhf.md)
10. **DPO** (frontier) — Direct Preference Optimization; closed-form implicit reward model; no PPO needed. [→](https://arxiv.org/abs/2305.18290)
11. **GRPO** (frontier) — group-relative advantage; eliminates the critic; DeepSeekMath breakthrough. [→](https://arxiv.org/abs/2402.03300)
12. **RLVR** (frontier) — RL with Verifiable Rewards; replace learned reward model with verification function. [→](https://arxiv.org/abs/2411.15124)
13. **DeepSeek-R1** (frontier) — pure RL produces reasoning ability without human reasoning labels. [→](https://arxiv.org/abs/2501.12948)
14. **DAPO** (frontier) — decoupled clipping + dynamic sampling; 50 pts on AIME 2024. [→](https://arxiv.org/abs/2503.14476)
15. **Dr. GRPO** (frontier) — removes length bias in GRPO; improves token efficiency. [→](https://arxiv.org/abs/2503.20783)

**Model-Based & Advanced**

16. **Model-based RL** (theoretical) — learn environment dynamics; Dyna; advantages of having a world model.
17. **MuZero** (frontier) — learned model + MCTS; planning in latent space. [→](https://www.nature.com/articles/s41586-020-03051-4)
18. **Offline RL** (applied) — learning from fixed datasets; conservative Q-learning; pessimism under distributional shift.

## Key figures

- **Richard Sutton** (Alberta) — co-author of RL: An Introduction; policy gradient theorem
- **John Schulman** (OpenAI) — TRPO, PPO, InstructGPT
- **Pieter Abbeel** (Berkeley) — inverse RL, imitation learning, RLHF precursors
- **DeepSeek AI** — GRPO, DeepSeekMath, DeepSeek-R1

## Essential reading sequence

1. [Reinforcement Learning: An Introduction](http://incompleteideas.net/book/the-book.html) — Sutton & Barto (Ch. 3–6) — MDP + TD foundations
2. [Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — Schulman et al. 2017 — the standard deep RL algorithm
3. [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) — Ouyang et al. 2022 — RLHF applied to LLMs
4. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023 — DPO
5. [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://arxiv.org/abs/2402.03300) — Shao et al. 2024 — GRPO
6. [DeepSeek-R1](https://arxiv.org/abs/2501.12948) — DeepSeek AI 2025 — pure RL for reasoning

## Current frontier anchors
> As of 2026-05-25

- **DeepSeek-R1** — demonstrates that pure RL (without supervised reasoning traces) produces strong reasoning ability in LLMs
- **DAPO** — 50 pts on AIME 2024 from Qwen2.5-32B; best public result on hard math reasoning
- **Dr. GRPO** — removes length bias artifact in GRPO; enables token-efficient training
- **V-JEPA 2 + RL** — RL control using a world model trained on video; zero-shot robot planning

## What you'll know when done

1. Implement Q-learning and PPO from scratch in Python
2. Explain the key distinction between on-policy and off-policy methods, and why it matters
3. Walk through the RLHF pipeline: reward model, KL penalty, PPO update step by step
4. Explain why GRPO eliminates the need for a value function and why this matters at scale
5. Articulate the open question: when does verifiable reward (RLVR) work vs. when do you need learned reward models?

## Branch points to other arcs

- **→ Language Models arc**: RLHF and GRPO are post-training steps in the LLM training pipeline
- **→ Robotics & Embodied AI**: RL fine-tuning of VLAs; model-based RL in robotic world models
- **→ ML Theory**: Decision-Estimation Coefficient as a theoretical unification of RL and online learning

## Where to go next

[Language Models arc →](../language-models/index.md) — Post-training in context of the full LLM pipeline

[Robotics & Embodied AI arc →](../scientific-ai/index.md) — RL for physical agents
