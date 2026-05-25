---
title: Reinforcement Learning
tags: [reinforcement-learning, policy-gradient, model-based-rl, rlhf, multi-agent]
---

# Track 06 · Reinforcement Learning

> Learning by interaction: value functions, policy optimization, model-based RL, and alignment via reinforcement learning from human feedback.

Reinforcement learning is the framework for learning through interaction with an environment. It has produced some of the most striking results in AI — from Go to protein folding to language model alignment — and remains an active frontier.

---

## Topics

### Foundations
- [Markov Decision Processes](mdp.md) — states, actions, rewards, Bellman equations, discounting
- [Dynamic Programming](dynamic-programming.md) — value iteration, policy iteration, exact solutions
- [Temporal Difference Learning](td-learning.md) — TD(0), TD(λ), Q-learning, SARSA

### Value-Based Methods
- [Deep Q-Networks](dqn.md) — DQN, Double DQN, Dueling DQN, prioritized replay
- [Distributional RL](distributional-rl.md) — C51, QR-DQN, IQN, distributional Bellman

### Policy Gradient Methods
- [Policy Gradient](policy-gradient.md) — REINFORCE, baseline, variance reduction
- [Actor-Critic Methods](actor-critic.md) — A2C, A3C, advantage estimation
- [Proximal Policy Optimization](ppo.md) — clipped objective, trust regions, TRPO

### Model-Based RL
- [Model-Based RL](model-based-rl.md) — world models, Dyna, planning with learned models
- [MuZero & AlphaZero](muzero.md) — learned model + MCTS, latent planning

### Modern RL
- [RLHF](rlhf.md) — reward modeling, PPO for alignment, Constitutional AI, DPO
- [Offline RL](offline-rl.md) — batch RL, conservative Q-learning, decision transformers
- [Multi-Agent RL](multi-agent-rl.md) — cooperative, competitive, MARL theory

---

## Connections to frontier research

- **LLM alignment** — RLHF, RLAIF, DPO as the dominant post-training paradigm
- **Embodied agents** — RL in robotics, sim-to-real, reward from vision
- **Game-theoretic AI** — multi-agent equilibria, mechanism design, AI safety via game theory

---

## Recommended entry points

Start with [Markov Decision Processes](mdp.md) and [Temporal Difference Learning](td-learning.md). For alignment relevance, jump to [RLHF](rlhf.md) after covering [Proximal Policy Optimization](ppo.md).
