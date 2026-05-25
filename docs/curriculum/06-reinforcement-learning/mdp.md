---
title: Markov Decision Processes
track: 06-reinforcement-learning
tags: [mdp, bellman, value-function, policy, reward, discount]
depth: foundations
prereqs: []
updated: 2026-05-25
---

# Markov Decision Processes
> **TL;DR:** The mathematical framework for sequential decision-making — states, actions, rewards, and the Bellman equations that connect value at one state to value at the next.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **State s** — complete description of the environment at a time step
- **Action a** — agent's choice; selected by policy π(a|s)
- **Reward r** — scalar feedback signal from the environment
- **Transition p(s'|s,a)** — the environment's dynamics
- **Discount factor γ** — weights future rewards; γ ∈ [0, 1)
- **Value function V^π(s)** — expected discounted return under policy π
- **Q-function Q^π(s,a)** — expected return starting with action a in state s
- **Bellman equation** — V^π(s) = Σ_a π(a|s) Σ_{s'} p(s'|s,a)[r + γV^π(s')]

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Reinforcement Learning: An Introduction (Ch. 3–4) | 2018 | Sutton & Barto | The textbook; available at [incompleteideas.net](http://incompleteideas.net/book/the-book.html) |

## Connected topics
- [[td-learning]] — learning V and Q from experience
- [[ppo]] — policy gradient methods optimize in the MDP framework
- [[model-based-rl]] — learning the transition function p(s'|s,a)
