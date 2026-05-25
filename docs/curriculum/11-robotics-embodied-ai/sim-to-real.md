---
title: Sim-to-Real Transfer
track: 11-robotics-embodied-ai
tags: [sim-to-real, domain-randomization, transfer, simulation, robotics]
depth: applied
prereqs: []
updated: 2026-05-25
---

# Sim-to-Real Transfer
> **TL;DR:** Training robot policies in simulation and deploying them in the real world — avoiding the cost and danger of real-world data collection, but facing the "reality gap" between simulated and real physics.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **Reality gap** — physical discrepancies between simulation and reality (friction, stiffness, perception)
- **Domain randomization** — vary simulation parameters randomly; force policies to be robust
- **Domain adaptation** — learn to transfer representations from sim to real
- **System identification** — calibrate simulator parameters to match real-world dynamics
- **Privileged information** — train in sim with access to ground-truth state; distill to real sensor inputs
- **Physics simulators** — MuJoCo, Isaac Gym/Sim, PyBullet; fidelity vs. speed tradeoffs

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) | 2017 | Tobin et al. (OpenAI) | The canonical sim-to-real technique |
| [Learning Dexterous In-Hand Manipulation (OpenAI Five)](https://arxiv.org/abs/1808.00177) | 2019 | OpenAI | Rubik's cube with domain randomization at scale |

## Connected topics
- [[rl-robotics]] — RL in simulation before real deployment
- [[imitation-learning]] — real demonstrations avoid sim-to-real gap entirely
- [[foundation-models-robotics]] — internet pretraining reduces reliance on sim

## Further reading
- [Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.05268) — Zhao et al. 2020
