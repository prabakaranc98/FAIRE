---
title: Robotics & Embodied AI
tags: [robotics, embodied-ai, manipulation, locomotion, world-models, sim-to-real]
---

# Track 11 · Robotics & Embodied AI

> Intelligence grounded in the physical world: manipulation, locomotion, perception-action loops, world models, and the challenge of embodied learning.

Embodied AI is the study of agents that perceive and act in physical environments. It connects deep learning, reinforcement learning, and control theory with the physical constraints of the real world. The frontier question: can we build systems that learn to act as flexibly as animals do?

---

## Topics

### Foundations
- [Robot Kinematics & Dynamics](kinematics.md) — rigid body, forward/inverse kinematics, Jacobians
- [Control Theory](control-theory.md) — PID, LQR, model predictive control, stability
- [SLAM](slam.md) — simultaneous localization and mapping, particle filters, visual SLAM

### Perception & Action
- [Visual Perception for Robots](visual-perception.md) — depth estimation, pose estimation, 6-DoF grasp
- [Tactile Sensing](tactile-sensing.md) — touch-based manipulation, contact-rich tasks
- [Imitation Learning](imitation-learning.md) — behavioral cloning, DAgger, diffusion policy

### Learning to Move
- [Locomotion](locomotion.md) — legged robots, sim-to-real transfer, terrain adaptation
- [Dexterous Manipulation](manipulation.md) — in-hand manipulation, multi-finger grasping
- [RL for Robotics](rl-robotics.md) — reward shaping, curriculum learning, domain randomization

### World Models & Planning
- [World Models for Robots](world-models-robotics.md) — predictive models, latent planning, RSSM
- [Foundation Models for Robotics](foundation-models-robotics.md) — RT-2, Octo, Pi0, generalist robot policies
- [Sim-to-Real Transfer](sim-to-real.md) — domain randomization, physics simulation, transfer gap

---

## Connections to frontier research

- **Generalist robot policies** — single models that control diverse robots across diverse tasks
- **Diffusion for robot actions** — diffusion policy as a flexible action generation framework
- **World models as planning substrate** — learning environment dynamics for model-based control
- **Embodied language grounding** — following natural language instructions in physical environments

---

## Recommended entry points

Start with [Control Theory](control-theory.md) and [Imitation Learning](imitation-learning.md). For frontier relevance, jump directly to [Foundation Models for Robotics](foundation-models-robotics.md).
