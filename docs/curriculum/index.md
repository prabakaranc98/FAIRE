---
title: Curriculum
description: The 10 canonical tracks per pracha.me/curriculum. Each subject has concepts, authors, arcs, builds.
state: active
updated: 2026-05-26
---

# Curriculum

The 10 canonical tracks from [pracha.me/curriculum](https://pracha.me/curriculum). Every subject has the same shape: `concepts/`, `authors/`, `arcs/`, `builds/`. See the [structure spec](../system/structure-v2.md) for the v2 design contract.

## The 10 tracks

| # | Track | Overview |
|---|---|---|
| 01 | AI | [Overview](core/01-ai/index.md) |
| 02 | Generative Modeling | [Overview](core/02-generative-modeling/index.md) |
| 03 | Representation Learning | [Overview](core/03-representation-learning/index.md) |
| 04 | Neural Networks & Deep Learning | [Overview](core/04-neural-networks-deep-learning/index.md) |
| 05 | Statistical & Probabilistic ML | [Overview](core/05-statistical-probabilistic-ml/index.md) |
| 06 | Reinforcement Learning | [Overview](core/06-reinforcement-learning/index.md) |
| 07 | Attention, Memory, Reasoning, Continual | [Overview](core/07-attention-memory-reasoning-continual/index.md) |
| 08 | Causal & Statistical Inference | [Overview](core/08-causal-statistical-inference/index.md) |
| 09 | Algorithms & Systems for AI | [Overview](core/09-algorithms-systems-for-ai/index.md) |
| 10 | Complexity, Cognition & Natural Intelligence | [Overview](core/10-complexity-cognition-natural-intelligence/index.md) |

Each row links to a subject overview where you'll find the concepts, authors, arcs, and builds for that track.

## The four artifact types

| Type | What it is | Where it lives |
|---|---|---|
| **Concept** | Self-contained Olah/Distill-grade explainer | `core/<track>/concepts/` |
| **Author** | Person-anchored reading guide (Pearl, Karpathy, Olah…) | `core/<track>/authors/` |
| **Arc** | Learning path through N concepts in order | `core/<track>/arcs/` |
| **Build** | MVB recipe — runnable, persona-tagged | `core/<track>/builds/` |

Pages converge: a concept lives inside an arc, links to its key authors, and ends with a build you can ship.
