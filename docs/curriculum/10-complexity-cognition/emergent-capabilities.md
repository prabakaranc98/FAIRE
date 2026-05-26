---
title: Emergent Capabilities in Large Models
track: 10-complexity-cognition
tags: [emergence, scaling, phase-transitions, grokking, in-context-learning]
depth: research
prereqs: [scaling-laws]
updated: 2026-05-25
---

# Emergent Capabilities in Large Models
> **TL;DR:** Abilities that appear discontinuously at certain model scales or training durations — not present in smaller models, then suddenly present in larger ones — a phenomenon that challenges smooth extrapolation from small-scale experiments.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Emergent capabilities are behaviors that appear to arise suddenly as a function of scale (parameters, data, or compute) rather than smoothly. Examples include in-context few-shot learning (GPT-3), chain-of-thought reasoning, multi-step arithmetic, and code generation. The phenomenon raises deep questions: do emergent abilities reflect genuine phase transitions, or artifacts of evaluation metrics?

## Why it matters at the frontier
Emergence complicates predictions about AI capability development. If most dangerous or valuable capabilities emerge discontinuously, they may not be foreseeable from smaller-scale experiments. This is central to AI safety, evaluation, and the science of scaling.

## Core concepts
- **Emergence** — a property that appears in larger systems not predictable from smaller versions
- **Sharp transition** — apparent discontinuity in capability vs. scale
- **Smooth vs. discontinuous** — debate over whether emergence is real or a metric artifact
- **Grokking** — delayed generalization; model appears to memorize, then suddenly generalizes
- **In-context learning** — few-shot learning without gradient updates; appears at scale
- **Phase transitions** — physical analogy; system undergoes qualitative change at a threshold

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Emergent Abilities of Large Language Models](https://arxiv.org/abs/2206.07682) | 2022 | Wei et al. | Catalogues emergent abilities; sparked the debate |
| [Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) | 2023 | Schaeffer et al. | Argues emergence is a metric artifact |

## Current SotA
> *Updated: 2026-05-25*
The debate continues. Schaeffer et al. showed that many emergent abilities disappear under continuous metrics. However, some abilities (like multi-step reasoning with chain-of-thought) appear genuinely discontinuous. Grokking is well-documented in controlled settings. The field is moving toward more rigorous evaluation frameworks.

## Connected topics
- [Scaling Laws](../04-neural-networks-dl/scaling-laws.md) — emergence appears against the backdrop of smooth scaling
- [Computational Complexity for AI](./complexity-classes.md) — formal treatment of phase transitions in computation
- [Cognitive Architectures](./cognitive-architectures.md) — theories of intelligence that might explain discontinuous leaps

## Further reading
- [Grokking: Generalization Beyond Overfitting on Small Algorithmic Datasets](https://arxiv.org/abs/2201.02177) — Power et al. 2022
