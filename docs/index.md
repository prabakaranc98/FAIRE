---
title: Frontier Wiki
template: home.html
---

## What is an Arc of Work?

An arc of work is a focused, sequenced journey through a domain — not a reading list, but a path with a destination. Each arc takes you from entry intuition to frontier capability, with a clear build at each stage. You finish an arc knowing *what you can do*, not just *what you've read*.

The wiki is organized around arcs — and every curriculum page feeds into one.

---

## Two ways in

### [Learning Arcs](arcs/index.md) — follow a path to capability

Focused sequences, 20–28 topics each. Every arc starts with the foundational question and ends at the current frontier — with a build at each stage.

<div class="grid cards" markdown>

- **[MLP → Transformer](arcs/mlp-to-transformer/index.md)**

    Architecture lineage: from perceptrons to attention. Understand *why* each step happened.

- **[Generative Stack](arcs/generative-stack/index.md)**

    VAEs → GANs → Diffusion → Flow Matching. Every generation paradigm, in the order they emerged.

- **[Language Models](arcs/language-models/index.md)**

    Pretraining → RLHF → alignment → frontier LLMs. The full arc from n-grams to GPT-4.

- **[Reinforcement Learning](arcs/reinforcement-learning/index.md)**

    Bandits → PPO → RLHF → GRPO → DeepSeek-R1. Decision-making from tabular to frontier.

- **[Causal AI](arcs/causal-ai/index.md)**

    SCMs → do-calculus → causal representation learning. From correlation to intervention.

- **[Systems for Scale](arcs/systems-for-scale/index.md)**

    FlashAttention → ZeRO → vLLM → speculative decoding. The engineering behind frontier models.

- **[Scientific AI](arcs/scientific-ai/index.md)**

    PINNs → FNO → AlphaFold 3 → V-JEPA. AI as a scientific instrument.

</div>

---

### [Curriculum](curriculum/index.md) — the full landscape by domain

15 tracks covering every major area of AI/ML. Enter at any topic, any depth. Use the curriculum when you want to understand how a domain is organized or look up a specific concept.

<div class="grid cards" markdown>

- **A · Foundations**

    [Algorithms & Systems](curriculum/09-algorithms-systems-ai/index.md) · [Complexity & Cognition](curriculum/10-complexity-cognition/index.md) · [ML Theory](curriculum/15-ml-theory-foundations/index.md)

- **B · Modeling**

    [Generative Modeling](curriculum/02-generative-modeling/index.md) · [Neural Networks & DL](curriculum/04-neural-networks-dl/index.md) · [Representation Learning](curriculum/03-representation-learning/index.md) · [Statistical ML](curriculum/05-statistical-probabilistic-ml/index.md) · [Causal AI](curriculum/08-causal-statistical-inference/index.md)

- **C · Decision**

    [Reinforcement Learning](curriculum/06-reinforcement-learning/index.md)

- **D · Perception & Action**

    [Language Models & Transformers](curriculum/07-attention-memory-reasoning/index.md) · [Robotics & Embodied AI](curriculum/11-robotics-embodied-ai/index.md) · [Graph & Relational AI](curriculum/13-graph-relational-ai/index.md)

- **E · AI for Science**

    [Physics & Scientific AI](curriculum/12-physics-scientific-ai/index.md) · [Biology & Life Sciences](curriculum/14-biology-life-sciences/index.md) · [AI Foundations](curriculum/01-ai/index.md)

</div>

---

## What every page gives you

Every topic page is built for **four reader types** — and always answers the same question: *what can I do with this?*

| Reader | Gets | Outcome |
|---|---|---|
| **Applied practitioner** | Key algorithms + MVB recipe | Something built today |
| **Curious generalist** | Clear intuition, no jargon | Genuine understanding |
| **Theory student** | Annotated LaTeX, derivations | Rigorous mental model |
| **Frontier researcher** | Named papers, open problems | Where to push next |

Pivotal pages include a **Minimum Valuable Build** — a concrete, runnable recipe with real HuggingFace model and dataset IDs, designed for a consumer GPU or free Colab tier.

---

## Source discipline

Every link traces back to a primary source:

- `arxiv.org` — papers and preprints
- `*.edu` — university lecture notes, course pages
- `huggingface.co` — model cards, datasets
- Official library docs (PyTorch, JAX, Diffusers...)
- *"In production" sections only:* official engineering blogs from frontier labs

No Medium. No Towards Data Science. No Wikipedia as a citation.
