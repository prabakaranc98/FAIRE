```yaml
---
title: Scaling Laws
track: 04-neural-networks-dl
tags: [scaling, laws, deep learning, model size, compute]
depth: applied
prereqs: [deep-learning, neural-networks]
updated: 2024-11-05
has_mvb: false
---

# Scaling Laws
> **TL;DR:** Scaling laws are empirical relationships that predict how model performance improves with increased computational resources, model size, and data volume, guiding efficient AI development.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [Code & implementations](#code--implementations) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is

Imagine you're building a chatbot, and you want it to answer complex questions. You could make the model bigger, but that costs a lot of money. Or, you could try to make the model smarter by using it in new ways. Scaling laws help us understand how to make AI systems better by changing how we use them, not just how big they are.

Scaling laws are empirical relationships that describe how model performance changes as we vary factors like the amount of training data, the size of the model (number of parameters), and the amount of computation used for training. These laws aren't just about making models bigger; they also encompass data strategies, compute budgets, and even inference techniques. Understanding these relationships allows us to predict the performance gains from increasing these resources, guiding efficient AI development.

These laws provide a framework for understanding the trade-offs between different resources. For instance, should you double the size of your model, or double the amount of training data? Scaling laws can help answer this question by predicting which approach will yield the greatest performance improvement for a given investment. They are crucial for optimizing the allocation of resources in AI projects.

## Why it matters at the frontier

Scaling laws are critical at the frontier of AI because they help researchers and practitioners make informed decisions about resource allocation when training large models. Training these models can cost millions of dollars, and scaling laws provide a way to predict whether the performance gains will justify the investment. This is particularly important as models continue to grow in size and complexity.

Furthermore, scaling laws are evolving to incorporate new factors, such as memory access costs and the impact of architectural choices. This is essential for optimizing models for real-world deployment, where efficiency and cost are paramount. The open problem is: How can we develop a unified scaling law that accurately predicts the performance of agent systems across diverse tasks and architectures, considering the trade-offs between tool coordination, model capability, and computational resources?

## Core concepts
- **Model Size** — The number of parameters in a neural network, often correlated with its capacity to learn complex patterns.
- **Data Volume** — The amount of training data used to train a model, influencing its ability to generalize to unseen examples.
- **Compute** — The total computational resources (e.g., FLOPs) used during training, affecting the model's ability to converge to an optimal solution.
- **Scaling Exponent** — A parameter in scaling laws that determines the rate at which performance improves with increasing resources.
- **Compute-Optimal Training** — Balancing model size and dataset size to maximize performance for a given compute budget.
- **Test-Time Scaling** — Adjusting model size or architecture during inference to optimize for latency or memory constraints.
- **Emergent Abilities** — Unexpected capabilities that arise in large models, often not present in smaller models.

## Mathematical foundations

While there aren't universal equations that capture all scaling laws, a common form relates model performance to model size, dataset size, and compute:

\[
\text{Performance} \propto (\text{Model Size})^{\alpha} \cdot (\text{Data Volume})^{\beta} \cdot (\text{Compute})^{\gamma}
\]

where \(\text{Performance}\) is a metric like accuracy or loss, \(\text{Model Size}\) is the number of parameters, \(\text{Data Volume}\) is the size of the training dataset, \(\text{Compute}\) is the amount of computation used for training, and \(\alpha\), \(\beta\), and \(\gamma\) are scaling exponents. This equation says that performance is proportional to the product of model size, data volume, and compute, each raised to a scaling exponent.

The scaling exponents \(\alpha\), \(\beta\), and \(\gamma\) determine the rate at which performance improves with increasing model size, data volume, and compute, respectively. These exponents are often empirically determined and can vary depending on the specific task and architecture.

## Key algorithms / techniques
- **Compute-Optimal Scaling (2022)** — Focuses on balancing model size and dataset size to maximize performance for a given compute budget, as highlighted by Chinchilla et al.
- **Parallel Scaling (2025)** — Increases a model's parallel computation during training and inference, offering improved inference efficiency compared to parameter scaling, as introduced by Li et al.
- **Kinetics Scaling (2025)** — Incorporates memory access costs into test-time scaling laws, proposing a new scaling paradigm centered on sparse attention, as presented by Zhu et al.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| On the Origin of Algorithmic Progress in AI | 2024 | Ho et al. | Investigates the sources of efficiency gains in AI, focusing on algorithmic advancements and scale-dependent improvements. |
| Kinetics: Rethinking Test-Time Scaling Laws | 2025 | Zhu et al. | Introduces the "Kinetics" scaling law that incorporates memory access costs, and proposes a new scaling paradigm centered on sparse attention. |
| Towards a Science of Scaling Agent Systems | 2025 | Gu et al. | Derives quantitative scaling principles for agent systems, analyzing the interplay between agent quantity, coordination structure, model capability, and task properties. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Scaling Laws for Neural Language Models | 2020 | Kaplan et al. | Established the foundational scaling laws relating model size, dataset size, and compute to language modeling performance. |
| Training Compute-Optimal Large Language Models | 2022 | Hoffmann et al. | Provided a comprehensive analysis of compute-optimal training, showing how to balance model size and dataset size for optimal performance. |

## Current SotA

Parallel Scaling (ParScale) achieves improved inference efficiency compared to parameter scaling (2025). Kinetics scaling law incorporates memory access costs into test-time scaling laws (2025). Agent systems scaling laws analyze the interplay between agent quantity, coordination structure, model capability, and task properties (2025).

## What's happening now

Research frontiers are focused on developing more comprehensive scaling laws that account for factors beyond model size, data volume, and compute, such as architectural choices, memory access costs, and the impact of different training techniques. This includes exploring the scaling behavior of agent systems and the interplay between agent quantity, coordination structure, and task properties.

Engineering and systems efforts are focused on developing efficient training and inference techniques that can take advantage of scaling laws to optimize model performance for real-world deployment. This includes techniques like parallel scaling, which increases a model's parallel computation during training and inference, and sparse attention, which reduces memory access costs.

The open problem is: How can we develop a unified scaling law that accurately predicts the performance of agent systems across diverse tasks and architectures, considering the trade-offs between tool coordination, model capability, and computational resources?

## In production
- Google — PaLM — Scales to 540 billion parameters, demonstrating improved performance on various NLP tasks — [https://ai.googleblog.com/2022/04/pathways-language-model-palm-scaling-to.html]
- OpenAI — GPT-3 — Scales to 175 billion parameters, showcasing emergent abilities in language generation and understanding — [https://openai.com/research/gpt-3]
- Meta — Llama — Open-source LLM family, demonstrating scaling laws in publicly available models — [https://ai.meta.com/blog/llama-open-source-language-model/]

## Code & implementations

For a hands-on build with this concept, see the MVB on [[large-language-models]].

## What comes next

- [[large-language-models]] — Scaling laws are most evident and impactful in large language models, guiding their development and optimization.
- [[efficient-transformers]] — Scaling laws motivate the development of efficient transformer architectures that can achieve high performance with reduced computational costs.

## Connected topics

- [Optimization](./optimization.md) — Scaling laws often involve optimizing large neural networks.
- [Backpropagation](./backpropagation.md) — Backpropagation is crucial for training the large models used in scaling laws.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers are a key architecture that benefits significantly from scaling laws.
- [Neural Tangent Kernel (NTK)](../15-ml-theory-foundations/ntk.md) — NTK relates to the behavior of neural networks in the large width limit, relevant to scaling laws.
- [Data Parallelism](../09-algorithms-systems-ai/data-parallelism.md) — Data parallelism is often used to train large models, enabling scaling laws.
- [KV Cache](../09-algorithms-systems-ai/kv-cache.md) — KV cache is used to improve the efficiency of large language models, which are subject to scaling laws.


## Further reading
- Kaplan et al. (2020) — "Scaling Laws for Neural Language Models" — [https://arxiv.org/abs/2001.08361] — This paper provides the foundational scaling laws relating model size, dataset size, and compute to language modeling performance.
- Hoffmann et al. (2022) — "Training Compute-Optimal Large Language Models" — [https://arxiv.org/abs/2203.15556] — This paper provides a comprehensive analysis of compute-optimal training, showing how to balance model size and dataset size for optimal performance.
- Ho et al. (2024) — "On the Origin of Algorithmic Progress in AI" — [https://arxiv.org/abs/2511.21622v1] — This paper investigates the sources of efficiency gains in AI, particularly focusing on the impact of algorithmic advancements and scale-dependent improvements.
- Li et al. (2025) — "Parallel Scaling Law for Language Models" — [https://arxiv.org/abs/2505.10475v1] — This paper introduces "Parallel Scaling" (ParScale), a new scaling paradigm that increases a model's parallel computation during training and inference, offering improved inference efficiency compared to parameter scaling.
```