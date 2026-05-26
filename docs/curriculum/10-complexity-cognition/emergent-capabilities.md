---
title: Emergent Capabilities
track: 10-complexity-cognition
tags: [scaling-laws, complexity, LLMs, intelligence-evolution]
depth: foundational
prereqs: [large-language-models, scaling-laws]
updated: 2025-05-14
has_mvb: true
---

# Emergent Capabilities

> **TL;DR:** Emergent capabilities are non-linear improvements in model performance on complex tasks that appear abruptly as a function of scale, challenging our ability to predict model behavior before deployment.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| Curious learner | [§What it is](#what-it-is) | Build intuition |
| CS student / tinkerer | [§MVB — CS student](#mvb-cs-student) | Run a scaling experiment |
| Applied engineer | [§In production](#in-production) | Deploy evaluation frameworks |
| Applied researcher | [§What's happening now](#whats-happening-now) | Test capability thresholds |
| Theory student | [§Mathematical foundations](#mathematical-foundations) | Understand phase transitions |
| Frontier researcher | [§Open questions](#open-questions) | Identify falsification criteria |
| PM / decision-maker | [§Why it matters at the frontier](#why-it-matters-at-the-frontier) | Assess deployment risks |

---

## What it is

Imagine you are teaching a child to solve puzzles. At first, they struggle with simple shapes, then suddenly, they grasp the concept of spatial rotation and begin solving complex jigsaws with ease. This "aha!" moment is the human equivalent of an emergent capability. In machine learning, we observe a similar phenomenon: as we increase the number of parameters—the internal weights the model adjusts during training—and the compute budget, performance on tasks like multi-step arithmetic or logical reasoning does not always improve steadily. Instead, it often remains near-random for a long time, then jumps abruptly to high accuracy.

This behavior is distinct from traditional software engineering, where system capabilities are bounded by explicit logic. In deep learning, the model learns internal representations—the mathematical patterns the network uses to encode information—that allow it to solve novel problems by interpolating across its training distribution. When a model reaches a specific threshold of scale, these representations coalesce into functional modules that enable complex task performance, a process that often catches developers by surprise.

The consequence is a fundamental shift in how we approach AI safety. If a model’s capabilities are not predictable through simple extrapolation of performance metrics, we cannot rely on testing small-scale versions to guarantee the safety of large-scale deployments. This unpredictability is why frontier labs now prioritize rigorous evaluation of capabilities at the edge rather than assuming that performance will remain within expected bounds.

## Why it matters at the frontier

Emergent capabilities represent the primary challenge in AI forecasting and safety. Because these abilities appear non-linearly, they create a capability gap where a model may seem benign at one scale but exhibit powerful reasoning skills at the next. This forces labs to treat scaling as a high-stakes experiment rather than a predictable engineering process.

Understanding these transitions is critical for the development of agentic systems. If we cannot predict when a model will gain the ability to plan, use tools, or execute code, we cannot design appropriate guardrails before the model is exposed to the internet. This uncertainty drives the current research focus on interpretability and robust evaluation, as labs attempt to map the internal state of models to their observable behaviors.

## Core concepts

- **Scaling Laws** — The empirical observation that model performance improves predictably with compute, data, and parameters.
- **Phase Transition** — A point in the scaling curve where a model’s performance on a specific task shifts from near-random to high-accuracy over a small increase in scale.
- **In-Context Learning** — The ability of a model to perform new tasks by processing examples in the prompt without weight updates.
- **Capability Threshold** — The minimum scale required for a model to reliably execute a specific reasoning or logic-based task.
- **Evaluation Gap** — The discrepancy between a model's performance on standard benchmarks and its actual capability in open-ended, real-world environments.

## Mathematical foundations

\[
P(\text{success}) \approx \sigma(\beta_0 + \beta_1 \log(C))
\]
where \(P(\text{success})\) is the probability of solving a task, \(\sigma\) is the sigmoid function, \(\beta_0\) and \(\beta_1\) are learned coefficients, and \(C\) is the compute budget (in FLOPs). This equation models the abrupt transition of capability as a logistic function of scale.

\[
\mathcal{L}(C) = \alpha C^{-\gamma} + \epsilon
\]
where \(\mathcal{L}\) is the loss, \(C\) is compute, \(\alpha\) and \(\gamma\) are scaling constants, and \(\epsilon\) is the irreducible error. This equation describes the baseline scaling law, which emergent capabilities often deviate from at specific thresholds.

## Key algorithms / techniques

- **Chain-of-Thought Prompting** — A technique that forces models to generate intermediate reasoning steps, often revealing emergent logic capabilities that are otherwise latent (Wei et al., 2022).
- **Scaling Analysis** — The systematic plotting of performance against compute to identify inflection points where task accuracy deviates from expected power-law trends, often used to probe for phase transitions in model behavior (Kaplan et al., 2020).

## Open questions

> **Researcher:** Can we define a universal metric for "emergence" that is invariant to the choice of evaluation task, or is emergence fundamentally tied to the specific structure of the benchmark?

> **Engineer:** How can we build "early warning" systems that detect the onset of emergent reasoning during the training process before the model reaches full convergence?

> **Open:** We currently have no way to look at a model's architecture and determine if it will exhibit emergent reasoning without training it to completion; what theoretical framework can bridge this gap?

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Emergent Abilities of LLMs | 2022 | Wei et al. | Defines the concept and provides the first systematic evidence of abrupt performance shifts. |
| Sparks of AGI | 2023 | Bubeck et al. | Critically analyzes whether emergence is a real phenomenon or a byproduct of evaluation metrics. |
| Emergent Abilities: A Survey | 2025 | Berti et al. | Synthesizes the current debate and categorizes the risks associated with unpredictable capability jumps. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Scaling Laws for Neural Language Models | 2020 | Kaplan et al. | Established the power-law relationship between compute and loss. |
| Language Models are Few-Shot Learners | 2020 | Brown et al. | Demonstrated that scale alone enables in-context learning. |

## Current SotA

Emergent reasoning is currently measured via benchmarks like MMLU and GSM8K. OpenAI's GPT-4o technical report (2024) indicates an 88.7% score on MMLU, demonstrating reasoning capabilities that were not present in models with <10B parameters, as detailed in the [OpenAI GPT-4o system card](https://openai.com/index/hello-gpt-4o/).

## What's happening now

Research is currently shifting from documenting emergence to predicting it. Berti et al. (2025) [https://arxiv.org/abs/2503.05788] argue that many "emergent" abilities are actually artifacts of evaluation metrics that are not continuous. By changing the metric from a binary pass/fail to a continuous score, researchers are finding that many capabilities actually scale smoothly.

Engineering efforts are focused on building "early warning" systems for model capabilities. Krakauer et al. (2025) [https://arxiv.org/html/2506.11135] propose using complex systems theory to model the internal dynamics of neural networks, looking for "critical slowing down" in training—a phenomenon that often precedes phase transitions in physical systems.

The open problem remains the lack of a predictive theory. We currently have no way to look at a model's architecture and determine if it will exhibit emergent reasoning without training it to completion, a limitation highlighted by the lack of mechanistic interpretability tools capable of mapping high-level reasoning to specific circuit activations (Olsson et al., 2022, [https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html]).

## In production

- **Amazon** — Uses an end-to-end production-grade framework for evaluating agentic AI systems to detect capability shifts before deployment ([AWS Blog](https://aws.amazon.com/blogs/machine-learning/evaluating-ai-agents-real-world-lessons-from-building-agentic-systems-at-amazon/)).
- **Google** — Implements Agentic AI Infrastructure to monitor autonomous agents as they transition from PoC to production environments ([Google Research](https://research.google/pubs/agentic-ai-infrastructure-in-practice-learn-these-key-hurdles-to-deploy-production-ai-agents-efficiently/)).
- **Databricks** — Utilizes coSTAR, an automated testing methodology for shipping AI agents without breaking existing logic ([Databricks Blog](https://www.databricks.com/blog/costar-how-we-ship-ai-agents-databricks-fast-without-breaking-things)).
- **Tools** — [nanoGPT](https://github.com/karpathy/nanoGPT) is the standard repository for training small-scale transformer models to study scaling laws; [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) is the industry-standard framework for evaluating emergent capabilities.

## Minimum Valuable Build

### MVB — CS student
**Goal:** Observe scaling behavior on a small-scale task.
1. Use the `TinyStories` dataset (HuggingFace: `roneneldan/TinyStories`).
2. Train three models of varying sizes (10M, 50M, 100M parameters) using `nanoGPT`.
3. Measure accuracy on a simple logic task (e.g., "Which word does not belong?").
4. **Artifact:** A plot of accuracy vs. parameter count.

### MVB — Curious learner
**Goal:** Visualize the "jump" in performance.
1. Use a pre-trained model family (e.g., `gpt2`, `gpt2-medium`, `gpt2-large`).
2. Run a zero-shot reasoning task (e.g., GSM8K subset) using the `lm-evaluation-harness`.
3. **Metric:** Compare the accuracy scores across the three model sizes to see if the performance gain is linear or exponential.

### MVB — Theory student
**Goal:** Test the "continuous metric" hypothesis.
1. Take a model that shows "emergent" behavior on a binary task.
2. Instead of binary pass/fail, extract the log-probability of the correct token.
3. **Metric:** Plot the log-probability as a function of model size to see if the "jump" smooths out.

### MVB — Applied researcher
**Goal:** Ablation study on emergent reasoning.
1. Select a model (e.g., `Qwen2-1.5B`).
2. Systematically mask attention heads to observe the degradation of reasoning capabilities.
3. **Artifact:** A heatmap of "critical" heads that cause a collapse in Chain-of-Thought performance.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/karpathy/nanoGPT) is the only signal we collect.*

---

## Code & implementations

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Official repo for small-scale transformer training.
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — Official repo for benchmarking LLM capabilities.

## What comes next

Understanding emergent capabilities allows for the design of safer, more predictable training regimes by identifying the thresholds where reasoning abilities appear.

- [[scaling-laws]] — Provides the empirical foundation for predicting how performance improves with compute.
- [[large-language-models]] — The primary architecture where emergent capabilities are most frequently observed.
- [[cognitive-architectures]] — Explores how emergent capabilities might be structured into goal-directed systems.
- [[circuit-complexity]] — Analyzes the computational limits of the circuits that give rise to emergent behaviors.
- [[complexity-classes]] — Categorizes the difficulty of the tasks that models suddenly master.
- [[chain-of-thought]] — A prompting technique that acts as a probe for latent emergent reasoning.
- [[convolutional-neural-networks]] — An older architecture that also exhibits emergent feature hierarchies.
- [[efficient-attention]] — Mechanisms that enable the scaling required for emergent capabilities to manifest.

## Connected topics

- [[../../arcs/complexity/step-01-scaling-laws.md]] — This concept appears in the scaling arc as the primary driver of non-linear performance jumps.

## Further reading

- [Lilian Weng's survey on LLM Evaluation (lil'log, 2023)](https://lilianweng.github.io/posts/2023-03-15-post-oai/) — A deep dive into the challenges of evaluating complex model behaviors.
- [Wei et al. (2022) Emergent Abilities of Large Language Models](https://export.arxiv.org/pdf/2206.07682v2.pdf) — The foundational paper defining the field.
- [Berti et al. (2025) Emergent Abilities in Large Language Models: A Survey](https://arxiv.org/abs/2503.05788) — The most comprehensive recent synthesis of the debate.