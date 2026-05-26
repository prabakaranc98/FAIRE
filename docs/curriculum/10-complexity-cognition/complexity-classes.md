---
title: Complexity Classes
track: 10-complexity-cognition
tags: [computational-complexity, scaling-laws, emergent-abilities, algorithm-analysis]
depth: foundational
prereqs: [scaling-laws, transformer-architecture]
updated: 2025-05-14
has_mvb: true
---

# Complexity Classes

> **TL;DR:** Complexity classes provide the formal framework for categorizing the computational resources required to solve problems, serving as a diagnostic tool for understanding why frontier models succeed at some reasoning tasks while failing at others.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is

Imagine you are trying to solve a jigsaw puzzle. Some puzzles are simple enough that you can finish them by matching colors and edges—a process that takes time proportional to the number of pieces. Other puzzles are designed to be deceptive, where every piece looks identical, and you must test every possible combination to find the right fit. This is the core intuition behind computational complexity: some problems are inherently "easy" to solve, while others require an amount of effort that explodes as the problem grows.

Large language models often exhibit a paradoxical performance profile. They can synthesize complex code or write fluent prose, yet they frequently struggle with simple arithmetic or multi-step logical deductions that a human child might solve effortlessly. This discrepancy arises because the underlying computational process of a transformer—a neural network architecture that uses self-attention to process sequences of data—is fundamentally different from the iterative, state-dependent algorithms required to solve problems in higher complexity classes.

The consequence is that model performance is not merely a function of parameter count, but a reflection of how well the model's internal representations can approximate the algorithmic structure of a given task. When a task requires a number of computational steps that grows polynomially or exponentially with input size, the model must "learn" an efficient heuristic or an iterative procedure within its weights. Understanding these classes allows us to predict which tasks will scale with compute and which will remain fundamentally out of reach for current architectures.

## Why it matters at the frontier

Complexity classes provide the diagnostic language for the reasoning gap observed in frontier models. As models are increasingly deployed as autonomous agents, their ability to navigate problems in classes like P (polynomial time) or NP (nondeterministic polynomial time) determines their reliability in high-stakes environments like software engineering or scientific discovery.

The field is shifting focus toward reasoning-heavy benchmarks because scaling laws are beginning to hit diminishing returns on simple prediction tasks. By formally characterizing the complexity of the tasks that trigger emergent abilities, researchers can move from empirical trial-and-error to a predictive science of capability acquisition. This transition is essential for building systems that do not just predict the next token, but reliably execute complex algorithmic procedures.

## Core concepts

- **P (Polynomial Time)** — The class of decision problems solvable by a deterministic Turing machine in time polynomial to the input size, representing problems considered efficiently solvable.
- **NP (Nondeterministic Polynomial Time)** — The class of decision problems for which a proposed solution can be verified in polynomial time, even if finding the solution is computationally expensive.
- **Complexity Hierarchy** — The nested structure of classes, such as P ⊆ NP ⊆ PSPACE, that defines the relative difficulty of computational problems based on resource requirements.
- **Emergent Ability** — A capability that appears suddenly at a specific scale of model training, often corresponding to the model discovering an efficient algorithm for a complex task.
- **Scaling Law** — The empirical relationship between compute, data, and model parameters that predicts performance on a given task class, helping to estimate the resources needed for specific complexity thresholds.

## Mathematical foundations

\[
T(n) = O(f(n))
\]
where \(T(n)\) is the time complexity function, \(n\) is the input size (number of bits), and \(f(n)\) is the upper bound on the number of elementary operations required to solve the problem.

\[
\text{P} = \bigcup_{k \in \mathbb{N}} \text{TIME}(n^k)
\]
where \(\text{P}\) is the class of polynomial-time problems, \(n\) is the input size, and \(k\) is a constant exponent representing the polynomial degree.

\[
\text{NP} = \bigcup_{k \in \mathbb{N}} \text{NTIME}(n^k)
\]
where \(\text{NP}\) is the class of nondeterministic polynomial-time problems, and \(\text{NTIME}\) denotes the time complexity on a nondeterministic Turing machine, representing problems where verification is easier than discovery.

## Key algorithms / techniques

- **Combinatorial Optimization** — Using LLMs to approximate solutions for NP-hard problems by framing them as sequence generation tasks, which leverages the model's ability to learn heuristics for complex search spaces (Li et al., 2025).
- **Chain-of-Thought (CoT) Prompting** — A technique that forces the model to decompose a complex problem into a sequence of intermediate steps, effectively increasing the "computational depth" of the inference process to handle tasks beyond the model's single-pass capacity.
- **Finetuning for Reasoning** — The process of aligning model weights to prioritize algorithmic paths over statistical correlations, often used to bridge the gap between P and NP-hard task performance by embedding specific reasoning heuristics (Snell et al., 2024).

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [A Theory for Emergence of Complex Skills](https://arxiv.org/abs/2307.15936) | 2023 | Arora et al. | Foundational theory on scaling and emergence. |
| [HeurAgenix](https://arxiv.org/abs/2506.15196v2) | 2025 | Li et al. | Practical application of complexity analysis. |
| [FrontierCS](https://arxiv.org/abs/2512.15699v1) | 2025 | FrontierCS | Benchmark suite for reasoning tasks. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| The Complexity of Theorem-Proving Procedures | 1971 | Cook — Introduced the concept of NP-completeness, which remains the bedrock of complexity theory. |
| Attention Is All You Need | 2017 | Vaswani et al. — Established the transformer architecture, which serves as the primary substrate for modern AI reasoning. |

## Current SotA

Frontier models like the o1-series and Claude 3.5 Sonnet achieve significant performance gains on algorithmic reasoning. OpenAI's o1 model scored 83.3% on the AIME 2024 math benchmark (OpenAI, 2024). These models outperform prior architectures by utilizing test-time compute to explore search spaces, effectively navigating problems that were previously considered intractable for standard transformers. These gains are directly linked to the techniques discussed in the Key algorithms section, specifically the use of iterative verification and search-based inference to bypass the limitations of fixed-depth computation.

## What's happening now

Research is currently focused on the "test-time compute" paradigm, where models are given the ability to perform iterative search or verification before outputting a final answer. Kim et al. (2025) argue that scaling agent systems requires a shift from static inference to dynamic, state-aware computation, which effectively allows models to operate outside the constraints of fixed-depth transformers.

Engineering efforts are directed toward "capability coevolution," where tasks are dynamically generated to challenge the model's current reasoning limits. Dai et al. (2026) showed that this coevolutionary approach accelerates the acquisition of novel expert capabilities by forcing the model to adapt to increasingly complex algorithmic constraints.

The open problem remains the formal characterization of the "complexity threshold" for emergent abilities. While we observe that models suddenly solve certain classes of problems at scale, we lack a predictive model that maps task complexity (e.g., circuit depth or logical depth) to the required parameter count or training compute.

## Open questions

> [!IMPORTANT]
> **Researcher:** Can we define a formal "logical depth" metric for LLM tasks that predicts the minimum parameter count required for successful zero-shot reasoning?

> [!IMPORTANT]
> **Engineer:** How can we implement efficient, low-latency search-based inference (e.g., Monte Carlo Tree Search) on consumer-grade hardware (10GB VRAM) without sacrificing model accuracy?

> [!IMPORTANT]
> **Open:** Is there a fundamental complexity class boundary that current transformer-based architectures cannot cross, regardless of the amount of compute or data provided?

## In production

- **Anthropic** — Claude 3.5 Sonnet — Used for complex code generation and reasoning tasks at scale.
- **OpenAI** — o1-series — Deployed for advanced mathematical and scientific reasoning, utilizing chain-of-thought and search-based inference.
- **Community Resource** — [FAIRE](https://github.com/prabakaranc98/FAIRE) — A community-maintained repository for exploring algorithmic reasoning in LLMs.

## Minimum Valuable Build

**Goal:** Build a reasoning-limited test harness to observe the "reasoning cliff" in a small model.

**Compute:** RTX 3080 (10GB VRAM) or free Colab T4.

1. **Install dependencies:** `pip install transformers torch datasets`
2. **Load model:** Use `meta-llama/Llama-3.2-1B` from Hugging Face.
3. **Prepare dataset:** Download the `gsm8k` dataset via `datasets.load_dataset("gsm8k", "main")`.
4. **Run inference:** Write a script to prompt the model with increasing logical steps (e.g., 1-step vs 5-step math problems).
5. **Measure:** Calculate accuracy vs. step count.
6. **Artifact:** A plot showing the accuracy drop-off, providing a visual representation of the model's reasoning limit.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations

- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — The standard library for loading and testing frontier models.
- [GSM8K Dataset](https://huggingface.co/datasets/openai/gsm8k) — The canonical benchmark for multi-step reasoning.

## This concept appears in

- [[../../arcs/reasoning/step-01-complexity-classes]] — This page serves as the foundational entry point for the reasoning arc, establishing the limits of current architectures.

## What comes next

Understanding complexity classes allows researchers to predict which tasks will require architectural changes versus simple scaling. Near-term progress is expected in the formalization of "logical depth" metrics, which will likely replace parameter count as the primary indicator of a model's reasoning potential.

- [[emergent-abilities]] — Emergent abilities are the empirical manifestation of a model crossing a complexity threshold.
- [[scaling-laws]] — Scaling laws provide the empirical data that complexity analysis seeks to explain.
- [[transformer-architecture]] — The transformer architecture defines the computational constraints that complexity classes help us analyze.

## Connected topics

- [[circuit-complexity]] — Circuit complexity is a related concept within the broader field of computational complexity, focusing on the size and depth of Boolean circuits.
- [[cognitive-architectures]] — Cognitive architectures often consider computational complexity in their design and analysis to mimic human-like reasoning.
- [[ai-hardware]] — AI hardware design is influenced by the computational complexity of algorithms, necessitating specialized chips for efficient execution.

## Further reading

- [Arora et al. (2023)](https://arxiv.org/abs/2307.15936) — "A Theory for Emergence of Complex Skills in Language Models"
- [Li et al. (2025)](https://arxiv.org/abs/2506.15196v2) — "HeurAgenix: Leveraging LLMs for Solving Complex Combinatorial Optimization Challenges"
- [FrontierCS (2025)](https://arxiv.org/abs/2512.15699v1) — "FrontierCS: Evolving Challenges for Evolving Intelligence"