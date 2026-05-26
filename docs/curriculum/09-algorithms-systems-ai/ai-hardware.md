---
title: AI Hardware
track: 09-algorithms-systems-ai
tags: [accelerators, systolic-arrays, inference, llm-serving, hardware-co-design]
depth: foundational
prereqs: [large-language-models, neural-networks]
updated: 2025-05-14
has_mvb: true
---

# AI Hardware

> **TL;DR:** AI hardware comprises specialized architectures designed to overcome the memory and computational constraints inherent in scaling modern neural networks.

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

## What it is

Imagine you are trying to solve a massive jigsaw puzzle, but you only have one hand to pick up pieces and a tiny desk to lay them out. This is the situation a standard central processing unit (CPU) faces when running modern AI. While these processors are excellent at handling complex, branching logic, they struggle when tasked with the millions of simple, repetitive calculations required to run a neural network. The "puzzle" of AI inference is too large for the "desk" of a general-purpose processor, leading to a bottleneck where the chip spends more time moving data than actually calculating.

To solve this, engineers build specialized hardware accelerators. Instead of one hand, these chips have thousands of tiny, synchronized workers—often arranged in a grid called a systolic array, which is a network of processing elements that rhythmically compute and pass data to neighbors—that move information in a highly efficient, rhythmic flow. By tailoring the physical silicon to the specific patterns of tensor math, these accelerators bypass the limitations of traditional hardware. This shift has transformed the chip from a passive component into a dynamic participant in the inference process, actively managing how data moves to keep the compute units busy.

## Why it matters at the frontier

Specialized AI hardware is the physical substrate upon which the current generation of frontier AI is built. The transition from general-purpose CPUs to domain-specific accelerators is the primary reason real-time LLM interaction is possible today. Hardware design choices, such as the ratio of compute-to-memory bandwidth, dictate the architectural constraints of the models themselves.

This matters because the hardware-software interface is the most significant bottleneck in AI research. As models move toward mixture-of-experts (MoE) architectures, the demand for hardware that can handle sparse, dynamic memory access patterns has intensified. Understanding these constraints is essential for researchers aiming to design algorithms that are not only mathematically sound but also physically efficient to execute. The synthesis of these fields suggests that the next generation of AI performance gains will come less from algorithmic novelty and more from the co-design of silicon that understands the specific dataflow of the models it executes.

## Core concepts

- **Systolic Array** — A network of data-processing units that rhythmically compute and pass data through the system, minimizing memory access.
- **Prefill Phase** — The initial inference stage where the model processes the prompt to generate the KV cache, characterized by high compute intensity.
- **Decode Phase** — The iterative stage where the model generates tokens one by one, characterized by high memory bandwidth intensity.
- **Memory Wall** — The performance limitation caused by the disparity between processor speed and memory bandwidth.
- **Disaggregation** — The architectural strategy of separating prefill and decode compute resources to optimize for their distinct workload profiles.

## Mathematical foundations

\[
T_{total} = T_{prefill} + T_{decode}
\]
where \(T_{total}\) is the total inference time, \(T_{prefill}\) is the time spent in the prefill phase, and \(T_{decode}\) is the time spent in the decode phase. This equation highlights the two distinct phases of LLM inference and the need for specialized hardware to optimize each.

\[
\text{Throughput} = \frac{\text{Tokens Generated}}{\text{Time}}
\]
where \(\text{Tokens Generated}\) is the count of output tokens and \(\text{Time}\) is the wall-clock duration. This equation defines throughput, the primary metric for evaluating the performance of LLM inference hardware.

\[
\text{Cost} = \text{Hardware Cost} + \text{Energy Cost}
\]
where \(\text{Hardware Cost}\) is the capital expenditure for the silicon and \(\text{Energy Cost}\) is the operational expenditure for power consumption. This equation represents the total cost of running an LLM, emphasizing the trade-offs in AI hardware design.

## Key algorithms / techniques

- **Systolic Dataflow** (Kung & Leiserson, 1978) — A method of mapping matrix operations onto a grid of processing elements to maximize data reuse; this technique is the basis for modern tensor cores.
- **Prefill-Decode Disaggregation** (SPAD, 2024) — A technique that routes prefill and decode tasks to different hardware clusters to prevent long-running decode tasks from blocking prompt processing.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Systolic Arrays for (VLSI) | 1978 | Kung & Leiserson | Foundational theory for parallel hardware. |
| SPAD: Specialized Prefill and Decode Hardware | 2024 | Zhang et al. | [https://arxiv.org/abs/2510.08544v1](https://arxiv.org/abs/2510.08544v1) |
| Serving LLMs on Huawei CloudMatrix384 | 2025 | Wang et al. | [https://arxiv.org/abs/2506.12708v1](https://arxiv.org/abs/2506.12708v1) |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Systolic Arrays for (VLSI) | 1978 | Introduced the systolic array, a paradigm that remains the standard for matrix multiplication in modern AI accelerators. |
| CNS-1 Architecture Specification | 1993 | Provided one of the first concrete specifications for neural network-specific hardware, detailing the trade-offs between compute and memory that still define modern chip design. |
| TPU: An Application-Specific Accelerator | 2017 | Demonstrated that domain-specific hardware could achieve orders of magnitude better performance-per-watt than general-purpose CPUs for neural network workloads. |

## Current SotA

The Huawei CloudMatrix384 supernode achieves a prefill throughput of 6,688 tokens/s per NPU and a decode throughput of 1,943 tokens/s per NPU (Wang et al., 2025, [https://arxiv.org/abs/2506.12708v1](https://arxiv.org/abs/2506.12708v1)). NVIDIA’s Blackwell Ultra platforms set the standard for extreme co-design, delivering the lowest token cost through tight integration of TensorRT-LLM and specialized hardware (NVIDIA, 2024, [https://developer.nvidia.com/blog/nvidia-platform-delivers-lowest-token-cost-enabled-by-extreme-co-design/](https://developer.nvidia.com/blog/nvidia-platform-delivers-lowest-token-cost-enabled-by-extreme-co-design/)).

## What's happening now

Research is exploring hardware that can handle the irregular memory access patterns of mixture-of-experts (MoE) models. Studies on routing costs in MoE architectures (Fedus et al., 2022, [https://arxiv.org/abs/2201.05596](https://arxiv.org/abs/2201.05596)) highlight that hardware must dynamically adapt to expert activation, a significant departure from static systolic designs.

Engineering efforts are shifting toward extreme co-design, where hardware is optimized for specific software kernels. Disaggregated inference hardware (Zhang et al., 2025, [https://arxiv.org/abs/2510.08544v1](https://arxiv.org/abs/2510.08544v1)) demonstrates that separating prefill and decode compute resources is critical for maintaining high utilization in multi-tenant environments.

The open problem remains the lack of hardware that can dynamically adapt to varying model architectures. As noted in recent surveys on hardware-software co-design (Hennessy & Patterson, 2019, [https://dl.acm.org/doi/10.1145/3282506](https://dl.acm.org/doi/10.1145/3282506)), current chips are often hard-wired for specific tensor shapes, making them inefficient when model architectures evolve rapidly.

## Open questions

:::admonition
**Researcher:** Can we design a reconfigurable systolic array that dynamically changes its dataflow topology based on the sparsity pattern of an MoE model at runtime?
:::

:::admonition
**Engineer:** How can you measure the impact of memory bandwidth on token generation latency using only consumer-grade hardware (e.g., RTX 3080) by artificially limiting the memory clock?
:::

:::admonition
**Think about this:** If hardware becomes perfectly specialized for current Transformer architectures, does that specialization create a "lock-in" effect that prevents the adoption of fundamentally different, more efficient neural architectures?
:::

## In production

- **NVIDIA** — Blackwell Ultra — Lowest token cost via extreme co-design — [NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-platform-delivers-lowest-token-cost-enabled-by-extreme-co-design/)
- **Huawei** — CloudMatrix384 — 6,688 tokens/s (prefill) per NPU — [arXiv:2506.12708v1](https://arxiv.org/abs/2506.12708v1)
- **vLLM** — [https://github.com/vllm-project/vllm](https://github.com/vllm-project/vllm) — High-throughput serving engine.
- **TensorRT-LLM** — [https://github.com/NVIDIA/TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) — NVIDIA's library for optimizing inference.

## Minimum Valuable Build

**Compute:** Runs on RTX 3080 (10GB VRAM) or free Colab T4.

1. **Install dependencies:** `pip install torch vllm`
2. **Run minimal inference:** `python -m vllm.entrypoints.openai.api_server --model meta-llama/Llama-3-8b`
3. **Benchmark:** Use `vllm/benchmarks/benchmark_throughput.py` to measure tokens/s.
4. **Expected outcome:** A running API server serving Llama-3-8B with a measured throughput of >100 tokens/s.

*For the curious generalist, the artifact is a local API server. For the math student, the success metric is observing the latency difference between prefill and decode phases.*

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/vllm-project/vllm) is the only signal we collect.*

---

## What comes next

Near-term hardware developments are moving toward "near-memory" compute, where processing logic is integrated directly into the memory die to eliminate the memory wall. This will likely enable the deployment of models with trillions of parameters on single-node systems, fundamentally changing the economics of local AI inference.

- [[attention-mechanisms]] — Understanding the compute-heavy nature of attention is the first step to designing hardware that accelerates it.
- [[quantization]] — Quantization reduces the memory bandwidth requirement, allowing hardware to process more tokens per second.

## Further reading

- [Kung & Leiserson (1978)](https://eecs.harvard.edu/htk/static/files/1978-cmu-cs-report-kung-leiserson.pdf) — The seminal paper on systolic arrays.
- [Lilian Weng's Survey on LLM Inference](https://lilianweng.github.io/posts/2023-01-10-inference/) — An overview of the inference bottlenecks that drive hardware design.