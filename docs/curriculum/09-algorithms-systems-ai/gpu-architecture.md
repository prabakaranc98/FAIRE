---
title: GPU Architecture
track: 09-algorithms-systems-ai
tags: [hardware, memory-hierarchy, compute-bound, memory-bound, gpu]
depth: foundational
prereqs: [tensor-operations, memory-hierarchy]
arc_refs: []
updated: 2025-05-14
has_mvb: true
---

# GPU Architecture

> **TL;DR:** Modern GPU architecture is a software-defined memory orchestration layer where performance is governed by data movement efficiency rather than raw floating-point throughput.

---

## Who this page is for

| Persona | What you get | Jump to |
|---|---|---|
| Curious learner | Intuition on the "Memory Wall" paradox | [§What it is](#what-it-is) |
| CS student / tinkerer | Laptop-GPU build with a target metric | [§MVB — CS student](#mvb-cs-student) |
| Applied engineer | Production framing + latency-shaped build | [§In production](#in-production), [§MVB — Applied engineer](#mvb-applied-engineer) |
| Applied researcher | Hypothesis + ablation build | [§What's happening now](#whats-happening-now), [§MVB — Applied researcher](#mvb-applied-researcher) |
| Theory student | Derivations + numerical verification | [§Mathematical foundations](#mathematical-foundations) |
| Frontier researcher | Open problems + falsifiers | [§Open questions](#open-questions), [§MVB — Frontier researcher](#mvb-frontier-researcher) |
| PM / decision-maker | Why it matters + SotA synthesis | [§Why it matters](#why-it-matters), [§Current SotA](#current-sota) |

---

## What it is

Modern GPUs are victims of their own success: an H100 can perform trillions of operations per second, yet it spends the vast majority of its time idling while waiting for data to arrive from memory. This "Memory Wall" paradox means that raw TFLOPS are increasingly irrelevant compared to the efficiency of the interconnects and cache management systems. To master modern GPU architecture, you must stop thinking of the GPU as a compute engine and start treating it as a software-defined memory orchestration layer.

This shift is a direct consequence of the widening gap between compute capability and memory bandwidth. As ALUs have scaled in density, the physical limits of HBM (High Bandwidth Memory) and the latency of data movement across the chip have become the primary constraints on throughput. Consequently, modern architecture is defined less by the raw number of cores and more by the sophistication of the memory hierarchy, including L1/L2 cache management and the interconnects that bridge the gap between compute units.

The consequence is that software performance is now dominated by IO-awareness. Algorithms that minimize data movement—by keeping tensors in fast SRAM as long as possible—consistently outperform those that rely on raw compute power. This insight has transformed the GPU from a static box of parallel processors into a dynamic system where the most successful models are those that treat the memory hierarchy as the primary compute resource.

## Why it matters

The transition to memory-centric architecture is the defining challenge for AI systems at the frontier. As models grow in parameter count and sequence length, the bottleneck shifts from the arithmetic logic units to the memory bus, making traditional compute-bound optimizations insufficient. This is why labs are increasingly prioritizing hardware-software co-design, where the memory access patterns of a specific model are baked into the kernel design itself.

This shift has unlocked new paradigms in inference, such as disaggregated hardware and specialized prefill/decode chips. Understanding these architectural constraints is no longer optional for researchers; it is the prerequisite for designing systems that can scale beyond the current limits of monolithic GPU clusters.

## Core concepts

- **SIMT (Single Instruction, Multiple Threads)** — A parallel execution model where a single instruction is executed across multiple threads, each operating on different data.
- **Memory Wall** — The performance bottleneck where the speed of the processor exceeds the speed of the memory system, causing the GPU to stall while waiting for data.
- **Arithmetic Intensity** — The ratio of floating-point operations to memory access operations, determining whether a kernel is compute-bound or memory-bound.
- **HBM (High Bandwidth Memory)** — A high-performance RAM interface for 3D-stacked DRAM, providing the massive bandwidth required for modern AI workloads.
- **IO-Awareness** — The design principle of optimizing algorithms to minimize data movement between slow HBM and fast on-chip SRAM.
- **SRAM (Static RAM)** — The fast, on-chip memory (cache) that acts as a buffer between the ALUs and the slower HBM.

## Mathematical foundations

\[
\text{Arithmetic Intensity} = \frac{\text{FLOPs}}{\text{Bytes}}
\]
where \(\text{FLOPs}\) is the total number of floating-point operations performed, and \(\text{Bytes}\) is the total amount of data moved from HBM. This equation defines the boundary between compute-bound and memory-bound regimes; if the ratio is lower than the hardware's compute-to-bandwidth ratio, the GPU is memory-bound.

\[
T_{\text{total}} = \max(T_{\text{compute}}, T_{\text{memory}})
\]
where \(T_{\text{compute}}\) is the time spent on ALU operations and \(T_{\text{memory}}\) is the time spent waiting for data transfer. This models the "Memory Wall" paradox, showing that the slowest component dictates the total execution time.

\[
\text{Throughput} = \frac{N \times \text{Ops}}{\text{Latency}_{\text{memory}} + \text{Latency}_{\text{compute}}}
\]
where \(N\) is the number of parallel threads, \(\text{Ops}\) is the operations per thread, and \(\text{Latency}\) represents the respective stall times. This quantifies the impact of memory latency on overall GPU throughput.

## Key algorithms / techniques

- **FlashAttention (2022)** — An IO-aware attention algorithm that tiles the attention matrix to keep data in SRAM, significantly reducing HBM access.
- **Kernel Fusion** — The process of combining multiple operations into a single GPU kernel to minimize the overhead of reading and writing intermediate results to HBM.
- **Quantization** — Reducing the precision of weights and activations to decrease the memory footprint and bandwidth requirements of model execution.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| NVIDIA Tesla: A Unified Graphics and Computing Architecture | 2008 | Lindholm et al. | Established the SIMT model and unified shader architecture. |
| FlashAttention: Fast and Memory-Efficient Exact Attention | 2022 | Dao et al. | Blueprint for IO-aware kernel design and memory-centric optimization. |
| SPAD: Specialized Prefill and Decode Hardware | 2025 | SPAD Research | Proposes disaggregated hardware for distinct inference phases. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| NVIDIA Tesla: A Unified Graphics and Computing Architecture | 2008 | Transitioned GPUs from fixed-function to general-purpose parallel processors. |
| FlashAttention: Fast and Memory-Efficient Exact Attention | 2022 | Proved that memory access patterns are the primary determinant of performance. |

## Current SotA

The NVIDIA Blackwell Ultra (GB300 NVL) platform achieves extreme performance through co-design, utilizing Quantum-X800 InfiniBand to minimize interconnect bottlenecks (2024). Huawei Cloud's CloudMatrix384 (Ascend 910C) achieves 1,943 tokens/s per NPU in decode throughput using a Unified Bus network (2025).

## What's happening now

Research is currently focused on "IO-aware" kernel design, where the memory hierarchy is treated as a first-class citizen in the compiler stack. Dao et al. (2022) demonstrated that exact attention can be computed with linear memory complexity by tiling, a technique now standard in all high-performance LLM kernels. Lin et al. (2024) have extended these IO-aware techniques to diverse hardware targets, including NPUs, proving that memory-centric design is portable across different silicon architectures.

Engineering efforts are shifting toward disaggregated hardware. The SPAD (2025) architecture separates compute-heavy prefill chips from memory-bandwidth-heavy decode chips, directly addressing the underutilization paradox where a single GPU cannot be optimized for both phases simultaneously. This disaggregation allows for more granular scaling of memory bandwidth versus compute power.

Open problems remain in the area of dynamic resource allocation. Current systems rely on static memory layouts, but researchers are investigating whether hardware can autonomously reconfigure its HBM cache hierarchy at runtime. This would allow the GPU to adapt to the specific sparsity patterns of an incoming request, potentially unlocking a new tier of inference efficiency.

## In production

- **NVIDIA** — Blackwell Ultra — Optimized for extreme co-design, utilizing Quantum-X800 InfiniBand to minimize interconnect bottlenecks — [NVIDIA Blog](https://developer.nvidia.com/blog/nvidia-platform-delivers-lowest-token-cost-enabled-by-extreme-co-design/)
- **Huawei Cloud** — CloudMatrix384 (Ascend 910C) — 384 NPUs interconnected via a Unified Bus (UB) network, achieving 1,943 tokens/s per NPU — [arXiv:2506.12708](https://arxiv.org/abs/2506.12708v1)

## Minimum Valuable Builds — by persona

### 1. For the curious learner (30 min · free tier)
**Build:** Visualize the difference between compute-bound and memory-bound operations.
**Artifact:** A Colab notebook comparing matrix multiplication (compute-bound) vs. a large-scale gather operation (memory-bound).
**Success:** Observing that the gather operation hits a performance ceiling significantly lower than the theoretical TFLOPS.
**Stack:** PyTorch on Google Colab T4.

### 2. For the CS student / tinkerer (1 day · RTX 4070 / M-series)
**Build:** Profile the arithmetic intensity of a custom CUDA kernel.
**Artifact:** A plot showing throughput vs. arithmetic intensity for varying matrix sizes.
**Success:** Identifying the "knee" in the curve where the kernel transitions from compute-bound to memory-bound.
**Stack:** PyTorch, `torch.profiler`, RTX 4070.

### 3. For the applied / production engineer (1 week · A10 / L4 / cloud)
**Build:** Deploy a vLLM endpoint and profile the memory bandwidth utilization.
**Artifact:** A latency report for Llama-3.1-8B-Instruct showing p50/p99 latency under varying batch sizes.
**Success:** Achieving p50 latency < 1.5s on A10 hardware.
**Stack:** `nvidia/Llama-3.1-8B-Instruct`, vLLM, A10 GPU.

### 4. For the applied researcher (3 days · A100)
**Build:** Ablate the impact of tile size on FlashAttention throughput.
**Artifact:** A comparison table showing throughput vs. tile size for different sequence lengths.
**Success:** Evidence that the optimal tile size is hardware-dependent and memory-bound.
**Stack:** `flash-attn` library, A100 GPU.

### 5. For the theory student (1 day · CPU)
**Build:** Derive the theoretical memory bandwidth limit for a matrix-vector multiplication.
**Artifact:** A plot comparing the theoretical bandwidth limit to a numerical simulation.
**Success:** Residual error below 5% between theory and simulation.
**Stack:** Python, NumPy.

### 6. For the frontier researcher (1 week+ · A100 cluster)
**Build:** Probe the "dynamic precision-aware memory allocation" hypothesis.
**Artifact:** Evidence that reconfiguring cache hierarchy based on sparsity improves decode throughput.
**Success:** A falsification criterion: if throughput does not increase by >10%, the hypothesis is rejected.
**Stack:** A100 cluster, custom CUDA kernels.

## Open questions

!!! researcher "For researchers"
    Can we design a hardware-software co-design that enables "dynamic precision-aware memory allocation," where the GPU architecture autonomously reconfigures its HBM cache hierarchy at runtime based on the specific sparsity patterns of an incoming LLM request?

!!! engineer "For engineers"
    How does the memory-bandwidth ceiling change when using 4-bit vs 8-bit quantization on a consumer GPU? Design an experiment to measure the throughput gain as a function of bandwidth reduction.

!!! open "Think about this"
    If memory bandwidth is the primary bottleneck, why do we continue to prioritize TFLOPS in GPU marketing? What would a "Bandwidth-per-Dollar" metric reveal about current hardware utility?

## This concept appears in

- Arc step pages for this concept are being generated.

## Connected topics

- [AI Hardware](./ai-hardware.md) — GPU architecture is a primary implementation of specialized AI hardware for machine learning.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Backpropagation is the core algorithm accelerated by efficient GPU architecture during model training.
- [Convolutional Neural Networks](../04-neural-networks-dl/cnn.md) — CNNs rely heavily on GPU architecture for parallelizing high-dimensional matrix convolution operations.
- [Efficient Attention](../07-attention-memory-reasoning/efficient-attention.md) — Efficient attention mechanisms are designed to optimize memory access patterns on GPU architectures.
- [Circuit Complexity](../10-complexity-cognition/circuit-complexity.md) — Circuit complexity provides the theoretical framework for analyzing hardware-level GPU computational limits.
- [Single-Head Attention](../07-attention-memory-reasoning/single-head-attention.md) — Single-head attention operations are mapped to GPU kernels to maximize parallel throughput.


## Further reading

- [NVIDIA Tesla Architecture (2008)](https://www.cs.cmu.edu/afs/cs/academic/class/15869-f11/www/readings/lindholm08_tesla.pdf) — The foundational paper on unified shader architecture.
- [FlashAttention (2022)](https://arxiv.org/pdf/2205.14135) — The definitive guide to IO-aware kernel design.
- [Lilian Weng's Blog on LLM Inference](https://lilianweng.github.io/posts/2023-01-10-inference/) — An excellent overview of the bottlenecks in modern LLM systems.