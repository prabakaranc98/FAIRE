---
title: Algorithms & Systems for AI
tags: [systems, distributed-training, inference, hardware, efficiency]
---

# Track 09 · Algorithms & Systems for AI

> The engineering of scale: distributed training, inference optimization, memory systems, hardware-aware algorithms, and the systems that make frontier AI possible.

Modern AI is as much a systems problem as an algorithms problem. Training a large model requires distributed compute, careful memory management, and hardware-aware kernels. Serving it requires inference systems that balance latency, throughput, and cost. This track covers the systems side of frontier AI.

---

## Topics

### Distributed Training
- [Data Parallelism](data-parallelism.md) — gradient synchronization, AllReduce, DDP, ZeRO
- [Model Parallelism](model-parallelism.md) — tensor parallelism, pipeline parallelism, 3D parallelism
- [Mixed Precision Training](mixed-precision.md) — FP16/BF16, loss scaling, numerical stability

### Memory & Efficiency
- [Gradient Checkpointing](gradient-checkpointing.md) — activation recomputation, memory-compute tradeoffs
- [Parameter-Efficient Fine-Tuning](peft.md) — LoRA, adapters, prefix tuning, QLoRA
- [Quantization](quantization.md) — INT8/INT4, post-training quantization, quantization-aware training

### Inference Systems
- [KV Cache](kv-cache.md) — attention caching, memory layout, multi-query and grouped-query attention
- [Speculative Decoding](speculative-decoding.md) — draft models, tree attention, acceptance criteria
- [Batching & Serving](inference-serving.md) — continuous batching, vLLM, PagedAttention, throughput optimization

### Hardware & Kernels
- [GPU Architecture](gpu-architecture.md) — CUDA hierarchy, memory bandwidth, roofline model
- [Custom Kernels](custom-kernels.md) — FlashAttention, Triton, CUTLASS, kernel fusion
- [AI Hardware](ai-hardware.md) — TPUs, Trainium, Groq, neuromorphic chips

### Algorithms
- [Sorting & Graph Algorithms](classical-algorithms.md) — complexity, data structures relevant to ML pipelines
- [Numerical Methods](numerical-methods.md) — linear algebra, eigendecomposition, iterative solvers

---

## Connections to frontier research

- **Training at the frontier** — what it takes to train 100B+ parameter models
- **Inference at scale** — serving millions of requests with sub-second latency
- **Memory-efficient architectures** — designing models that fit within hardware constraints
- **Co-design** — hardware and algorithm development as a joint optimization

---

## Recommended entry points

Start with [Data Parallelism](data-parallelism.md) and [Mixed Precision Training](mixed-precision.md) for training systems. For inference, start with [KV Cache](kv-cache.md) and [Batching & Serving](inference-serving.md).
