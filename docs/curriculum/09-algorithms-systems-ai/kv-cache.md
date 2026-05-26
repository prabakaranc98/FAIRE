```yaml
---
title: KV Cache
track: 09-algorithms-systems-ai
tags: [LLM, inference, optimization, memory management]
depth: foundational
prereqs: [attention-mechanism, transformer-architecture]
updated: 2024-07-03
has_mvb: false
---
# KV Cache
> **TL;DR:** The KV cache is a memory optimization technique used in transformer-based language models to store key and value pairs from previous layers, enabling efficient inference by avoiding redundant computations.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [In production](#in-production) | Understand how to optimize LLM inference |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Grasp the role of KV cache in LLMs |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Learn the underlying principles |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Explore the latest advancements and open challenges |

---

## What it is
Imagine you're chatting with a sophisticated AI, and it suddenly forgets details from earlier in the conversation. This happens because large language models (LLMs) struggle to remember everything from long interactions. The "KV cache" is a crucial technique designed to help LLMs retain and efficiently access information from previous turns, improving their ability to maintain context. Without it, the AI would have to recompute everything, slowing down responses and making the conversation feel disjointed.

The KV cache is a memory optimization strategy used during the inference phase of transformer models. In each layer of a transformer, the attention mechanism requires access to the keys (K) and values (V) from all previous tokens in the sequence. Instead of recomputing these K and V pairs for every new token generated, the KV cache stores them in memory. This allows the model to quickly retrieve and reuse these intermediate results, significantly reducing the computational cost and latency of generating text.

This optimization is particularly important for long sequences, where the memory and computational requirements of the attention mechanism can become prohibitive. By caching the K and V pairs, the model avoids redundant calculations, enabling it to generate longer and more coherent sequences with greater efficiency.

## Why it matters at the frontier
The KV cache is a critical component in enabling the deployment of large language models in real-world applications. As models grow in size and context length, the memory footprint of the KV cache becomes a major bottleneck. Efficient KV cache management is essential for reducing inference latency, increasing throughput, and minimizing the hardware resources required to serve these models.

Researchers are actively exploring techniques to compress, offload, and dynamically manage the KV cache to overcome these limitations. The open problem is: How can we dynamically and adaptively manage the KV cache across heterogeneous hardware (e.g., GPUs, CPUs, and SSDs) to optimize for both latency and throughput under varying workload conditions and context lengths? Addressing this challenge will unlock new possibilities for deploying LLMs in resource-constrained environments and enabling more interactive and engaging AI experiences.

## Core concepts
- **Key (K)** — A matrix representing the encoded input sequence, used in the attention mechanism to compute attention weights.
- **Value (V)** — A matrix representing the encoded input sequence, used in the attention mechanism to compute the weighted context vector.
- **Attention Mechanism** — A process that allows the model to focus on different parts of the input sequence when generating each output token.
- **Inference** — The process of using a trained model to generate new outputs, such as text or images.
- **Context Length** — The maximum number of tokens that a model can process at once, influencing its ability to understand and generate long sequences.
- **Memory Footprint** — The amount of memory required to store the model and its intermediate computations, including the KV cache.
- **Quantization** — A technique for reducing the memory footprint of a model by representing its parameters and activations with lower precision.

## Mathematical foundations
The attention mechanism, which utilizes the KV cache, can be expressed as:
\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

where \(Q\) is the query matrix, \(K\) is the key matrix, \(V\) is the value matrix, and \(d_k\) is the dimension of the keys. This equation calculates the attention weights by taking the softmax of the scaled dot product of the query and key matrices, and then uses these weights to compute a weighted sum of the value matrix.

The KV cache stores the key and value matrices from previous layers, allowing the model to reuse them in subsequent computations. The memory required for the KV cache can be expressed as:
\[
M = L \times N \times (d_k + d_v) \times B
\]

where \(M\) is the total memory required, \(L\) is the number of layers in the transformer, \(N\) is the sequence length, \(d_k\) is the dimension of the keys, \(d_v\) is the dimension of the values, and \(B\) is the batch size. This equation shows that the memory footprint of the KV cache grows linearly with the sequence length and the dimensions of the key and value matrices.

The KV cache optimization aims to reduce the memory footprint \(M\) while maintaining the performance of the attention mechanism. This can be achieved through techniques such as quantization, which reduces the size of \(d_k\) and \(d_v\), or through selective caching, which reduces the effective sequence length \(N\).

## Key algorithms / techniques
- **Quantization** — Reduces the memory footprint of the KV cache by storing the key and value matrices with lower precision (e.g., INT8 instead of FP16).
- **Key and Value Offloading** — Transfers the KV cache from GPU memory to CPU or disk storage to reduce GPU memory usage, trading off latency for memory savings.
- **Windowed Attention** — Limits the attention to a fixed-size window of previous tokens, reducing the memory required to store the KV cache.
- **FlashAttention** (Dao et al. 2023) — Reorders the attention computation to reduce memory reads and writes, improving performance and enabling longer sequence lengths.
- **MiKV** (Yang et al. 2024) — Compresses the KV cache using mixed-precision quantization to balance compression ratio and generation quality.

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | Dao et al. | Introduces an optimized attention algorithm that improves performance by leveraging on-chip memory and parallel computation. |
| No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization | 2024 | Yang et al. | Addresses the critical problem of KV cache compression and its impact on generation quality, introducing a practical solution. |
| Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving | 2024 | Qin et al. | Highlights the system-level challenges and solutions for KV cache management in production LLM serving, including disaggregation and scheduling. |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| Attention is All You Need | 2017 | Introduced the transformer architecture and the attention mechanism, laying the foundation for modern LLMs. |
| FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness | 2022 | Introduced FlashAttention, a memory-efficient attention mechanism that reduces memory reads and writes. |

## Current SotA
MiKV achieves significant KV cache compression with minimal impact on generation quality, enabling the deployment of long-context LLMs on resource-constrained devices (Yang et al., 2024). FlashAttention-2 further improves the performance of attention computations, reducing latency and increasing throughput (Dao et al., 2023). Mooncake demonstrates a KVCache-centric disaggregated architecture for LLM serving that separates prefill and decoding clusters and uses a disaggregated cache of KVCache (Qin et al., 2024).

## What's happening now
Research is focused on developing more efficient KV cache compression techniques that minimize the impact of eviction on generation quality while maximizing compression ratios, particularly for long-context LLMs. Engineering efforts are centered on designing system-level architectures that can effectively manage the KV cache across heterogeneous hardware resources, such as GPUs, CPUs, and SSDs, to optimize for both latency and throughput. A key open problem is: How can we develop KV cache compression techniques that minimize the impact of eviction on generation quality while maximizing compression ratios, particularly for long-context LLMs?

## In production
- NVIDIA — Dynamo — KV cache offloading to AWS S3 — [https://developer.nvidia.com/blog/nvidia-dynamo-adds-support-for-aws-services-to-deliver-cost-efficient-inference-at-scale/]
- Superhuman and Databricks — Production-scale inference platform — 200K+ QPS with end-to-end sub-1s P99 latency — [https://www.databricks.com/blog/how-superhuman-and-databricks-built-200k-qps-inference-platform-together]

## Minimum Valuable Build
For a hands-on build with this concept, see the MVB on [[FlashAttention]].

## Code & implementations
- FlashAttention: [https://github.com/Dao-AILab/flash-attention](https://github.com/Dao-AILab/flash-attention)

## What comes next
- [[Attention Mechanism]] — provides the foundation for understanding how the KV cache optimizes attention computations.
- [[FlashAttention]] — implements a memory-efficient attention mechanism that directly improves KV cache performance.

## Connected topics
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — KV-cache is a key component used within the Transformer architecture.
- [Data Parallelism](./data-parallelism.md) — Data parallelism can be used to manage and scale KV-cache storage.
- [Agent Architectures](../01-ai/agent-architectures.md) — KV-cache can be used in agent architectures for memory and context.
- [Cognitive Architectures](../10-complexity-cognition/cognitive-architectures.md) — KV-cache relates to memory and context within cognitive architectures.
- [Foundation Models in Robotics](../11-robotics-embodied-ai/foundation-models-robotics.md) — Foundation models in robotics may utilize KV-cache for efficient processing.
- [GNN Expressivity](../13-graph-relational-ai/gnn-expressivity.md) — KV-cache concepts can be applied to improve GNN expressivity.


## Further reading
- Dao et al. (2023) — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" — [https://arxiv.org/pdf/2307.08691.pdf] — Provides a detailed explanation of the FlashAttention-2 algorithm and its performance benefits.
- Yang et al. (2024) — "No Token Left Behind: Reliable KV Cache Compression via Importance-Aware Mixed Precision Quantization" — [https://arxiv.org/abs/2402.18096v1] — Explores the MiKV method for compressing the KV cache using mixed-precision quantization.
- Qin et al. (2024) — "Mooncake: A KVCache-centric Disaggregated Architecture for LLM Serving" — [https://arxiv.org/abs/2407.00079v3] — Presents a KVCache-centric disaggregated architecture for LLM serving.
- Hooper et al. (2024) — "Towards Efficient Large Language Model Serving: A Survey on System-Aware KV Cache Optimization" — [https://www.academia.edu/144683467/Towards_Efficient_Large_Language_Model_Serving_A_Survey_on_System_Aware_KV_Cache_Optimization] — Offers a comprehensive overview of system-aware KV cache optimization techniques.