---
title: KV Cache & Attention Efficiency
track: 09-algorithms-systems-ai
tags: [kv-cache, attention, inference, mqa, gqa, mla, paged-attention]
depth: applied
prereqs: [transformer, gpu-architecture]
updated: 2026-05-25
---

# KV Cache & Attention Efficiency
> **TL;DR:** The key-value cache makes autoregressive generation fast by storing computed key/value tensors across steps — but it grows linearly with sequence length and batch size, making memory the primary constraint of LLM inference.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
During autoregressive generation, each token attends to all previous tokens. Without caching, this requires recomputing K and V for every prior token at every step — O(N²) total work. The KV cache stores these tensors, making each new token only O(1) in attention computation. The tradeoff: memory usage scales as O(N × d_model × num_layers × 2 × precision).

## Why it matters at the frontier
KV cache memory is the bottleneck of LLM inference at scale. Serving a 70B model with 32K context requires ~80GB of KV cache alone, often exceeding model weights. This drives: MQA/GQA (reduce heads), MLA (compress KV), PagedAttention (dynamic memory management), and speculative decoding (amortize prefill).

## Core concepts
- **KV cache** — store K and V for all past tokens; reuse at each generation step
- **Multi-Query Attention (MQA)** — single K,V head shared across all Q heads (Shazeer 2019)
- **Grouped-Query Attention (GQA)** — K,V shared across groups of Q heads (Ainslie et al. 2023)
- **Multi-Head Latent Attention (MLA)** — compress KV into a low-rank latent (DeepSeek-V2 2024)
- **PagedAttention** — manage KV cache like virtual memory pages; vLLM's key innovation
- **Prefill vs. decode** — prefill: parallel processing of prompt; decode: sequential generation

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Efficient Streaming Language Models with Attention Sinks](https://arxiv.org/abs/2309.17453) | 2023 | Xiao et al. | KV cache management for long-context streaming |
| [Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | 2023 | Kwon et al. | PagedAttention / vLLM |

## Current SotA
> *Updated: 2026-05-25*
MLA (DeepSeek-V2/V3) achieves 5-13× KV cache compression vs. MHA with no quality loss — the current most efficient attention variant. vLLM with continuous batching + PagedAttention is the production standard. Disaggregated prefill/decode (Mooncake, DistServe) is the next frontier for serving efficiency.

## Connected topics
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — KV cache is a direct consequence of transformer autoregressive generation
- Speculative Decoding — reduces decode latency complementary to KV cache optimizations
- Inference Serving — batching strategies that interact with KV cache management

## Further reading
- [FlashAttention-2: Faster Attention with Better Parallelism](https://arxiv.org/abs/2307.08691) — Dao et al. 2023
