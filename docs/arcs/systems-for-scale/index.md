---
title: "Arc: Systems for Scale — FlashAttention to Inference at Scale"
arc: systems-for-scale
super_domain: A-Foundations
tracks: [09-algorithms-systems-ai, 04-neural-networks-dl, 07-attention-memory-reasoning]
estimated_depth: "5-7 weeks, ~20 papers"
prereqs: [transformer, backpropagation, basic-gpu-concepts]
---

# Arc: Systems for Scale
> **What this arc builds:** The engineering knowledge required to train and serve frontier AI models — GPU architecture, distributed training, memory optimization, and the inference systems that serve millions of requests.

## Why this arc exists

Training a frontier model isn't just an algorithms problem — it's a systems problem. A 70B model doesn't fit in any single GPU's memory. Serving it at scale requires careful management of KV cache, speculative decoding, and continuous batching. FlashAttention changed training by making attention IO-aware; PagedAttention changed serving by managing KV cache like virtual memory.

This arc makes those systems decisions legible. If you want to work at a frontier lab, or simply understand *why* models are designed the way they are, you need to understand the systems constraints they're designed for.

## Prerequisites

Understand the transformer architecture. Basic familiarity with GPU programming concepts is helpful (CUDA is not required). Mixed precision concepts (FP16, BF16) should be familiar.

## The sequence

**Hardware Foundations**

1. **GPU architecture** (foundational) — SMs, memory hierarchy (HBM vs. SRAM), bandwidth constraints, Tensor Cores.
2. **Roofline model** (theoretical) — arithmetic intensity; memory-bound vs. compute-bound operations. [→](https://arxiv.org/abs/2002.09498)
3. **CUDA programming basics** (applied) — warps, thread blocks, shared memory; why kernel fusion matters.
4. **Mixed precision** (applied) — FP16/BF16/FP8; loss scaling; numerical stability. [→](https://arxiv.org/abs/1710.03740)

**Efficient Training**

5. **FlashAttention** (frontier) — IO-aware attention using tiling; avoids materializing N×N attention matrix; 2-4× speedup. [→](https://arxiv.org/abs/2205.14135)
6. **FlashAttention-2/3** (frontier) — better parallelism; more hardware-aware; 3-9× over baseline.
7. **Gradient checkpointing** (applied) — recompute activations during backward; trade compute for memory. [→](../../curriculum/09-algorithms-systems-ai/data-parallelism.md)
8. **Data parallelism** (applied) — AllReduce, DDP, ring topology; scales throughput linearly. [→](../../curriculum/09-algorithms-systems-ai/data-parallelism.md)
9. **ZeRO / FSDP** (applied) — shard optimizer states, gradients, parameters; enables trillion-parameter training.
10. **Tensor parallelism** (applied) — split weight matrices across GPUs; Megatron-LM; requires model-aware sharding.
11. **Pipeline parallelism** (applied) — split layers across GPUs; GPipe, PipeDream; pipeline bubbles.
12. **3D parallelism** (applied) — data × tensor × pipeline combined; the recipe for largest training runs.

**Efficient Inference**

13. **KV cache** (foundational) — store and reuse key-value tensors across steps; memory scales O(N × L). [→](../../curriculum/09-algorithms-systems-ai/kv-cache.md)
14. **MQA / GQA** (applied) — reduce KV heads; trade quality for memory efficiency.
15. **Quantization** (applied) — INT8/INT4 weights and activations; GPTQ, AWQ, FP8. [→](../../curriculum/09-algorithms-systems-ai/quantization.md)
16. **Speculative decoding** (applied) — draft model generates candidates; main model verifies; ~3× speedup.
17. **PagedAttention / vLLM** (frontier) — manage KV cache as virtual memory pages; continuous batching. [→](https://arxiv.org/abs/2309.06180)
18. **Disaggregated prefill/decode** (frontier) — separate servers for prefill (parallel) and decode (sequential); Mooncake.
19. **LoRA / PEFT** (applied) — fine-tune large models with <1% parameters; rank-1 updates. [→](../../curriculum/09-algorithms-systems-ai/peft.md)

## Key figures

- **Tri Dao** (Princeton → Together AI) — FlashAttention series; Mamba-2
- **Samyam Rajbhandari** (DeepSpeed) — ZeRO memory optimizations
- **Woosuk Kwon** (Berkeley) — PagedAttention / vLLM
- **Zhuohan Li, Stephanie Wang** (Anyscale/Together) — disaggregated inference

## Essential reading sequence

1. [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) — Dao et al. 2022 — the key breakthrough
2. [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) — Rajbhandari et al. 2019
3. [Megatron-LM: Training Multi-Billion Parameter LMs Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Shoeybi et al. 2019
4. [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — Kwon et al. 2023
5. [GPTQ: Accurate Post-Training Quantization](https://arxiv.org/abs/2210.17323) — Frantar et al. 2022

## Current frontier anchors
> As of 2026-05-25

- **FlashAttention-3** — H100-optimized with async computation and FP8 support
- **vLLM with continuous batching** — the production standard for serving
- **DeepSeek-V3's MLA** — 5-13× KV cache compression; most efficient attention variant
- **Mooncake / DistServe** — disaggregated prefill/decode at scale

## What you'll know when done

1. Explain why FlashAttention is faster without approximation (it's about memory access, not FLOP count)
2. Describe ZeRO Stage 3 and what "sharding" the model weights means
3. Explain PagedAttention's insight: KV cache fragmentation is a memory management problem
4. Implement a basic speculative decoding loop and explain the acceptance criterion
5. Given a model and hardware spec, estimate if training fits in memory and what parallelism strategy to use

## Branch points to other arcs

- **→ MLP → Transformer arc**: Architecture choices (MLA, GQA, MoE) are driven by systems constraints
- **→ Language Models arc**: Serving LLMs requires all the inference systems here
- **→ Algorithms arc**: FlashAttention as an example of algorithm-hardware co-design

## Where to go next

[Language Models arc →](../language-models/index.md) — Serving the LLMs trained with these systems

[MLP → Transformer arc →](../mlp-to-transformer/index.md) — Architecture decisions motivated by systems constraints
