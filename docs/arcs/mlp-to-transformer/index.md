---
title: "Arc: MLP to Transformer — The Architecture Lineage"
arc: mlp-to-transformer
super_domain: B-Modeling
tracks: [04-neural-networks-dl, 07-attention-memory-reasoning, 09-algorithms-systems-ai]
estimated_depth: "5-7 weeks, ~22 papers"
prereqs: [backpropagation, optimization, basic-linear-algebra]
---

# Arc: MLP to Transformer
> **What this arc builds:** The full architecture lineage — from universal approximation to attention to modern hybrid models — and why the transformer became the universal backbone for nearly every frontier AI system.

## Why this arc exists

Modern AI runs on transformers. But why transformers? Why attention? Why not deeper MLPs or better RNNs?

This arc answers those questions by tracing the actual historical development: each architecture existed because the previous one had a specific failure. Vanishing gradients → residual networks. Sequential bottleneck → attention. Quadratic attention cost → SSMs. Following this sequence, you understand not just what each architecture is but why it was invented and what it solved — which tells you how to think about what comes next.

## Prerequisites

Linear algebra (matrix multiplication, eigenvalues), calculus (chain rule), probability basics. Having implemented a neural network once (any framework) will help.

## The sequence

**Universal Approximation**

1. **Perceptrons & MLPs** (foundational) — linear classifier → universal approximation; why depth matters.
2. **Backpropagation** (foundational) — reverse-mode autodiff; the algorithm that makes training possible. [→](../../curriculum/04-neural-networks-dl/backpropagation.md)
3. **Optimization dynamics** (theoretical) — SGD, Adam, loss landscape; what makes neural networks trainable. [→](../../curriculum/04-neural-networks-dl/optimization.md)

**Convolutional & Recurrent**

4. **CNNs** (applied) — local connectivity, weight sharing, translational equivariance; AlexNet moment (2012).
5. **Residual Networks** (applied) — skip connections; solved vanishing gradients for very deep nets. [→](https://arxiv.org/abs/1512.03385)
6. **Batch Normalization** (applied) — normalize activations; smooth loss landscape; faster convergence.
7. **RNNs & LSTMs** (theoretical) — sequential processing; memory cells; still bottlenecked by sequence length.

**Attention & Transformer**

8. **Bahdanau attention** (foundational) — first attention mechanism; encoder-decoder alignment for translation.
9. **Self-attention** (foundational) — every token attends to every other; O(N²) but parallelizable. [→](../../curriculum/07-attention-memory-reasoning/transformer.md)
10. **Transformer** (applied) — multi-head attention + FFN + residual + LayerNorm; replaces RNNs. [→](../../curriculum/07-attention-memory-reasoning/transformer.md)
11. **Positional encodings** (theoretical) — absolute (sinusoidal), relative (RoPE, ALiBi); why position information must be injected.
12. **BERT vs. GPT** (applied) — encoder-only (masked LM) vs. decoder-only (causal LM); different pretraining objectives for different tasks.
13. **Scaling laws** (theoretical) — performance scales as power law with compute, data, model size. [→](../../curriculum/04-neural-networks-dl/scaling-laws.md)
14. **ViT** (applied) — patches as tokens; transformer for vision without convolutions.

**Modern Variants**

15. **Efficient attention** (applied) — FlashAttention, linear attention, sliding window; making O(N²) tractable.
16. **Grouped-query attention / MQA** (applied) — reduce KV heads; inference memory efficiency.
17. **Multi-Head Latent Attention (MLA)** (frontier) — KV cache compression; DeepSeek-V2's key innovation.
18. **State Space Models (Mamba)** (frontier) — input-selective SSMs; O(N) alternative to attention. [→](../../curriculum/07-attention-memory-reasoning/state-space-models.md)
19. **Titans** (frontier) — test-time learnable memory; >2M context via neural long-term memory. [→](https://arxiv.org/abs/2501.00663)
20. **Hybrid models** (frontier) — Jamba, Zamba; combining attention and SSM layers for best of both.

## Key figures

- **Ashish Vaswani, Illia Polosukhin** — Attention is All You Need
- **Jacob Devlin** — BERT
- **Alec Radford** — GPT series
- **Albert Gu** — S4, Mamba
- **Tri Dao** — FlashAttention, Mamba-2
- **DeepSeek AI** — MLA, MoE architectures

## Essential reading sequence

1. [Attention is All You Need](https://arxiv.org/abs/1706.03762) — Vaswani et al. 2017 — the transformer
2. [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) — He et al. 2015 — skip connections
3. [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) — Dosovitskiy et al. 2020
4. [FlashAttention](https://arxiv.org/abs/2205.14135) — Dao et al. 2022 — IO-aware attention
5. [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — Gu & Dao 2023
6. [Titans: Learning to Memorize at Test Time](https://arxiv.org/abs/2501.00663) — Behrouz et al. 2024

## Current frontier anchors
> As of 2026-05-25

- **DeepSeek-V3** — MLA + MoE + multi-token prediction; most efficient frontier LLM architecture
- **Mamba-2 (SSD)** — State Space Duality; formal equivalence between SSMs and attention
- **Titans** — neural long-term memory enabling >2M token contexts

## What you'll know when done

1. Implement a transformer from scratch, including multi-head attention and positional encoding
2. Explain why residual connections solve vanishing gradients
3. Explain the I/O-bound nature of attention and how FlashAttention fixes it
4. Describe the key difference between Mamba's selective SSMs and standard attention, and when each is preferred
5. Read a new architecture paper and identify which problem in the lineage it's addressing

## Branch points to other arcs

- **→ Language Models arc**: Transformers are the backbone; this arc explains the machinery
- **→ Systems for Scale arc**: FlashAttention, MLA, and MoE are where architecture meets systems
- **→ Generative Stack arc**: Diffusion Transformers (DiT) apply the transformer backbone to generation

## Where to go next

[Language Models arc →](../language-models/index.md) — How transformers become LLMs

[Systems for Scale arc →](../systems-for-scale/index.md) — How transformers are trained and served efficiently
