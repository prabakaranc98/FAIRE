---
title: Language Models, Transformers & Multimodal AI
tags: [transformers, attention, language-models, multimodal, reasoning, ssm]
---

# Track 07 · Language Models, Transformers & Multimodal AI

> The architecture of modern AI: self-attention, transformers, large language models, multimodal systems, and the frontier of machine reasoning.

This track covers the dominant computational paradigm in modern AI — the transformer and its descendants — together with the language models and multimodal systems built on top of it. It spans architecture (how attention works) to system scale (how LLMs are trained) to behavior (how they reason).

---

## Topics

### Attention & Transformers
- [Self-Attention](self-attention.md) — queries, keys, values, scaled dot-product, complexity
- [Transformer Architecture](transformer.md) — encoder-decoder, positional encoding, layer normalization
- [Efficient Attention](efficient-attention.md) — linear attention, FlashAttention, sliding window, sparse patterns

### Sequence Models
- [State Space Models](state-space-models.md) — S4, Mamba, selective SSMs, recurrent vs. attention tradeoffs
- [Positional Encodings](positional-encodings.md) — absolute, relative, RoPE, ALiBi

### Language Models
- [Language Model Pretraining](lm-pretraining.md) — masked LM, causal LM, next-token prediction at scale
- [Scaling Laws](scaling-laws-llm.md) — Chinchilla, compute-optimal training, emergent capabilities
- [Instruction Tuning & Alignment](instruction-tuning.md) — SFT, RLHF, DPO, Constitutional AI

### Multimodal AI
- [Vision-Language Models](vision-language-models.md) — CLIP, contrastive pretraining, zero-shot transfer
- [Multimodal Generation](multimodal-generation.md) — image generation from text, video, audio-language models
- [Multimodal Reasoning](multimodal-reasoning.md) — chain-of-thought with images, visual question answering

### Reasoning & Memory
- [Chain-of-Thought Reasoning](chain-of-thought.md) — CoT prompting, scratchpads, reasoning traces
- [Memory Architectures](memory-architectures.md) — external memory, retrieval augmentation, long context
- [Tool Use & Agents](tool-use.md) — function calling, ReAct, agent scaffolds, code execution

---

## Connections to frontier research

- **Frontier LLMs** — GPT-4, Claude, Gemini, Llama — all are transformer-based language models with multimodal extensions
- **Mechanistic interpretability** — understanding what attention heads and MLP layers actually compute
- **World models** — transformers as predictive models of sequences of observations
- **Reasoning capabilities** — whether LLMs reason or recall, and what the difference means

---

## Recommended entry points

Start with [Self-Attention](self-attention.md) and [Transformer Architecture](transformer.md). Then [Language Model Pretraining](lm-pretraining.md). For multimodal, [Vision-Language Models](vision-language-models.md) is the entry point.
