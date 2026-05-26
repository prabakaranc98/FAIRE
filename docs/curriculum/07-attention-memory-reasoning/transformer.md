---
title: Transformer Architecture
track: 07-attention-memory-reasoning
tags: [transformer, self-attention, multi-head-attention, positional-encoding, encoder-decoder]
depth: all
prereqs: [self-attention, backpropagation, linear-algebra]
updated: 2026-05-25
has_mvb: true
---

# Transformer Architecture
> **TL;DR:** The architecture that replaced RNNs for sequence modeling — multi-head self-attention stacked with feedforward blocks — now the universal backbone for language, vision, audio, protein structure, and multimodal AI.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms-techniques) → [MVB](#minimum-valuable-build) | Build and fine-tune a transformer |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Understand why transformers beat RNNs |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Derive multi-head attention and understand the residual stream |
| Researcher / frontier | [Current SotA](#current-sota) → [What's happening now](#whats-happening-now) | Know where transformers are being challenged |

---

## What it is

The transformer replaces sequential recurrence with **self-attention**: every position in a sequence attends to every other position simultaneously, computing a weighted sum of value vectors. This parallelism is what makes transformers fast to train — unlike RNNs, there's no sequential bottleneck.

The full architecture stacks two sublayers in each block: **(1) Multi-head self-attention** — the core mechanism; **(2) Position-wise feedforward network** — two linear layers with a nonlinearity between them. Both sublayers are wrapped with residual connections and layer normalization. Stack N of these blocks, and you have a transformer.

The encoder-decoder structure in the original "Attention Is All You Need" paper (Vaswani et al. 2017) was designed for translation: the encoder produces representations, the decoder attends to them with cross-attention. Modern LLMs use only the **decoder stack** (GPT family) or only the **encoder stack** (BERT family), dropping the cross-attention.

## Why it matters at the frontier

The transformer is the architecture. Every frontier AI system — GPT-4, Claude, Gemini, Llama, Stable Diffusion (DiT backbone), AlphaFold 2 (Evoformer), Sora — uses a transformer or transformer variant as its core module. Understanding the transformer is not background knowledge — it's the vocabulary every paper in the field is written in.

Why did it win? Two reasons: (1) self-attention parallelizes over sequence length, enabling training on massive datasets on GPUs; (2) the architecture is flexible enough to be adapted — with small modifications — to language, vision, audio, graphs, and molecules.

## Core concepts

- **Self-attention** — each token attends to all others; produces contextualized representations; the mechanism that enables parallelization
- **Query, Key, Value (Q, K, V)** — three learned linear projections of each token's embedding; attention score = dot product of Q with all K; output = weighted sum of V
- **Scaled dot-product attention** — scale by 1/√d_k to prevent vanishing gradients from large dot products
- **Multi-head attention (MHA)** — h parallel attention heads with independent Q/K/V projections; each head attends in a different representation subspace; concatenated and projected
- **Causal masking** — in decoder-only models (GPT), future positions are masked out; prevents attending to positions not yet generated
- **Position-wise FFN** — two linear layers (expand → compress) with activation; standard is SwiGLU in modern LLMs; this is where ~2/3 of parameters live
- **Residual connections** — x + Sublayer(x); critical for gradient flow in deep stacks
- **Layer normalization** — pre-norm placement (before sublayer, not after) is standard in modern LLMs; stabilizes training
- **Positional encoding** — transformers have no built-in notion of order; absolute (sinusoidal), learned, or relative (RoPE, ALiBi) encodings inject position information

## Mathematical foundations

Scaled dot-product attention:
\[
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

where Q, K ∈ ℝ^{N×d_k}, V ∈ ℝ^{N×d_v}, N is sequence length, d_k is key dimension.

Multi-head attention (h heads, each of dimension d_k = d_model / h):
\[
\text{head}_i = \text{Attention}(QW_i^Q,\; KW_i^K,\; VW_i^V)
\]
\[
\text{MHA}(Q,K,V) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h)\, W^O
\]

Full transformer block (pre-norm variant, used in GPT-2 and beyond):
\[
x \leftarrow x + \text{MHA}(\text{LayerNorm}(x))
\]
\[
x \leftarrow x + \text{FFN}(\text{LayerNorm}(x))
\]

**Complexity:** Self-attention is O(N²·d) in time and O(N²) in memory — the quadratic bottleneck that FlashAttention and SSMs address.

## Key algorithms / techniques

- **Encoder-only (BERT-style)** — bidirectional context; MLM pretraining; best for classification, NER, retrieval
- **Decoder-only (GPT-style)** — causal (left-to-right) attention; autoregressive generation; best for language modeling, chat, reasoning
- **Encoder-decoder (T5/BART-style)** — cross-attention; best for translation, summarization, seq2seq tasks
- **FlashAttention** (Dao et al. 2022) — IO-aware attention: tiles Q/K/V to avoid materializing N×N matrix; same output, 2-4× faster, O(N) memory
- **Grouped-query attention (GQA)** — multiple query heads share a single key/value head; reduces KV cache memory by 8-32×; standard in Llama 3, Mistral
- **Multi-Head Latent Attention (MLA)** (DeepSeek-V2) — compress K/V into a latent vector; 5-13× KV cache reduction; most memory-efficient attention variant
- **RoPE** (Su et al. 2021) — rotary positional embeddings; relative position via rotation; enables length generalization; default in Llama, Mistral, DeepSeek

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Vaswani et al. | The transformer paper — the foundation |
| [BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) | 2018 | Devlin et al. | Encoder-only; how transformers become pretrained representations |
| [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 2020 | Brown et al. | Decoder-only scaling; in-context learning |
| [FlashAttention: Fast and Memory-Efficient Exact Attention](https://arxiv.org/abs/2205.14135) | 2022 | Dao et al. | How attention is actually computed at scale |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | The transformer — ended the RNN era |
| [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | 2020 | Transformer applied to vision without convolutions |
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Why bigger transformers are predictably better |
| [RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) | 2021 | RoPE — the default positional encoding in modern LLMs |

## Current SotA
> *Updated: 2026-05-25*

All frontier LLMs are decoder-only transformers with: pre-norm, RoPE, SwiGLU activations, GQA or MLA, RMSNorm. DeepSeek-V3 (671B MoE) represents the efficiency frontier: MLA (5-13× KV compression) + Mixture-of-Experts + multi-token prediction. Mamba (Gu & Dao 2023) and hybrid Mamba-attention models (Zamba, Jamba) are the main architectural challengers.

## What's happening now
> *Research · Engineering · Systems*

**Research:** State Space Models (Mamba) and hybrid attention-SSM architectures challenge transformers for long-context efficiency. Multi-token prediction (training to predict k tokens simultaneously) improves sample efficiency. Multi-Head Latent Attention (DeepSeek-V2) shows that compressing the KV cache 13× incurs minimal quality loss.

**Engineering & Systems:** FlashAttention-3 (H100-optimized, async computation, FP8) is the production standard. PagedAttention (vLLM) manages KV cache at serving time as virtual memory pages. KV cache quantization (INT4/FP8) reduces serving memory 2-4×.

**Open problems:** Can attention scale to 10M+ token contexts efficiently? When do SSMs beat transformers, and vice versa? Is multi-token prediction a free lunch or does it trade off something?

## In production
> *How top labs and companies have deployed this at scale*

- **OpenAI (GPT-4, ChatGPT):** Decoder-only transformer at scale; speculative decoding + continuous batching for serving. [openai.com/research/gpt-4](https://openai.com/research/gpt-4)
- **Google DeepMind (Gemini):** Multi-modal transformer handling text, image, audio, video natively; Flash attention variants for long context. [arxiv.org/abs/2312.11805](https://arxiv.org/abs/2312.11805)
- **Meta AI (Llama 3):** 70B-405B open-weight decoder-only transformer; GQA, RoPE, SwiGLU; the reference open model. [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)
- **DeepSeek (V3):** 671B MoE transformer with MLA and multi-token prediction; trained for \(6M vs. GPT-4's ~\)100M. [arxiv.org/abs/2412.19437](https://arxiv.org/abs/2412.19437)
- **NVIDIA (NeMo framework):** Tensor parallelism + pipeline parallelism + FlashAttention for training 70B+ transformers on H100 clusters. [developer.nvidia.com/nemo](https://developer.nvidia.com/nemo)

## Minimum Valuable Build

**What you're building:** A character-level GPT (decoder-only transformer) trained on Shakespeare — generates Shakespeare-style text after 10 minutes of training on a laptop GPU.

**Why this is valuable:** Andrej Karpathy's nanoGPT is 300 lines of PyTorch and contains every key component of GPT-2. After building it, you will have implemented: multi-head causal attention, the residual stream, positional embeddings, and the autoregressive generation loop. This is the fastest path from zero to "I understand transformers."

**Stack:**
- **Model:** Implement from scratch following [nanoGPT](https://github.com/karpathy/nanoGPT) — ~300 lines
- **Dataset:** [tiny_shakespeare](https://huggingface.co/datasets/Trelis/tiny-shakespeare) or the raw text file (1MB)
- **Framework:** Pure PyTorch — no abstractions beyond `nn.Module`

**The recipe:**

1. **Implement the attention block:** `CausalSelfAttention` — Q/K/V projections, scaled dot-product attention, causal mask, multi-head concat + output projection. ~50 lines.
2. **Stack blocks:** `TransformerBlock` = attention + FFN + two LayerNorms + residuals. `GPT` = embedding + N blocks + LM head.
3. **Train on Shakespeare:** Cross-entropy loss on next-character prediction. ~10 minutes on M1 MacBook or Google Colab.
4. **Generate:** Sample autoregressively with temperature and top-k — watch Shakespeare emerge.
5. **Scale it:** Try 6-layer vs. 12-layer. Observe loss improvement. Now you understand scaling laws in your fingers.

**Expected outcome:** A working character-level language model generating plausible Shakespeare, with full understanding of every line of the transformer code.

**Stretch goals:**
- Fine-tune [gpt2](https://huggingface.co/openai-community/gpt2) on a custom dataset using HuggingFace `Trainer` — understand how fine-tuning differs from pretraining
- Add RoPE positional embeddings instead of learned absolute embeddings; see if performance changes
- Implement Flash Attention manually for the attention block; compare throughput vs. vanilla attention

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations

- [nanoGPT](https://github.com/karpathy/nanoGPT) — Karpathy's minimal GPT-2; best pedagogical resource
- [The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/) — harvard.edu; code-annotated walkthrough of the original paper
- [huggingface/transformers](https://huggingface.co/docs/transformers) — production library; `GPT2Model`, `BertModel`, etc.

## What can you build next?
> *Your arc of work continues here.*

You built a transformer and understand attention. The natural next question is: how do you take this architecture and turn it into a model that actually follows instructions? That requires alignment — and that's RLHF.

**Go deeper on this concept:**
→ Flash Attention — the single optimization that makes transformers viable at scale; implement it and see the memory/throughput difference directly

**Build a system with this:**
→ [Reinforcement Learning from Human Feedback (RLHF)](../06-reinforcement-learning/rlhf.md) — take your fine-tuned GPT-2 and run DPO on a preference dataset; that's the arc from "model that predicts tokens" to "model that follows instructions"

**The arc this page belongs to:**
→ [MLP → Transformer arc](../../arcs/mlp-to-transformer/index.md) — the transformer is the arc's culmination; the Language Models arc picks up here

## Connected topics

- Self Attention — the core mechanism inside the transformer
- [State Space Models](./state-space-models.md) — Mamba; O(N) alternative to O(N²) attention
- Flash Attention — IO-aware computation of attention; the reason transformers scale
- Lm Pretraining — how transformers are trained on tokens at scale
- Rotary Position Embedding — RoPE; the default positional encoding in modern LLMs

## Further reading

- [FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Dao 2023
- [GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — Ainslie et al. 2023
- [DeepSeek-V2: A Strong, Economical, and Efficient MoE LLM](https://arxiv.org/abs/2405.04434) — DeepSeek AI 2024 — MLA
