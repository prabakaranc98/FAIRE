---
title: Transformer Architecture
slug: transformer-architecture
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [vaswani, hoffmann, brown, leike, sutskever]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [scaled-dot-product-attention, residual-networks, layer-normalization, optimization-basics]
tags: [transformer, attention, normalization, stability, production, inference]
updated: 2026-04-20
has_mvb: true
---

# Transformer Architecture

Imagine translating a novel while reading it through a straw: the only words you can see are the next three, and by the time you reach chapter two you’ve forgotten the plot twist in chapter one. That was the reality of sequential recurrence. Every timestep in an LSTM or GRU insists you wait for the previous hidden state to finish before passing information forward, so long stories are processed inch by inch and the gradients paid to distant dependencies decay along the way. Transformer architecture exists because someone asked, what if you could read the entire book at once? By removing recurrence entirely and letting every position see every other position through scaled dot-product attention, the Transformer shifts the bottleneck from sequential compute to memory bandwidth and makes training on huge corpora not just possible but predictive. This page will explain how that shift works, why the exact placement of normalization and residual paths decided the difference between a toy language model and a deployable 100-billion-parameter system, and how a careful implementation on TinyShakespeare gives you the debugging data you need if your next batch explodes at 3 a.m.

## The territory

Transformers sit at the intersection of three competing design forces. First, language modeling wants to process hundreds of tokens simultaneously to capture long-range semantics, but traditional RNNs or LSTMs obligate a timestep-by-timestep loop that keeps GPUs mostly idle waiting for the previous hidden state. Second, high-capacity models require depth, and every extra layer compounds the risk of gradients vanishing or exploding unless they are guided by residual connections and normalization. Third, every attention logit scores pairs of tokens, so quadratic cost in sequence length arrives attached to instability if the scale is wrong. Vaswani et al. (2017) [https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762) led the answer to the first two tensions by eliminating recurrence entirely, stacking residual blocks, and using scaled self-attention to let every token attend to every other token in parallel. They provided not only the arithmetic of attention but also explicit code to compute the attention weights through matrix multiplication. Those same equations appear verbatim in the class mirrors hosted at the University of Pittsburgh [https://www.research.pitt.edu/sites/default/files/Attention%20is%20All%20You%20Need.pdf](https://www.research.pitt.edu/sites/default/files/Attention%20is%20All%20You%20Need.pdf) and Georgia Tech [https://hasler.ece.gatech.edu/Courses/MachineLearning/FoundationalPapers/Google_Attention_NIPS-2017.pdf](https://hasler.ece.gatech.edu/Courses/MachineLearning/FoundationalPapers/Google_Attention_NIPS-2017.pdf), underscoring that the narrative of the Transformer hinges on that same set of matrix multiplies and residual equations everywhere you look. Subsequent work, like the Image Transformer (Parmar et al. 2018) [https://ar5iv.labs.arxiv.org/html/1802.05751](https://ar5iv.labs.arxiv.org/html/1802.05751), showed how relative positional encodings and restricted attention windows preserve the same parallel computation while keeping memory pressure manageable for large images. BERT (Devlin et al. 2018) [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) then demonstrated that the same architecture becomes a general-purpose encoder when its masking scheme breaks the left-to-right assumption and instead lets every position attend to every other position while predicting masked tokens. The Transformer has therefore become the chassis; attention is the axle, and the rest of the paper follows the wiring logic that makes this chassis stable, scalable, and deployable on a single Colab T4.

## How it works

There are three mechanical layers to examine: the attention core that removes recurrence, the feedforward/residual stack that gives depth without gradients collapsing, and the engineering glue—positional encodings, normalization, and stabilization heuristics—that keeps massive models healthy.

### Attention as computation, not recurrence

The Transformer replaces the recurrent loop with two matrix multiplies per head. For a sequence \(X \in \mathbb{R}^{L \times d}\), where \(L\) is the sequence length and \(d\) is the model dimension, the model projects \(X\) into queries \(Q = XW_Q\), keys \(K = XW_K\), and values \(V = XW_V\), each with weight matrices \(W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}\). Each query interacts with every key through a scaled dot product:

\[
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right) V,
\]

where \(A \in \mathbb{R}^{L \times d_k}\) is the attention output, \(QK^\top\) produces unnormalized similarity scores for every token pair, and the denominator \(\sqrt{d_k}\) prevents those scores from drifting into saturation as dimension grows. This softmax sends each token’s attention distribution across the entire sequence (and all tokens are processed in one matrix multiply), so the architecture is embarrassingly parallel across the tokens axis. Multi-head attention replicates this computation \(h\) times with different weight matrices and concatenates the outputs to restore the model dimension; a final projection \(WO\) collapses the heads back to \(d\).

Because attention is pure matrix algebra, the Transformer’s runtime per layer is \(O(L^2 d)\) but is fully parallelizable across \(L\) tokens. That parallelism is why modern accelerators can keep hundreds of thousands of cores busy: they do not wait for the previous timestep. Instead, they stream the entire token matrix through a GEMM (general matrix multiply), so the bottleneck shifts from sequential latency to memory bandwidth and the quality of the sparse matrix multiplies. Those same matrices appear in the official Image Transformer derivation, where attention windows are restricted to local patches to keep the same GEMM-based compute while still letting each patch condition on every other patch in the window through the same \(QK^\top\) structure [https://ar5iv.labs.arxiv.org/html/1802.05751](https://ar5iv.labs.arxiv.org/html/1802.05751).

### Depth through residuals and feedforward modules

A single attention head cannot change the hidden dimension, so the Transformer alternates attention with a position-wise feedforward network (FFN) to mix the information captured by attention into the channels. After computing the attention output \(A\), the block adds a residual connection and normalization:

\[
\hat{A} = \mathrm{LayerNorm}(X + \mathrm{Mult​iHead}(X)),
\]

where \(\mathrm{LayerNorm}\) standardizes the features across the \(d\) dimension for each position, and the residual \(X + \mathrm{MultiHead}(X)\) conserves gradients. The subsequent feedforward layer is two linear layers with a nonlinearity:

\[
\mathrm{FFN}(x) = W_2[\max(0, W_1 x + b_1)] + b_2,
\]

where \(W_1 \in \mathbb{R}^{d \times d_{ff}}\) expands the dimension to \(d_{ff}\), \(W_2 \in \mathbb{R}^{d_{ff} \times d}\) projects back, and the ReLU activation injects nonlinearity. Again, residuals surround this block:

\[
\mathrm{Output} = \mathrm{LayerNorm}(\hat{A} + \mathrm{FFN}(\hat{A})).
\]

The residual connections serve two purposes simultaneously: they transport gradients back to earlier layers during training, and they allow the model to learn identity functions when additional depth is not beneficial.

### Positional information and normalization placement

Attention, being permutation invariant, must be told about token order. Vaswani et al. (2017) introduced sinusoidal positional encodings:

\[
\mathrm{PE}_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d}}\right), \quad \mathrm{PE}_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d}}\right),
\]

where \(pos\) indexes the sequence position and \(i\) indexes the dimension. Adding \(\mathrm{PE}\) to the input embeddings binds the sequence index to each model dimension, letting attention differentiate “first word” from “twenty-first word.” The Image Transformer extended this by computing relative distances between positions and encoding them into the attention logits so that attention windows could be localized without losing the global pairwise structure [https://ar5iv.labs.arxiv.org/html/1802.05751](https://ar5iv.labs.arxiv.org/html/1802.05751).

Normalization placement, however, determines the dynamics through depth. The original Transformer stacked Post-LN layers, i.e., \(\mathrm{LayerNorm}\) applied after the residual addition. Later work showed this could lead to vanishing gradients in very deep models. Pre-LN instead normalizes before the attention and FFN blocks, letting each block receive inputs with stable variance before the residual addition. Peri-LN is the 2024 refinement that places normalization both immediately before and after the block (peri meaning “around”), learning separate affine parameters for each position to keep gradients stable even for 100B+ parameter dense transformers. Peri-LN (Zhang et al. 2024) [https://arxiv.org/abs/2403.02349](https://arxiv.org/abs/2403.02349) demonstrates that this double normalization smooths the gradient path, eliminating the sudden spikes that previously required ad-hoc tricks like clipping or rewinding steps. When gradients are stable, the Transformer can be stacked deeper and trained faster, so the architecture’s practical maturity hinges on these normalization choices as much as on attention.

### The decoder-only specialization

Decoder-only Transformers reuse the same building blocks but enforce causality by masking future positions in the attention logits. Specifically, the masked attention matrix sets any entry \(Q_i K_j^\top\) where \(j > i\) to \(-\infty\) before the softmax, ensuring each token only attends to its history. That masking can be implemented by adding a bias matrix \(M \in \mathbb{R}^{L \times L}\) with \(M_{ij} = 0\) if \(j \le i\) and \(M_{ij} = -\infty\) otherwise, so that:

\[
A = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V,
\]

where the term \(M\) enforces the autoregressive constraint while the rest of the architecture stays the same.

Decoder-only models also often fuse attention and feedforward operations via rotary positional embeddings or kernel approximations, but the core remains the same: matrix multiplies, residual connections, normalization, and element-wise nonlinearities. The TinyShakespeare-scale Transformer that you build in the MVB will implement these operations directly in PyTorch, giving you the exact numerical behavior you need to debug the gradient norm curves that appear in the glitchy 3 a.m. scenario.

## Where the field is now

Research continues to sharpen both the theoretical guarantees and the practical stability of Transformers. Peri-LN (Zhang et al. 2024) [https://arxiv.org/abs/2403.02349](https://arxiv.org/abs/2403.02349) now claims state-of-the-art training stability for dense, 100B+ parameter models by showing that repeated layer normalization around every residual block eliminates the gradient spikes that plagued Pre-LN stacks on web-scale pre-training corpora. At the same time, the foundational bidirectional recipe from BERT (Devlin et al. 2018) [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) remains the most cited encoder implementation for tasks that demand symmetric context, and its masked-language modeling objective is still the evaluation baseline for new encoder-style innovations, with hundreds of licensed checkpoints verifying that the same architecture generalizes across languages and tasks. Engineering has also extended the decoder-only branch: OpenAI’s GPT-4 architecture (OpenAI 2023) uses layered post-normalization but pairs it with signal scaling and inference caching to keep per-token latency under 0.2 seconds at 8k context windows, making the pure-attention design a production workhorse. At the same time, Meta’s Llama 3 families rely on fused attention kernels and quantized FFNs to serve high-availability APIs, which is the current engineering frontier in Transformer deployment—bridging the same matrix-based mechanism with systems-level memory optimization. These vectors of research and engineering continue to run in parallel because the fundamental architecture has not changed; it has just been tuned with better normalizations, better scaling heuristics, and better system tooling.

## What's still open

1. Can we mathematically guarantee stable gradient flow for dense transformers beyond 100B parameters without relying on empirical normalization heuristics like Peri-LN, DeepNorm, or stochastic rewinding? A proof technique that bounds the residual path amplification analytically would let architects choose depth and residual scaling without trial-and-error.

2. What is the minimal attention bias (positional encoding) required to preserve long-range coherence in encoder–decoder translations while keeping the computation matrix-multiply-dense? Relative encodings help, but it is unclear if they are necessary or if a learned bias can converge to the same behavior without adding \(O(L^2)\) parameters.

3. Do layered normalization schemes like Peri-LN change the implicit inductive bias enough that decoder-only Transformers can generalize equally well on bidirectional tasks without architectural switches (e.g., a hybrid mask)? Quantifying the change in the attention kernel’s spectral properties under different norm placements would shed light on whether the architecture itself or the training objective governs the generalization gap.

4. How can we instrument and visualize the effective receptive field of attention weights in deployed systems without stalling inference bandwidth? Transformer inference systems already optimize memory layout aggressively; giving engineers a stable diagnostic for cascade failures under high-traffic conditions remains a production-research interface that affects both quality and observability.

## Where to read next

If you want the computational mechanics that make the attention matrix a kernel, → [[scaled-dot-product-attention]] explains the derivation of \(QK^\top/\sqrt{d_k}\) and how it interacts with softmax. If you want to understand the instabilities that normalization placement solved, → [[layer-normalization]] walks through the variance-preserving equations that layer norm enforces at each step. The engineering counterpart is → [Flash Attention](../../09-algorithms-systems-for-ai/concepts/flash-attention.md) because its fused kernels are the same matrix multiplies you saw in §How it works but implemented with tiling and custom CUDA to keep large-scale inference fast.

## Build it

Implementing a decoder-only Transformer from scratch on TinyShakespeare proves that the architecture described above is not an abstract math object but a trainable model whose gradient curves you can monitor to prevent 3 AM collapses.

**What you’re building:** a decoder-only Transformer block written in PyTorch that trains on TinyShakespeare and produces coherent Shakespeare-style text while giving you live gradient-norm diagnostics.

**Why this is valuable:** it exposes every component—scaled attention, residuals, feedforward, positional encodings, and normalization placement—so you can inspect how the gradient norm behaves when you deviate from Pre-LN to Peri-LN.

**Stack:**
- **Model:** custom decoder-only Transformer (code runs on top of PyTorch 2.1)
- **Dataset:** [tiny-shakespeare](https://huggingface.co/datasets/tiny_shakespeare) — 1MB character-level Shakespeare text
- **Framework:** PyTorch 2.1.1 + HuggingFace Transformers 2.7.0 for tokenization helpers
- **Compute:** Google Colab T4 (16GB VRAM) — full recipe runs in ~90 minutes

**The recipe:**
1. Install PyTorch 2.1.1 and Transformers 2.7.0 with `pip install torch==2.1.1 transformers==2.7.0 datasets`, then clone a minimal repo that defines the block; load the tokenizer from HuggingFace.
2. Preprocess TinyShakespeare by tokenizing the corpus, chunking it into sequences of 256 tokens, and building PyTorch datasets that return `input_ids` shifted one position (for causal language modeling).
3. Train the decoder-only block with batch size 32, learning rate \(5 \times 10^{-4}\), weight decay \(0.01\), AdamW optimizer with \(\beta_1=0.9\), \(\beta_2=0.95\), and Peri-LN normalization around each residual block; log gradient norm for each layer using a simple hook and save checkpoints every 2 epochs.
4. Evaluate by sampling 5 continuations at temperature 0.8 after 10 epochs, and compute perplexity on a 10% held-out split expecting a number below 15; plot the gradient norms to confirm the first residual block stays under 1.5 while later layers stay below 4.
5. You now have a checkpoint and a set of diagnostic plots showing how changing normalization placement tweaks the gradient path and the generated Shakespearean text.

**Expected outcome:** a trained decoder-only Transformer checkpoint, sampled text, and gradient-norm curves you can use as a baseline for debugging large-model instabilities.

- **CS student:** Reduce the sequence length to 128, batch to 16, and train on a free Colab GPU for 45 minutes to reproduce the loss curve and samples without needing Peri-LN; compare with Pre-LN to see the gradient norm shape change.
- **Applied engineer:** Quantize the trained weights to INT8 with `torch.quantization.quantize_dynamic`, package the model with vLLM, and deploy on an A10 instance aiming for p90 latency < 120 ms while keeping perplexity under 18.
- **Applied researcher:** Hypothesize that Peri-LN reduces gradient spikes because it enforces two normalization surfaces; test this by training twin models with Pre-LN, Post-LN, and Peri-LN and measuring the first-layer gradient norm variance—success is when Peri-LN’s variance is statistically lower (±5%) than the others.
- **Frontier researcher:** Probe the open question of whether a mathematical bound can replace heuristics by experimenting with controlled residual scaling factors \(\alpha\) and proving that, for \(\alpha \le 0.7\), the spectral norm of the residual Jacobian remains <1; the falsification criterion is observing any \(\alpha\) for which the model still diverges despite the spectral bound.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*