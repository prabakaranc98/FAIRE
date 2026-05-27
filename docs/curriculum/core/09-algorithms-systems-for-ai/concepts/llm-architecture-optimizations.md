---
title: LLM Architecture Optimizations
slug: llm-architecture-optimizations
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [vaswani, barroso, armbrust, gentry, liu]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [attention, sparse-attention-patterns, memory-hierarchies]
tags: [llms, sparse-attention, memory-hierarchy, hardware-software-codesign, kv-cache, scaling-laws]
updated: 2025-02-22
has_mvb: true
---

# LLM Architecture Optimizations

Imagine issuing a single long-context query to a decoder-only model and watching every GPU tick as the KV cache balloons to several gigabytes while the compute units idle waiting for memory. The query completes not because the model needs more FLOPs, but because the cache, spread across HBM and DRAM, cannot be streamed fast enough; each new token has to read the entire set of past key/value vectors before the softmax denominator even awakens. That choking effect is the memory wall of modern LLM inference—attention still costs \(O(L^2)\) memory even if the arithmetic can be pruned out, and every architecture optimization from now on must treat storage, bandwidth, and parameter layout as a joint optimization surface. By the end of this page you will see how recent work restructures layers, re-parameterizes weights, and compresses KV storage so that throughput rises without retraining from scratch, and you will be ready to build a light post-training pruning + distillation flow that proves these principles on a real small model.

## The territory

Early transformer pretraining, such as BERT (Devlin et al. 2018) [https://arxiv.org/pdf/1810.04805v1], taught the community that scaling depth and width together unlocks richer contextual representations, but it also concretely anchored every subsequent system to a uniform architecture: fixed \(Q/K/V\) dimensions, dense attention, and a one-size-fits-all feed-forward block. When the model grew to half a trillion parameters, the parameter tensors were no longer the issue—the KV cache and inference buffers became the limiting factor. Attention computes the weight matrix \(A = \text{softmax}(QK^\top / \sqrt{d_k})\), where \(Q\) and \(K\) have query and key vectors of dimension \(d_k\); the auto-regressive decoder must see every cached \(K\) and \(V\) pair for every new token, and the cached tensors must live on the fastest storage available because each step touches them. The practical consequence is that memory and bandwidth rather than FLOPs determine how large a batch and how long a sequence you can reasonably run at 40 ms latency. That observation moves the reward from inventing new activation functions to reshaping the architecture after pretraining—pruning heads, replacing attention kernels, and moving the cache resident to cheaper storage tiers.

This page sits in the field of LLM infrastructure, where sparse attention, memory hierarchies, and parameter-efficient finetuning now converge. Instead of naively scaling layers uniformly, the techniques here ask: can parts of a decoder be restructured, pruned, or parameterized after pretraining so that each layer consumes less cache, fewer parameters, or both, while still matching downstream accuracy? The territory therefore includes KV-cache compression, post-training neural architecture search, structured pruning, and parameter-efficient adapters. How does it actually work? The mechanism is best understood by starting from the inference bottleneck—KV cache storage and movement—and revealing how each optimization eventually reduces the amount of data shuttled across the memory hierarchy without redoing the entire pretraining investment.

## How it works

The bottleneck for autoregressive inference is not the number of parameters but the mobility of the KV cache. A single decoder layer stores, for every past token \(t\), a key vector \(k_t \in \mathbb{R}^{d_k}\) and a value vector \(v_t \in \mathbb{R}^{d_v}\). When the sequence length \(L\) grows, the cached tensors \(K \in \mathbb{R}^{L \times d_k}\) and \(V \in \mathbb{R}^{L \times d_v}\) occupy \(O(L)\) memory per layer, so the access pattern becomes a high-bandwidth streaming problem rather than a compute-heavy dense matmul. Moving those tensors between cache levels—tightly-coupled SRAM, high-bandwidth HBM, slower DRAM, or even CPU memory—causes latency spikes. The first strategy reduces the size of \(K\) and \(V\) via structured pruning of attention heads.

### Head and block pruning

Head pruning starts with the observation that not every head contributes equally to the softmax output. The softmax weights depend on \(QK^\top\), so if a head consistently produces near-uniform attention, its contribution is effectively noise. During a pruning pass, one evaluates a utility score \(U_h\) for head \(h\) as the KL divergence between the logits produced with and without head \(h\). Removing the head yields a new attention weight \(A_{-h} = \text{softmax}((Q_- K_-^\top)/\sqrt{d_k})\), where \(Q_-\) and \(K_-\) drop the pruned head's contribution. The new cached tensors shrink by \(d_k\) per head. To keep accuracy, pruning must be coupled with either magnitude-aware fine-tuning or knowledge distillation: e.g., the original \(p_{\text{teacher}}(y|x)\) acts as soft targets when training the pruned student, and the distillation loss \(\mathcal{L}_{\text{KD}} = \text{KL}(p_{\text{teacher}} \| p_{\text{student}})\) ensures the student still mirrors the teacher’s multi-step reasoning.

Pruning often leaves the model susceptible to “lazy learning,” where some layers see vanishing gradients because their outputs were scaled down in expectation. Parameterization strategies restore gradient flow without re-training the whole model. The parameter-efficient transfer learning framework introduced by Houlsby et al. (2019) [https://arxiv.org/pdf/1902.00751] inserts trainable adapter layers, re-scaling layer outputs without touching millions of frozen weights. LoRA (Hu et al. 2021) [https://arxiv.org/pdf/2106.09685.pdf] recasts a weight increment \(\Delta W\) as two low-rank matrices \(A \in \mathbb{R}^{d \times r}\) and \(B \in \mathbb{R}^{r \times k}\) so that \(W' = W + BA\) and only \(BA\) is trained while \(W\) remains frozen; when \(r \ll \min(d,k)\), the number of trainable parameters shrinks dramatically, allowing fine-tuning on a single GPU. This re-parameterization can also stabilize pruned models by providing lightweight shortcuts that the optimizer can adjust to offset the missing heads.

### Cache compression and caching tiering

After pruning, the remaining KV cache is still the largest resident data structure, but the observation from SentenceKV-style compression is that much of the information in \(K, V\) is redundant across nearby tokens. You can project local windows onto a smaller sentence-level embedding \(s_j = \phi(T_j)\), where \(T_j\) contains the tokens for sentence \(j\), and store only \(s_j\) in GPU memory while keeping the raw token vectors on CPU. Retrieval during inference then becomes a two-stage process: (1) fetch the relevant \(s_j\) from GPU and (2) reconstruct the detailed \(K, V\) by streaming from CPU only when necessary. The approximation naturally introduces quantization error, so the reconstruction can take the form \(K_t \approx W_k s_j + b_k\) with trainable matrices \(W_k, b_k\), and the same for \(V_t\). Distillation against the original cache ensures these approximations do not degrade perplexity. This tiered caching technique trades store-read bandwidth for a lightweight linear layer per sentence, enabling sequences twice as long for the same HBM footprint.

### Post-training architecture search

Post-training neural architecture search, such as NVIDIA’s Jet-Nemotron (Wang et al. 2025) [https://arxiv.org/abs/2508.15884], swaps entire high-resolution attention blocks with cheaper variants after pretraining. The search inspects each layer, measures its sensitivity (e.g., the increase in perplexity when switching the layer to a linear attention kernel), and applies the swap only where the accuracy drop is within a small epsilon. The new kernel might compute the attention weights \(\tilde{A} = \text{softmax}(QW^1 (K W^2)^\top / \sqrt{d_h})\) with \(W^1, W^2\) learned low-rank projections, reducing the internal cache storage because the projections compress \(K\) before it enters the softmax. Because the swap happens post-training, the search relies on a light evaluation loop on a held-out validation set rather than full retraining: freeze all weights except the new \(W^1, W^2\), run a few epochs of distillation using outputs from the original attention blocks, and only keep the new architecture if the validation loss stays within a preset margin.

### Distillation without full retraining

Distillation itself serves both to recover accuracy and to regularize the pruned/modified backbone. Hinton’s “Outrageously Large Neural Networks” (Hinton et al. 2015) [https://www.cs.toronto.edu/~hinton/absps/Outrageously.pdf] showed that even when the student model is much smaller, the softened logits from the teacher reveal richer training signals than hard labels. When you prune or replace layers post-training, the teacher is the original checkpoint and the student is the modified one. The distillation loss keeps the new architecture aligned with the teacher’s multi-step reasoning, which is essential because structural pruning can otherwise remove the very degrees of freedom needed to reason through long contexts. Mathematically, the softened logits are computed as \(z_i = \frac{\exp((W_i x)/T)}{\sum_j \exp((W_j x)/T)}\), where \(T\) is the temperature, \(W_i\) are the final layer weights, and \(x\) is the hidden state; training the student to match these logits encourages it to interpolate the teacher’s conditional distributions.

### Putting it all together

The optimization surface therefore includes three knobs: (1) head/block pruning to reduce \(d_k\), (2) parameter-efficient adapters/LoRA matrices to preserve gradient flow, and (3) cache compression or PostNAS-led kernel swaps to reduce the size of \(K, V\) without retraining. Each knob affects both compute and memory. The combined strategy is to prune, inject parameter-efficient trainables, compress the cache into sentence-level vectors that can be streamed, and finally, distill to recover accuracy. The interplay between memory footprint and accuracy is what distinguishes these architecture optimizations from earlier scale-based research: once your inference is bound by cache movement, improving the arithmetic doesn’t help until the cache itself shrinks or becomes cheaper to access.

## Where the field is now

Jet-Nemotron (Wang et al. 2025) [https://arxiv.org/abs/2508.15884] represents the current research frontier: NVIDIA showed that PostNAS can swap 12 of 24 attention layers to linear kernels after pretraining and still match quality on MT-Bench while delivering a 53.6× generation throughput speedup and a 6.1× prefilling speedup, all without retraining from scratch. The paper benchmarks on a suite of reasoning and coding tasks, and the linear-swapped layers use learned low-rank projections that compress the cached key tensors before softmax, so each increment in throughput is directly tied to reduced memory movement. That result frames the central research challenge—how to prove that these post-training swaps do not undermine the intricate multi-step reasoning behaviors that emerged in the dense model.

The engineering frontier is roughly where SentenceKV-style systems are entering production at scale. In these deployments, models store sentence-level semantic summaries in GPU memory while pushing raw token vectors to CPU or even NVMe. Retrieval then reconstructs the necessary \(K, V\) pairs from summaries with a micro-layer of linear transformations, which cuts HBM usage while keeping the GPU busy with attention since the reconstruction proceeds in a fused kernel. Production inference engines combine this compression with quantized activations and Triton kernels that reorder the cache layout to maximize burst bandwidth, letting them serve 13B+ models at sub-100 ms latency using a single A100 while maintaining 93–95% of the dense accuracy reported in the evaluation suite.

Together these fronts show that optimizing LLM architecture is a systems math problem: the new layers and caches must be both memory-aware and programmable in frameworks like Triton, FlashAttention, or DeepSpeed’s inference pipelines so that pruned architectures can still run efficiently on commodity accelerators.

## What's still open

Can we predict, from the structure of the teacher and the pruning mask, whether the post-trained student will preserve emergent capabilities such as multi-hop reasoning without running large benchmark suites? Existing validation loops rely on evaluating a handful of tasks, which is expensive; a theoretical guarantee would let infrastructure teams prune with confidence before deployment.

How can cache compression be made adaptive to token semantics in real time—switching between fine-grained KV storage for reasoning-heavy spans and coarse-grained sentence summaries for boilerplate content—without introducing oscillations and latency spikes?

Is there a principled way to schedule the PostNAS layer swaps (attention-to-linear, dense-to-sparse) so that stability in activation norms is preserved across the network, preventing “lazy learning” where early layers stop updating because downstream gradients vanish due to the nonlinear swaps?

## Where to read next

If you want a deeper look at how flash kernels keep those compressed caches moving, → [Flash Attention](flash-attention.md) explains the Triton-level rearrangements that make fused attention possible at low bandwidth. For the broader system-level memory story, → [[memory-hierarchies]] traces how HBM, DRAM, and CPU caches must be co-designed with KV compression schemes. If you need the sparse-attention background that motivates these optimizations, → [[sparse-attention-patterns]] lays out the families of masks and kernels that we now prune, replace, or distill. Finally, the parameter-efficient adapter story in → [[adapters]] connects the LoRA and Houlsby-style insertions we use to stabilize pruned layers.

## Build it

This build lets you feel the memory wall: you will prune a TinyLlama-1.1B checkpoint, remap its remaining heads into a compressed cache tier, and distill back to regain perplexity on WikiText-2.

**What you're building:** A post-training optimization script that prunes redundant attention heads, compresses the remaining KV cache into sentence-level vectors, and distills TinyLlama-1.1B on WikiText-2 in under two hours on a Colab T4.

**Why this is valuable:** Because the artifact occupies a handful of GBs of cache instead of tens of GBs, you can run long-context inference on a single GPU, proving that structural pruning + cache compression works without retraining from scratch.

**Stack:**
- **Model:** [OpenAssistant/tinylama-1.1B](https://huggingface.co/OpenAssistant/tinylama-1.1B) — 3.8k downloads
- **Dataset:** [wikitext](https://huggingface.co/datasets/wikitext) (WikiText-2 subset) — lightweight, well-documented
- **Framework:** Diffusers + transformers 4.42 + bitsandbytes 0.40 (for quantized checkpoints)
- **Compute:** Free Google Colab T4 (16 GB VRAM) — expect ~90 minutes total

**The recipe:**
1. Install + load: `pip install transformers==4.42 diffusers bitsandbytes torch==2.2` then use `from transformers import AutoModelForCausalLM, AutoTokenizer`.
2. Data: tokenize WikiText-2 with the TinyLlama tokenizer, then create sentence-level batches of ~64 tokens and compute sentence embeddings via a frozen RoBERTa encoder to seed cache compression.
3. Train/fine-tune: apply magnitude-based head pruning until 25% of heads are removed, then insert LoRA-style low-rank adapters (rank 8) around the attention outputs, and distill against the full TinyLlama logits with temperature 2. The loss should drop from ~3.4 to ~3.1 cross-entropy over ~3 epochs.
4. Evaluate: generate on the WikiText-2 validation split and measure perplexity; aim for ≤3.5, within 5% of the original checkpoint despite the pruning.
5. What you now have: a compressed TinyLlama checkpoint plus a cache-reconstruction module that can be loaded into Triton or vLLM for fast inference with reduced KV footprint.

**Expected outcome:** A distilled TinyLlama-1.1B checkpoint that runs long-context decoding on a single T4 with reduced memory, demonstrating the impact of post-training architecture tweaks.

- **CS student:** Run the same recipe on RTX 4070 without LoRA (just pruning + cache compression) and compare the perplexity curves to see how much LoRA adds to stability.
- **Applied engineer:** Quantize the distilled checkpoint with bitsandbytes 8-bit and serve it through vLLM, targeting p95 latency ≤120 ms at batch size 2.
- **Applied researcher:** Hypothesis: sentence-level cache compression performs as well as token-level caching for non-repetitive text; test this by measuring perplexity on two WikiText splits and plotting the ratio of cache reconstruction error vs. bit savings.
- **Frontier researcher:** Investigate the open question whether PostNAS swaps undermine emergent multi-step reasoning by designing a probe dataset; treat the distilled TinyLlama as the pruned student and compare its output consistency against the dense teacher, measuring failure cases.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*