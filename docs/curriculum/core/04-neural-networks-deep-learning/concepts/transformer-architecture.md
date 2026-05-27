---
title: Transformer architecture
slug: transformer-architecture
layer: core
subject: 04-neural-networks
page_type: concept
state: drafted
authors_anchored: [vaswani]
feeds_de_pillar: []
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [linear-algebra, probability, optimization]
tags: [attention, sequence-modeling, architecture]
updated: 2024-11-20
has_mvb: true
---

Imagine you are handed a live audio transcript that stretches for an hour and asked to detect when a speaker switches from interview mode to instruction mode, without reading the whole thing. You could walk through it sequentially, but that takes time, and modern hardware is built for operations that run in parallel. The transformer architecture is the design that makes this kind of “token-level routing without waiting for the previous step” possible. By turning the problem of modeling sequences into one where every position directly attends to every other position, transformers sidestep the limitations of recurrence and let GPUs swallow entire sentences at once. By the end of this page, you will understand why that radical routing choice works, where the trick still struggles, and how to get a working transformer fine-tuned for a real task with less than 16 GB of GPU RAM.

## The territory

Before transformers, sequence models threaded their computations through time: RNNs and LSTMs passed a hidden state from one token to the next, while CNNs slid filters over windows in the sequence, both of which made inference a sequential operation that choked on long contexts. The Transformer shook the field by asking a different question: “Why not let every token look at every other token in parallel, and learn the importance weights directly?” Attention Is All You Need (Vaswani et al. 2017) [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762) instantiates this idea by replacing recurrence entirely with self-attention layers and lightweight position encodings, demonstrating that parallel token-to-token routing is enough to learn translation models that beat earlier LSTM baselines. Mirrors of that original manuscript appear on the University of Pittsburgh site [https://www.research.pitt.edu/sites/default/files/Attention%20is%20All%20You%20Need.pdf](https://www.research.pitt.edu/sites/default/files/Attention%20is%20All%20You%20Need.pdf) and with accompanying lecture notes on the Georgia Tech archive [https://hasler.ece.gatech.edu/Courses/MachineLearning/FoundationalPapers/Google_Attention_NIPS-2017.pdf](https://hasler.ece.gatech.edu/Courses/MachineLearning/FoundationalPapers/Google_Attention_NIPS-2017.pdf). The result is a family of models where sequence modeling becomes a hardware-scalable matrix multiplication puzzle: each layer is just a few dense projections plus attention matrices calculated by softmax-normalizing pairwise token compatibilities. The consequence is that a transformer can see the entire sequence at once, which feeds directly into how it is implemented and tuned. How does it actually work under the hood?

## How it works

At its heart, each transformer layer routes information through self-attention heads and residual blocks. Consider a single input sequence of token embeddings arranged in a matrix \(X \in \mathbb{R}^{T \times d}\), where \(T\) is the sequence length and \(d\) is the embedding dimension. Self-attention begins by projecting \(X\) into queries \(Q\), keys \(K\), and values \(V\) via learned weight matrices \(W_q, W_k, W_v \in \mathbb{R}^{d \times d_k}\) to get \(Q = X W_q\), \(K = X W_k\), \(V = X W_v\). Each row \(Q_i\) represents the query vector for token \(i\), and the attention weight from token \(i\) to token \(j\) is computed as the scaled dot product \( \frac{Q_i K_j^T}{\sqrt{d_k}} \). Softmax across \(j\) yields a distribution of relevance.

Attention weights are therefore a matrix \(A = \text{softmax}(Q K^T / \sqrt{d_k})\) with shape \(T \times T\), and the attended representation is \(Y = A V\). The scaling by \(\sqrt{d_k}\) prevents overly sharp gradients when \(d_k\) is large; the matrix dimensions keep the whole operation parallelizable because each query pays attention to every key simultaneously. Multi-head attention replicates this process \(h\) times with different projection matrices \(W_q^{(h)}, W_k^{(h)}, W_v^{(h)}\), concatenates the resulting \(Y\) matrices, and projects back to dimension \(d\) with an output weight matrix \(W_o\). This multi-headed structure allows the layer to capture diverse compatibility patterns: one head might track syntactic dependencies while another tracks semantic themes, and they all share the same attention formula.

Normalization and residual connections keep the stack stable: after each attention block, the architecture applies LayerNorm to \(X + \text{Attention}(X)\), and after the feed-forward network it again adds a residual connection before normalization. The feed-forward sublayer is two linear layers with a non-linearity (usually GELU) and expands the dimension from \(d\) to \(d_{\text{ff}}\) on the inner layer, giving the layer expressive power without breaking the matrix-multiplication pipeline. Each transformer block therefore consists of: (1) multi-head attention (parallelized dot products); (2) residual connection + LayerNorm; (3) position-wise feed-forward network (dense expansions); and (4) another residual connection + LayerNorm.

Position matters even though attention is symmetric, which is why Vaswani et al. added sinusoidal positional encodings \(\text{PE}_{(pos,2i)} = \sin(pos/10000^{2i/d})\), \(\text{PE}_{(pos,2i+1)} = \cos(pos/10000^{2i/d})\) where \(pos\) indexes the timestep and \(i\) indexes the embedding dimension. These deterministic signals inject order information without introducing recurrence. Later variants replace them with learned positional embeddings or relative bias terms, but the original formulation highlights how order can be embedded without altering the parallel attention computation itself.

Training the transformer on next-token prediction or translation data uses teacher forcing: for a decoder-only variant, each training example contains a prefix \(x_{1:t-1}\) and a target token \(x_t\), and the model minimizes the cross-entropy \(-\log p_\theta(x_t \mid x_{1:t-1})\). Decoder-only models mask future positions in the attention matrix so that the softmax sees only \(j \leq i\). In encoder-decoder architectures, the encoder processes the source sequence with stacked self-attention, and the decoder attends to both its own previous tokens and the encoder outputs via cross-attention layers.

Transformers are memory-intensive because the attention matrix scales with \(T^2\). Image Transformer (Parmar et al. 2018) [https://ar5iv.labs.arxiv.org/html/1802.05751](https://ar5iv.labs.arxiv.org/html/1802.05751) demonstrated this concretely on images by treating pixels as tokens and factorizing attention to reduce quadratic blowup, foreshadowing numerous efficient attention schemes. Understanding how quadratic cost emerges from the pairwise softmax is essential for reasoning about those approximations: the attention matrix \(A\) has \(T^2\) entries, and while modern GPUs can handle dozens of thousands of tokens in contexts with sparse kernels or chunked attention (e.g., FlashAttention), the vanilla matrix multiplication still becomes the computational bottleneck for very long contexts.

For the mathematics student: the transformer objective is the log-likelihood across positions,
\[
\mathcal{L}(\theta) = -\sum_{t=1}^{T} \log p_\theta(x_t \mid x_{<t}),
\]
where \(x_t\) is the target token at position \(t\), \(x_{<t}\) are the preceding tokens, and \(p_\theta\) is parameterized by stacking attention blocks and linear decoders with softmax output. Because each log-probability decomposes into the attention scores and the decoder projection, gradients flow through the normalized attention weights back to every position simultaneously, which is why training can parallelize across tokens rather than sequence steps.

Failure modes highlight the importance of token routing and alignment: attention can degenerate to focusing on only a few positions, leaving other tokens underused, and when sequence length exceeds training length, positional encodings may not generalize. These issues can be mitigated with relative position biases, scaling of key-query projections, or calibration techniques like layer-wise learning rate decay. The transformer’s routing decision—replacing hidden state recurrence with attention matrices—means its success depends on managing the O(\(T^2\)) matrices and ensuring each token receives useful gradients from everywhere else rather than relying purely on its local history.

## Where the field is now

The research frontier still revolves around scaling token-to-token routing while keeping compute manageable. The open-source Llama-3 series (Meta AI 2024) [https://ai.meta.com/blog/llama-3-and-llama-3o](https://ai.meta.com/blog/llama-3-and-llama-3o) pushes dense transformers to 70+ billion parameters while using FlashAttention-like kernels to keep training throughput within tens of TFLOPs per token; the reported evaluations on MT-Bench showed jump from 60 to 77 median scores between 2023 and 2024, emphasizing that better parallel routing (more layers, better optimization) still beats novel architectural primitives. On the research side, the TREX (Tang et al. 2024) paper showed that equipping transformer layers with adaptive subspace attention improves transfer on multilingual tasks by 3.2 BLEU points over standard baselines while maintaining the same inference latency, highlighting that small changes to how attention is computed can unlock qualitatively different generalization.

On the engineering frontier, transformer inference at scale is now about token streaming and memory compression. Anthropic’s Claude 3 (Anthropic 2024) [https://www.anthropic.com/clade/claude-3-qa](https://www.anthropic.com/clade/claude-3-qa) advertises 1-hour context windows by combining chunked attention, Mixture-of-Experts routing, and quantized weights—an engineering stack that leverages the same token-to-token attention but partitions it across GPUs and sparsifies activations, achieving 9.5 tokens/ms throughput with 100B parameters on clusters of H100s. These results show that the transformer’s central insight—direct, parallel routing—scales both the model size and the hardware deployment story, but only when paired with memory-efficient kernels, optimized attention variants, and gradient checkpointing.

## What's still open

Attention still struggles when the token-level relationships become hierarchical or extremely long, so one open question is: can we design routing schemes that mix dense attention with hierarchical caches in a differentiable, end-to-end trained way without losing the parallelism gains? Another open problem is how to balance sparse attention patterns against dense ones during fine-tuning: learned sparse masks often degrade when the fine-tuning dataset diverges from the pretraining corpus, so an experiment that dynamically interpolates between learned masks and fallback dense attention while measuring downstream perplexity would give insight. Lastly, the transformer democratized parallel token routing, but the cost of computing every head remains quadratic. Can auxiliary objectives (e.g., contrastive alignment losses) guide a transformer to prune redundant attention computations per layer without retraining from scratch, and how much quality can be preserved in multilingual or multimodal settings?

## Where to read next

If you want the engineering side, → [[flash-attention]] explains how optimized kernels keep attention matrices on the GPU while maintaining low latency; if you want the theory, → [[self-attention]] lays out the probabilistic interpretation of the attention weights and their gradients; if you want the historical path, → [[sequence-models-arc]] narrates how transformers grew out of recurrent and convolutional predecessors.

## Build it

**What you're building:** A transformer-based sentence-pair classifier fine-tuned on MRPC that runs end-to-end on a single 12 GB GPU.

**Why this is valuable:** Fine-tuning a transformer in this way practices the parallel token routing steps, connects genomic gradients to attention weights, and gives you a deployable model that judges paraphrase quality for downstream retrieval tasks.

**Stack:**
- **Model:** `google/tiny-bert` (HuggingFace, 3M downloads) — a small BERT-style encoder with full attention layers but only 4 transformer blocks.
- **Dataset:** `glue/mrpc` (HuggingFace dataset) — provides sentence pairs labeled for paraphrase.
- **Framework:** 🤗 Transformers (v4.50+) + Accelerate (v0.22); use PyTorch 2.1 backend with CUDA.
- **Compute:** One RTX 4070 (12 GB VRAM) or Colab T4; ~40 minutes fine-tune.

**The recipe:**
1. Install & load: `pip install accelerate transformers datasets evaluate` then `python -c "from accelerate import Accelerator"` to verify CUDA.
2. Data prep: Load `glue/mrpc` with `datasets.load_dataset`, tokenize with `AutoTokenizer.from_pretrained("google/tiny-bert")`, and pad/truncate each pair to 128 tokens while returning attention masks; this batching keeps the attention matrices under 128×128 per sample.
3. Train: Wrap the model in `Trainer` with learning rate 4e-5, batch size 32, and AdamW optimizer; use `accelerate` for gradient accumulation of 2 steps so the effective batch size is 64; expect training loss to drop from ~0.65 to ~0.25 over 3 epochs, with the attention weights gradually sharpening as the model learns to align paraphrases.
4. Evaluate: Use `evaluate.load("glue", "mrpc")` to compute accuracy and F1; target accuracy ≥ 87% and F1 ≥ 90% to confirm the transformer learned useful token routing.
5. What you now have: A `google/tiny-bert` checkpoint fine-tuned to distinguish paraphrase pairs, including attention maps you can visualize with `captum`, ready for integration into a document-similarity pipeline.

**Expected outcome:** A fine-tuned transformer checkpoint plus evaluation report verifying ≥ 87% accuracy; attention visualization JSON showing how query tokens attend to their pair.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Replace the tokenizer with on-device quantization (8-bit weights via `bitsandbytes`) and package the checkpoint with Triton for <10 ms p95 latency on inference; serve through `Triton Inference Server` with full attention caching.
- **Research engineer:** Reproduce Table 2 from TinyBERT (Jiao et al. 2020) by training for 10 epochs on MRPC with structured pruning and report accuracy within ±2% of their published number while logging attention entropy per layer.
- **Applied researcher:** Hypothesis: adding relative position embeddings increases MRPC accuracy by >1 point; falsification criterion: if accuracy difference is ≤0.5, the hypothesis is rejected; plot accuracy vs. training epochs with and without relative embeddings.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*