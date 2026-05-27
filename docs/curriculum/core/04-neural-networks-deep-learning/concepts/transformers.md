---
title: Transformers
slug: transformers
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [vaswani, devlin, raffel, sutskever, glm-4-5-team]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [sequence-to-sequence-learning, recurrent-neural-networks, positional-encoding, layer-normalization]
tags: [self-attention, parallelization, transformer, foundation-models, sequence-modeling, natural-language-processing]
updated: 2025-06-15
has_mvb: true
---

# Transformers

Imagine reading a thousand-word essay and trying to summarize it while only being allowed to keep one number in mind—the entire history of the story squashed into a single, fixed-length vector. That, more or less, was the design pressure behind the RNNs and LSTMs that dominated sequence modeling. Every new sentence needed to shove its nuance through the same tiny bottleneck, so long-range dependencies frayed and gradients vanished. Then Vaswani et al. (2017) — Attention Is All You Need — [https://arxiv.org/pdf/1706.03762](https://arxiv.org/pdf/1706.03762) (with accessible mirrors at [https://www.research.pitt.edu/...](https://www.research.pitt.edu/sites/default/files/Attention%20is%20All%20You%20Need.pdf) and [https://hasler.ece.gatech.edu/...](https://hasler.ece.gatech.edu/Courses/MachineLearning/FoundationalPapers/Google_Attention_NIPS-2017.pdf)) asked a different question: what happens if every token can glance back at every other token instantaneously, and we throw recurrence out the window? By the end of this page you will know why that insight freed engineers to parallelize training, why bidirectional pre-training doubled the practical effectiveness of those layers, how modern 355B-parameter mixtures-of-experts still obey the same routing logic, and how to concretely build a single-head self-attention block you can train on Colab to demystify queries, keys, and values.

## The territory

Transformers live on the ridge between recurrent sequence models and memory-augmented retrieval. Before their debut, Sutskever et al. (2014) — Sequence to Sequence Learning with Neural Networks — explained how stacking LSTM encoders and decoders could learn translation, but they still had to carry every token through a chain of hidden states. Each step became a sequential dependency, limiting parallelism to the length of the sentence. Transformers answer the same engineering question—modeling the probability of an output sequence given an input sequence—but they do it by replacing recurrence with self-attention, effectively turning the whole sequence into a soft-addressable database where every position can read from any other position at once. This is why scaling laws like those reported in Raffel et al. (2020) — Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer — map cleanly from small encoder-decoder jobs to billion-parameter language models: the mechanism that routes context, not the recurrent loop, is now the bottleneck. How does that routing work beneath the hood?

## How it works

The answer to the bottleneck is a simple set of matrix multiplications that encode “what I want to attend to,” “what each position offers,” and “what I will take away.” The transformer takes a sequence of token embeddings \(X \in \mathbb{R}^{N \times d}\), projects them into three spaces via learned matrices \(W^Q, W^K, W^V \in \mathbb{R}^{d \times d_k}\) (with \(d_k = d_v = d\) in the original formulation), and then matches queries to keys through dot products to compute attention weights.

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

where \(Q = XW^Q\) is the query matrix of dimension \(N \times d_k\), \(K = XW^K\) is the key matrix of dimension \(N \times d_k\), \(V = XW^V\) is the value matrix of dimension \(N \times d_v\), and \(\sqrt{d_k}\) scales the logits so their variance stays near one. The softmax weighs each key relative to the query it is paired with, producing \(N\) distributions that let every token gather a new representation by mixing the entire sequence’s values. This is what makes recurrence unnecessary—the entire input is accessible without waiting for a hidden state to propagate.

A single attention “head” learns one kind of alignment, which is why the transformer multiplies, slices, and concatenates multiple heads in parallel:

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(head_1, \dots, head_h)W^O
\]

where each \(head_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)\) uses its own projection matrices \(W_i^{\{Q,K,V\}}\), \(h\) is the number of attention heads, and \(W^O\) merges the concatenated heads back to dimension \(d\). The key idea is that each head can learn a different alignment pattern—one head can track syntax, another can track related nouns—while the concatenation keeps the final dimension fixed. Because every head reads the entire sequence in one matrix multiplication, the transformer processes \(N\) tokens in parallel, unlike an LSTM that must step through them one at a time.

Layer normalization (Ba et al.) and residual connections wrap each attention block to keep gradients stable, and a position-wise feedforward network \( \text{FFN}(x) = \text{GELU}(xW_1 + b_1)W_2 + b_2 \) adds depth without entangling sequence length. Positional encodings \(\text{PE}(pos, 2i) = \sin(pos/10000^{2i/d})\), \(\text{PE}(pos, 2i+1) = \cos(pos/10000^{2i/d})\) inject order, letting the attention kernels discriminate “earlier” from “later” even though they see every token simultaneously.

Training preserves the parallelism: rather than unfolding through time, the transformer minimizes cross-entropy on the output sequence using teacher forcing or the masked language modeling objective. Devlin et al. (2018) — BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding — showed that masking random tokens and forcing the model to predict them while still seeing both left and right context at every layer yields bidirectional representations that are easily fine-tuned for downstream tasks. The masked language model loss is simply \(\mathcal{L}_{MLM} = -\sum_{t \in M} \log P_\theta(x_t | X_{\setminus t})\) where \(M\) is the set of masked positions, \(x_t\) is the true token, and \(X_{\setminus t}\) is the sequence with \(t\) replaced by [MASK]. Because attention already lets the model see all unmasked tokens simultaneously, there is no longer a directional bias—the network can integrate context from both sides in one pass, which was impossible with unidirectional LSTMs without complicated tricks.

When scaling from BERT-sized models to 355B-parameter giants such as GLM-4.5 (GLM-4.5 Team 2025 — GLM-4.5: Agentic, Reasoning, and Coding (ARC) Foundation Models), engineers still rely on the same attention routing. What changes is the infrastructure around it: mixture-of-experts layers replace dense feed-forward blocks with sparse routers that activate only subsets of experts per token, while MoE gating preserves the quadratic compute of softmax attention but keeps activation costs manageable. The routing logic that the transformer introduced survives precisely because it treats the sequence as a shared context bank rather than a serialized chain.

Parallelism is not the only gain. Because attention attends over the whole sequence, it naturally forms similarity matrices that can be interpreted as soft retrieval. Each row of the attention matrix is a probability distribution over the entire input, and downstream tasks can treat attention weights as “importance scores.” That interpretability pays dividends when debugging: if a model misclassifies a token, inspecting its attention gives evidence of which tokens it actually “looked at,” a luxury absent from opaque RNN gates.

The only remaining sequential dependence is the softmax itself, which still requires \(\mathcal{O}(N^2)\) operations to compute the pairwise similarities. Many recent innovations, such as FlashAttention, optimize this kernel through fused CUDA operations, but the architecture leaves the door open for future variants like linear attention or efficient transformers that keep the retrieval ability without quadratic memory.

## Where the field is now

Attention-based transformers reigned by the end of the 2020s because they were easy to scale, easy to parallelize, and amenable to pre-training and fine-tuning across diverse tasks. Research frontiers now test how far the same core routing can extend. GLM-4.5 (GLM-4.5 Team 2025 — [https://arxiv.org/pdf/2508.06471](https://arxiv.org/pdf/2508.06471)) demonstrates that mixtures-of-experts with 355B parameters (32B activated per token) can juggle agentic planning, reasoning, and coding by alternating between “thinking” and “response” modes within the same stack of transformer layers. Its attention blocks stay standard, but the routing between modes is itself learned, showing that the transformer’s soft retrieval remains the workhorse even when new meta-controllers sit on top.

The engineering frontier is equally active. Meta AI’s “Introducing Llama 3” blog (Meta AI 2024 — [https://ai.meta.com/blog/introducing-llama-3/](https://ai.meta.com/blog/introducing-llama-3/)) describes how production deployments combine quantized transformer weights, FlashAttention kernels, and custom serving that keeps end-to-end p95 latency under 70 ms for 1.2M daily users. That system is still a stack of transformer encoder-decoder layers, but it pairs them with aggressive kernel fusion, tensor parallelism, and caching to meet real-world throughput. The route from the original Vaswani et al. formulation to this deployment is direct: each optimization still hinges on the fact that attention reduces the sequential bottleneck, so hardware engineers can batch tokens at inference time and exploit the same dense matrix multiplies that training uses.

| Model | Notable achievement | Source |
|---|---|---|
| BERT (Devlin et al. 2018) | GLUE dev score 80.3 — first bidirectional pre-training | [https://arxiv.org/abs/1810.04805](https://arxiv.org/abs/1810.04805) |
| GLM-4.5 | 355B parameters with hybrid reasoning/coding modes | [https://arxiv.org/pdf/2508.06471](https://arxiv.org/pdf/2508.06471) |
| Llama 3 | Production quantized inference with FlashAttention | [https://ai.meta.com/blog/introducing-llama-3/](https://ai.meta.com/blog/introducing-llama-3/) |

The comparison underscores that transformers’ value is not in clever recursion but in routing entire sequences with matrix multiplies—this is what lets researchers push from GLUE to agentic systems and corporations ship low-latency chat services.

## What's still open

Can we keep the precise token-to-token routing that softmax attention provides while reducing the \(\mathcal{O}(N^2)\) memory and compute requirements to \(\mathcal{O}(N)\) for infinite context windows without sacrificing retrieval fidelity?

Is there a self-attention parameterization that is both interpretable (each head maps to a linguistically meaningful relation) and trainable at the scale of GLM-4.5 without requiring hand-tuned biases per head?

How do we verify that the sparse expert routing in MoE transformers preserves the same robustness to adversarial prompts that dense models exhibit, especially when gating decisions are input-dependent?

Can a structured compression of the attention matrix (e.g., low-rank or kernel approximations) be learned jointly with the transformer without degrading the model’s ability to judge fine-grained token similarity, and can such compression generalize across domains?

## Where to read next

If you want the probabilistic foundation that clocked the same softmax attention as score matching, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) gives the likelihood-free training perspective underlying every transformer gradient; the engineering counterpart is → [Flash Attention](../../09-algorithms-systems-for-ai/concepts/flash-attention.md) showing how to fuse those kernels for production speed; for the next scaling curve, → [Mixture of experts](../../01-ai/concepts/mixture-of-experts.md) explains how sparse routing layers plug into the same transformer backbone.

## Build it

This build proves that you can distill a single-head self-attention block down to a few hundred lines of PyTorch and train it on a synthetic sentiment dataset, locking in the intuition that queries, keys, and values are the primitive operations that transformer stacks repeat. The artifact is a checkpoint whose attention maps you can visualize and whose forward pass matches the class token encodings from modern sentence transformers.

**What you're building:** A mini transformer block with a single self-attention head trained to classify 4-turn synthetic sentiment dialogs, with attention maps compared to off-the-shelf sentence-transformers embeddings.

**Why this is valuable:** It forces you to implement the query-key-value math yourself, monitor softmax weights, and see how a learned attention distribution compares to compressed semantic embeddings, which is the core intuition the transformer architecture encodes.

**Stack:**
- **Model:** sentence-transformers/all-MiniLM-L6-v2 (3.7M downloads) as the reference embedding extractor and sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (2.9M downloads) for multilingual similarity checks.
- **Dataset:** emotion (Hugging Face dataset ID `emotion`) for rapid sentiment labels that fit in Colab RAM.
- **Framework:** PyTorch 2.1 + Hugging Face Datasets + Matplotlib 3.9 for visualization.
- **Compute:** Free Colab T4 (16GB VRAM) — training runs in ~18 minutes.

**The recipe:**
1. Install `pip install torch==2.1.0 datasets transformers matplotlib` and download the `emotion` dataset; cache the splits in `/tmp/emotion`.
2. Preprocess by padding every dialog to four turns, tokenize with a lightweight BERT tokenizer, and augment each sample with a “class prompt” token prepended so you can read off the sentiment representation from the attention block.
3. Define a single-head attention module: project inputs into \(Q/K/V\), compute the scaled dot-product attention matrix, apply dropout, and pass the result through a two-layer MLP with GELU; train with AdamW lr=1e-4, batch size 32, for 10 epochs, monitoring cross-entropy loss and attention entropy (expect loss to fall from ~1.1 to <0.45 and entropy to stabilize).
4. Evaluate by comparing the block’s class token embedding to the fixed sentence-transformers reference using cosine similarity; expect accuracy above 78% on the holdout and mean cosine similarity within 0.18 of the reference model.
5. What you now have is a checkpoint (`single_head_attention.pt`) whose attention weights you can visualize alongside the sentence-transformers embeddings, a training log for loss/entropy, and a short script that maps unseen sentences to the block’s attention-weighted sentiment representation.

**Expected outcome:** A runnable PyTorch module that matches sentence-transformers’ semantic signal on the emotion dataset, plus attention heatmaps that expose how the block routes context.

- **CS student:** Run the same recipe on an RTX 4070 (or Colab Pro) but shrink the dataset to 1,000 examples and add gradient-checkpointing to see how memory scales; the training script now also saves computable attention distributions that you can export to TensorBoard.
- **Applied engineer:** Quantize the single-head block to INT8 using PyTorch TensorRT backend, serve it through vLLM at 0.9 tokens/ms latency, and compare that latency to a cached sentence-transformers/all-MiniLM-L6-v2 inference to demonstrate the transformer block’s production readiness.
- **Applied researcher:** Hypothesize that attention entropy increases when the inputs are semantically ambiguous; test this by duplicating examples with paraphrased contexts and plotting entropy vs. accuracy—falsify if entropy does not rise by at least 12% on ambiguous cases.
- **Frontier researcher:** Extend the current build toward the open question above by replacing the softmax attention matrix with a kernelized linear attention and measure whether the learned similarity still matches the sentence-transformers baselines within 0.2 cosine distance; if it does not, that falsifies the hypothesis that linear attention can preserve semantic fidelity on short dialogs.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*