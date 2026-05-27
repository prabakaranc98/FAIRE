---
title: Transformer
slug: transformer
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [vaswani, schmidhuber, radford, bommasani]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [attention-mechanisms, positional-encoding, optimization-basics]
tags: [transformer, self-attention, language-models, scaling-laws, inference]
updated: 2025-01-01
has_mvb: true
---

# Transformer

Imagine the "forgetful translator" who is handed a 1,000-word paragraph, writes the translation to the first sentence, then drops it into a drawer and starts translating the next sentence with no reminder what came before. That is what recurrent models used to do: compress the entire prefix into a fixed-size vector and hope the downstream layers remembered the beginning by the time they reached the end. The transformer walks into the room with a different metaphor. Rather than shrink the story into one vector, it pins every token to a nametag on a giant bulletin board and lets every token ask every other token for help at the same time. By the end of this page you will see how that bulletin-board view turns into precise matrix multiplications, why the architecture collapses the sequential bottleneck, and how to build the very attention module that makes this routing possible.

## The territory

Language models long ago stopped being about recurrence and gating. The central problem was never “how to remember the prefix” but “how to let every decision point read any other relevant part of the context without waiting for dozens of sequential steps.” The transformer (Vaswani et al. 2017) [arxiv:1706.03762](https://arxiv.org/abs/1706.03762) answers that by framing computation as soft routing: each token learns to query a global memory of tokens and pull back the few numbers it needs, making every layer a redistribution of information across positions. This routing view inherits ideas from Schmidhuber’s fast-weight controllers, which had already hinted in 2022 that some form of associative access could be learned to replace recurrence [arxiv:2203.15702](https://arxiv.org/abs/2203.15702). The transformer's family consists of stacked self-attention layers plus small position-wise feed-forward networks, giving it the ability to both relate arbitrary tokens and to project these relationships into the higher-order feature space that language understands. This section positions the transformer not as another sequential model but as a routing architecture that treats context as a soft memory, and that leads directly into the question: how does this routing get calculated?

## How it works

The transformer begins with a matrix of token embeddings \(X \in \mathbb{R}^{N \times d_{\text{model}}}\), where \(N\) is the sequence length and \(d_{\text{model}}\) is the hidden dimension. Instead of running through \(X\) left-to-right, each layer builds three projections: queries, keys, and values. Writing these projections as \(Q = XW^Q\), \(K = XW^K\), and \(V = XW^V\) makes the next step transparent: every position \(i\) computes interaction scores against every position \(j\) by taking the dot product \(Q_i K_j^\top\), scaling those scores by \(\sqrt{d_k}\) (to keep gradients well-behaved), and passing them through \(\text{softmax}(\cdot)\) to turn them into a weighted average. Formally,

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
\]

where \(Q \in \mathbb{R}^{N \times d_k}\) holds the queries, \(K \in \mathbb{R}^{N \times d_k}\) the keys, \(V \in \mathbb{R}^{N \times d_v}\) the values, and the softmax operates row-wise so that each query becomes a distribution over keys. The direct consequence is that each output position is a convex combination of all values, with routing weights learned to focus on the few tokens that matter.

A single attention head is already powerful, but the transformer multiplies its expressivity by assembling \(h\) such heads in parallel, each with its own projection matrices. The multi-head attention output is

\[
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)W^O
\]

where \(\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)\), \(W_i^Q \in \mathbb{R}^{d_{\text{model}} \times d_k}\), \(W_i^K \in \mathbb{R}^{d_{\text{model}} \times d_k}\), \(W_i^V \in \mathbb{R}^{d_{\text{model}} \times d_v}\) are the per-head projection matrices, \(W^O \in \mathbb{R}^{h d_v \times d_{\text{model}}}\) is the output projection, and \(h\) is the number of heads. Each head learns its own routing, so one can specialize in attending to the subject, another to dependencies farther back, and another to positional structure. The concatenation followed by \(W^O\) lets the layer recombine these different routes into the single-space representation that feeds the next layer.

Constructing queries, keys, and values in this way assumes absolute context positions don’t matter, which is why the transformer adds positional information to the embeddings before the projections. The most common scheme, sinusoidal positional encoding, sets \(P_{2i} = \sin(n / 10000^{2i/d_{\text{model}}})\) and \(P_{2i+1} = \cos(n / 10000^{2i/d_{\text{model}}})\) for position \(n\). Since these encodings are deterministic, they allow interpolation to unseen lengths and ensure that every dot product \(Q_i \cdot K_j\) remains sensitive to relative distances via the periodic patterns. Later variations such as rotary position encoding add the positional signal directly inside the attention computation, but all of them preserve the transformer’s premise: the routing weights should be a function of the absolute and relative positions inside the learned projection space.

Once the attention module outputs a new representation, the transformer applies residual connections and normalization to stabilize training. Residuals add \(X\) back to the attention output so that gradients can flow through identity paths, and layer normalization makes each dimension have zero mean and unit variance, which prevents the growing variance that would render the attention softmax ineffective. After normalization, a small two-layer feed-forward network processes each position independently:

\[
\text{FFN}(x) = \text{GeLU}(xW_1 + b_1)W_2 + b_2,
\]

where \(W_1 \in \mathbb{R}^{d_{\text{model}} \times d_{\text{ff}}}\), \(W_2 \in \mathbb{R}^{d_{\text{ff}} \times d_{\text{model}}}\), \(b_1, b_2\) are biases, and \(\text{GeLU}\) is the Gaussian error linear unit. This position-wise projection re-weights and mixes the channel dimension after the attention routing has redistributed information across the sequence. The combination of attention, residual, normalization, and feed-forward layers forms a transformer block. Stacking dozens or hundreds of such blocks deepens the routing capability while keeping the cost per token manageable, because each block still only performs \(O(N^2)\) dot products when computing attention.

The training objective reflects this architecture. Autoregressive transformers optimize the log-likelihood of the next token given all previous ones. If \(x = (x_1, \dots, x_T)\) is a training sequence, the probability factorizes as \(\prod_{t=1}^T P_\theta(x_t \mid x_{<t})\), and every step uses the same attention mechanism to dig into the prefix. Masked language models—like BERT—train on predicting randomly masked tokens by letting attention see both sides of the token while removing it in the output. Regardless of directionality, the key bottleneck that the transformer removes is the need to pass a hidden state sequentially through time; the softmax attention can refer directly to any token with a single matrix multiplication.

Because the transformer can connect any pair of tokens, it also avoids the vanishing gradient problems that plagued recurrent nets. Gradients flow backward through the same attention paths that pass information forward, so the model can learn long-range dependencies without gating functions. Yet this flexibility incurs the \(O(N^2)\) cost of computing all pairwise scores. Practical implementations mitigate this by limiting \(N\) (chunked contexts), pruning attention weights, or introducing sparse variants, but the canonical transformer accepts the quadratic cost as the price for a routing mechanism that still parallelizes across the sequence. The rest of this page walks through how practitioners make that mechanism work in practice, including the training recipes and inference challenges of today's massive models.

## Where the field is now

Transformers dominate both research and production. On the research frontier, OpenAI’s GPT-4o (OpenAI 2024) [https://openai.com/research/gpt-4o] illustrates the same architecture scaling beyond text into simultaneous language, vision, and speech by concatenating multi-modal embeddings before the attention layers and training with multi-task objectives. The paper emphasizes that the same multi-head attention that processed text can route information across pixel patches and audio frames, so the transformer truly becomes a universal router. OpenAI also reports that attention-based routing remains critical to the agentic behaviors observed in GPT-4o, because it allows the model to focus its computation on modality-specific features within each layer rather than forcing a shared recurrent summary.

The engineering frontier is visible in how Google deploys Gemini 1.5 (Google Research blog 2024) [https://ai.googleblog.com/2024/10/introducing-gemini-1-5.html] for long-context agentic reasoning. Gemini 1.5 uses a transformer stack with the same softmax attention per layer, but pairs it with a staged caching layer that retains the biggest values across context windows to avoid recomputing the full attention matrix at every token. This system-level innovation shows that production transformers can stay fundamentally faithful to the routing paradigm while engineering around the quadratic expense: the attention weights still compute interactions between all token pairs, but selective replay and quantized caches keep latency and cost reasonable during dialogue and agent loops.

A second research frontier is the set of post-training alignment recipes that refine transformer outputs into safe, useful agents. The Allen Institute’s Tülu 3 (AI21 2024) blog [https://www.ai21.com/blog/tulu-3] documents how they fine-tune transformer checkpoints via supervised fine-tuning (SFT), then apply direct preference optimization (DPO), and finally incorporate reinforcement learning via the Reward and Value Regularization (RLVR) losses, all while keeping the same transformer backbone. This trilogy of post-training steps proves that the underlying transformer architecture can serve as a substrate for alignment—inference is still routed via attention—but the output distribution is shaped afterward by carefully calibrated optimization.

Together, these fronts show that transformers are not a single paper but a living stack: Vaswani et al.’s architecture supplies the routing core; GPT-4o extends the domain to multi-modal agentic tasks; Gemini 1.5 shows the system-level choreography required at scale; and Tülu 3 proves that downstream losses can bend the same routing mechanism toward alignment without re-architecting the model. The SOTA narrative of 2024 is not a different architecture but a set of plumbing improvements—context caching, alignment losses, quantization, long-context replays—wrapped around the original transformer block.

## What's still open

Can we preserve the precise cross-token retrieval of softmax attention while reducing the inference cost from \(O(N^2)\) to \(O(N)\) or ideally \(O(1)\)? Most efficient transformers either prune attention heads, compress the context into a summary, or rely on sparse kernels, but these steps introduce approximation noise that causes needles-in-a-haystack information—like rare factual details or code tokens—to vanish. The question is whether there exists a routing mechanism that selects tokens through learned hashing or recurrence-free memory addressing while still allowing exact comparisons when needed, so that long documents can be processed quickly without losing the accuracy of full softmax attention.

A second open question stems from the training/inference mismatch. During training, transformers compute full attention across the entire context; during inference, production pipelines often chunk, cache, or compress the context to stay within latency budgets. How can we train a transformer such that its learned attention patterns remain valid after aggressive context caching without retraining the entire stack on cached permutations? The ideal answer would allow a single trained checkpoint to operate efficiently in both academic and production contexts.

Finally, transformers still struggle to control when and how much each layer routes different types of information. Multimodal, multi-objective agents like those described in GPT-4o and Gemini 1.5 show that some layers need to focus on modality alignment while others must preserve text semantics. What architectural knobs or regularizations can enforce this routing discipline, enabling each head not only to compute the right weights but to avoid redundancy and catastrophic attention collapse across modalities or tasks?

## Where to read next

If you want the probabilistic foundation behind this routing, → [[score-matching]] explains how score-based models estimate gradients of the log density without ever computing partition functions, which mirrors how attention avoids accumulating hidden states. The engineering counterpart is → [[flash-attention]] where low-level kernels keep the attention routing fast enough for billion-scale models. For the next research paradigm, → [[flow-matching]] generalizes the noising process that transformers implicitly learn when treating sequences as routing problems over continuous paths.

## Build it

Transformers are often perceived as opaque blocks, so the best first experiment is to rebuild the multi-head attention routing and see how attention weights realign during training on real text. The following build proves that even a PyTorch implementation of a single transformer block can learn dynamic routing over arbitrary character contexts.

**What you're building:** a bare-bones multi-head attention module trained end-to-end to generate Shakespeare characters, with live inspection of the attention weights.

**Why this is valuable:** it forces you to implement the routing matrices (queries, keys, values, and the softmax) yourself, showing how the bullet-point view of a transformer block becomes actual matrix operations whose gradients change the routing probabilities.

**Stack:**
- **Model:** a custom PyTorch transformer block (no pretrained checkpoint required)
- **Dataset:** [tiny_shakespeare](https://huggingface.co/datasets/tiny_shakespeare) — 3 KB of Shakespeare snippets
- **Framework:** PyTorch 2.1 + TorchScript-friendly modules
- **Compute:** Free Colab T4 (16 GB VRAM, ~30 minutes to train 10 epochs)

**The recipe:**
1. `pip install torch datasets matplotlib` and download Colab-ready helper that loads tiny_shakespeare with tokenizer `char` and batches sequences of length 64.
2. Normalize token IDs to embeddings of size 256, add sinusoidal positional encodings, and compute \(Q, K, V\) by projecting the batched embeddings with learnable matrices \(W^Q, W^K, W^V\).
3. Implement the scaled-dot-product attention, reshape into 8 heads with \(d_k = d_v = 32\), and multiply by \(W^O \in \mathbb{R}^{256 \times 256}\); wrap this in a Transformer block with residuals, layer norm, and a two-layer FFN of hidden size 1024.
4. Train with cross-entropy on the next-character prediction target, using AdamW with learning rate \(5 \times 10^{-4}\), batch size 32, 10 epochs; attention weights should start uniform and then concentrate on the preceding characters (visualize with `matplotlib` heatmaps after each epoch).
5. Evaluate by sampling characters autoregressively for 512 tokens and compare sample perplexity to the validation text; you now have a checkpoint and attention visualizations that show routing in action.

**Expected outcome:** a trained transformer block checkpoint, perplexity ~1.2 on the tiny dataset, and plots of how attention weights shift from uniform to sharply focused distributions.

- **CS student:** Replace the visualization with a Colab widget that animates a single sample’s attention matrices over training, so you can explore the routing with only free-tier compute.
- **Applied engineer:** Export the trained block via TorchScript, quantize the weights to int8, and wrap it in a vLLM-style API serving inference at <50 ms latency for 64-token prefixes on an A10 cloud instance.
- **Applied researcher:** Use the same recipe but add a learnable gating scalar per head that multiplies the attention logit, then ablate whether head pruning hurts perplexity more or less than disabling entire layers.
- **Frontier researcher:** Probe the open question from §What's still open by replacing the softmax in attention with a fixed-length routing list (a softmax over a single learned 16-entry cache) and measure where this \(O(1)\) approximation breaks the retrieval of rare tokens; falsify the design if perplexity jumps by more than 20%.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*
