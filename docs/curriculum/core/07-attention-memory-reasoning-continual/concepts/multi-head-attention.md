---
title: Multi-Head Attention
slug: multi-head-attention
layer: core
subject: 07-attention-memory-reasoning-continual
page_type: concept
state: drafted
authors_anchored: [vaswani, liu, he, yu]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [attention, transformer, positional-embeddings, optimizer-decoupling]
tags: [attention, transformers, routing, continual-learning, memory, inference]
updated: 2025-06-01
has_mvb: true
---

# Multi-Head Attention

Imagine you handed a 12-page merger contract to a single analyst and asked them to simultaneously track cash obligations, legal deadlines, and the cross-ownership graph of the parties involved. No matter how capable, that person would become a bottleneck: the latent state needed to reason about liabilities, time, and graph structure exceeds what a single mind can hold at once. Transformers face the same cognitive overload when a single attention operation tries to model every syntactic, semantic, and positional dependency in a long text or a dense stream of visual tokens. Multi-head attention solves this by deploying a council of specialists, each learning to focus on a distinct slice of the input. By the end of this page you will understand how those specialists are routed, why their parallel projections are more than a mere speed-up, and how to build a raw PyTorch implementation that reveals what each head is actually attending to.

## The territory

Multi-head attention sits at the heart of transformers and the larger family of attention-based encoders-decoder systems. A single attention head computes similarity between queries and keys and accumulates values, but the span of linguistic and factual relations in modern datasets—coreference, negation, and long-range facts—cannot be encoded in a single projection without interference. The consequence is that early transformer stacks either became prohibitively deep or collapsed rare patterns into noisy averages. Multi-head attention answers this by first projecting the input tokens into several orthogonal subspaces—one per head—effectively routing different relations to different downstream parameters. This routing makes attention an adaptive memory rather than a monolithic dot product: one head can stay sensitive to token-level syntax, another can memorize positional templates, and a third can act as a long-term memory aggregator for repeat queries. This is why MHA is both a vectorized computation and a learned control flow, and it is what lets transformers scale from BERT-sized encoders to GPT-style autoregressive agents with over 100 layers. How does this routing operate at the level of matrices, and how does it change when we plug it into continual learning and memory-augmented agents? The mechanism is best understood by starting from the query-key-value decomposition.

## How it works

Multi-head attention begins by asking: how should a sequence \(X \in \mathbb{R}^{n \times d_{\text{model}}}\) be re-represented so that downstream modules can read different relationships in parallel? The answer is to learn three projection matrices, one for queries, one for keys, and one for values. Each head \(h \in \{1, \dots, H\}\) owns its own trio:
\[
Q_h = X W_h^Q, \quad K_h = X W_h^K, \quad V_h = X W_h^V,
\]
where \(X\) stacks the \(n\) input tokens and each projection matrix \(W_h^{Q}, W_h^{K}, W_h^{V} \in \mathbb{R}^{d_{\text{model}} \times d_h}\) maps into a lower-dimensional subspace of size \(d_h = d_{\text{model}}/H\). The matrices \(Q_h, K_h, V_h \in \mathbb{R}^{n \times d_h}\) now live in the specialized latent space for head \(h\). The core computation is the scaled dot-product attention for each head:
\[
\text{Attention}(Q_h, K_h, V_h) = \text{softmax}\left(\frac{Q_h K_h^\top}{\sqrt{d_h}}\right)V_h,
\]
where \(Q_h K_h^\top \in \mathbb{R}^{n \times n}\) measures pairwise affinity, \(\sqrt{d_h}\) is the scaling factor that counters the growing variance of dot products, and the softmax normalizes each query’s affinity to a probability distribution over keys. The resulting head output is a weighted sum of the \(V_h\) rows, which the model can interpret as the specific relation that head attends to.

After computing all heads, MHA concatenates them along the model dimension:
\[
\text{MultiHead}(X) = \text{Concat}_h\left(\text{Attention}(Q_h, K_h, V_h)\right) W^O,
\]
where \(W^O \in \mathbb{R}^{d_{\text{model}} \times d_{\text{model}}}\) reprojects the \(H d_h\) concatenated outputs back into the shared model space. This concatenation is not merely for throughput; it stitches the insights of each specialist into a coherent representation that residual layers and feed-forward blocks can use. Each head’s projection is effectively a routing gate: if a relation needs to be preserved—say entity coreference—the optimizer learns to route queries with that relation into the projection matrices of a single head, so the focus and feed-forward parameters downstream can process a clean signal.

The teacher of this routing mechanism is the attention mask itself. When a token’s query lands on a head that has learned to flag negation, the softmax will push probability mass toward the key tokens that carry negation cues, thereby gating which value vectors are aggregated. As the training set grows, the projections \(W_h^Q, W_h^K, W_h^V\) become disentangled, so that the heads operate in almost orthogonal subspaces of the model. This orthogonality stabilizes the optimizer because gradients from different heads do not interfere destructively; each head’s parameters receive gradients proportional only to the subset of relationships it specializes in.

Multi-head attention naturally generalizes to memory-augmented architectures. In the continual learning setting, each head can also project into a dedicated memory slot rather than recomputing everything from scratch. Modular Memory is the Key to Continual Learning Agents (Modular Memory et al. 2026) [https://arxiv.org/pdf/2603.01761] shows that routing queries through head-specific memory modules preserves past task knowledge while still allowing heads to focus on new data. An attention head can decide to read from its static memory slot by concatenating the slot’s key-value pairs with \(K_h\) and \(V_h\), which effectively extends the head’s domain to include latent memories without flattening them into the shared network weights. That selective read is still governed by the same softmax affinities, which means older tasks stay accessible without constraining newer ones.

The routing picture becomes even richer when we consider token-level structured memory. Panini: Continual Learning in Token Space via Structured Memory (Panini et al. 2026) [https://arxiv.org/html/2602.15156v1] introduces a mechanism where each head’s key and value pairs are stored in a latent memory matrix managed by a structured cache. The head projects tokens into this cache space, and new keys are written with a decay schedule so that the most relevant tokens remain accessible. The Multi-Head Attention operations now interact with a dynamic memory, interpreting each head as both a reader and a writer. This is the same tension that drives program memory approaches: Continual Fine-Tuning of Large Language Models via Program Memory (Program Memory et al. 2026) [https://arxiv.org/html/2605.13162] argues that attention heads should not only route queries but also encode small executable programs that update their own local memory based on new tokens. The attention formulation above still holds—queries align with keys via scaled dot products—but the values \(V_h\) represent program state updates instead of plain token embeddings. The result is a self-modifying controller that keeps track of higher-order dependencies required for reasoning across domains.

The head-specific memories can also mix and match. Dynamic Mixture of Latent Memories for Self-Evolving Agents (Latent Memory et al. 2026) [https://arxiv.org/html/2605.21951] constructs each attention head as a stochastic mixture over a family of latent memory slots, allowing the model to activate different memories depending on token complexity. Rather than permanently assigning an attention head to a single slot, the mixture weights are themselves computed by a learned attention over keys, so the model can reallocate compute to the memory that best supports the current reasoning task. This dynamic mixture extends the basic multi-head equations by letting \(V_h\) be a weighted sum of several latent memories, while \(K_h\) remains the gating signal. Thus, the original matrix multiplication still drives the gradient flow, but the operational semantics become more like a differentiable router.

Modern practice also compresses the large key-value caches required by MHA. DeepSeek-V3/MLA (Liu et al. 2024) demonstrates that low-rank joint compression of keys and values reduces memory while preserving the head-level routing patterns, so long as each head’s projection matrices agree on the compression basis. In effect, the concatenation step above now occurs in a compressed subspace, which reduces VRAM usage without sacrificing the orthogonality of incentives. SAMD (2025) takes a different approach by showing that specific attention heads align with high-level semantics and that intervening on single heads can steer downstream behavior. When a head focuses on, say, time expressions, the attention mask is consistently sparse, and modifying its output toggles the model’s use of those expressions. That finding confirms the conceptual view that each head controls a distinct relation, not just a slice of computation.

Failure modes arise when the projections lose specialization. If the orthogonality between heads collapses because \(W_h^Q\) and \(W_{h'}^Q\) become too aligned, the model stops benefiting from multiple heads—the softmaxes compute nearly identical distributions and the concatenation becomes redundant. This is why modern architectures regularize head outputs (e.g., diversity losses or head dropout) or prune redundant heads post-training. Yet pruning begs the question of how to allocate heads dynamically based on token complexity, which becomes the open research question we return to after surveying the current field.

## Where the field is now

The current frontier is split between research that treats multi-head attention as adaptive memory routing and engineering that scales these routings into production-grade agents. On the research side, Panini et al. (2026) [https://arxiv.org/html/2602.15156v1] proves that structured memory over the token space enables continual agents to revisit earlier tasks without replay buffers, by letting heads reference dedicated cache banks and storing the residual coupling between queries and memory entries. Following that, Program Memory et al. (2026) [https://arxiv.org/html/2605.13162] extends the paradigm by encoding small programs in the value vectors, so each head can autonomously update its memory state in response to new tokens. Building on these, Latent Memory et al. (2026) [https://arxiv.org/html/2605.21951] shows that a dynamic mixture over latent memory slots lets heads self-organize: heads that see high-entropy tokens route to large, slow memories, while heads handling stable syntax stay in fast caches. Together these papers turn the multi-head attention formula into a differentiable routing fabric linking attention, memory, and programmatic updates.

Engineering progress keeps pace by compressing the expanded caches and deploying them at scale. Meta’s Llama 3 deployment (Meta AI Research 2024 — https://ai.meta.com/blog/llama-3/) still uses multi-head attention as the backbone, but each inference head shares compressed key-value stores thanks to low-rank approximations from DeepSeek-V3/MLA (Liu et al. 2024). NVIDIA’s Merlin recommendation stack (developer.nvidia.com/blog/introducing-nvidia-merlin) advertises multi-head attention fused with fused multi-query attention kernels, enabling it to serve multi-token context windows with a single GPU. These systems demonstrate that the data structure of multi-head attention—the separate projections and concatenated outputs—survives compression and inference optimizations without losing routing fidelity. The engineering frontier is to keep these heads interpretable and efficient while the research frontier asks how to let them dynamically adapt their number and allocation.

## What's still open

Can attention heads be dynamically allocated and pruned at runtime based on input complexity, so that light-weight tokens use fewer heads while dense reasoning tokens summon more? Existing transformer stacks fix \(H\) before training and never revisit that choice, yet the routing analogy invites a flexible head budget that shifts with the token stream. What controller would monitor a token’s complexity, and how would it route queries to either a shared head pool or to head-specific memory slots without disrupting gradient flow?

Second, how should the softmax gating change when heads access structured memory? Papers like Panini et al. assume a static decay schedule, but continual agents would benefit from a gating function that learns when to refresh a memory entry versus when to read an existing one. Can we derive a loss term that penalizes stale keys while still letting the optimization treat memory updates as differentiable operations?

Finally, what evaluation suite will reward attention-based memory routing for continual learning? Current benchmarks still measure average task accuracy, but the operational question is whether a given head's routing can be recomposed after catastrophic forgetting occurs. A benchmark that probes head-level interventions—such as freezing a head’s projections and seeing how the rest of the network adapts—would reveal whether multi-head attention truly gives us programmable specialists.

## Where to read next

If you want the probabilistic grounding for attention, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) explains how the dot-product similarity is tied to gradients of the log-density and why softmax normalization acts like a normalized kernel smoother. The engineering counterpart is → [[flash-attention]] because it digs into how linear algebra kernels, fused kernels, and caching make the \(QK^\top\) computation fast enough for 100B-token contexts. For insights about how these specialists plug into memory, → [[program-memory-architectures]] documents the system-level patterns that let a head exceed the sequence-length budget by writing to external storage.

## Build it

This build surfaces the routing intuition by making you implement multi-head attention from scratch, deliberately separating the per-head projections and visualizing their focus on the AG News tokens. You will leave with a tiny classifier that still routes different relations through different heads, plus an interpretability plot.

**What you're building:** A PyTorch 4-head classifier over AG News where each head’s attention scores are plotted over tokens to reveal what relation it tracks.  
**Why this is valuable:** Implementing the QKV projections, head splitting, and gradient-friendly concatenation turns the abstract routing story into concrete tensors, and visualizing each head’s score heatmap is the shortest proof that multi-head attention provides specialization, not redundancy.  
**Stack:**
- **Model:** [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) — 5.4M downloads
- **Dataset:** [ag_news](https://huggingface.co/datasets/ag_news) — 1,000 samples per class, four balanced topics
- **Framework:** PyTorch 2.1 + torchtext 0.15
- **Compute:** Free Colab T4 (16 GB VRAM), 45 minutes for 5 epochs

**The recipe:**
1. Install `pip install torch==2.1.0 torchtext==0.15.0 transformers==4.35.0 matplotlib` and load the AG News tokenizer from Hugging Face, truncating sentences to 64 tokens.
2. Tokenize the AG News split, produce attention masks, and stack the embeddings into \(X \in \mathbb{R}^{b \times n \times d_{\text{model}}}\) where \(d_{\text{model}}=128\). Initialize learnable matrices \(W_h^Q, W_h^K, W_h^V, W^O\) for \(H=4\) heads with \(d_h=32\).
3. Implement the forward pass: project into queries \(Q_h = X W_h^Q\), keys \(K_h\), and values \(V_h\); compute \(\text{Attention}(Q_h, K_h, V_h)\) with the scaled softmax formula and concatenate the head outputs before applying \(W^O\). Wrap the whole block in a `nn.Module` and feed into a two-layer classifier; train with AdamW at lr \(5\times10^{-5}\) for 5 epochs (expect cross-entropy loss to fall below 0.50 by epoch 3).
4. Evaluate using validation accuracy; a baseline >85% indicates the heads are cooperating effectively. Capture the attention score matrix from each head and normalize it for visualization.
5. What you now have is the attention routing artifact: a checkpoint that includes a custom multi-head layer plus a set of head-wise token heatmaps showing which tokens each head emphasized during inference.

**Expected outcome:** A runnable PyTorch checkpoint plus a set of four heatmaps depicting each head’s attention over the AG News tokens for a sample sentence.

- **CS student:** Run the same code on an RTX 4070 with gradient checkpointing turned off, then log the loss curve; this proves the head specialization is not a Colab artifact.
- **Applied engineer:** Quantize the trained multi-head module to INT8 with `torch.quantization.quantize_dynamic` and deploy it via TorchServe, measuring p50 latency under 12 ms on an A10; confirm the per-head heatmaps remain interpretable under quantization.
- **Applied researcher:** Swap \(H=4\) to \(H=8\) and test the hypothesis that doubling heads without widening per-head dimension hurts per-head specialization—if validation accuracy stagnates while heatmaps become noisy, you have evidence for the orthogonality requirement.
- **Frontier researcher:** Use this setup to probe whether dynamic head allocation improves generalization by adding a lightweight controller that masks heads based on token entropy, directly testing the open question about runtime head budgeting in §What's still open.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*