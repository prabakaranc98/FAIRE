---
title: Long Context
slug: long-context
layer: core
subject: 07-attention-memory-reasoning-continual
page_type: concept
state: drafted
authors_anchored: [vaswani, weston, graves, longfunceval]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [attention, memory-networks, retrieval-augmented-generation]
tags: [long-context, kv-cache, tool-calling, evaluation, memory, scale]
updated: 2025-11-05
has_mvb: true
---

# Long Context

Imagine handing an enterprise AI assistant a 100-page catalog of tools, invoices, and policies, then asking it to call the exact API that books a flight on behalf of a busy executive. In controlled benchmarks, the assistant matches queries to the right entry. In the field, its tool-calling accuracy plummets by as much as 85% before the context window even fills, and it starts hallucinating API parameters from earlier paragraphs. What changed? The window is still within the advertised 128K tokens, but the agent can no longer keep up with the non-linear dependencies it was reasoning over. Long-context design is not just about scaling token budgets; it is about understanding why the apparent hardware capacity is the least of the problem and what to do when attention, memory, and retrieval all conspire to “forget” the question before the answer appears.

## The territory

Long-context modeling sits at the intersection of large-scale attention, episodic memory, and interaction-heavy agents. The Transformer encoder-decoder stack that propelled language models to fluency introduced a quadratic self-attention cost, which meant that naively extending the context window required either prohibitive compute or new architectural tricks. That same architecture also fixed all intermediate states (queries, keys, and values) to token positions inside the window, so there was no baked-in mechanism to recall, shuffle, or evict tokens once they passed the line of sight. Long-context research, therefore, is not just another scale-up; it is where attention meets memory management, retrieval augmentation, and tool-aware execution. Techniques borrowed from Memory Networks decouple storage from processing, and dynamic cache systems try to maintain the illusion that every token is still “in view” even after it has been replaced in the physical buffer. The territory differs from vanilla retrieval in that it has to preserve step-by-step inference—multi-hop chains, tool calls, and agentic dialogues—over thousands of tokens without letting the model’s “working set” degrade catastrophically. How does it actually work?

## How it works

Scaling a context window begins with confronting the quadratic term in self-attention. In a Transformer layer, attention weights are computed as

\[
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V,
\]

where \(Q\), \(K\), and \(V\) are matrices of queries, keys, and values with shapes \((N, d_k)\), \((M, d_k)\), and \((M, d_v)\) respectively, \(N\) is the number of query tokens (often equal to the number of context tokens), and \(M\) is the number of context positions that the model attends to. Vaswani et al. (2017) [arxiv:1706.03762](https://arxiv.org/abs/1706.03762) showed that computing \(QK^\top\) requires \(O(N \cdot M \cdot d_k)\) operations and \(O(N \cdot M)\) memory, so doubling the context length quadruples the compute and memory budget inside each layer. When \(N = M = L\), the cost becomes \(O(L^2)\). The consequence is that the base Transformer simply cannot keep processing at the same depth once \(L\) reaches tens of thousands unless we redesign how it stores and updates the key-value pairs.

One line of attack, inspired by Memory Networks (Weston et al. 2014) [arxiv:1410.3916](https://arxiv.org/abs/1410.3916), is to decouple the “memory” from the compute. In their setup, external facts \(m_i\) are stored as vectors in memory, and a controller selects which facts to read by computing soft attention over the set

\[
p_i = \mathrm{softmax}\left(f_r(q, m_i)\right),
\]

where \(q\) is the question embedding and \(f_r\) is a scoring function (often a dot product or parameterized similarity). The controller can perform multiple hops, writing new contents back into memory between hops. Since the external store can be arbitrarily large, the architecture sidesteps the quadratic blow-up by keeping the active attention window small and only selecting the relevant facts to pass to the controller at each hop. This decoupling is the conceptual ancestor of how modern language models maintain a KV cache: the “memory” now lives in the cache, and the “controller” is the cross-attention head that chooses which past tokens matter.

Dynamic Memory Networks (Kumar et al. 2015) [arxiv:1506.07285](https://arxiv.org/abs/1506.07285) extended this idea to question-answering by having episodic memories \(c^t\) that are updated as the network iterates. Each episode attends over a fact set \(S\) with attention weights computed via a gating function \(g\left(c^{t-1}, q, s_i\right)\), and then updates the episode vector using

\[
c^t = \mathrm{GRU}\left(c^{t-1}, \sum_i g\left(c^{t-1}, q, s_i\right) s_i \right),
\]

where \(s_i \in S\) are candidate facts, \(q\) is the question vector, and the GRU integrates the selected facts into the next episode vector. The episodic dynamics give the model a way to focus on a small subset of facts per reasoning step and remember the outcome of each step explicitly. When translated to Transformer KV caches, this corresponds to refreshing the cache with the distilled result of a reasoning step rather than leaving every token in place; stale cache entries are effectively “evicted” by overwriting them with more salient representations.

Neural Turing Machines (Graves et al. 2014) [arxiv:1410.5401](https://arxiv.org/pdf/1410.5401v2) introduced a differentiable memory matrix \(\mathbf{M}_t \in \mathbb{R}^{N \times W}\) that could be read and written with soft attention mechanisms, and the model could choose dynamic weighting \(w_t\) over memory slots via content-based addressing plus optional shift mechanisms. Every read operation produced a vector

\[
r_t = \sum_i w_t(i) \mathbf{M}_t(i),
\]

where \(w_t\) is a probability distribution over the \(N\) slots, ensuring differentiability. The write operations could combine erase and add gates, allowing the model to discard old information and record new events. This is the first analog in the family of architectures that treat attention as a dynamic, overwriteable structure rather than a static matrix computed once per layer. Modern KV caches inherit this overwrite behavior: each token’s key and value pair is an entry in the cache, and when the cache becomes full, we must choose which entries to keep. Without a good policy, we retain tokens that will never be attended again simply because they still “fit” in the buffer.

The practical lever of this concept is the cache management strategy employed by most inference stacks. During generation, each token \(x_t\) produces a key \(k_t\) and value \(v_t\); when computing attention for the next token \(x_{t+1}\), the model attends over the cache entries \(\{(k_i, v_i)\}_{i=1}^t\). The size of the cache grows linearly with \(t\), but the hardware limit enforces a maximum \(T_{\max}\). To keep inference tractable, implementations often evict the oldest entries (a FIFO policy) or the least frequently accessed entries (an LRU policy). However, this treats every token as equally important for future reasoning, which is false in multi-hop and tool-call scenarios—the dependencies are hierarchical and non-linear. The result is that, long before the cache is full, critical summary tokens or tool parameters fall out of the attention span.

LongFuncEval (2025) [arxiv:2505.10570](https://arxiv.org/abs/2505.10570) quantified what this looks like in practice. They measured tool-calling accuracy for long-context agents when the tool description, API schema, and previous tool use were shuffled into different positions in a 128K-token buffer. While synthetic benchmarks in a static retrieval setup held accuracy stable across positions (within 5% of best-case), tool-calling accuracy in agentic tasks dropped by up to 85% when the relevant tool spec appeared in the first 20% of the context window. The agent was still within the advertised context length, but it lacked the dynamic recall mechanism necessary to reason over the non-linear dependencies of the execution trace. The key takeaway is that hardware context is a necessary but not sufficient condition—without cache-aware strategies that preserve reasoning-critical tokens, the agent becomes forgetful as soon as the transcript exceeds a few thousand tokens.

The “Needle-in-a-Haystack” phenomenon is apparent when we evaluate how retrieval accuracy changes with the target’s position in the context. Suppose we inject a relevant paragraph \(p\) at position \(i\) in a context of length \(L\), and we define retrieval accuracy as the probability \(R(i)\) that the model attends to \(p\) while answering a question \(q\). If the cache uses a simple age-based eviction, \(R(i)\) will depend heavily on \(i\), with \(R(i)\) decaying as \(i\) falls outside the most recent tokens. Instead, a content-aware strategy would aim for \(R(i) \approx 1\) regardless of \(i\). Because attention costs \(O(L^2)\), the strategy cannot be “always attend to all tokens”—instead it must select a small working set via heuristics such as “keep tokens with high cross-attention scores to the current reasoning chain” or “retain tokens that were part of previous tool invocations.” The more the cache mirrors the episodic memory dynamics described above, the more stable \(R(i)\) becomes.

Finally, retrieval-augmented systems layer an external knowledge base \(D\) on top of the cached context. At each step, the retriever issues a query \(q_t\) and fetches documents \(d_j\) ranked by similarity. The retrieved passages are appended to the context, but they carry the same eviction risk—if \(d_j\) is stale or if the tokens representing the retrieved facts are pushed out before they can be reasoned over, the model effectively “loses” the retrieval. Synchronizing retrieval, reasoning, and cache eviction forms the crux of long-context work: without a policy that keeps the tokens needed for multi-hop inference, reasoning chains break even though the retrieval itself succeeded. This is why the physical window is a promise of capacity, not a guarantee of comprehension.

## Where the field is now

LongFuncEval’s degradation curves have become the new baseline for what counts as a robust long-context system. Their multi-agent benchmark showed that once the context exceeded roughly 32K useful tokens, agentic tool-calling sharply dropped, a result that has forced teams to rethink both evaluation and deployment. The 2025 LONGCODEU benchmark [arxiv:2503.04359](https://arxiv.org/abs/2503.04359) extended the problem into code understanding, showing that popular long-context claims (128K to 1M tokens) do not translate into the sustained comprehension needed when the input exceeds 32K tokens of interleaved code, comments, and schema. Models optimized purely for maximum window length tend to miss the multi-step dependencies that appear in long-form code because the attention matrix can no longer track symbol definitions, loop boundaries, and API contracts simultaneously. MLRBench (2025) added another dimension, evaluating multilingual contexts and finding that the cost of maintaining coherent representations for long sequences in multiple languages is roughly 1.5× the cost in a single language, due to fragmented tokenization and variable-length encodings. Together, these research frontiers articulate a new evaluation regime: instead of asking “how many tokens can you fit,” we now ask “how well can you reason over the tokens that matter, regardless of their position?”

On the engineering side, large vendors are pairing long-context-trained models with aggressive cache strategies. Nvidia’s Eagle3 line, exemplified by the long-context GPT-OSS 120B release, deploys hardware-aware inference kernels that fold KV caches across attention heads and pipelines asynchronous eviction policies to keep the most salient tokens alive through the entire dialogue. The deployment in real-time call routing shows that doubling the context window without cache management merely spreads the mistake-rate over a longer conversation; the precision gains only appear when the evicted token list is pruned by relevance heuristics similar to those described in dynamic memory networks. Meanwhile, research labs such as Anthropic (Claude 3.5) fuse long-context reasoning with retrieval augmentation by backfilling retrieval hits into the last 4K tokens and issuing periodic “summaries” of old tokens, effectively creating a manual eviction policy; the anecdotal evidence is that tool accuracy stabilizes around 87% when the buffer is kept at this smaller size with purposeful retention, even if the full model supports 100K tokens.

The common thread through these advances is a shift from raw window length to “effective working set” management: the model must decide which tokens to keep alive for each new reasoning step, which is exactly the question the open terrain still needs to answer in a principled way.

## What's still open

Can we design a dynamic KV cache eviction policy that discards redundant tokens during inference while preserving the complex, non-linear token dependencies required for multi-hop reasoning and tool-calling? The challenge is simultaneously combinatorial and contextual—redundancy cannot be judged purely by age or frequency, and dependencies span tokens that arrived minutes apart. Another question is how to evaluate long-context comprehension beyond retrieval hits: can we define a benchmark where completion quality is tied explicitly to the ordering and presence of rare tokens across thousands of steps, similar to LongFuncEval but with finer-grained tool-call predicates? Lastly, how do we quantify the trade-off between cache sparsity and attention fidelity when using retrieval augmentation? If we reduce the number of tokens entering cross-attention to maintain latency, where is the tipping point at which we lose the reasoning chain, and can the model learn to approximate that tipping point dynamically?

## Where to read next

For the core attention mechanics, → [Attention](attention.md) explains how the \(QK^\top\) computation grows with context length and what alternatives exist; for the memory-management side, → [KV cache](../../09-algorithms-systems-for-ai/concepts/kv-cache.md) tracks how caches are structured, sharded, and refreshed in practice; if you want to see retrieval tied explicitly to reasoning, → [[retrieval-augmented-generation]] shows how augmented passages are punished when they fall out of the working set; and for a view of what comes after long-context reasoning, → [[agentic-tool-calling]] traces how agents coordinate retrieval, memory, and execution.

## Build it

The build proves that context-position sensitivity is measurable and actionable: by sweeping a relevant document’s insertion point across a long transcript while keeping the rest of the context fixed, you observe the drop in retrieval accuracy that mirrors real-world tool failures.

**What you're building:** A “Needle-in-a-Haystack” evaluator that measures how a model’s retrieval accuracy changes as the target information drifts from the top to the bottom of a long context window.

**Why this is valuable:** The evaluation surfaces the exact failure mode that kills agents with long histories—the map between token position and recall probability—so you can experiment with eviction heuristics and see if they make the curve flatter.

**Stack:**
- **Model:** [nvidia/gpt-oss-120b-Eagle3-long-context](https://huggingface.co/nvidia/gpt-oss-120b-Eagle3-long-context) — downloads: 5.4k (HF model card)
- **Dataset:** [squad_v2](https://huggingface.co/datasets/squad_v2) — used to assemble synthetic 65K-token documents
- **Framework:** Transformers 4.43.0 + Accelerate 1.19.0 powered by PyTorch 2.1
- **Compute:** Free Google Colab T4 (16GB VRAM) — 2 hours for evaluation sweep

**The recipe:**
1. `pip install transformers accelerate datasets matplotlib` and initialize `from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline`.
2. Load `squad_v2`, randomly sample 25 passages, concatenate them with separators to create a 65K-token context, and insert the chosen question paragraph at ten positions evenly spaced across the buffer.
3. Use `AutoTokenizer` to tokenize the context, stream each insertion position into a generation loop, and record the model’s log-likelihood of the ground-truth answer span while preventing attention from exceeding 65K tokens by truncating tokens beyond the limit.
4. Evaluate accuracy by measuring whether the answer appears in the top-k generated tokens for each position and plot accuracy vs. insertion position; expect a drop of at least 30% between early and late positions when no cache policy is applied.
5. You now have the Needle-in-a-Haystack artifact: a visualization plus aggregated accuracy numbers that prove reasoning degrades before the window is full.

**Expected outcome:** A chart showing the decline of retrieval accuracy as the target paragraph moves toward the older end of the context, highlighting the need for better cache eviction.

- **CS student:** Run the same evaluation on an RTX 4070 but replace the 120B model with [akamb/long-context-nano-1](https://huggingface.co/akamb/long-context-nano-1) to keep inference under 8GB VRAM and compare the shape of the accuracy curve.
- **Applied engineer:** Add a quantization step (4-bit QLoRA) to nvidia/gpt-oss-120b-Eagle3-long-context, host the model inside a vLLM server, and measure the same accuracy curve while also recording p50 latency (target < 550 ms) for each insertion position.
- **Applied researcher:** Hypothesize that cross-attention rerouting (resetting attention to the latest 4K tokens when a tool call fails) reduces the accuracy drop, and run an ablation where you reset the KV cache midway through the insertions and observe whether the curve flattens.
- **Frontier researcher:** Test the open question about dynamic KV eviction by developing a heuristic that keeps tokens with high mutual information to the current attention span and falsify it if the accuracy drop between the 10th and 90th percentile positions remains > 20%.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*