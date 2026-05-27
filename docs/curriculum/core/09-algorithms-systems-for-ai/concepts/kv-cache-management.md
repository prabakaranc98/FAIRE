---
title: KV Cache Management
slug: kv-cache-management
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [zhang, li, wang, chen]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [llm-inference-basics, attention-mechanisms, distributed-systems-for-llms, inference-optimization]
tags: [kv-cache, memory-management, llm-inference, hybrid-caching, runtime-systems, tiered-memory]
updated: 2025-10-01
has_mvb: true
---

# KV Cache Management

Imagine a speed-reader who, instead of skimming for themes, must memorize every single word of a 1,000-page novel in active memory. By chapter three their brain starts dropping sentences not because they forgot the plot but because the working-memory limit has been breached—every word, even the redundant ones, requires storage now. That is precisely what autoregressive large language models demand from GPUs: the key-value cache keeps a dense representation of every past token so attention can attend to it again, and the cache grows linearly with sequence length. With longer context windows and agents that chain dozens of prompts, we can no longer treat the cache as a fixed-size bag of vectors; instead, we must triage: keep the crucial semantic pieces in the fastest tier and push redundant details down. This page makes that shift visible. By the end, the reader will understand which dimensions of KV cache management matter, how recent systems like SentenceKV, SpindleKV, and Apt-Serve turn that insight into hybrid caching, and how to prototype a tiered cache that dynamically offloads low-attention blocks in a memory-constrained setting.

## The territory

The KV cache sits between the model and every new token generated; every autoregressive head must recompute attention weights against all previous tokens, so the cache holds one key and one value vector per head per past token. The naive view—store every token forever—worked when context windows were hundreds of tokens, but as we push to thousands or even tens of thousands of tokens for research agents, summarization chains, and instruction-following flows, the cache becomes the dominant memory burden. GenAI for Systems: Recurring Challenges and Design Principles from Software to Silicon (2026) [arxiv:2602.15241v1](https://arxiv.org/html/2602.15241v1) documents that GPU memory budgets stop scaling, which is why every inference stack is forced to re-architect KV handling. It names KV cache overflow as the recurring choke point across Microsoft, Meta, and Google deployments, where the bottleneck is not compute but the inability to keep a ten-thousand-token history in 80 GB of GPU DRAM.

Therefore KV cache management bridges two families of techniques: memory-tiered systems from traditional operating systems (LRU, mirroring, prefetch) and semantic compression methods borrowed from retrieval-augmented generation (similarity search, sparse retrieval). This concept also overlaps with work on hidden-state caching, because the decision of what KV entries to evict determines which intermediate activations must be recomputed or fetched from slower storage. The rest of this page explains how the modern mechanism knits these ideas together—sentence-level grouping, layer-aware reduction, and adaptive scheduling—so that tokens once evicted can be recalled without ruining downstream accuracy. How does it actually work?

## How it works

A first insight is that the KV cache introduces a predictable growth pattern. For a transformer with \(L\) layers, \(H\) heads, and head dimension \(d_h\), the memory per token is
\[ \text{mem}_{\text{token}} = 2 \cdot L \cdot H \cdot d_h \cdot b \]
where the factor of two accounts for both keys and values and \(b\) is the byte size of the floating-point representation (e.g., 2 for BF16). This cost is unavoidable: every future attention call needs both key and value. The consequence is simple arithmetic: adding another 1,000-token chunk multiplies GPU memory usage, and model architects hit out-of-memory before their compute is saturated. The mechanism in practice is not about reducing \(L\) or \(H\) but rethinking what "stores all \(t\)" means.

### Sentence-level semantic batching

SentenceKV (Zhang et al. 2025) [arxiv:2504.00970](https://arxiv.org/abs/2504.00970) introduces the first tier of this rethinking. Instead of storing token-level keys and values in GPU memory, SentenceKV groups contiguous tokens that belong to the same semantic unit—a sentence—into a single compact representation. The algorithm maintains two pools: (1) a GPU-resident sentence cache holding high-importance semantic vectors and (2) a CPU-resident raw token cache that keeps the detailed tokens but is consulted only when the GPU-level sentence cache misses. During a prefill phase, tokens are grouped by delimiting punctuation and embeddings; each group's mean key and value are computed and stored in the GPU cache, while the raw per-token KV pairs stream to CPU memory with a compressed representation.

Retrieval in SentenceKV uses a similarity score \(s(q, c)\) between the current query \(q\) and each cached sentence context \(c\):
\[ s(q, c) = \frac{q \cdot c}{\|q\|\|c\| + \epsilon} \]
where \(q\) is derived from the decoder's current query projection, \(c\) is the sentence-level key vector, and \(\epsilon\) prevents division by zero. High similarity retains the sentence in the GPU cache; low similarity triggers a swap, moving the sentence back to CPU and pulling in another with higher expected reuse. The advantage is twofold: (a) the number of entries in GPU RAM drops by roughly the average number of tokens per sentence, and (b) the retrieval is semantic rather than positional, so even if the query refers to a token far back in the sequence, it can still hit if it belongs to an influential sentence.

This sentence-level tier is the first "semantic-aware routing" layer. The GPU-level cache acts like a working set of the story, while CPU stores the book on a slower shelf. When the decoder revisits a sentence that was evicted, SentenceKV recomputes keys and values from the CPU store; the recomputation only happens when semantic similarity suggests the sentence matters again, so most evictions are safe. The trade-off is the extra CPU-GPU bandwidth for rare fetches, but SentenceKV measured a 40–60% reduction in GPU memory without degrading quality on their 14B-parameter proxy.

### Layer-aware redundancy handling

SpindleKV (Wang et al. 2025) [arxiv:2507.06517](https://arxiv.org/abs/2507.06517) reveals the second insight: not every layer suffers equally from redundancy. Deeper layers accumulate attention across tokens, so their keys and values are more semantically unique (they "remember" the long context), whereas shallower layers mostly capture local, redundant features.

SpindleKV splits the cache into two policies. For deep layers, it uses an **attention-based eviction** score: each key-value pair keeps the maximum attention weight received in the last \(T\) steps,
\[ a_t^* = \max_{\tau \in [t-T, t-1]} \text{softmax}\left(\frac{q_\tau k_t^\top}{\sqrt{d_k}}\right) \]
where \(k_t\) is the key at timestep \(t\), \(q_\tau\) is a recent query, \(d_k\) is the key dimension, and \(T\) is a sliding window length. Pairs with low \(a_t^*\) are evicted first because they have not contributed to attention recently. This ensures the algorithm retains those tokens that actually participate in attention flows, not just the latest ones.

For shallow layers, SpindleKV performs **codebook merging** instead of eviction. It clusters keys whose cosine similarity exceeds a threshold, collapses them into a single representative vector, and updates the values by averaging or reweighting based on attention mass. The key idea is that early layers have many near-duplicate features (e.g., word prefixes, stopwords), so collapsing them only reduces redundancy rather than information. A codebook entry includes a support count, so the model can later reconstruct approximations if needed. This asymmetric policy—evict deep layers, merge shallow layers—delivers the memory savings without losing the tokens that drive high-level planning.

Integrating SentenceKV's sentence-level tier with SpindleKV's layer-level policies gives a hybrid structure: GPUs hold sentence vectors and deep-layer KV pairs selectively retained by attention, while shallow-layer tokens are compressed and optionally shunted to CPU. The dynamic is a multi-tiered pipeline: tokens flow from raw decoder outputs to CPU storage, get batched into sentences for GPU promotion, and either stay there or get reduced via layer-aware policies. The next element we need is scheduling.

### Adaptive scheduling and TTFT awareness

Apt-Serve (Li et al. 2025) [arxiv:2504.07494](https://arxiv.org/abs/2504.07494) extends hybrid caching with adaptive request scheduling. It observes that hybrid caches—combining fast GPU memory, slower CPU RAM, and occasionally disk—introduce latency variance when fetching evicted tokens. Generation systems with strict Time-To-First-Token (TTFT) guarantees cannot stall for a cache miss, so Apt-Serve uses a queue-aware scheduler.

The scheduler maintains two queues: one for requests whose queries have cache hits (non-blocking) and one for cache misses requiring CPU retrieval. Each request is tagged with a priority \(p\) derived from service-level objectives, and the scheduler allocates GPU compute to high-priority hits first, while asynchronously prefetching CPU-backed entries for upcoming misses. Apt-Serve balances throughput by modeling the probability \(P_{\text{miss}}(q)\) that a given query \(q\) will miss and orchestrating CPU fetches ahead of demand, using historical similarity features.

Another mechanism from Apt-Serve is **cache warming**: before serving a batch, the system computes synthetic queries and preloads sentence-level contexts known to be likely based on the prompt pattern. This reduces cold-start TTFT spikes considerably. The scheduler ties into the SpindleKV attention metrics: it de-prioritizes evicting tokens with rising attention scores, effectively letting the scheduler own the trade-off between memory savings and latency.

### Global attention scoring and compression shortcuts

Even with layers and sentences accounted for, eviction risks forgetting rare but crucial tokens. G-KV (Liu et al. 2025) [arxiv:2512.00504](https://arxiv.org/abs/2512.00504) adds a safeguard by computing a global attention importance score that blends recent attention history with a long-term decay. Each KV entry stores an aggregate score:
\[ g_t = \lambda \cdot \max_{\tau \in [t-T, t-1]} a_\tau + (1 - \lambda) \cdot \nu_t \]
where \(a_\tau\) is the attention weight at time \(\tau\) (like SpindleKV’s max), \(\nu_t\) is a cumulative importance (e.g., number of past hits), and \(\lambda \in [0,1]\) balances recency vs. longevity. This score prevents both transient high attention and persistent but low-attention contexts from being evicted before the scheduler can consider them, which is critical when auto-regressive decoding suddenly references an earlier token that had low attention earlier but becomes pivotal now.

Compression also plays a role. HACK (Chen et al. 2025) [arxiv:2502.03589](https://arxiv.org/abs/2502.03589) shows how to operate directly on quantized KV data to avoid expensive dequantization in disaggregated inference systems. The system stores KV entries in compressed form (e.g., 4-bit quantization with block floating scaling) and performs matrix multiplications via homomorphic-friendly approximations. During cache hits, the GPU can multiply query vectors against quantized keys without full-density reconstruction, reducing memory bandwidth and letting more entries reside in the cache. When needed, the scheduler temporarily promotes the raw (dequantized) value to GPU memory for fine-grained attention, but the majority of operations live in the compressed tier.

By combining sentence-level semantic batching, layer-aware policies, adaptive scheduling, global attention scoring, and quantized operations, modern KV cache management transitions from storing every token to orchestrating a multi-tiered memory pipeline that balances GPU pressure with long-context fidelity.

## Where the field is now

SentenceKV (Zhang et al. 2025) already leads the research frontier by proving that semantic re-batching can reduce GPU cache pressure without hurting downstream quality. SpindleKV (Wang et al. 2025) follows by layering attention-based retention on top of SentenceKV’s layout and demonstrating through ablations that shallow-layer merges keep perplexity gains intact. Apt-Serve (Li et al. 2025) then anchors the benchmark: it is the first to tie hybrid caches to TTFT SLOs and document scheduling gains on multi-tenant clusters serving billions of tokens. These papers collectively form the current research narrative: semantic structure matters more than raw token counts, and caches must be aware of both temporal attention and service-level constraints.

The engineering frontier is playing out in production-scale inference backends. GenAI for Systems: Recurring Challenges and Design Principles from Software to Silicon (2026) names Microsoft, Meta, and Google as companies already deploying tiered caches, highlighting that GPU memory limits are now the dominant bottleneck for both self-hosted deployments (Meta’s Llama 3 API) and hyperscale services (Microsoft + OpenAI). Their systems share the observation that real workloads rarely reuse raw token-level KV entries, so modern inference servers invest in routing logic above the hardware layer—in effect, what this page describes as semantic-aware eviction and adaptive scheduling. The consequence is that caching policies are no longer static: they are dynamic services that monitor attention scores, latency budgets, and queue depth in real time.

The DeepResearch-9K benchmark (2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152) provides a hard evaluation suite for these systems, forcing agents to span long interactions and multi-step reasoning. Running an inference stack against this benchmark exposes KV cache failures immediately; if cache management underestimates attention shifts, TTFT spikes, or retrieval errors surface, the agent fails on DeepResearch tasks. Reinforcement Learning Foundations for Deep Research Systems: A Survey (2025) [https://export.arxiv.org/pdf/2509.06733](https://export.arxiv.org/pdf/2509.06733) argues that robust KV cache policies are an essential component for any reinforcement-learned deep research agent, because policy rollouts need to query histories longer than those seen during training, and the cache must decide which parts of that history stay on the GPU. Finally, A Decade of Deep Learning: A Survey on The Magnificent Seven (2024) [https://arxiv.org/html/2412.16188](https://arxiv.org/html/2412.16188) frames these advances as part of the larger trend toward hardware-aware systems; KV cache management has now joined the classic seven axes because it directly determines whether a system can scale context windows while preserving latency.

## What's still open

Can we combine sentence-level grouping with online compression to maintain exact retrieval guarantees for any token that becomes important after being evicted? The current hybrids trade some fidelity for memory savings, but we lack a principled way to decompress or reconstruct pruned tokens when a long-context query suddenly references them. That capability would be the difference between "approximate" retrieval and the hard constraint of "never drop anything you'd need later."

How do we incorporate reinforcement learning directly into the scheduling policy so that cache hits are treated as actions in an RL environment? Apt-Serve’s scheduler models miss probabilities heuristically, but a policy gradient approach could learn, from data, which evictions generalize across prompts and which ones break under adversarial queries in DeepResearch-type benchmarks.

Finally, can we orchestrate multi-agent scenarios where each agent has different TTFT and throughput requirements, yet the shared GPU cache still respects all of them? Current schedulers bulk requests or assign weights, but the question is whether a single hybrid cache can maintain per-agent guarantees without replicating data.

## Where to read next

For the probabilistic backbone of attention, → [Attention Mechanisms](attention-mechanisms.md) explains how dot-product weighting defines the KV relevance signals that drives cache eviction priorities. If you want the systems rules that scale a scheduler, → *llm inference systems* <!-- [[llm-inference-systems]] --> shows how queues, GPUs, and disks align to serve large workloads with strict latency budgets. To understand how context compression compounds across an entire arc, → *compressed contexts for llms* <!-- [[compressed-contexts-for-llms]] --> traces from inference-time tricks all the way to training-time data shaping.

## Build it

This build proves that a multi-tiered KV cache manager can run on a single Colab session by combining sentence-level grouping, attention-aware eviction, and query similarity retrieval without requiring a fleet. The recipe winds a lightweight simulation around `HuggingFaceTB/SmolLM-135M` on a short WikiText sample so you can feel memory pressure and observe whether evicted sentences return correctly.

**What you're building:** A tiered KV cache simulator that demotes low-attention sentence blocks from GPU into CPU buffers and retrieves them via similarity during generation.

**Why this is valuable:** It forces you to implement both the sentence grouping that reduces GPU load and the attention/similarity logic that decides when to fetch sentences back, so you touch every layer described earlier.

**Stack:**
- **Model:** [HuggingFaceTB/SmolLM-135M](https://huggingface.co/HuggingFaceTB/SmolLM-135M) — 1,240 downloads, 135M tiny llama-style model
- **Dataset:** [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/2-raw-v1) — standard long-context challenge
- **Framework:** Hugging Face `transformers 4.40`, `accelerate 0.21`, `faiss-cpu 2.12` for similarity search
- **Compute:** Colab T4 (16 GB VRAM) or any single RTX 3060 (12 GB) — ~90 minutes total

**The recipe:**
1. Install the stack via `pip install transformers==4.40 accelerate==0.21 faiss-cpu==2.12 torch==2.2.1` and load `SmolLM-135M` with `device_map="auto"`; verify a simple generation works to confirm the KV cache exists on GPU.
2. Preprocess `wikitext-2-raw-v1` into sentences using a tokenizer-based sentence splitter; for each sentence, store its tokens, compute mean key and value vectors per layer/head, and insert them into a GPU dictionary keyed by a sentence ID while pushing the full token vectors into a CPU reservoir keyed by timestamp.
3. During token generation, maintain per-KV attention statistics \(a_t^*\) as the maximum attention weight seen in the last 32 steps (use the transformer's `outputs.attentions`); mark sentences with \(a_t^* < 0.02\) for demotion. When demoting, move their mean vectors to a `demoted_sentences` queue and replace the GPU entry with the next higher-scoring sentence from the CPU reservoir.
4. Implement similarity retrieval by embedding the current query \(q\) (projected via the final query head) and measuring cosine similarity against the demoted sentences using FAISS; if the top similarity exceeds 0.85, pull that sentence back into the GPU cache and reconstruct its per-token KV pairs from the CPU reservoir to continue generation.
5. Evaluate by generating continuations on three prompts from Wikitext and measuring two metrics: (A) average GPU KV memory usage before/after demotion (use `torch.cuda.memory_allocated()` after each chunk) and (B) response consistency by checking that with and without the tiered cache, the logits over a held-out token differ by less than 0.05 in KL divergence.

**Expected outcome:** A notebook that outputs a plot of GPU KV memory versus token position (showing plateauing after demotion) and logs similar logits between tiered and full caches, proving the simulation works.

- **CS student:** Run the same notebook on Colab with a 4 GB T4 by decreasing sentence batch size to four sentences before demotion and stashing CPU buffers in pin-memory to avoid swapping.
- **Applied engineer:** Extend the notebook to quantize the sentence vectors to 8-bit using PyTorch’s quantization utilities, then wrap the model in `vllm` and measure p95 latency at batch 1 with and without the tiered cache to verify you meet a 150 ms TTFT target.
- **Applied researcher:** Test the hypothesis that attention-aware eviction (SpindleKV) outperforms pure similarity-based eviction by toggling the attention threshold \(a_t^*\) between 0.01 and 0.05, plotting per-token retrieval latency vs. perplexity on Wikitext tokens.
- **Frontier researcher:** Investigate the open question by integrating a reinforcement-learned scheduler that treats evictions as actions—use a simple policy gradient to decide which sentences to demote and measure retrieval failures on synthetic prompts that intentionally reference pruned tokens.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*