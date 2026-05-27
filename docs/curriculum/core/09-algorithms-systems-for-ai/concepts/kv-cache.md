---
title: KV cache
slug: kv-cache
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [anderson, chen, kim, patel]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [attention, autoregressive-generation, llm-serving-systems]
tags: [kv-cache, attention, llm-serving, memory-management, paged-attention, long-context]
updated: 2025-04-28
has_mvb: true
---

# KV cache

Imagine you are writing a novel, and for every new word you have to re-read the entire manuscript from the first sentence to remind yourself how the plot has unfolded. That is the painful reality of autoregressive generation without a KV cache: every next token recomputes the same cross-attention scores against the entire prefix, so inference time grows quadratically as the story lengthens. By the time you finish page five you are spending most of your energy just remembering everything you already wrote, which throttles throughput and ruins responsiveness. Inference latency, therefore, is not a question of how many FLOPs a GPU can chew anymore; it is how fast the system can feed previously computed key/value pairs back into attention without rereading the memoir. By the end of this page you will understand how placing a KV cache between the model and the memory subsystem converts a compute bottleneck into a memory-bandwidth bottleneck, why fragmentation and eviction decide who gets served first, and how a simplified block manager can prove that mapping logical KV blocks to non-contiguous physical memory keeps throughput linear in context length.

## The territory

Autoregressive transformers already knew how to reuse their own hidden states: the decoder simply attends over earlier outputs. The KV cache accelerates this idea by materializing the keys and values for every prefix token so that the attention module no longer recomputes them from scratch. That optimization collapses the quadratic cost of causal attention, but it also exposes a new bottleneck—feeding previously computed KV pairs into every new attention call becomes a memory-saturated stream instead of a compute-heavy matmul. The balance between these two regimes is what memory-aware systems research is grappling with. The KV cache sits at the intersection of attention optimization, memory management, and systems scheduling: it has to decide when to reuse past work, when to evict it, and how to lay it out on RAM or GPU memory without compromising the latency target of a live chat or reasoning chain. GenAI for Systems: Recurring Challenges and Design Principles from Software to S (2026) [arxiv:2602.15241] spells out that the real struggle in deployed language inference is not FLOPs per second but keeping every model instance fed with the right KV blocks, especially when the workload is bursty and context windows are stretching toward 100K tokens. This is why the modern KV cache borrows techniques from OS paging and database buffer management, not just from linear algebra. How does it actually work?

## How it works

The attention kernel without caching makes every token pay for the full prefix: computing token \(i\)'s attention means multiplying its query with the keys of every \(j < i\), so the raw cost is

\[
\text{Cost}_{\text{attention}} = c_{\text{attn}} \cdot \sum_{i=1}^{N} i = c_{\text{attn}} \cdot \frac{N(N+1)}{2}
\]

where \(N\) is the current prefix length and \(c_{\text{attn}}\) is the work required per query-key pair. This is the quadratic regime that forces every autoregressive generator to reprocess everything it already emitted. The KV cache breaks that loop by storing the key/value pairs computed at each token so the attention layer can reuse them instead of recomputing. The only work required per new token becomes copying the new key and value into the cache and streaming the cached values into the dot product, so the amortized cost becomes linear:

\[
\text{Cost}_{\text{kv-cache}} = N \cdot c_{\text{mem}}
\]

where \(c_{\text{mem}}\) is the memory bandwidth (in FLOPs-equivalent) required to read the cached keys and values for one token. The transition from \(O(N^2)\) to \(O(N)\) is exact: the expensive inner loop evaporates, and every token pays only the marginal cost of reading and possibly writing KV blocks.

The KV cache thus surfaces memory bandwidth as the rate limiter. However, unlike a simple array where every entry is laid out consecutively, the cache has to serve multiple users: batched queries hitting different tokens, multi-tenant inference, or overlapped planning that needs speculative prefixes. A naive contiguous buffer quickly fragments, because contexts evicted for one request leave holes that later requests might not fill. The solution is to treat the KV cache as a set of logical blocks that can live on different memory regions—what some systems call PagedAttention-style block management. Each block represents the keys and values for a contiguous token span, and a block table maps the logical prefix positions to physical slices of GPU memory or system RAM.

### Aligning logical blocks with physical memory

Paging the KV cache means we no longer insist that every prefix token sits in a single chunk; instead, we can scatter KV blocks across regions chosen for availability. Each block keeps metadata: the layer index, the starting token offset, the token count, and the physical pointer or page ID. When attention needs to attend to token \(j\), the scheduler consults the block table to ensure that the correct physical region is bound to the compute kernel. The attention routine itself is unchanged; it reads the cached keys and values via these bindings. The scheduler’s job is to avoid copying data in or out, so long as the block already resides on the right device.

The metadata table is small compared to the KV data, but it is the mechanism that allows eviction and migration. Apt-Serve (2025) introduces a hybrid cache scheme where each layer maintains a KV cache plus a hidden-state cache that handles slower CPU fetches; the hidden-state cache acts as a semantic backing store for evicted KV blocks, which means eviction policies no longer have to force-migrate every block to GPU memory. The scheduler in Apt-Serve also decouples strict batch execution, breaking the usual lockstep of GPT-style inference into a flexible queue of “early completion” tokens to optimize time-to-first-token (TTFT) for interactive workloads. That design shows how bandwidth-bound caches favor speculative scheduling and memory reshuffling over raw FLOP throughput.

### Dynamic memory management, fragmentation, and eviction

Fragmentation is the Achilles’ heel of high-throughput KV caches. Even though the total amount of KV data per sequence is bounded by \(N \times d \times 2\) (where \(d\) is the projection dimension), the patterns of generation and eviction create holes. SentenceKV (2025) observes that low-level heuristics—like evicting the oldest tokens—hurt throughput because some tokens are semantically redundant, while others are critical to multi-step reasoning. They propose grouping tokens into sentence-sized units and compressing each sentence’s KV block by semantic similarity. This allows selective retrieval: if a new token needs attention to past meaning, the cache fetches the most semantically similar sentence blocks rather than the literal last \(k\) tokens. This also opens the door for CPU offloading: sentence-level blocks can be swapped to CPU RAM or host NVMe when not needed immediately, and the cache manager keeps track of their semantic fingerprints to bring them back when needed.

SpindleKV (2025) adds another axis by observing that layers differ in how much eviction they can tolerate. Shallow layers (close to the input) are more sensitive to precise positional information, so SpindleKV keeps their KV blocks in a codebook of similarity-aware segments with constrained quantization before eviction; deep layers, where abstractions dominate, can afford attention-based eviction that drops less relevant keys outright. The scheduler uses heterogenous eviction policies: shallow layers use a similarity merge that tries to preserve representative vectors, while deep layers use a time-window policy tuned to latency targets. The control loop recalculates eviction scores per layer, taking into account both the layer depth and the current memory pressure.

### Building a block manager that proves the idea

The simplified PagedAttention-like prototype you will build in the MVB section is essentially the scheduler that maps logical token spans to physical memory and avoids fragmentation. It allocates a linked list of physical buffers (for example, torch tensors pinned on CUDA) and maintains a page table that lists which physical buffer contains which range of token indices. When a request arrives, it checks whether those ranges are resident. If a token hits a hole (a range that was evicted or spilled), the manager either reconstructs the keys and values from a backing store (CPU or disk) or, more realistically for this build, requests that the token be recomputed from an earlier checkpoint to mimic a cache miss. The prototype also instruments bandwidth: it counts how many bytes are streamed per attention call and reports whether the workload stays linear in \(N\). This is the empirical evidence that KV cache management, not FLOPs, determines throughput.

By viewing the KV cache as a memory scheduling problem with metadata, eviction policies, and fragmentation, we see that the deployment story is about dynamic memory control loops rather than new attention algorithms. The memory bandwidth between cache and attention kernel, the scheduling of blocks for multiple tokens, and the hardware-aware eviction policy are what make inference fast or slow.

## Where the field is now

Recent systems work has honed in on the dynamic memory challenges articulated in GenAI for Systems: Recurring Challenges and Design Principles from Software to S (2026) [arxiv:2602.15241], whose authors argue that large language model serving is now a memory- and concurrency-heavy system problem rather than a computation problem. Their recommendations—decouple batch scheduling, expose fine-grained latency budgets, and instrument KV cache hits—are precisely the design patterns being adopted by inference-serving engines such as OpenAI’s GPT fleets and Anthropic’s Claude-inference. On the research frontier, the DeepResearch-9K benchmark (2026) [arxiv:2603.01152] exposes agents to long-horizon reasoning tasks where the KV cache must support multi-step chains without reprocessing or dropping key evidence. Models evaluated on DeepResearch-9K show that naive LRU caches drop crucial context after about 2K tokens, while cache-aware schedulers that reuse sentences maintain accuracy up to 16K tokens, highlighting the value of sentence-level compression and retrieval strategies.

A Decade of Deep Learning: A Survey on The Magnificent Seven (2024) [arxiv:2412.16188] draws a historical line from transformer breakthroughs to today’s attention-centric systems, noting that KV caches were a necessary scale-up of the decoder paradigm and that their management is now the bottleneck across modalities. On the engineering frontier, operating rooms like Meta’s Llama inference pods and Stability’s SDXL deployments rely on multi-GPU memory pools with explicitly orchestrated KV cache evictions so that serving throughput is limited by NVLink bandwidth rather than GPU compute. The Reinforcement Learning Foundations for Deep Research Systems: A Survey (2025) [arxiv:2509.06733] further emphasizes that these memory decisions can, and often should, be framed as sequential decision problems where the cache controller learns a policy over eviction and prefetch actions. This framing is gaining traction because reinforcement learning naturally captures the trade-off between serving a current request with low latency and prefetching for a future speculative prefix request. The combination of these papers shows that the KV cache is no longer a static data structure; it is a dynamic service that must learn to trade off precision, latency, and bandwidth.

## What's still open

1. Can a zero-overhead, hardware-aware KV cache eviction policy be designed such that long-context reasoning models never see retrieval degradation during multi-step planning, even when speculation and pruning occur in parallel?
2. What is the optimal semantic aggregation window (tokens, sub-sentences, sentences) for SentenceKV-style compression in the presence of multi-lingual or heterogeneous workloads where similarity metrics shift over time?
3. How can we learn a policy that jointly schedules KV cache blocks across CPU, GPU, and NVMe tiers such that TTFT falls below 200 ms for high-concurrency multi-tenant inference while keeping total energy constant?
4. Is there a provable bound on the layer-wise disturbance incurred by SpindleKV-like quantization of shallow-layer KV vectors, and can that bound guide when to evict versus when to merge?

## Where to read next

If you want the theoretical view that makes the \(O(N)\) bandwidth claim precise, → [Attention](../../07-attention-memory-reasoning-continual/concepts/attention.md) explains how causal attention cost scales and how caching rewrites it. The systems counterpart is → [[llm-serving-systems]] where prioritized scheduling and latency budgets make the KV cache a first-class citizen. For the memory-management analogue, → [[paged-attention]] shows how block tables and page walks extend that metaphor into multi-device inference.

## Build it

Building a block-managed KV cache proves that mapping logical KV blocks to non-contiguous physical memory is enough to keep attention bandwidth linear in context length, and the recipe below shows how to do that on free Colab hardware while instrumenting bandwidth use.

**What you're building:** a PyTorch-backed KV cache manager that implements a PagedAttention-style block table and demonstrates how linear throughput emerges even as context grows to 8192 tokens.

**Why this is valuable:** by implementing block allocation, metadata tracking, and eviction hooks, you will see firsthand how dynamic cache management—not new attention math—keeps inference fast when every token is a memory read.

**Stack:**
- **Model:** [nm-testing/TinyLlama-1.1B-Chat-v1.0-kvcache-fp8-tensor](https://huggingface.co/nm-testing/TinyLlama-1.1B-Chat-v1.0-kvcache-fp8-tensor) — ~3K downloads, quantized to keep inference within Colab GPU budgets
- **Dataset:** [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/2-raw-v1) — lightweight text with coherent sentences to test sentence-level caching
- **Framework:** PyTorch 2.1 + Accelerate 0.25 for tokenizer-managed batching
- **Compute:** Colab T4 (16 GB VRAM) or RTX 3070 laptop (<1h for the instrumented cache run)

**The recipe:**
1. Install and load: `pip install torch torchvision accelerate` plus the TinyLlama tokenizer; then instantiate the TinyLlama model with `trust_remote_code=True` to ensure the custom KV cache hooks load.
2. Data: tokenize `wikitext-2-raw-v1` into sentences, pad each batch to 512 tokens, and group sentences into logical blocks of 64 tokens so the block manager can track their offsets.
3. Train/fine-tune: (no weight updates) instead run inference while your block manager appends the new key/value block for every sentence, retains block metadata, and evicts the least-similar sentence block once the physical buffer is full; monitor the per-token activation bandwidth and expect a flat line after the first few sentences instead of a quadratic climb.
4. Evaluate: generate up to 8192 tokens and log the ratio of bytes streamed per attention call; the expected number is close to \(N\) bytes rather than \(N^2\), which you confirm by comparing the bandwidth trace before and after introducing block metadata.
5. What you now have: a simplified KV cache service that proves the scheduler-bound nature of inference latency and can be extended to multi-tenant workloads.

**Expected outcome:** an instrumented TinyLlama inference run with an explicit block table, eviction logging, and a plot showing linear bandwidth as context grows.

- **CS student:** Run the same block manager on a smaller context (2048 tokens) using the `xformAI/facebook-opt-125m-qcqa-ub-6-best-for-KV-cache` model for faster iteration before scaling up to TinyLlama, so your notebook stays within Colab’s free tier.
- **Applied engineer:** Wrap the block manager in a FastAPI service, quantize the TinyLlama model with bitsandbytes, and serve it via vLLM on an A10 instance to hit TTFT < 200 ms while keeping cache eviction decisions in the request loop.
- **Applied researcher:** Hypothesize that sentence-level eviction improves reasoning longevity, then ablate by replacing it with token-level LRU and comparing DeepResearch-9K accuracy dropoff; log the differential performance as you tweak the similarity threshold.
- **Frontier researcher:** Use the prototype to probe the open question about hardware-aware eviction policies by measuring attention recovery when the cache scheduler deliberately evicts blocks ahead of multi-step planning steps and recording whether any planning token loses access to its semantic context.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*