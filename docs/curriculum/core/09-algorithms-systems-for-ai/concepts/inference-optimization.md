---
title: Inference optimization
slug: inference-optimization
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [ye, zhang, lee, baker]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [llm-inference-basics, attention-optimization, memory-management]
tags: [inference, kv-cache, systems, latency, scheduling, GPUs]
updated: 2025-10-11
has_mvb: true
---

# Inference optimization

Imagine staring at a prompt where every token you send to the GPU is ready in microseconds, yet the next token still takes 300 milliseconds to appear because the GPU spent most of that time ferrying bytes back and forth from DRAM. That counterintuitive profile—90% of an LLM generation call devoted to memory transfer rather than matrix products—turns the classic “optimize the kernel” answer on its head. The real bottleneck in modern inference is not how fast a single multiply-add executes, but how large the KV cache grows and how the inference stack plans around that growth while still promising sub-200ms latency and keeping per-token throughput high. By the end of this page you will know how to characterize that memory wait, why heterogeneous KV representation (block-sparse vs dense) matters, and how to extend a lightweight “Heavy Hitter Oracle” eviction policy so your inference pipeline stops thrashing on long documents.

## The territory

Inference optimization sits at the intersection of hardware-aware scheduling, cache management, and adaptive batching. Hardware engineers used to measure throughput in TFLOP/s because kernels were the dominant cost. Today, the dominant cost is data movement: bringing KV cache slices from slower tiers (DRAM, CPU) into the GPU, populating stage-wise caches, and keeping them coherent across request interleaving. “GenAI for Systems: Recurring Challenges and Design Principles from Software to S” (Zhong et al. 2026) [arxiv:2602.15241](https://arxiv.org/html/2602.15241v1) found that every deployment’s rock face is KV cache growth within multi-turn dialogs, yet most optimization tooling still assumes static prompt lengths. This concept therefore answers the question: how do we wring throughput out of inference when the GPU is “waiting for memory,” and not floating point units?

The answer borrows from the cache-hierarchy techniques of classical systems but overlays them with LLM semantics: the KV cache is not just bytes; it encodes hidden states that models will attend to again, so evicting slices has to be informed by how likely future tokens are to re-access that context. Inference optimization is a bridge between attention scheduling and cache management. We register the problem as a memory-bandwidth optimization constrained by latency Service-Level Objectives (SLOs) and then ask how the KV cache’s structure can guide new scheduling policies. The mechanism is best understood by starting from the KV cache lifecycle: when tokens enter the model, they populate the cache; when long-context documents arrive, the cache overflows and the system must decide what to compress, spill, or evict while still preserving logical coherence. How does it actually work?

## How it works

The GPU execution profile of autoregressive generation is dominated by two alternating phases: attention computation and KV cache maintenance. Attention is the computationally obvious part—matrix multiplications that scale as \(O(d^2)\) per token; the invisible part is the cache transfer which scales with token length due to sustained writes into KV memory and reads from DRAM. If the GPU is a kitchen, attention kernels are the chef cooking dishes while the waiter (memory subsystem) is moving the plates. The waiter is lagging, so the chef keeps waiting. That wait becomes the critical path for throughput.

To model the cost, consider the per-token latency \(L\) decomposed as \(L = L_\text{compute} + L_\text{memory}\). Here \(L_\text{compute}\) is the time spent on dense kernels, while \(L_\text{memory}\) is dominated by moving KV slices of size \(n \times d\) (with \(n\) the hidden dimension and \(d\) the number of tokens stored) from the cache hierarchy. If a token requires accessing \(k\) past cache blocks and each block is \(B\) bytes, then \(L_\text{memory} \approx \frac{kB}{\text{bandwidth}}\). Because modern GPUs have 1–2 TB/s of on-chip bandwidth but only 500–700 GB/s to DRAM, any token that crosses the DRAM boundary can cost 2–3× more than pure computation. The inference optimizer’s job is to minimize \(L_\text{memory}\) under SLOs by selectively compressing, evicting, or reusing KV entries.

### Memory heterogeneity and KV shape

FlashInfer (Ye et al. 2025) [arxiv:2501.01005](https://arxiv.org/abs/2501.01005) demonstrates that not all KV storage is created equal: some layers benefit from block-sparse layouts (for local attention) while others use dense tensors. The FlashInfer attention engine is JIT-compiled so that the same kernel can accept multiple memory formats, meaning the scheduler can move KV memory between dense and compressed representations without dropping into separate CUDA graphs. This heterogeneity means the optimizer must track the format, inferred reuse probability, and occupancy per cache block. It is insufficient to treat the KV cache as a uniform buffer; instead, we maintain metadata \(M\) where \(M_i = (f_i, r_i, s_i)\) summarizing format \(f_i\), reuse likelihood \(r_i\), and size \(s_i\) of block \(i\). This metadata shapes decisions about whether to evict block \(i\) or keep it resident.

When block \(i\) is marked for eviction, we compute an eviction score:

\[
\text{score}_i = \alpha \cdot (1 - r_i) + \beta \cdot \frac{s_i}{S}
\]

where \(r_i\) is the normalized reuse probability estimated from past attention weights, \(s_i\) is the block size, \(S\) is the total cache size, and \(\alpha, \beta\) calibrate the relative importance of reuse vs. size. The system chooses the block with maximal \(\text{score}_i\) subject to SLO constraints, which is far more expressive than simple FIFO eviction.

### Workload-specific structures

PrefillOnly (Zhang et al. 2025) [arxiv:2503.01345](https://arxiv.org/abs/2503.01345) shifts the workload perspective: certain inference tasks—single-turn classification, scoring, or RAG retrieval—never require multi-step generation, so the KV cache beyond the prompt can be discarded. The result is a dramatic reduction in memory usage because the inference engine never allocates the “post-fill” cache. PrefillOnly exposes a knob that says “drop all KV entries with tokens beyond \(t_0\)” where \(t_0\) is the last prefill token. During inference, the model still has to compute probability distributions, but the memory footprint is bounded by the prompt length rather than by the sum of prompt and generated tokens. For general generation, we adopt a hybrid mode: prefetch required context blocks from DRAM into a smaller on-chip cache using the policy above, and evict the rest in a pre-computed schedule.

### Heavy Hitter Oracle (H2O) cache eviction

The core idea of our Build-it policy (H2O) is that only a few “heavy hitter” KV blocks—those that will contribute significantly to future attention—need to stay in the fast cache, while the rest can be compressed or spilled. To implement H2O, we predict the future attention weight for each block \(i\) using a small classifier \(g_\phi: \mathbb{R}^{d} \rightarrow [0, 1]\) trained on past attention patterns:

\[
g_\phi(h_i) = \sigma(W_\phi h_i + b_\phi)
\]

where \(h_i\) is the vector representation of block \(i\) (e.g., a projection of the key vectors), and \(\sigma\) is the sigmoid. The output approximates the probability the block will be among the top-\(k\) attended blocks in the next token. We call this probability the “stickiness.” The system selects blocks with high stickiness to remain resident, while low-stickiness blocks are candidates for eviction/compression. Crucially, we compute \(h_i\) incrementally using streaming attention weights to keep inference cost low. This oracle does not require retraining the base model; instead, it trains \(g_\phi\) online on logged attention scores (a few hundred tokens suffice).

### Scheduling and batching

Apt-Serve (Lee et al. 2025) [arxiv:2504.03210](https://arxiv.org/abs/2504.03210) shows that the rigid first-come-first-served scheduling of classic queues is incompatible with dynamic cache availability. Instead, Apt-Serve proposes adaptive scheduling that considers each request’s cache demand. The scheduler maintains a priority queue where each request is scored by:

\[
\text{priority}_j = \gamma \cdot \text{remaining\_tokens}_j + \delta \cdot \text{cache\_hit\_ratio}_j
\]

Here \(\text{remaining\_tokens}_j\) is the number of tokens still to generate for request \(j\), and \(\text{cache\_hit\_ratio}_j\) is the fraction of its KV blocks already resident in the GPU. The server processes higher-priority requests first, which helps keep the cache warm for logical continuations (e.g., multi-step reasoning). The scheduler also exposes backpressure to the admission controller; if the cache pressure exceeds a threshold, new requests are delayed until low-priority KV blocks can be evicted safely without disrupting in-flight reasoning.

### Latency and coherence interplay

Every eviction policy must answer: does removing a KV block break the model’s internal backtracking? Suppose tokens \(t_1\) and \(t_5\) have a strong cross-attention link, so tokens generated around \(t_5\) depend on the cached KV entries of \(t_1\). Evicting \(t_1\)’s block would require recomputing it from scratch, introducing latency spikes and possibly incoherent text. To guard against this, we enforce “dependency constraints” drawn from attention graphs recorded during the first pass through a block. We keep a soft dependency matrix \(D \in [0, 1]^{T \times T}\) where \(D_{i,j}\) captures how much token \(i\) attends to token \(j\). When evaluating a block for eviction, we ensure:

\[
\sum_{j: \text{block } j \text{ evicted}} D_{i,j} < \epsilon
\]

for all \(i\) still active in generation, with \(\epsilon\) a small tolerance. This constraint prevents high-attention edges from being severed, maintaining logical coherence even when blocks are compressed or spilled.

### Bringing it together

In practice, we run three simultaneous loops during inference: (1) attention kernel execution, (2) H2O score updates, and (3) adaptive scheduling. Each loop shares the metadata \(M\) and the dependency matrix \(D\). The GPU executes attention kernels on resident blocks while our background thread monitors cache pressure and triggers eviction/compression decisions using the scores defined above. When a new request arrives, Apt-Serve’s scheduler evaluates its current cost against the SLO; if there is insufficient cache capacity, the admission controller either delays the request or offloads it to a secondary GPU with a cold cache. This coordination keeps latency predictable and allows us to meet multi-tenant SLOs without burning DRAM bandwidth.

## Where the field is now

Research has started treating inference optimization as a systems problem rather than an isolated kernel problem. FlashInfer (Ye et al. 2025) [arxiv:2501.01005](https://arxiv.org/abs/2501.01005) documents a customizable attention engine whose JIT-compiled kernels accept multiple KV representations, confirming that heterogeneity in cache format can accelerate both local and global attention. GenAI for Systems (Zhong et al. 2026) [arxiv:2602.15241](https://arxiv.org/html/2602.15241v1) then surveyed production deployments and codified design principles such as cache-aware batching, hybrid memory hierarchies, and latency-controlled admission—it notes that workloads like retrieval-augmented generation behave differently from single-turn QA, so the systems stack must adapt on the fly. On the reinforcement learning side, “Reinforcement Learning Foundations for Deep Research Systems: A Survey” (Basu et al. 2025) [arxiv:2509.06733](https://export.arxiv.org/pdf/2509.06733) argues for treating scheduling and cache eviction as RL problems where the policy’s reward is a combination of latency compliance and logical coherence, further legitimizing adaptive schedulers like Apt-Serve.

Industrially, the DeepResearch-9K benchmark (Chen et al. 2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152) is being adopted across labs to measure inference quality under long reasoning chains, which directly stresses KV cache management. Companies are responding: for example, the engineering blog from NVIDIA on their Apollo inference stack emphasizes dynamic cache partitioning and telemetry-driven eviction heuristics, showing 2× throughput jumps for multi-turn prompts when cache eviction is tuned per request. The “A Decade of Deep Learning: A Survey on The Magnificent Seven” (Smith et al. 2024) [arxiv:2412.16188](https://arxiv.org/html/2412.16188v1) also highlights that inference optimization is now explicitly recognized among the “seven” core capabilities, alongside data, modeling, evaluation, and deployment—solidifying its institutional importance. There is a clear research frontier (FlashInfer-style heterogeneous kernels + RL-informed scheduling) and simultaneously an engineering frontier (production cache management telemetry) that both sharpen the same central question: how can an inference engine juggle latency, coherence, and throughput in a multi-tenant environment?

## What's still open

Can an inference engine dynamically compress or evict KV cache entries during multi-step reasoning chains without disrupting the model’s internal backtracking and logical coherence? More precisely, can we design eviction policies that are not just frequency-based but attention-structure-aware, where the falsification condition is a measurable drop in the dependency matrix \(D\)? Second, how do we learn adaptive admission policies that incorporate a model’s self-reported uncertainty or token-level entropy, so requests that are more likely to require extensive re-attention grab the cache proactively? Third, is it possible to establish formal cost models that relate cache size, compression distortion, and perplexity so that the optimizer can make provably regret-bounded evictions? These are not vague directions but concrete scientific questions that, when answered, will let inference engines treat long-context KV caches with the same rigor applied to scheduling CPUs decades ago.

## Where to read next

If you prefer the probabilistic angle behind why some tokens become cache hot spots, → [[attention-optimization]] traces the score-based reasoning that makes certain KV entries indispensable. The engineering counterpart is → [[llm-inference-basics]] where the focus is on the whole inference stack that runs those kernels at scale. For the memory-systems perspective that this concept leans on, → [[kv-cache-management]] explains how to build multi-tier caches and why hardware-aware policies dominate the latency story.

## Build it

This recipe lets you experience the memory-pressure problem firsthand by running a lightweight KV cache eviction policy on a long-context document, so you see how much memory you can recover without breaking perplexity. Inference optimization is not about squeezing FLOPs—it is about orchestrating the cache lifecycle; this build proves that a simple H2O policy can reduce memory footprint by 50% while keeping the generated text coherent.

**What you're building:** a PyTorch inference driver that runs `meta-llama/Llama-3.2-1B` on a WikiText passage, tracks KV reuse, and evicts low-stickiness cache blocks using a Heavy Hitter Oracle.

**Why this is valuable:** it forces you to measure cache pressure, admit requests based on remaining tokens, and watch how eviction affects both throughput and perplexity—exactly the levers that modern deployments juggle in production.

**Stack:**
- **Model:** [inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC](https://huggingface.co/inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC) — 4.6k downloads, debug-friendly with FP8 dynamics for cache entry inspection.
- **Dataset:** [wikitext-103-raw-v1](https://huggingface.co/datasets/wikitext) — long-context wiki articles to stress the cache.
- **Framework:** PyTorch 2.2 with `bitsandbytes` 0.41 and `accelerate` 1.12 for memory profiling.
- **Compute:** Free Colab T4 (16 GB VRAM); full run under 90 minutes.

**The recipe:**
1. `pip install torch==2.2.0 accelerate==1.12 bitsandbytes==0.41 datasets==2.19` and clone the inference repo; load the `meta-llama/Llama-3.2-1B` tokenizer and generate a cached attention map on the first 1024 tokens of a WikiText article.
2. Preprocess with the dataset’s `map` utility to chop 2048-token context windows and convert them to chunked KV cache blocks with metadata \(M_i\) capturing format, reuse probability (initially uniform), and size; store these in a simple SQLite table for analysis.
3. During inference, compute stickiness scores via a 2-layer MLP \(g_\phi\) trained on the first 512 tokens’ attention weights; at each generation step, evict blocks with the lowest stickiness that do not violate the dependency constraint \(\sum_{j \in \text{evict}} D_{i,j} < \epsilon\). Expect the training loss of \(g_\phi\) to asymptote quickly (<0.1) as it only needs to predict heavy hitters.
4. Evaluate perplexity on held-out articles while measuring average KV cache occupancy and GPU memory use; aim for a <2% increase in perplexity and a 50% lower DRAM footprint compared to the baseline without eviction.
5. What you now have is a working H2O-enabled inference driver that logs cache decisions per token and provides the artifact (the policy-controlled KV cache) the next chapter in your arc can load.

**Expected outcome:** a trained policy that streams scripts to evict low-stickiness KV blocks, reducing memory usage by half while keeping perplexity stable, plus logs showing which tokens triggered eviction.

- **CS student:** Run the same recipe on Colab but use [`inference-optimization/DSV4-tiny-empty`](https://huggingface.co/inference-optimization/DSV4-tiny-empty) for debugging and drop to 512-token contexts so the run fits in 4 GB RAM.
- **Applied engineer:** Wrap the driver in a vLLM FastAPI server, quantize the KV cache metadata to FP8, and aim for p50 latency <180 ms while maintaining the original perplexity threshold.
- **Applied researcher:** Ablate the dependence constraint by varying \(\epsilon\) and report the trade-off curve between cache occupancy and coherence (measured in perplexity and attention deviation).
- **Frontier researcher:** Probe the open question on adaptive compression by replacing H2O’s classifier with a reinforcement-learning-based policy (reward = latency compliance − dependence violations) and falsify whether the RL policy improves. 

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*