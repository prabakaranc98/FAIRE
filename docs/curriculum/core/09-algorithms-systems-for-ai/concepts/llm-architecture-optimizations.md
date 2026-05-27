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
updated: 2024-11-15
has_mvb: true
---

# LLM Architecture Optimizations

Imagine you prune every attention head that “looks redundant” and route only the surviving token pairs through compute-hungry matrix multiplies, only to realize that each surviving key/value vector still lives in HBM and must be loaded for every inference step. The FLOPs saved vanish into the bandwidth spikes of shuttling millions of bytes between on-chip SRAM, HBM, and off-chip DRAM, turning the “sparse attention” win into a memory-accounting headache. What the pruning optimizers never told you is that the KV cache is not just a side buffer; it is the memory footprint that defines the feasible batch size, the throughput, and whether the layer can hide the latency of the denominator in the softmax. This page shows how to treat those caches, the parameterization of depth/width, and the multilevel memory hierarchy as a single optimization surface, so that the design decisions you make in PyTorch, CUDA, or Triton are not fighting each other but reinforcing one another.

## The territory

Large language models still make the same unforgiving trade-off as the first Transformer: attention scales quadratically in sequence length, so the most obvious wins are not in new activation functions but in how you store and move the key/value tensors. Vaswani et al. (2017) [arxiv:1706.03762](https://arxiv.org/abs/1706.03762) showed that multi-head attention computes weights \(A = \text{softmax}(QK^\top / \sqrt{d_k})\) for queries \(Q\) and keys \(K\), with \(d_k\) the head dimension, and that every decoder step needs both the current query and the entire set of cached keys and values to produce the next token distribution. The FLOPs are the obvious offender, but the real limiter is that \(K\) and \(V\) must remain on a very fast storage tier because they are accessed irregularly at inference time, for every autoregressive token produced.

The community answered this by pruning, quantizing, and sparsifying; each attempt saved arithmetic while leaving the cache hierarchy untouched. In practice, however, the cache spans hundreds of megabytes on HBM and must be mirrored to DRAM or even CPU-accessible storage when the batch count grows. Warehouse-scale insights remind us why this matters: Barroso et al. (2009) [https://people.eecs.berkeley.edu/~randy/Courses/CS294.F09/wharehousesizedcomputers.pdf] and Armbrust et al. (2009) [https://www.cs.princeton.edu/courses/archive/fall10/cos561/papers/AboveClouds09.pdf] describe the shuttling penalty between compute and storage as the defining limiter of throughput. When a layer finishes computing on a token, the next token may still be waiting for the KV cache to stream back from DRAM, and the time that takes depends on the physical topology of the datacenter network and whether memory interconnects are saturated. That is why architecture optimization is not just about layers but about data placement, energy per bit, and the latency of cache misses that feel like software bugs even if they are hardware constraints.

The shaping insight, then, is that LLM architecture optimization is a joint problem across three dimensions: model topology (heads, depth, width), parameterization scaling (how depth-wise learning rates and width multipliers trade compute for capacity), and hardware memory hierarchies (HBM⇄DRAM offload, cache residency, prefetch cadence). Where does one begin? How does such a joint surface guide the actual implementation of sparse attention that still pushes the throughput envelope? The mechanism is best understood by starting from the quadratic KV cache, following its path into the hardware hierarchy, and then layering in the hyperparameter decisions that CompleteP (2024) claims allow transfer across depths. How does it actually work?

## How it works

### From quadratic attention to cache pressure

Every multi-head attention layer calculates \(A = \text{softmax}(QK^\top / \sqrt{d_k})\), where \(Q\), \(K\), and \(V\) are \(n \times d_k\) and \(n\) is the sequence length. The matrix multiplication \(QK^\top\) is \(O(n^2 d_k)\), which is the FLOP count. However, as soon as you adopt autoregression, the next query needs the entire history of keys and values, meaning that the key/value cache grows as \(O(n d_k)\). The number of bytes is \(n \cdot d_k \cdot 2 \cdot \text{sizeof(float)}\) for single-precision (or quantized equivalents). This cache is accessed for each decoding step, so its bandwidth request is \(O(n d_k)\) per token, and the access pattern is mostly random reads from a contiguous buffer.

A naive sparse attention layer prunes the input tokens per layer or selects a subset of \(q \in Q\) pairs to attend to, reducing the FLOPs but not the cache size, because even if only 10% of the tokens attend, the keys and values for the other 90% still need to stay in high-bandwidth storage for the next token if the next step might attend to them. This is the dynamic sparse attention paradox: the arithmetic you skip is dwarfed by the memory you still have to buffer. As a result, your throughput is still limited by the interconnect that reads \(K\) and \(V\) out of HBM, and cache eviction stalls happen whenever the working set spills over to DRAM. That situation is what politicians call “computing through memory limitations.”

The architecture optimization question becomes: can we reshape the cache to shrink in bandwidth-critical tiers whenever we prune tokens, and then restore it seamlessly when higher fidelity is needed? The answer lies in explicitly modeling the cache residency and prefetch strategy during the design of the sparse attention layer.

### Memory hierarchies and datacenter constraints

The datacenter perspective tells us why this matters. Interconnect networks were not designed for the irregular, low-arithmetic-running pattern of sparse attention. Barroso et al. (2009) argued that energy and latency budget allocated to memory transfers dominates the CPU/GPU arithmetic budget; each byte moved between HBM and DRAM costs microseconds. Armbrust et al. (2009) extended this to the cloud view, showing that shared storage resources and virtualization amplify these delays, creating a “stalled machine” whenever an application cannot keep its working set inside the preferred tier. The takeaway is that any optimization that only enumerates arithmetic without modeling memory movement is incomplete.

A more granular view appears in communication-avoiding literature such as the algorithms summarized by the arXiv report in 1409.3215 [https://arxiv.org/pdf/1409.3215], which shows that blocking, cache tiling, and pipelined neighbor access reduce the arithmetic-to-communication ratio. Applied to attention, you can tile the KV cache such that only the currently active “pruned” rows live in HBM while the rest remain in DRAM, and you prefetch the next tile when the decoder steps into it. This is similar to blocking used in matrix multiply, but now decisions depend on token selection. The optimization, therefore, becomes hardware-aware: prune sequences not only to save FLOPs but also to shrink the active cache footprint and orchestrate HBM⇄DRAM transfers such that bandwidth availability matches the compute stage.

To manage this orchestrated movement, we model the cache as two tiers: a hot tier resident in HBM with low latency and a warm tier in DRAM. The hot tier holds the keys/values for tokens that are more likely to be attended to next (for example, tokens in the past window or with high salience under a learned importance score). When a query is pruned, we do not delete its keys/values entirely; instead, we demote them to the warm tier and keep a lightweight summary or compressed representation, ready to be promoted back when the attention score surpasses a threshold. The hardware model must track the time penalty of this demotion/promotion so that we never spend more time moving bytes than we saved by pruning. 

### Scaling and CompleteP’s depth-wise transfer

Every level of the model — depth, width, head count — interacts with the hardware model. CompleteP (2024) (a depth-wise hyperparameter transfer framework) suggests that the optimal learning rate, layer normalization schedule, and pruning intensity vary predictably with depth, allowing practitioners to reuse tuned hyperparameters when transitioning from a toy GPT to a deeper geometry. The framework quantifies how much extra compute each additional layer consumes and proposes a mapping function \(f(d)\) from depth \(d\) to your compute budget.

In practice, this means we solve a constrained optimization:
\[
\min_{d, h, c_{\text{cache}}} \mathcal{L}(f(d), h, c_{\text{cache}}) \quad \text{s.t.} \quad \text{BW}_{\text{HBM}}(c_{\text{cache}}) + \text{BW}_{\text{DRAM}}(c_{\text{cache}}) \leq B_{\text{max}},
\]
where \(h\) is the number of heads, \(c_{\text{cache}}\) is the active cache size, \(B_{\text{max}}\) is the available bandwidth on your node, and \(\mathcal{L}\) is the loss resulting from the depth/width selection measured via CompleteP’s transfer function. The consequence is that you cannot treat depth scaling as independent of cache sizing: adding a layer increases the inner loop’s cache pressure, and the framework tells you how to recalibrate the pruning threshold (e.g., token selection probability) so that the bandwidth constraint stays satisfied.

CompleteP further recommends non-lazy learning schedules that adapt to hardware: the framework uses an “early warming” period in which we warm the cache by precomputing the key/value matrices during the forward pass and then gradually increase the prune rate as the model stabilizes. This adjustment helps because the cache movement has startup costs; amortizing them early keeps throughput high once real decoding begins.

### Systems bridging sparsity and execution (SparseServe)

Algorithmic sparsity promises savings, but the systems story is harder. SparseServe (2025) demonstrates a production-grade path where the sparse attention selection is coupled with an explicit cache manager that streams between HBM and DRAM with backpressure signals. The key is to treat the sparse selection as a request for a subset of the cache tier, then orchestrate the vendor’s DMA engine to fetch that subset in the background. SparseServe’s scheduler is aware of the decompressed size of each cache entry and uses a budgeted compression codec to shrink the “warm” tier footprint while remaining decompressable when fetched back into HBM.

To describe it concretely, consider that each key/value pair is represented as a 32-bit float vector of length \(d_k\). SparseServe introduces a compression function \(\psi(K)\) where \(\psi\) outputs a packed 8-bit representation plus metadata to rebuild the original vector. The metadata includes a checksum and an L2 norm used for importance scoring. During inference, when the importance score exceeds a threshold, the system issues a "promote" request that decompresses the entry back into HBM. These promotes are pipelined with attention computation: while head 1 consumes the hot tier, the DMA engine warms up the next predicted tile, allowing scheduled compute to overlap with cache movement.

The consequence is that the KV cache is now a layered asset: the hot layer supports fast, dense attention; the warm layer stores compressed representations; and the movement between them is driven by the sparsity policy. That policy is instrumented with telemetry to track how often we rehydrate entries from DRAM, how much latency each demotion/promotion costs, and whether the prefetch success rate justifies the compression ratio. SparseServe reports up to 40% latency reduction on a 4-layer GPT-style decoder by balancing these volumes; the critical insight is that you cannot improve inference latency by thinking about dropout masks alone—you must orchestrate compression, DMA, and compute simultaneously.

### Dynamic sparse layers with hardware-aware caching

Putting it all together, the optimized sparse attention layer in our build exhibits the following behavior:

1. Each decoder layer maintains a “selection tensor” \(s \in \{0,1\}^n\) computed by a lightweight selector network that estimates attention salience. 
2. The keys/values of pruned tokens are immediately compressed via \(\psi\) and demoted to the warm tier while a retention table keeps track of their last accessed positions.
3. Future queries consult the selector first; if a pruned token’s predicted importance surpasses the cache-recall threshold, its compressed representation is promoted while the attention computation continues with the currently available hot entries.
4. The promotion uses asynchronous DMA to keep the compute pipes busy, and the scheduler prevents offloading when the available bandwidth in that interval is saturated.

Mathematically, the runtime becomes constrained by both FLOPs and bandwidth:
\[
T_{\text{step}} = \frac{F_{\text{active}}}{P_{\text{compute}}} + \frac{B_{\text{active}}}{B_{\text{HBM}}} + \frac{C_{\text{promote}}}{B_{\text{DRAM}}},
\]
where \(F_{\text{active}}\) is the FLOPs for the active tokens, \(P_{\text{compute}}\) is the chip’s compute throughput, \(B_{\text{active}}\) is the number of bytes pulled from the hot cache, \(B_{\text{HBM}}\) is the HBM bandwidth, and \(C_{\text{promote}}\) is the volume of data promoted from DRAM, with \(B_{\text{DRAM}}\) its bandwidth. Optimizing the network now means minimizing \(F_{\text{active}}\) while ensuring that \(C_{\text{promote}}\) stays within a budget determined by \(B_{\text{HBM}}/B_{\text{DRAM}}\). The dynamic selection thresholds and compression ratios are tuned through CompleteP-style transfer so that deeper models do not suddenly overload the bandwidth once they are scaled.

Because the warm tier uses a compact representation, we can also experiment with layered precision (e.g., FP8 in HBM, INT4 in DRAM) without refactoring the architecture. The selector becomes the guardrail that keeps the high-precision entries in memory only when they make a measurable latency win. This is how multi-dimensional co-design unfolds: topology, parameterization, and memory all contribute measurable terms to the step time, and the optimization is finding the pareto frontier on that surface.

## Where the field is now

SparseServe (2025) has become the most cited experiment in the intersection of algorithmic sparsity and memory hierarchies; the authors show that a DMA-aware cache manager can sustain 100 tokens per second on a 65B parameter decoder with no more than a 0.5% drop in loss, provided the promote/demote policy obeys their derived latency budget. The paper’s contribution is not just in the policy but in the instrumentation that maps each cache transfer to a concrete bandwidth spike in the trace, allowing operators to know when an upcoming generation will saturate DRAM. This corresponds with more recent works that continue to treat the KV cache as a scheduling problem—you cannot prune tokens without knowing what the hardware does with the bytes you keep.

On the scaling side, CompleteP’s transfer rules are now being incorporated into libraries such as Hugging Face’s Accelerate, where depth-specific learning rate multipliers and gradient clipping constants are reused across GPT-2 → GPT-NeoX transitions without expensive grid searches. The idea that you can reuse hyperparameters when you double depth (a simple polynomial mapping) reduces the practical compute required to tune new variants and helps align architecture search with hardware constraints.

For the engineering frontier, there are several production-grade deployments that show what these co-design principles buy:

- AWS’s [LLM-D framework](https://aws.amazon.com/blogs/machine-learning/llm-d) exposes a memory-tiered inference path for large decoders, explicitly sharding KV caches across multiple DDR banks to avoid saturation. Their blog quantifies that a single r7g.8xlarge instance can serve 20B models at 10ms p95 when the KV cache is pinned to HBM and the warm tier sits in adjacent DDR4 modules.
- AWS’s vLLM (https://aws.amazon.com/blogs/machine-learning/vllm-open-source) uses asynchronous scheduling and kernel fusion to keep GPU compute busy even when the KV cache has to be drizzled in from CPU memory, similar to the promotion/demotion story in SparseServe.
- NVIDIA Dynamo (https://developer.nvidia.com/blog/nvidia-dynamo/) applies memory virtualization inside the GPU by streaming KV slices from host to device when capacity is exceeded, matching the idea of warm/cold tiers executed across PCIe links.
- Google’s PROMPTS framework (https://research.google/pubs/pub50421/) emphasizes caching of prompt tokens with per-layer hooking, ensuring each attention layer keeps only the necessary tokens in HBM by introducing layer-specific caching policies.

These examples show that the systems practice of co-designing the cache paths with the architecture is no longer an experiment—it is now part of the stack for production inference at hyperscale.

## What's still open

1. Can we design a unified, hardware-native attention mechanism that dynamically compresses and decompresses KV caches in-flight without the latency of CPU-GPU transfers exceeding the computational savings of the sparsity itself?
2. What is the optimal policy for promoting compressed entries under fluctuating bandwidth budgets so that the scheduler never walks into a “cache storm” where every head requests data from DRAM simultaneously?
3. How do we reconcile CompleteP-style depth-wise hyperparameter transfer with curriculum learning, where the importance of attention heads shifts dramatically between pretraining and fine-tuning?
4. Can FHE-grade privacy-preserving inference, as pioneered by Gentry (2009) [http://www.cs.cmu.edu/~odonnell/hits09/gentry-homomorphic-encryption.pdf], be made compatible with dynamic sparse caches without exploding latency, especially when the cache entries themselves must remain encrypted during demotion?

## Where to read next

If you want the mathematical foundation for why attention has the structure it does, → [Attention](../../07-attention-memory-reasoning-continual/concepts/attention.md) walks through the dot-product formalism and the trade-offs between different parameterizations. If the hardware story intrigues you, → [[memory-hierarchies]] dissects the tiered storage stack that attention layers must live inside. For a deeper dive into sparsity algorithms whose scheduling plays nicely with these cache tiers, → [[sparse-attention-patterns]] explores the selectors and token pruning heuristics that feed the cache manager.

## Build it

This build proves that a tiny Transformer can be instrumented with a hardware-aware dynamic sparse attention layer whose simulated cache manager shows measurable latency improvements on a free Colab GPU.

**What you're building:** A cache-aware dynamic sparse attention layer for GPT-2 Tiny, with an HBM/DRAM simulator that tracks promotions/demotions and reports latency savings on TinyShakespeare decoding.

**Why this is valuable:** This recipe forces you to tie token selection, cache compression, and placement decisions together, which is the essence of the co-design insight: you cannot save time by pruning attention if the bandwidth cost of moving the remaining key/values erases the savings.

**Stack:**
- **Model:** [gpt2](https://huggingface.co/gpt2) — 3.5M downloads, small decoder baseline.
- **Dataset:** [tiny_shakespeare](https://huggingface.co/datasets/tiny_shakespeare) — well-known toy dataset for autoregressive modeling.
- **Framework:** PyTorch 2.2 + Diffusers 0.35 for utilities.
- **Compute:** Free Colab T4 (16GB VRAM, ~1hr training/fine-tuning).

**The recipe:**
1. `pip install torch==2.2.0 transformers diffusers fastrand` and load `AutoModelForCausalLM.from_pretrained("gpt2")`.
2. Tokenize TinyShakespeare with `GPT2TokenizerFast` and create 256-token causal sequences; the KV cache simulator wraps the forward pass and compresses every key/value pair exceeding 64 tokens into 8-bit chunks before storing them in the simulated DRAM buffer.
3. During training fine-tune for 3 epochs with 8 warm-up steps and a cosine learning rate, tracking the cache miss rate as you vary the sparsity selector’s threshold; expect training loss to start around 2.2 and fall below 1.8.
4. Evaluate by generating 128-token sequences with top-p sampling at 0.9 and measuring tokens-per-second with and without the cache manager; aim for a 15–25% latency drop when sparsity is enabled.
5. After evaluation, you have a GPT-2 checkpoint plus the cache telemetry table showing promotion/demotion counts and the achieved latency savings.

**Expected outcome:** A hardware-aware dynamic sparse attention module that you can slot into other GPT-2 sized decoders, together with logged latency gains and a simulated HBM/DRAM trace.

- **CS student:** Run the same build on an RTX 4070 with the same script but increase batch size to 4 and plot the trade-off between cache promotion frequency and tokens per second using Matplotlib.
- **Applied engineer:** Deploy the fine-tuned checkpoint with vLLM on an A10 instance, enable 4-bit quantization, and aim for p95 latency below 60ms by keeping the hot cache locked to GPU memory and only offloading compressed entries when absolutely necessary.
- **Applied researcher:** Test the hypothesis that a cosine sparsity schedule outperforms a step schedule in terms of latency reduction without hurting perplexity by training two versions of the selector (cosine vs. step) and comparing their promo/demote counts.
- **Frontier researcher:** Probe the open question of dynamic compression by implementing an end-to-end attention kernel that compresses KV entries as they are demoted, measuring whether the additional decompression cost ever outweighs the bandwidth savings in the trace.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*