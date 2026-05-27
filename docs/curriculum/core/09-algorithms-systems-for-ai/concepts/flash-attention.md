---
title: Flash Attention
slug: flash-attention
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [dao, krueger, lin, zhang]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [attention-mechanisms, transformer-architecture, gpu-memory-optimizations]
tags: [attention, memory-bandwidth, gpus, inference, pipelining, performance]
updated: 2025-04-01
has_mvb: true
---

# Flash Attention

Imagine trying to train a 30‑billion‑parameter language model and watching the GPU die of thirst even though every Tensor Core is ready for work. The arithmetic throughput of Hopper or Blackwell chips runs at petaflops, yet each attention layer is forced to stall because every query needs to read a huge \(N \times N\) affinity matrix out of High-Bandwidth Memory (HBM), then write it back before the softmax can be applied. The surprising truth is that the GPU is not short on math; it is short on bandwidth, so much so that standard attention spends up to 90% of its runtime waiting on I/O. Flash Attention answers the question: how do we keep the mathematical meat of transformers flowing while never materializing that massive matrix? By the end of this page the reader will be able to see the bandwidth problem, follow the math for online softmax tiling, and run a PyTorch proof-of-concept that computes exact attention with \(O(1)\) extra memory.

## The territory

Modern generative systems hang on attention because it correlates every token with every other token, but the correlation matrix grows quadratically in length. The deep-learning survey on “The Magnificent Seven” (Diaz et al. 2024) [arxiv:2412.16188](https://arxiv.org/abs/2412.16188) highlights how attention migrated from academic prototypes to every deployed LLM precisely because it expresses long-range structure without sampling. That same quadratic surface is the source of the compute bottleneck: GPU Tensor Cores can process an entire row of the matrix faster than HBM can stream the next cache line of keys. Generator-to-system reports such as “GenAI for Systems: Recurring Challenges and Design Principles from Software to Systems” (Zheng et al. 2026) [arxiv:2602.15241](https://arxiv.org/abs/2602.15241) catalogue this mismatch as the main limiter for scaling inference at low latency. The consequence is not simply slower training; it is complex logic in serving fabrics, as autotuning engineers spend weeks aligning attention kernel throughput with KV-cache resource usage.

Flash Attention sits at the intersection of algorithmic attention (the \( \text{softmax}(QK^\top)V \) formula) and systems engineering for bandwidth. Instead of claiming a faster asymptotic complexity, it promises the same exact attention while never writing the full \(N \times N\) matrix to memory, which is the chunk responsible for the bandwidth drain. The same strategy belongs to the family of streaming and block-sparse variants, but Flash Attention’s innovation is to orchestrate the compute, copies, and softmax reduction inside a single kernel so that the GPU sees only the pieces it needs when it needs them. How does it actually work? The mechanism is best understood by breaking the attention computation into pipelined stages and maintaining softmax statistics online.

## How it works

### Why memory bandwidth dominates

The vanilla attention matrix is
\[
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d}}\right) V,
\]
where \(Q \in \mathbb{R}^{N \times d}\) contains the query vectors for an \(N\)-token sequence, \(K \in \mathbb{R}^{N \times d}\) contains keys, \(V \in \mathbb{R}^{N \times d_v}\) contains values, and \(d\) is the model dimension. This formula is elegant, but computing \(QK^\top\) naively writes \(N^2\) scalars before the softmax and the subsequent matrix multiplication can proceed, which is \(8N^2\) bytes for float32. On a 2048-token context that is 32 MB just for one affinity matrix, and every layer repeats it. Even when a single GPU has the arithmetic headroom to compute 8 trillion operations per second, the HBM bandwidth (typically under 2 TB/s) cannot keep up with streaming those billions of scalars back and forth. The plateau described in GenAI for Systems (Zheng et al. 2026) [arxiv:2602.15241](https://arxiv.org/abs/2602.15241) is not a theoretical limit; it is a memory bandwidth wall.

Flash Attention turns the problem sideways: rather than computing the entire matrix before applying softmax, it computes attention block-by-block and collapses the softmax normalization while the GPU is still holding the relevant blocks in shared memory. The key difference is that the GPU no longer needs to write all \(N^2\) entries out to HBM; it streams blocks through the GPU, performs partial softmax normalization, updates running statistics, and writes only the final outputs.

### The tiled attention pipeline and online softmax

The sequence of \(N\) tokens is partitioned into blocks of size \(b\) so that \(N = B \times b\), and attention is computed block-row by block-row. Let \(Q_i \in \mathbb{R}^{b \times d}\) denote the queries from block \(i\), \(K_j \in \mathbb{R}^{b \times d}\) the keys from block \(j\), and \(V_j\) the corresponding values. The partial affinity between \(Q_i\) and \(K_j\) is \(S_{ij} = Q_i K_j^\top\). The core challenge is to evaluate
\[
\mathrm{softmax}_j(S_{ij}) = \frac{\exp(S_{ij})}{\sum_{j'} \exp(S_{ij'})}
\]
without materializing \(S_{ij}\) for all \(j\). This is solved by maintaining, for each output row \(i\), two scalars: \(m_i\), the largest score seen so far, and \(l_i\), the cumulative sum of exponentials scaled relative to \(m_i\). When a new block \(S_{ij}\) arrives, compute
\[
m_{\text{new}} = \max(m_i, \max(S_{ij})),
\quad
l_{\text{new}} = \exp(m_i - m_{\text{new}}) \cdot l_i + \sum_{j} \exp(S_{ij} - m_{\text{new}}),
\]
where \(\max(S_{ij})\) is taken element-wise over the block’s rows and each term is annotated immediately. The exponentials are computed inside the kernel while \(Q_i\) and \(K_j\) sit in shared memory. Once all blocks \(j\) have been processed, the normalized attention for block \(i\) is
\[
A_i = \sum_j \frac{\exp(S_{ij} - m_{\text{final}})}{l_{\text{final}}} V_j,
\]
where \(m_{\text{final}}\) and \(l_{\text{final}}\) are the final statistics. This combination of max-plus scaling and streaming accumulation is mathematically equivalent to the full matrix softmax but requires only \(O(Nd)\) storage for \(Q_i\), the running \(m_i\) and \(l_i\), and the accumulating output \(A_i\). Each block \(V_j\) is multiplied by the normalized weights on the fly, so neither the full attention nor the intermediate sums must touch HBM beyond what can be pipelined.

### Pipelining GEMMs, softmax, and memory copies

Flash Attention’s performance leap is not only algorithmic; it is about keeping every execution unit busy. Standard kernels wake up, copy \(Q\), \(K\), \(V\) from HBM into registers, perform \(QK^\top\), write the result out, and then read it back for softmax and the downstream GEMM with \(V\). Flash Attention overlaps these actions: while one block is running the softmax reduction in register-heavy shared memory, the next block is asynchronously loading \(K\) and \(V\) from HBM. The dimension of these asynchronous copies is tuned so Tensor Cores are kept busy computing \(Q_i K_j^\top\) while DMA engines stream the next block in the background. FlashAttention-1 (Dao et al. 2022) [arxiv:2205.14135](https://arxiv.org/abs/2205.14135) introduced this triple-stage pipeline and showed that bandwidth constraints disappear once the GPU stops materializing the entire affinity surface.

The asynchronous design also plays well with FP16 and FP8 quantization: weights can be stored in the lower precision, but blocks within shared memory are promoted to FP32 for the softmax reduction so that \(m\) and \(l\) updates remain stable. On Hopper hardware or Blackwell accelerators, the kernel schedules separate computation stages as CUDA graphs so that a single kernel launches (1) load, (2) compute, and (3) reduction phases without going back to the CPU. Each of these phases is bounded by either shared-memory capacity or the bandwidth of a single block, meaning the GPU sees a steady pipeline rather than a stop-and-go process.

### Tiling for KV cache and tiled attention

Inference in production adds another dimension: transformer serving engines typically store KV caches for each request and interleave attention computation across multiple requests. Flash Attention keeps the same pipeline for a single request, but the scheduler ensures that while request A is computing the reduction for block \(i\), request B’s blocks are simultaneously being materialized in HBM or processed. In practice, this requires a small scheduler, which some inference frameworks implement using reinforcement learning-style policies over request priorities. The reinforcement learning survey for deep research systems (O’Neil et al. 2025) [arxiv:2509.06733](https://arxiv.org/abs/2509.06733) provides the theoretical grounding for these multi-objective policies: the reward balances latency, throughput, and bandwidth usage. Flash Attention kernels expose a simple API that lets such schedulers decide (1) which KV caches should be advanced, (2) how much of the cache to keep on-chip, and (3) whether to quantize values for a particular request.

### Online softmax as a proof-of-concept

The numerical proof that attention is unchanged is straightforward to test: the streaming softmax described above can be implemented in PyTorch as a sequential loop over block columns, computing the same \(m\) and \(l\) that full attention would use. Each iteration updates the partial output \(A_i\) with
\[
A_i \leftarrow A_i + \left(\frac{\exp(S_{ij} - m_{\text{final}})}{l_{\text{final}}}\right) V_j,
\]
where only the current block \(S_{ij}\) is in shared memory. The updates can be vectorized so that entire blocks are handled at once, and the final result matches the dense attention within numerical precision, proving that the bandwidth-saving trick is exact, not approximate. Later sections will show how to turn this proof into a deployable kernel and why modern inference engines care about that property.

## Where the field is now

Flash Attention is now the de facto baseline for training transformers on longer contexts. Flash Attention (Dao et al. 2022) [arxiv:2205.14135](https://arxiv.org/abs/2205.14135) replaced dense attention kernels in Hugging Face Accelerate and in training stacks like DeepSpeed and Megatron, showing a 2× speed-up on 8K tokens simply by removing the \(N^2\) writes. The most recent engineering push keeps this idea in production: AWS SageMaker’s new Model Parallel Library update uses a Flash Attention–style kernel internally, as described in their blog post on throughput improvements for partitioned models. The production story is similar at Databricks’ Superhuman flow, where the inference stack compresses KV caches into blocks and runs them through Flash Attention’s pipeline before any other operation; this is the kind of engineering-level change documented in the GenAI for Systems report (Zheng et al. 2026) [arxiv:2602.15241](https://arxiv.org/abs/2602.15241), which emphasizes that systems engineering and algorithmic primitives must co-design to overcome bandwidth walls.

Research continues to push Flash Attention’s pipeline toward adaptive behavior. The DeepResearch-9K benchmark (Kim et al. 2026) [arxiv:2603.01152](https://arxiv.org/abs/2603.01152) surfaces contexts with long mathematical derivations, where the bandwidth cost is dominated by infrequent attention spikes. Models evaluated on this benchmark need kernels that treat common tokens differently from rare ones, or else the attention stage becomes the bottleneck in the entire training curve. The latest production engines such as SGLang, vLLM, and MLC-Engine integrate Flash Infer–style schedulers that repost requests to the kernel whenever a new KV block becomes hot, relying on reinforcement-learning-style heuristics summarized in “Reinforcement Learning Foundations for Deep Research Systems” (O’Neil et al. 2025) [arxiv:2509.06733](https://arxiv.org/abs/2509.06733) to trade off freshness and bandwidth.

The interaction between attention kernels and the surrounding system is also a research frontier. “A Decade of Deep Learning: A Survey on The Magnificent Seven” (Diaz et al. 2024) [arxiv:2412.16188](https://arxiv.org/abs/2412.16188) notes that the seven major primitives now have to negotiate shared hardware resources; attention’s bandwidth-frugal implementation is the first one to be fully integrated with the scheduler, while others like convolutions and MLP layers still rely on static partitioning. Superhuman/Databricks and AWS SageMaker demonstrate the engineering frontier: their inference stacks compose Flash Attention with query routing, quantized KV caches, and multi-tenant scheduling so that each request sticks to a fixed memory budget while still lowering latency in practice.

## What's still open

Can we weave Flash Attention’s pipelined kernel into a single unified operator that dynamically senses token sparsity at runtime and skips whole KV blocks without causing warp divergence? This question is urgent because memory-bound workloads do not scale when the kernel has to break the static execution pipeline that GPUs expect. What is the minimal metadata that the scheduler must maintain on top of the kernel so that rare tokens can trigger blocks to be skipped entirely? Finally, can we guarantee that hardware-level asynchronous copies, which rely on predetermined block sizes, never starve the Tensor Cores when attention patterns shift rapidly mid-inference? These questions are the frontier of bandwidth-aware attention.

## Where to read next

The practical grounding for many of these points appears in [[attention-mechanisms]], which explains how query/key/value scores are constructed in the first place; → [[gpu-memory-optimizations]] drills into the bandwidth limits that Flash Attention is trying to solve; the engineering counterpart is → [[llm-serving]], which shows how schedulers and KV caches are orchestrated in large-scale deployments; and → [[block-sparse-attention]] explains the sparsity structures that future Flash Attention kernels might adapt to.

## Build it

This build proves that streaming attention with online softmax is not just a theoretical trick: you will write the tiled pipeline yourself and watch a free Colab T4 compute exactly the same output as dense attention while allocating only a constant amount of attention workspace.

**What you're building:** A pure PyTorch proof-of-concept that updates an attention block row by row using online \(m\)/\(l\) statistics and shows exact equivalence to the dense \(QK^\top\) result on a 2048-token input.

**Why this is valuable:** It forces you to implement the normalized softmax accumulation, block GEMMs, and DK-structured reduction that distinguish Flash Attention from legacy kernels, rather than just calling a library function.

**Stack:**
- **Model:** [quietflamingo/dnabert2-no-flashattention](https://huggingface.co/quietflamingo/dnabert2-no-flashattention) — downloads: 1.3K, use its tokenizer and embedding layers to produce synthetic tokens
- **Dataset:** [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/2-raw-v1) — short articles that you can concatenate into a single 2048-token sequence
- **Framework:** PyTorch 2.1 with the `torch.compile` backend and `torch.autograd.profiler` for bandwidth tracing
- **Compute:** Free Colab T4 (16 GB VRAM, <30 min for the full pipeline)

**The recipe:**
1. Install PyTorch 2.1 and `transformers` in a Colab instance (`pip install torch==2.1.0 transformers==4.41.0`) and load the tokenizer/model through `quietflamingo/dnabert2-no-flashattention`.
2. Sample one 2048-token chunk from `wikitext-2-raw-v1`, embed it with the model’s tokenizer, and split the token indices into blocks of \(b=256\); convert selected embeddings into query/key/value matrices \(Q, K, V\) of shape \((2048, d)\).
3. Write a function that loops over block rows \(Q_i\) and block columns \(K_j, V_j\), computes \(S_{ij} = Q_i K_j^\top / \sqrt{d}\), and updates \(m_i, l_i\) by tracking \(\max\) and rescaling the exponentials (use the equations given above). Accumulate the normalized output into \(A_i\) while copying only the blocks you need.
4. After the loop, compare \(A = [A_0; \dots; A_B]\) to `torch.nn.functional.scaled_dot_product_attention(Q, K, V)` using `torch.allclose(abs_tol=1e-5)` and plot both the softmax normalizers `m_i` and `l_i` over the block index to confirm numerical stability.
5. Use PyTorch’s profiler to measure bytes read/written and show that the streaming kernel reads roughly \(O(Nd)\) data instead of \(O(N^2d)\); save the profiler trace as the artifact.

**Expected outcome:** A Colab-ready notebook that prints the attention discrepancy (should be <1e-5), plots the running softmax statistics, and reports the profiler’s bandwidth estimate for the streaming kernel.

- **CS student:** Swap the T4 for an RTX 4070, increase block size to \(b=512\), and verify the profiler trace still shows near-linear bandwidth scaling; show that the Colab notebook runs within 1 hour.
- **Applied engineer:** Extend the code to emit serialized blocks that can be pre-fetched by an inference scheduler, then wrap the kernel in a simple HTTP endpoint (Flask + `torchserve`) and target p99 latency under 150 ms for a 2-request queue.
- **Applied researcher:** Treat a custom gating policy as a hypothesis: run the same pipeline with a token-level mask that forces block skipping for low-variance rows, and measure whether profiler bandwidth drops without breaking the all-close check.
- **Frontier researcher:** Use the kernel to test the open question from §What’s still open—define a runtime sparsity detector that dynamically skips blocks when the per-row entropy stays below a threshold and verify whether the scheduler can predict the resulting warp divergence by measuring CUDA kernel occupancy.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*