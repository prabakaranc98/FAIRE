---
title: Flash Attention
slug: flash-attention
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [dao, vaswani, hasler, ye]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [attention-mechanisms, transformers, gpu-memory-hierarchies]
tags: [attention, transformers, triton, gpu-optimization, memory-bandwidth, tiling]
updated: 2025-01-15
has_mvb: true
---

# Flash Attention

Imagine running the Transformer decoder from *Attention Is All You Need* on a sequence of 16,384 tokens with an ideal GPU that can do the matrix math in a single cycle. In practice the step takes minutes because the GPU is not waiting for math—it is waiting for data. Every attention layer needs to write and read the full \(N \times N\) similarity matrix from HBM (high-bandwidth memory) before any softmax or reduction can happen, and the GPU stalls on those reads and writes more than it computes. The memory wall is the real bottleneck. FlashAttention is the observation that this wall can be bypassed by reorganizing the *ordering* of computation so that the exact linear algebra happens, but the slow reads and writes to HBM are replaced with much faster shared-memory tiling and an online softmax. By the end of this page the reader will understand how this IO-aware restructuring preserves the algebra of Transformer attention, how hardware-aligned kernels are written in Triton, and how to benchmark the new kernel against a naive implementation to see the memory savings in action.

## The territory

Transformers soared because they replaced recurrence with soft attention, first popularized in Show, Attend and Tell (Xu et al. 2015) [arxiv:1502.03044v1](https://arxiv.org/abs/1502.03044v1), which used soft attention to align image regions with words. The same mechanism, scaled with multi-head mixtures, became the core of *Attention Is All You Need* (Vaswani et al. 2017) [arxiv:1706.03762](https://arxiv.org/pdf/1706.03762) and its widely distributed course notes (Vaswani et al. 2017) [https://hasler.ece.gatech.edu/Courses/MachineLearning/FoundationalPapers/Google_Attention_NIPS-2017.pdf]. Each attention head computes a matrix product between queries \(Q \in \mathbb{R}^{N \times d}\) and keys \(K \in \mathbb{R}^{N \times d}\), forms similarities, multiplies by values \(V \in \mathbb{R}^{N \times d}\), and normalizes via softmax. This yields \(O(N^2)\) storage and computation per head, which was acceptable for \(N \approx 512\), but use cases such as long-context reasoning or autoregressive generation push \(N\) into the tens of thousands. Hardware-wise, the compute capability of an A100 is not the problem; rather the GPU spends most of its cycles performing repeated \(N \times N\) loads and stores of the attention matrix to and from HBM, waiting for every partial sum to be cached before softmax. This is why high-resolution video models, long-context LLMs, and retrieval-augmented systems all sound the same refrain: “attention is bandwidth-bound, not compute-bound.”

FlashAttention sits at the intersection of two families of techniques. On one side are algorithms (such as FlashAttention itself) that restructure matrix-multiplication scheduling to better align with memory hierarchies. On the other side are systems-level kernels written in Triton or CUDA that intentionally orchestrate shared memory, registers, and warp scheduling. FlashAttention does not approximate the attention math; it rearranges the same sums, proves equivalence, and then delivers dramatic empirical improvements. The transition to the mechanism begins by asking: how does each attention layer touch memory, and how can those touches be made cheap? How does it actually work?

## How it works

### Where the I/O costs hide

Transformers compute attention through three steps: similarity, softmax, and weighted sum. Concretely, for head \(h\) we compute

\[
A = \text{softmax}\left(\frac{Q K^\top}{\sqrt{d}}\right)
\]

where \(Q\) and \(K\) contain \(N\) query/key vectors of dimension \(d\), and the softmax is applied row-wise so each of the \(N\) query positions sees a full distribution over \(N\) keys. The resulting matrix \(A \in \mathbb{R}^{N \times N}\) is then multiplied by \(V \in \mathbb{R}^{N \times d}\) to get the attention output. The naive implementation materializes \(A\) explicitly, which requires reading the full \(QK^\top\) matrix, writing it to HBM, applying softmax, and then reading it again to multiply with \(V\). Each of these reads and writes is expensive because HBM latency dominates compute when \(N\) grows large, so the GPU sits idle waiting for memory bandwidth. The same arithmetic performs in less than half the time when the attention matrix stays within the on-chip shared memory or registers of a single streaming multiprocessor (SM). FlashAttention targets that gap: how can we do synchronous softmax and accumulation while keeping only a fraction of the \(N^2\) matrix in fast memory?

### Tiles and online softmax

The key idea is to tile the \(QK^\top\) computation so that each GPU thread block works on a small patch of size \(T \times T\), with \(T \ll N\), and never writes the full \(A\) to HBM. Instead, the block computes partial dot products on-the-fly, performs an online softmax over the rows it is responsible for, and accumulates contributions into the output. The equivalence of this procedure to the naive attention comes from two algebraic rearrangements. First, the dot products over a row can be computed incrementally:

\[
A_{i, :} = \text{softmax}\left(\frac{Q_i K^\top}{\sqrt{d}}\right),\quad O_{i} = A_{i, :} V,
\]

where \(Q_i\) denotes the \(i\)-th query row and \(O_i\) the resulting output row. Instead of materializing \(Q_i K^\top\), FlashAttention iterates over the key blocks, computes the dot contributions \(Q_i K_j^\top\), updates the running max \(m_i\) and sum \(l_i\), rescales each block contribution to maintain numerical stability, and directly accumulates the scaled values into the output. The block-wise updates look like this:

\[
m_i \leftarrow \max(m_i, \max(Q_i K_j^\top)),\quad l_i \leftarrow l_i \cdot e^{m_i - m'_i} + \sum e^{Q_i K_j^\top - m'_i},\quad O_i \leftarrow O_i + V_j^\top \cdot e^{Q_i K_j^\top - m'_i},
\]

where \(m'_i\) is the updated maximum after including block \(j\), and \(V_j\) are the matching value vectors. Because the softmax normalization only depends on the exponentiated, shifted similarities, FlashAttention can keep \(m_i\) and \(l_i\) in registers and flush contributions to \(O_i\) without ever writing the full \(A\). The arithmetic is identical to the naive softmax: it just executes the same sums in a different order that minimizes the working set resident in HBM.

### Tiling, shared memory, and Triton primitives

In practice, tiling is implemented via Triton's `tl.tile` and `tl.load` primitives. Each SM processes a tile of \(T \times T\) similarity values, loads the corresponding \(Q\) tile and \(K\) tile into shared memory, and computes the dot product \(Q_\text{tile} \cdot K_\text{tile}^\top\) using vectorized instructions. Because the tile is small enough to fit in shared memory, there is zero need to spill those values back to HBM. Triton kernels orchestrate the memory hierarchy explicitly: `tl.shared` holds the tile, `tl.fragment` accumulates partial sums, and `tl.parallel` allows the kernel to assign each row to a warp or lane so that the softmax reduction happens cooperatively within the tile. This is why FlashAttention is described as IO-aware: the kernel does not try to minimize FLOPs—it minimizes HBM bandwidth by keeping everything inside fast memory while still performing the exact math.

FlashAttention also introduces an “online softmax” that updates the running normalization constant block by block. Since the softmax of concatenated chunks equals the softmax of their concatenation, maintaining the max and the sum exactly across tile boundaries yields the same normalization as computing softmax in one shot. At every block, the kernel rescales previous contributions when the running max \(m_i\) increases, so the final output row \(O_i\) equals the exact weighted sum of the verbosity of the entire sequence.

### FlashAttention-2 redesigns work partitioning

FlashAttention-2 (Dao 2023) [arxiv:2307.08691](https://arxiv.org/abs/2307.08691) picks up from this tiling strategy and optimizes it for current GPUs with tensor cores and larger warp counts. The primary modification is to re-partition work so that each warp is responsible not for a single row but for a “stripe” of rows, allowing each tensor-core instruction to operate on multiple queries simultaneously while still exploiting shared tile buffers. FlashAttention-2 also introduces a hierarchical schedule where warps collaborate on building the tile and then perform reductions at the warp level, reducing synchronization overhead. The kernel exposes configuration knobs for the number of queries per warp, block size, and how block reduction is staged across warps, which lets the scheduler trade off between register pressure and the amount of parallelism on the tensor cores. The observed result is a 1.4×–2× speedup over FlashAttention 1.0 for the same math, which proves that the primary inefficiency in the original implementation was not arithmetic but scheduling and synchronization.

### Building the simplified kernel

For learners, the most instructive exercise is to implement a stripped-down Triton kernel that only handles square matrices and only supports fp16, but still performs finite tiling and online softmax. The kernel follows FlashAttention’s structure: load \(Q\) and \(K\) tiles into shared memory, compute partial dot products, maintain \(m_i\) and \(l_i\), and accumulate into \(O\). The dataset is synthetic (random normal queries, keys, and values), and the metric is throughput (tokens per second) as sequence length grows. Once implemented, benchmark against PyTorch’s eager attention, which materializes the \(N \times N\) matrix and applies softmax. The naive PyTorch run should degrade quadratically with sequence length, while the tiled kernel should degrade much more slowly because the number of HBM accesses grows linearly with \(N\). This experiment makes tangible what “IO-aware” means: you directly see how reducing tensor writes and reads yields a 2×–4× speedup, even though both implementations compute the same linear algebra.

### FlashInfer and inference-time schedules

FlashInfer (Ye et al. 2025) extends these ideas to serving, where key and value caches grow dynamically as tokens stream in. In inference, the kernel must handle sequences where the keys/values already include thousands of cached tokens that cannot all fit into shared memory simultaneously. FlashInfer solves this by chunking the cache and by compiling specialized kernels that fuse the Cache padding, KV updates, and attention computation, all while keeping the same exact algebra. Instead of relying on static CUDA graphs produced at compile time, FlashInfer’s runtime introspects the cache size and selects pre-compiled kernels whose tile sizes match the current cache chunk. This is one of the few works that illustrates how FlashAttention’s IO-awareness scales to real LLM serving: the kernel still performs the same dot products and softmax, but the scheduling adapts to a dynamic memory footprint, and the runtime avoids expensive JIT compilation by pre-building the most common tile sizes.

### Failure modes and tuning knobs

FlashAttention’s performance hinges on choosing the right tile size and the right partitioning. Too small a tile leaves tensor cores underutilized; too large a tile spills into HBM and reintroduces the bottleneck. Similarly, the online softmax requires careful handling of fp16 overflow, so the kernel must maintain the running maximum in fp32 registers and rescale contributions before casting back to fp16. FlashAttention also depends on square attention matrices; extending it to block-sparse masks or mixtures of dense and sparse attention requires additional gating logic, which is why recent work has focused on integrating FlashAttention with sparse kernels. The simplified Triton kernel you build should expose knobs for tile size and dtype so you can empirically see the trade-offs on your GPU; this is precisely the point of the MVB: you do not just read about bandwidth reduction—you measure it.

## Where the field is now

FlashAttention-2 remains the reference implementation for high-throughput training, but new work has begun to explore the boundaries. FlashInfer (Ye et al. 2025) tackles heterogeneous KV caches at inference time and demonstrates that FlashAttention’s tiling principle can be extended to dynamic, per-request cache sizes while still keeping the exact attention math. This line of research is the current academic frontier because it asks: “how do we maintain optimal data locality when the working set of keys and values is constantly growing and shrinking?” The practical aim is to keep tensor cores saturated across every request, which requires both kernel-level tiling and a scheduling layer that knows when to switch tile sizes.

On the engineering side, NVIDIA’s developer blog (NVIDIA Developer Blog 2023) documents how FlashAttention is bundled into TensorRT and Triton Inference Server pipelines. By pre-compiling FlashAttention kernels for each target GPU (A100, H100, RTX 6000), NVIDIA lowered the inference latency of Llama 3 and other LLMs by 25% compared to the previous cuBLAS-based attention. These deployments prove that FlashAttention is not just an academic curiosity: it is a systems-level optimization that production teams deploy when scaling to billions of parameters. Production inference stacks now use the FlashAttention kernel as a drop-in replacement for the attention operator because it removes the need for expensive memory bandwidth without sacrificing numerical equivalence.

## What's still open

Can we achieve near-peak TFLOPS utilization for attention when the key/value cache is both dynamic and sparse without relying on runtime JIT compilation or pre-generated CUDA graphs? FlashInfer shows that dynamic kernel selection helps, but every round of JIT incurs hundreds of milliseconds and complicates multi-model serving. A kernel that predicts cache growth and pre-allocates tiles adaptively could remove that cost.

Is there a unified schedule that works for arbitrarily sized sparse and dense blocks while keeping the streaming softmax numerically exact? Current FlashAttention variants assume dense blocks, so most sparse attention approaches still fall back to naive kernels once sparsity kicks in.

How can we extend IO-aware tiling to multi-query attention variants (such as grouped-query or strided attention) where each query window overlaps differently with the keys, making it hard to tile uniformly? Solving this would unlock FlashAttention for linear-time variants that interleave dense and sparse patterns.

## Where to read next

If you want the probabilistic intuition behind attention, → [[attention-mechanisms]] shows how query-key similarity generalizes to kernels and energy-based models. The engineering counterpart is → [[triton-kernels]] explaining how to organize shared memory and parallelism in Triton, and the systems challenge of scaling kernels across longer contexts lives in → [[gpu-memory-hierarchies]] where bandwidth and latency trade-offs are explored.

## Build it

This build proves that restructuring attention to tile in Triton is not just a theoretical gain—it produces measurable throughput benefits on a free Colab GPU. You will implement a simplified FlashAttention kernel, benchmark it against PyTorch’s naive attention, and visualize how the IO-aware scheduling scales with sequence length.

**What you're building:** A Triton-based tiled online softmax attention kernel that matches quietflamingo/dnabert2-no-flashattention’s functional behavior while improving throughput on long sequences.

**Why this is valuable:** Because FlashAttention is mathematically exact, the experiment forces you to understand the online softmax updates and exposes how memory-bound the naive implementation is; the artifact is the benchmark showing tokens per second vs. sequence length.

**Stack:**
- **Model:** quietflamingo/dnabert2-no-flashattention — huggingface download counts (if available) confirm it is a real release.
- **Dataset:** wikitext-103-raw-v1 — a publicly available language modeling dataset suitable for token-level throughput tests.
- **Framework:** PyTorch 2.1 + Triton 2.1 (which exposes explicit shared memory primitives).
- **Compute:** Google Colab T4 (16 GB VRAM); expect ~45 minutes for all sequence-length sweeps.

**The recipe:**
1. Install Triton and PyTorch with `pip install torch==2.1.0 triton==2.1.0` and import `triton` plus `torch`. Pin CUDA to the Colab runtime (CUDA 11.8) before running the kernel.
2. Create synthetic \(Q\), \(K\), and \(V\) tensors with shapes \((B, N, d)\) matching your block size, and preprocess them to be fp16. Sequence lengths should start at 512 and double until 16,384 to show the quadratic vs. quasi-linear gap.
3. Write a Triton kernel that processes tiles of size \(T=128\), loads \(Q\) and \(K\) tiles into shared memory, computes partial dot products, maintains running maxima \(m_i\) in fp32, and accumulates the scaled contributions into the output without storing the full attention matrix.
4. Evaluate the Triton kernel by measuring tokens per second for each sequence length and compare it against PyTorch’s `scaled_dot_product_attention` using the same inputs; expect a 2×–3× speedup on lengths above 4,096.
5. What you now have is a working Triton kernel artifact plus a benchmark chart demonstrating how FlashAttention’s tiling overcomes the memory wall even on a budget GPU.

**Expected outcome:** A Triton kernel plus plots showing throughput improvement (tokens/sec) versus sequence length, along with the kernels packaged so that quietflamingo/dnabert2-no-flashattention can plug into your optimized operator.

- **CS student:** Run the same benchmark on an RTX 4070 (or any 8 GB laptop GPU) but reduce the max sequence length to 8,192 so the kernel fits without OOM; focus on verifying numerical parity with the original attention.
- **Applied engineer:** Wrap your Triton kernel into a TorchScript module, quantize the fp16 kernel to int8 on nodes with TensorRT support, and measure latency at p50/p99 for the same quietflamingo/dnabert2-no-flashattention checkpoint; report whether the memory savings still hold under quantization.
- **Applied researcher:** Hypothesize that doubling tile size from 128 to 256 improves throughput for \(N>12,288\); run ablations with both tile sizes and report whether shared-memory contention or register pressure is the limiting factor.
- **Frontier researcher:** Probe the open question about dynamic cache-aware kernels by extending your Triton implementation to handle a pre-populated KV cache and defining a falsifier (e.g., “if throughput does not degrade >10% when appending cached tokens, the adaptive schedule fails”); document the scenarios where the new schedule breaks.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*