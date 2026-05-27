---
title: Tensor Cores
slug: tensor-cores
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [markidis, mo, niu, casas]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [cuda-programming, matrix-multiplication-kernels, mixed-precision-arithmetic, triton-programming]
tags: [gpu-hardware, tensor-operations, mixed-precision, performance-engineering, triton-kernels, matrix-decompositions]
updated: 2025-05-01
has_mvb: true
---

# Tensor Cores

Imagine every GEMM (general matrix-multiply) that feeds your transformer being scheduled like a parade of single multiply-add instructions: each multiply and each add pays for instruction decode, issue, and retirement, swamping the arithmetic units in overhead instead of letting them do work. That is still what happens on a CUDA core when you write `c[i] += a[i] * b[i]`—the control logic is busier than the math. Tensor Cores refuse to play that game. They bundle the entire \$D = A \times B + C\$ tile into one warp-level primitive, then execute hundreds of multiply-accumulate operations in a single cycle. By the time you finish this page you will see why that trade—limited instruction flexibility for enormous arithmetic throughput—changed how datacenter GPUs are organized, how compiler teams expose mixed-precision math, and how you can target those pipelines with your own Triton kernel on a free Colab L4.

## The territory

For decades, GPUs were programmed as wide arrays of floating-point ALUs. Each multiply-add was an instruction, and the matrices that underpin convolutions, attention, or recommendation systems were nothing but loops of scalar operations. The instruction-overhead bottleneck only exacerbated as datacenters tracked up the roofline: The Datacenter as a Computer (Barroso & Hölzle 2009) [https://people.eecs.berkeley.edu/~randy/Courses/CS294.F09/wharehousesizedcomputers.pdf] notes that as compute grew faster than memory, the pipelines feeding the arithmetic needed to scale with fewer cycles wasted on control. At the same time, large matrix-heavy workloads such as those described in Building High-level Features Using Large Scale Unsupervised Learning (Le et al. 2011) [https://ar5iv.labs.arxiv.org/html/1112.6209] showed the appetite for dense linear algebra and the limits of regular CUDA looping: the hardware was starved not for FP throughput but for the ability to dispatch entire tiles with a single control decision.

Tensor Cores answer this by specializing. They are not scalar ALUs. Instead, they are hardwired matrix-multiply-accumulate (MMA) units: a warp-wide instruction consumes entire tiles of input, multiplies them, accumulates the result, and writes back within a single cycle, trading away the flexibility of general arithmetic. That specialization hard-anchors them in the space between roofline-limited dense compute and the specialized tensor operations required by modern AI. To operate, software must layer tiles and decompositions on top of the rigid MMA primitive, which is why tensor mathematics such as the mode-3 unfolding from Tensor Decompositions and Applications (Kolda & Bader 2009) [https://www.math.ucdavis.edu/~saito/data/tensor/kolda-bader_tensor-decomp-siamrev.pdf] becomes a natural language for thinking about how to slice data into the tiles these cores accept. How does this pipeline function end-to-end, and how do modern frameworks surface it? That is the mechanism we now unpack.

## How it works

Tensor Cores are born from the realization that the arithmetic you actually care about in deep learning is matrix multiplication, and that packing an entire tile into a single instruction lets you keep the data in product registers instead of repeatedly shuffling values in and out. From the hardware point of view, a Tensor Core executes:

\[
D = A \times B + C
\]

where \(A\), \(B\), and \(C\) are dense tile registers (typically 16×16 or 8×8, depending on the generation), and the operation performs the entire multiply-accumulate on a single warp-level instruction. Each operand lives inside registers or shared memory; there is no per-element instruction fetch, decode, or retirement, which lets the core reach multiple teraflops per square millimeter. The key constraint is that you can feed only specific tile sizes and data types (FP16, BF16, INT8, even INT4 for the latest generations) without branching mid-operand—this is why tensor mathematics about unfolding and slicing matters. When you tile a large matrix, you choose a blocking factor that matches the warp-friendly tile, much like how Kolda & Bader describe mode decompositions that reduce a dense tensor into sequences of matrix multiplies. Tensor Cores are therefore a set of hardened, fixed-size operators, not general-purpose ALUs.

### Mapping tiles to warps

The programmer’s first interface to a Tensor Core is the warp matrix-multiply-accumulate (WMMA) API introduced in early Volta hardware. WMMA lets each warp own a tile of \(A\), \(B\), and \(C\), load them from shared or global memory, call a single \(mma.sync\) instruction, and write back the result. The API looks like:

```
mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32
```

which says: with 16×8×16 tiling, multiply two FP16 tiles and accumulate in FP32. The first instruction fetch loads the entire tile specification, the core performs the multiply-accumulate entirely inside registers, and the final store commits the result. Whereas consuming scalar instructions requires significant control overhead, WMMA’s single instruction reduces that overhead by an order of magnitude. Within a single warp (32 threads), each thread loads a piece of the tile, and the Tensor Core threads collaborate to complete the multiply. All 32 threads literally become a single unit of arithmetic, not 32 independent ones, which is why the throughput is measured in TMACs (trillion multiply-accumulates per second) rather than FLOPs.

This is also why warp scheduling—how the GPU threads are grouped—is crucial. Since a Tensor Core instruction spans the entire warp, triggering the instruction requires proper alignment. CUDA itself cannot chain a single Tensor Core operation without arranging the data in warp-friendly layout, and herein emerges the role of software frameworks like Triton, cuBLAS, and Ampere’s mma instruction decoders.

### Mixed precision, accumulation, and the precision guardrail

One of the painful consequences of packing operations into Tensor Cores is the loss of per-element precision control. For a single warps’s tile multiply, you cannot vary the precision mid-instruction. That is baked into the hardware because the multiplier is a fixed width array of units. To balance precision against throughput, NVIDIA designed the instruction to accept varying formats: the inputs \(A\) and \(B\) might be FP16 or INT8, while the accumulation \(C\) is typically FP32 (Volta) or BF16/FP32 in later generations. The hardware essentially computes:

\[
D_{ij} = \sum_{k=1}^K A_{ik} \cdot B_{kj} + C_{ij}
\]

where the \(A_{ik}\) and \(B_{kj}\) multipliers are interpreted in a lower-precision type, while the sum \(D_{ij}\) uses higher precision to avoid catastrophic cancellation. Markidis et al. (2018) — "NVIDIA Tensor Core Programmability, Performance & Precision" [https://arxiv.org/abs/1803.04014] quantify how the precision trade-offs manifest: when you accumulate in FP32, the error is limited to \(\mathcal{O}(\mathrm{ulp}_{FP32})\) even if the operands were FP16, because the hardware promotes the partial sums. They also provide tuning knobs such as the `mma.sync.f32.f16` instruction that lets you choose whether the inputs are FP16 or BF16.

Since the hardware cannot mix precisions mid-operation, the software must do work to keep the data in the correct format. For instance, if your values start as FP32, you either convert them offline or use specialized quantization to pack them into INT8 before hitting the Tensor Core. That is where software-hardware co-design shines: Mo et al. (2024) — "LUT Tensor Core: A Software-Hardware Co-Design for LUT-Based Low-Bit LLM Inference" [https://arxiv.org/abs/2408.06003] add a lookup-table stage so you can simulate low-bit operations without native support. In that design, the Tensor Core still performs the dense tile multiply, but a LUT stage recombines the pieces, effectively letting you instruct the hardware with a precomputed conversion for each quantized value.

### From dense kernels to irregular workloads

Because Tensor Cores only understand dense tiles, irregular workloads—sparse graphs, unstructured fields—must be remapped into dense operations. A standard approach is to pad the data into block-sparse tiles, but this bloats memory and compute. Niu & Casas (2025) — "BerryBees: Breadth first search by bit-tensor-cores" [https://hdl.handle.net/2117/42676] show that you can encode the frontier of a BFS directly into bit-packed row slices, then pack those slices into dense tiles that the Tensor Core can process. The idea is to interpret the adjacency matrix as a tensor, break it into mode-3 slices (recalling the decompositions from Kolda & Bader), and use bit-level arithmetic to represent the frontier. The Tensor Core never knows it's processing a graph; it just sees a dense \(A \times B\) multiply. This pattern proves that you can coerce irregular workloads onto Tensor Cores if you can rebuild the problem in the language of matrix decompositions and tile packing.

### Compiler and kernel support

Given the rigid requirements—tile size, data type, warp alignment—the compiler must orchestrate loads, conversions, and stores. In CUDA, this means writing kernels that explicitly manage shared memory buffers sized to the Tensor Core tiles and launching them with dimensions that result in entire warps participating. Triton, however, raises the level of abstraction: its `tl.dot` primitive (with arguments `tl.dot(a, b, trans_a=..., trans_b=...)`) abstracts away the warp-level pipelining by scheduling the load-store loops for you. Usage looks like:

```
@triton.jit
def kernel(a, b, c, M, N, K):
    pid = tl.program_id(0)
    # compute tile offsets
    a_tile = tl.load(a + pid * M * K + tl.arange(0, M)[:, None] * K + tl.arange(0, K))
    b_tile = tl.load(b + ...)
    acc = tl.dot(a_tile, b_tile)
    tl.store(c + ..., acc)
```

Under the hood, Triton's `tl.dot` lowers to Tensor Core instructions when compiled for Ampere or later hardware. This is how you get to the single-cycle throughput without writing WMMA instructions explicitly. The compiler still has to ensure the tile sizes match `tl.dot`’s built-in blocking (typically 16×16), but for kernels that stay within Triton’s layout, the complexity is hidden.

### Failure modes and strategies

There are three common failure modes when targeting Tensor Cores:

1. **Misaligned warps**: if your tile is not aligned to 16, the Tensor Core instruction cannot execute, and the compiler falls back to scalar operations, losing the throughput advantage. The fix is to pad the matrix—think of the mode-3 unfolding as slicing into blocks of the right size.

2. **Precision mismatch**: Tensor Cores cannot mix FP16 and FP32 on the fly. If you stream FP32 inputs without conversion, hot paths will revert to FP32 CUDA cores. The solution is mixed-precision aware schedules and global aborts if the value range is too big; some teams convert to FP16 with dynamic range scaling (loss scaling), while Mo et al. let a LUT layer simulate low-bit operations over them without rerunning training.

3. **Warp-level register pressure**: each Tensor Core instruction uses a large register file. If the compiler spills registers, the instruction cannot issue at full speed. This is why high-performance kernels often use shared memory to stage inputs and only keep registers for the tile being processed.

By understanding these failure modes, you can modify the kernel or the Triton scheduling to stay within the Tensor Core’s expectations. The net result is a multiply-accumulate pipeline that requires precise data layout but delivers order-of-magnitude throughput improvements over general-purpose cores.

## Where the field is now

Today’s benchmark leaderboards read like a tensor-core manifesto. Mo et al. (2024) explain how co-design reopened Tensor Cores to low-bit inference; their LUT stage and tailored quantization let 4-bit and even 3-bit models run on hardware that only has official support for FP16/BF16. That means that modern edge deployments now use Tensor Cores not just for training but for inference by treating each low-bit operation as a dense tile multiply plus lookup. As a result, companies like NVIDIA and Google build inference stacks whose scheduling layers must generate the right tile sizes and load/store patterns before feeding the hardware.

On the research frontier, Niu & Casas (2025) map irregular graph computations like BFS onto Tensor Cores by encoding the frontier in binary tile slices. They report throughput comparable to state-of-the-art sparse graph engines while using only dense matrix multiply hardware, pointing toward a new frontier where Tensor Cores do not just serve dense neural networks but also structured algorithms. While this is still proof-of-concept, it suggests that the core's performance advantage can sometimes outweigh the extra encoding work for highly parallel workloads.

Engineering teams in datacenters already treat Tensor Cores as a separate subsystem. Their throughput—dominated by MMA instructions—drives datacenter design much like Barroso & Hölzle (2009) advocated for system-level co-design. Systems such as NVIDIA’s Hopper and Blackwell chips now dedicate more area to Tensor Core arrays than to traditional CUDA cores, acknowledging that the arithmetic game has shifted. On the software side, Triton, cuBLAS XT, and PyTorch’s `torch.compile` can now emit Tensor Core-bound kernels automatically, giving engineers a production path to sustained teraflop throughput.

## What's still open

1. Can a compiler automatically infer the right tensor decomposition (tile partitioning and data layout) for highly irregular sparse matrices so that the resulting tiles naturally fit Tensor Core tile sizes without wasteful padding?

2. What are the limits of LUT-based mixed-precision inference: can you achieve the same throughput benefits on Tensor Cores for dynamic bitwidth models (e.g., per-layer FP8/INT4) without explosive table sizes or extra memory traffic?

3. How can Tensor Core scheduling be fused with sparse scheduling so that workloads with both dense and sparse components—like transformer blocks with sparse attention—share the same tile execution pipeline instead of switching back to CUDA cores mid-graph?

## Where to read next

If you want the engineering companion that shows how the Tensor Core-aware kernel becomes a production microservice, → *triton perf* <!-- [[triton-perf]] --> breaks down the scheduling patterns and kernel launch parameters that Triton emits. The mathematical backbone of these decompositions lives in → *tensor decompositions* <!-- [[tensor-decompositions]] -->, which explains how to slice a tensor into matrix blocks that can be mapped to specialized hardware. For a systems-level view of how these accelerators reshape datacenter budgets, → *datacenter systems* <!-- [[datacenter-systems]] --> lays out the co-design of cooling, power, and compute that the tensor-core revolution demands.

## Build it

Tensor Cores can only sing when you explicitly tile and schedule the data they expect, so this build proves that you really understand the concept by writing a Triton kernel that targets Tensor Cores directly and compares it to a scalar baseline.

**What you're building:** a custom Triton kernel that loads two FP16 tiles, runs `tl.dot` (which emits Tensor Core MMA instructions on an L4 GPU), accumulates in FP32, and benchmarks throughput and register pressure against a naive element-wise CUDA-like loop.

**Why this is valuable:** tuning the blocking factor, shared-memory prefetch, and `tl.dot` call teaches you how Tensor Cores trade instruction overhead for throughput, and benchmarking against the baseline makes the performance gain tangible.

**Stack:**
- **Model:** none; the workload is a dense matrix multiply microbenchmark
- **Dataset:** synthetic matrices sampled from `torch.randn((8192, 8192), dtype=torch.float16)` to stress large tiles
- **Framework:** Triton 2.0 (use `pip install triton==2.0.0`)
- **Compute:** Google Colab L4 (8 GB of GPU memory) — expect about 90 minutes to tune and measure both kernels

**The recipe:**
1. Install Triton, PyTorch 2.1, and NVIDIA’s cuBLAS wrappers (`pip install triton==2.0.0 torch==2.1.0`), and verify the L4 runtime is available via `torch.cuda.get_device_name()`.
2. Synthesize two large matrices, pad them to 16×16 boundaries, and create a CuPy-like iterator that yields tiles aligned to the Triton kernel’s block size.
3. Implement the Tensor Core kernel with `tl.dot` and the naive baseline that multiplies each element in Python with `einsum`. Tune the shared-memory tile loading to reduce bank conflicts and insert `cuda.sync_threads()` at the warp boundary.
4. Measure GFLOPs by timing the kernels over several runs; expect the Triton `tl.dot` kernel to be 6-10× faster on Tensor Cores while the naive kernel, limited by control logic, should remain flat regardless of matrix size.
5. Collect the Triton kernel’s register and shared-memory usage via Nsight Compute (or `nvprof`), and package the results into a comparison table.

**Expected outcome:** a runnable Triton microbenchmark that outputs throughput numbers showing the Tensor Core kernel’s advantage, plus a short report documenting register usage and precision behavior.

- **CS student:** On a free Colab RTX 4070, reduce the matrix size to 4096×4096 so the kernel fits the smaller shared memory budget; measure how the tile padding affects GFLOPs and explain the drop.
- **Applied engineer:** Extend the build by adding INT8 quantization (with pre-scaling) and deploy the kernel through Triton Inference Server; target ≤80 ms latency at batch 1 for 4K×4K tiles.
- **Applied researcher:** Hypothesize that doubling the tile size from 16×16 to 32×32 will increase throughput despite higher register pressure; test by swapping Triton’s block size and falsify the hypothesis if the register spill negates the benefit.
- **Frontier researcher:** Use the kernel as a base to explore the open compiler question: automatically derive the tile partitioning for a sparse matrix (with varying nonzero patterns) and report the padding overhead; does a heuristic tiling strategy exist that keeps waste under 10%?

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*