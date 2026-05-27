---
title: Tensor Parallelism
slug: tensor-parallelism
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [shazeer, ho, krizhevsky, chen]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [distributed-training, transformer-architecture, communication-primitives, linear-algebra]
tags: [tensor-parallelism, megatron, distributed-matrix-multiply, gloo, all-reduce, all-gather]
updated: 2024-11-20
has_mvb: true
---

# Tensor Parallelism

What does it mean for eight gigabytes of high-bandwidth memory (HBM) to vanish in a single matrix multiply? Walk into any lab running a 70B-parameter transformer on an 80GB GPU and you immediately feel the wall: a single \(W \in \mathbb{R}^{4096 \times 32768}\) weight matrix no longer fits on the card, and yet the next layer still needs both its inputs and outputs computed in tens of milliseconds. Tensor parallelism is the choreography that slices that enormous matrix itself, assigns each piece to a different GPU, and performs the GEMM while the chips exchange partial results like musicians trading phrases mid-concert. By the end of this page you will understand how today's Megatron-style implementations partition individual linear layers, why communication collectives cancel themselves out, which hardware affordances they lean on, and how to implement and test a simple column-parallel and row-parallel block yourself on a Colab instance using PyTorch’s collective primitives.

## The territory

The fundamental problem that tensor parallelism solves is memory locality at the level of a single layer. When scaling transformers beyond a few billion parameters, data parallelism alone can no longer keep every shard of a weight matrix resident on a single GPU, and pipeline parallelism introduces latency and implementation complexity. The territory we occupy sits between memory-aware sharding and fine-grained operator fusion: instead of treating a whole linear layer as a monolith, we partition its weight tensor—either by slicing across the input dimension (row parallelism) or the output dimension (column parallelism)—and compute the resulting partial matrix multiplies concurrently. This family of techniques borrows from early CNN days where each layer’s spatial extent dictated whether to partition across channels or filters, but adds the distributed-GPU choreography that supercomputers use for dense linear algebra.

The effectiveness of that choreography depends on a high-bandwidth intra-node network (NVLink or Infinity Fabric) and on collective algorithms—All-Reduce, All-Gather—that keep shards synchronized without serializing the GEMM. Tensor parallelism therefore sits at the intersection of distributed linear algebra and accelerator-aware operator design; it is what allows a single transformer layer to span multiple chips while still returning a single \(y = xW\) vector. How does it actually work? We start by writing down the math for a single linear layer and then show how each tensor-parallel variant rewrites that computation across GPUs, revealing the precise points where communication collectives enter the graph.

## How it works

Consider a standard dense layer, \(y = xW + b\), where \(x \in \mathbb{R}^{B \times d_\text{in}}\) is a minibatch of \(B\) vectors of dimension \(d_\text{in}\), \(W \in \mathbb{R}^{d_\text{in} \times d_\text{out}}\) is the weight matrix, and \(b \in \mathbb{R}^{d_\text{out}}\) the bias. To run this layer across \(P\) GPUs, tensor parallelism breaks the weight matrix into \(P\) shards along either its rows or columns, performs \(xW_i\) locally, then uses collectives to compose the final output. The original insight that such a split could bypass device memory limits appeared already in Krizhevsky’s “One weird trick for parallelizing convolutional neural networks” (Krizhevsky 2014) [https://arxiv.org/abs/1404.5997], showing that different layer types benefit from different partition axes. Tensor parallelism applies that lesson to dense transformers by aligning shard axes with the dimensions that dominate each layer’s memory footprint.

### Column-parallel linear layers

In column parallelism each GPU holds a subset of the output channels. Let \([W_1, W_2, \dots, W_P]\) denote a partition of \(W\) along the output dimension so that \(W_i \in \mathbb{R}^{d_\text{in} \times (d_\text{out} / P)}\). Each GPU computes \(y_i = xW_i\) independently using its local inputs, so the local computation cost is \(O(B \cdot d_\text{in} \cdot d_\text{out} / P)\). The GPUs then concatenate their partial outputs along the channel axis to recover the full \(y\). The bias \(b\) is likewise partitioned and added locally before concatenation. No collective is required for the forward pass because concatenation reconstructs the complete vector, but the gradients of \(W\) with respect to \(y\) decompose across shards during backpropagation: the local gradient \(\partial \mathcal{L}/ \partial W_i = x^T \delta_i\) only depends on the local slice \(\delta_i\), so no All-Reduce is needed until the optimizer step. Column partitioning therefore trades off increased output parallelism (good for projecting to large hidden size) with the need for contiguous allocation of the final output vector before the next layer.

### Row-parallel linear layers

Row parallelism slices \(W\) across its input dimension. Define \(W = \begin{bmatrix} W^{(1)} \\ \vdots \\ W^{(P)} \end{bmatrix}\), where \(W^{(p)} \in \mathbb{R}^{(d_\text{in} / P) \times d_\text{out}}\). The minibatch input \(x\) must also be partitioned such that each GPU holds \(x^{(p)} \in \mathbb{R}^{B \times (d_\text{in} / P)}\). Each GPU computes a partial result \(y^{(p)} = x^{(p)} W^{(p)}\) and then all GPUs perform an All-Reduce sum over \(p\) to produce the full output \(y = \sum_p y^{(p)}\). The All-Reduce introduces a communication step, but it occurs on the much smaller \(B \times d_\text{out}\) tensor rather than on the huge weight matrix. During backpropagation, the gradient with respect to \(x^{(p)}\) depends on the global \(d\mathcal{L}/dy\) that was reconstructed via All-Reduce, so row parallelism ensures all shards see the same downstream error signal. Row parallelism therefore scales when \(d_\text{in} \gg d_\text{out}\), which is common in feed-forward or attention projection layers, and is the partner for column parallelism in many Megatron-style implementations.

### Communication choreography and fusion

Mesh-TensorFlow (Shazeer et al. 2018) [https://arxiv.org/abs/1811.02084] abstracted tensor parallelism by mapping each tensor axis to a logical processor mesh, which formalizes how column and row parallelism can coexist: the mesh dimension assigned to \(W\)'s output axis carries column parallelism, while the mesh dimension assigned to its input axis carries row parallelism. This abstraction allows a compiler to rewrite tensor contractions so that they respect the mesh layout, automatically inserting the All-Reduce or All-Gather collectives required to realign tensors across layers.

The most delicate part of the choreography is ensuring that the communication cost does not dominate the compute. All-Reduce is mathematically placed so that sum operations collapse the partial results that would otherwise represent duplicates; for example, if the feed-forward network computes \(z = \text{GELU}(xW^{(1)})W^{(2)}\), splitting both \(W^{(1)}\) and \(W^{(2)}\) requires alternating row and column partitioning with intermediate All-Reduce and All-Gather steps. In practice, these collectives are overlapped with the GEMM so that the GPU’s Tensor Memory Accelerator (TMA) pushes data while the compute unit finishes the current block. The ThunderKittens “One Kernel for All Your GPUs” work [https://export.arxiv.org/pdf/2105.14500v2.pdf] demonstrates that modern hardware can fuse these collectives into the GEMM, bypassing NCCL entirely and shuttling data through the GPU's TMA during the GEMM pipeline.

### Adaptive layouts and dynamic routing

ATP: Adaptive Tensor Parallelism (Zheng et al. 2023) [https://arxiv.org/html/2301.08658] pushes the idea further by automatically selecting between 1D and 2D partitioning strategies for each layer based on its shape and the current batch size. The model observes that the ratio of compute to communication changes across layers and input lengths, so ATP monitors runtime statistics, chooses new layouts at compile time, and inserts the necessary All-Reduce/All-Gather calls accordingly. It also rewrites dropout, layernorm, and bias additively so that their parameters can remain sharded without extra synchronization, keeping the communication as close to the GEMM as possible.

Folding Tensor and Sequence Parallelism for Memory-Efficient Transformer Training (2026) [https://arxiv.org/html/2604.26294] adds yet another layer: when attention sparsity or MoE branching causes GPUs to wait on a subset of specialists, this paper shows that folding tensor parallelism with sequence parallelism keeps both weight and activation sharding aligned, allowing each device to process only the tokens it owns while still participating in the same GEMM. Collectively, these works show that the tensor parallel split is not static: it is a live schedule of matrix operations where communication is fused, overlapped, and sometimes entirely bypassed by hardware accelerators.

### Failure Modes and Debugging

The typical failure arises when communication and compute are misaligned: if the All-Reduce over \(y^{(p)}\) finishes after the next layer begins, the model sees stale activations. To avoid this, frameworks stage the communication on CUDA streams that are synchronized with the GEMM completion event, and they insert `torch.cuda.synchronize()` before feeding the partial outputs. Another failure is inconsistent shard ordering: the row partitioning on \(W\) must match the slicing used by the optimizer; otherwise, the local gradient update writes to the wrong subset of weights. Logging tensor shapes and verifying `torch.chunk` boundaries during initialization quickly catches this mismatch. Amplified latencies appear when the batch size is too small relative to \(P\): the communication cost \(\mathcal{O}(d_\text{out} \cdot P)\) no longer hides behind compute, so model parallelism becomes slower than data parallelism. That is why production deployments begin with an analysis of \(B \cdot d_\text{out}\) per GPU before enabling tensor parallelism.

## Where the field is now

The research frontier now focuses on adaptive layouts where the mesh changes mid-weight update. ATP (Zheng et al. 2023) [https://arxiv.org/html/2301.08658] reported that switching between 1D and 2D parallelism per layer reduced the communication volume by 28% and improved throughput on OPT-175B by 20% at batch sizes where static sharding saturated the interconnect. Following ATP, LightningNetworks (2024) fused Mesh-TensorFlow-style layout selection with hardware telemetry to decide at runtime whether a layer should split on \(d_\text{in}\), \(d_\text{out}\), or both, using a reinforcement signal derived from GPU occupancy. On the systems side, Folding Tensor and Sequence Parallelism (2026) [https://arxiv.org/html/2604.26294] translated that adaptive mindset into production by combining tensor parallelism with sequence sharding and showing that a single NVIDIA DGX-A100 could sustain a 70B parameter transformer’s forward and backward passes without out-of-memory faults, despite heavy MoE sparsity.

From the engineering perspective, ThunderKittens’ “One Kernel for All Your GPUs” [https://export.arxiv.org/pdf/2105.14500v2.pdf] proved that the communication collectives themselves could be fused into the GEMM pipeline, bypassing NCCL altogether. A prototype fused the All-Gather into the TMA access pattern, reducing latency by 15% even though it was running the same 2D partitioning layout that higher-level frameworks provided. Stability AI’s teams have adopted similar fusion to make SDXL Turbo’s MLPs run with sub-millisecond inter-layer latency when moving from NVLink-connected A100s to NVSwitch pods, accepting the extra engineering cost because the alternative was to duplicate billions of weights. These papers illustrate the current boundary: the math of tensor partitioning is stable, but its engineering still hinges on how closely operators can couple computation and communication.

## What's still open

Can dynamic tensor layouts be extended to handle conditional compute—the token-dependent sparsity generated by MoE layers and adaptive attention windows—without sacrificing determinism? Each GPU currently chooses its shard based on static position, so when an MoE router sends an imbalanced number of tokens to a subset of experts, execution stragglers appear and large All-Reduce barriers force idle devices. Solving this requires a runtime-aware mesh scheduler that can remap tensor partitions on the fly while keeping gradients consistent.

Another open question is whether the communication collectives can be replaced by register-level transfers on future hardware. ThunderKittens fused GEMMs with collectives inside the GPU, but only for single-node setups. The next step is to extend that fusion across NVLink fabrics so that an All-Gather is literally an intra-GPU shuffle through a shared register file instead of a library call. If that is not possible, what is the minimum set of collective semantics that must still run as separate operations?

Finally, how can tensor-parallel layouts be prepared during training so that deployment remains deterministic? Static sharding requires retracing the entire model for inference, but dynamic ATP-style remapping would generate different fusion kernels per sequence length. The open research question is: can schedulers commit to a canonical layout family whose union covers all lengths, so that the deployment binary carries a small set of kernels instead of one per observed configuration?

## Where to read next

If you want the broader distributed-training picture that combines tensor with pipeline and data parallelism, → [Distributed Training](distributed-training.md) lays out the trade-offs and the orchestration patterns that tensor parallelism plugs into. The engineering counterpart is → [[communication-primitives]] which explains how All-Reduce, All-Gather, and NCCL primitives are implemented on NVLink and why that latency matters for tensor splits. For a deeper dive into the transformer blocks whose matrices you are slicing, → [[transformer-architecture]] unpacks attention heads, projection matrices, and MoE routers so you can decide which dimension to shard first.

## Build it

The core lesson of tensor parallelism becomes concrete when you implement the column-parallel and row-parallel linear layers that Megatron uses, and then verify that using PyTorch’s Gloo backend with two processes reproduces the same outputs as the single-shard baseline while timings show the collectives canceling communication. This build proves you understand both the math of the shard (the local GEMMs) and the engineering (the All-Reduce/All-Gather choreography) by making the barriers explicit.

**What you're building:** A PyTorch Megatron-style MLP block that runs column and row parallel linear layers across two simulated GPUs, demonstrating how All-Reduce and All-Gather collectives align to form a correct global matrix multiply.

**Why this is valuable:** Because tensor parallelism trades compute for coordinated communication, building and timing the block under Gloo convinces you that the combined workload is still deterministic and lets you observe the collectives’ bandwidth usage when each shard produces only a fraction of the output.

**Stack:**
- **Model:** `facebook/opt-125m` (643k downloads) — the architectural template whose MLP block you replicate.
- **Dataset:** `wikitext-2-raw-v1` (accessible, short sequences).
- **Framework:** PyTorch 2.1.0 with TorchDistributed (gloo backend).
- **Compute:** Free Colab T4 (16GB VRAM) or CPU-only Colab (no NCCL); `torchrun --nproc_per_node=2` simulates two GPUs for ~1 hour of training including sanity checks.

**The recipe:**
1. Install + load PyTorch 2.1.0 (`pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`) and download the `wikitext-2-raw-v1` dataset via `datasets`.
2. Preprocess by tokenizing headlines with a small byte-level tokenizer (e.g., HuggingFace’s `RobertaTokenizerFast`) and stacking a minibatch \(B=32\) of sequences padded to length 128 so each shard sees the same batch size.
3. Create `ColumnParallelLinear` and `RowParallelLinear` modules that split \(W\) and \(x\) using `torch.chunk`, wrap them in `DistributedDataParallel` (even though the weight refresh is local), and run `torchrun --standalone --nnodes=1 --nproc_per_node=2 train_mp.py` where each forward includes the All-Gather (column) or All-Reduce (row) as explicit `torch.distributed` collectives; expect the loss to drop by ~0.05 per epoch for 3 epochs.
4. Evaluate by running the same input through a single-shard (no distributed) version and comparing the outputs; the per-token mean squared difference should stay below \(1 \times 10^{-6}\), and logging `torch.distributed.get_world_size()` shows the collectives invoked exactly once per forward.
5. What you now have is a working manual implementation of Megatron’s MLP block ready to extend (e.g., to include fused activations or mixed precision) and a script that logs the time spent in each collective versus the GEMM kernel.

**Expected outcome:** A pair of scripts that log the forward/backward loss curves, show the All-Reduce/All-Gather timings, and assert numerical equivalence to the non-parallel baseline, producing a checkpoint that implements the distributed MLP block.

- **CS student:** Swap the Colab T4 for a single 4070 and reduce the batch size to \(B=16\); you can then run the script on CUDA with NCCL by leaving `--nproc_per_node=1` while still observing the Discipline of manually calling `torch.distributed.all_reduce` to ensure you understand the math.
- **Applied engineer:** Extend the script by exporting the assembled MLP block as a TorchScript module and hosting it behind Triton Inference Server or vLLM with Quantized weights; measure the latency for a 4-token prompt target of <4 ms on an A10 when all communication is fused into one All-Reduce call per token.
- **Applied researcher:** Hypothesize that overlapping All-Reduce with the backward GEMM halves runtime; add CUDA events to the training loop to measure overlapping degree and run the train script with `torch.cuda.Stream`-managed collectives to confirm whether throughput improves by at least 10%.
- **Frontier researcher:** Probe the open question of dynamic sparsity by extending the script so that each process routes tokens only to its assigned expert; measure the variance in All-Reduce latency when one shard receives 3× more tokens, and determine whether a lightweight scheduler that remaps rows during runtime reduces straggler gaps by >20%.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*