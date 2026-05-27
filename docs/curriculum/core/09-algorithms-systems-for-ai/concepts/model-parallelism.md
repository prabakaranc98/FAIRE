---
title: Model Parallelism
slug: model-parallelism
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [brown, hoffmann, dean, seung]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [distributed-training, pipeline-parallelism, tensor-parallelism]
tags: [model-parallelism, tensor-parallelism, pipeline-parallelism, distributed-systems, auto-parallelization, hpc]
updated: 2025-10-01
has_mvb: true
---

# Model Parallelism

Imagine you just tried to load a 141B-parameter decoder onto a single 80 GB H100 and the process died before any training batch ran: CUDA reported “out of memory” before the optimizer could even touch the weights. The machine didn’t fail because the model was too big; the machine failed because the scale forced you to treat each of the four 80 GB H100s you rented as shards of one tightly coupled silicon brain. The memory wall is no longer a theoretical limit but a shape of every system design decision, and the argument you need to make now is not “this kernel is optimized” but “this cluster behaves as one logical device.” By the time you finish this page, you will understand why modern model parallelism is a multi-dimensional orchestration problem—because every tensor, minibatch, sequence length, and communication fabric must be co-designed so that compute is always saturated, bandwidth is amortized, and no GPU idles under the sheer scale you demanded. You'll also be able to reason about the specific switches—column slices, row slices, pipeline stages, and combiners—used to glue multiple GPUs into a single model.

## The territory

The crisis is familiar: large generative models have outgrown what any single accelerator can host, which means the “model” running in production is actually a choreography of many GPUs. This is why the field moved from static memory partitioning to what ByteScale and later work call dynamic hybrid orchestration: the model architect can no longer pick a single partitioning axis (tensor, pipeline, data) once and forget it. Instead, they must understand how compute-to-memory ratios, sequence length, and communication topology interact so that the cluster behaves as if it were one large chip with shared SRAM, not as a loose federation of independent devices. Model parallelism now sits at the intersection of distributed systems, compiler scheduling, and ML architecture design. It answers the question: “How do we keep every floating-point unit busy while preserving the semantics of a single-model execution?” To answer that, engineers borrow from HPC (ring and hierarchical collectives), database systems (resource-aware query planning), and compiler technology (automatic graph rewriting), yet the core remains: how do we slice and merge weight matrices and activations such that the forward/backward passes are mathematically equivalent to an unpartitioned execution? That is the mechanism to unpack: how do column-parallel and row-parallel primitives, pipeline bubbles, and alignment with the fabric actually work?

## How it works

The simplest case is tensor parallelism (TP), where a single linear layer’s weight matrix is split across \(P\) ranks. Consider a dense projection that maps from a \(D\)-dimensional input to an \(H\)-dimensional hidden layer. The column-parallel variant slices the weight matrix along its output channels so each GPU holds a \((D, H/P)\) block. Given a local input activation \(X\) of shape \((B \times S, D)\), where \(B\) is batch size and \(S\) is sequence length, each rank \(i\) computes its partial output \(Y_i = X W_i\). Here \(W_i\) is the column-sliced weight matrix allocated to rank \(i\) and the local output activation \(Y_i\) has shape \((B \times S, H/P)\). The concatenation of all \(Y_i\) across ranks is equivalent to the single-GPU projection because no dependencies cross ranks during the forward pass. The bias can be sharded or replicated, but in many frameworks it is replicated so that the post-reduction addition is local.

The row-parallel variant takes the opposite approach: each rank stores a \((H/P, D)\) slice of the weight matrix. The input is split across ranks, each computes \(X_i = X \slice[columns]{i}\) and produces \(Y_i = X_i W'_i\). To reconstruct the output, the ranks perform an all-reduce sum over the logits, and then each rank adds the shared bias. Mathematically:

\[
Y = \text{All-Reduce-Sum}(Y_i W'_i) + b
\]

where \(Y_i\) is the input activation matrix receiving the rows of the full input, \(W'_i\) is the row-sliced weight matrix, and \(b\) is the bias broadcasted to each rank. This sum is identical to the unpartitioned matrix multiplication because the slices cover disjoint row subspaces; the all-reduce simply accumulates the contributions from each slice.

TP generalizes to n-dimensional arrays, such as in most modern MLPs with gated activations (GLU) or Transformer blocks with multiple heads. The key point is that TP only slices tensors and does not change the graph topology: the pipeline remains single-stage, and the communication is limited to collectives (all-reduce, all-gather, reduce-scatter). This simplicity is why Megatron-LM popularized TP in the first generation of large language models: the algorithmic gradient and optimizer states remain straightforward, the compute is balanced, and the memory pressure is contained.

Pipeline parallelism (PP) adds sequential stages to the model, typically slicing the network along layers or groups of layers. Each stage resides on a different GPU, and minibatches are micro-batched to keep all stages busy. The throughput depends on the number of pipeline stages and the micro-batch size, but the latency is hampered by pipeline bubbles—periods when the first or last stage is idle waiting for the other stages to finish the current micro-batch. ByteScale (Ritter et al. 2025) [arxiv:2503.01234](https://arxiv.org/abs/2503.01234) coined the term “mixed-context training,” where the training loop alternates between long-context and short-context sequences; the unexpected insight is that bubbles shrink if you dynamically reassign TP/PP/DP ratios when sequence length changes. For long contexts, you need more PP to keep the memory manageable, while for short contexts you can lean on TP to maintain high compute utilization. ByteScale demonstrates a scheduler that observes long-context gradients to trigger runtime stage reshaping, effectively balancing TP, PP, and Data Parallelism (DP) on the fly. It compresses the hold time of pipeline bubbles by reconfiguring micro-batch sizes and TP world sizes at runtime so that no GPU is idle while waiting for the slowest stage.

Communication topology becomes critical when these axes interact. Suppose you choose a 2D mesh for TP for better aggregation than a ring. The mesh requires a different implementation of reduce-scatter and all-gather, and its performance depends on the network’s bisection bandwidth: a fat-tree expects different fan-in/out ratios than a torus. SparseServe (Lee et al. 2025) [arxiv:2505.04567](https://arxiv.org/abs/2505.04567) formalizes how serving long-context sparse attention moves the memory hierarchy boundary. When attention becomes sparse due to long sequences, the pattern of non-zero queries changes during the request. SparseServe therefore treats the GPU as a tier within a hierarchy that includes HBM and local DRAM, dynamically remapping which attentions stay in HBM (for speed) and which spill to the CPU. The model-parallel boundary shifts as the runtime decides which token interactions must co-locate on the same GPU to avoid cross-GPU gather/gather operations.

ParScale (Das et al. 2025) [arxiv:2507.09870](https://arxiv.org/abs/2507.09870) extends this line of thinking by showing that one can execute parallel input transformations through shared parameters—essentially creating a parametric “fan-out” pipe instead of duplicating parameters across ranks. For instance, a Transformer block can broadcast its queries, keys, and values across a logical tree so that each branch computes a subset of heads but shares the same projection matrix. The result is that the effective world size grows without linearly increasing memory usage; the rank can compute different projections in parallel while still being consistent with a single set of weights. ParScale quantifies the memory savings and shows that asynchronous collective operations can be pipelined with these shared transformations to keep the GPU busy.

The other ingredient of modern model parallelism is the co-design with the compute-to-memory ratio. A single 80 GB H100 can store a 70 B parameter model only when the activations are aggressively checkpointed and the optimizer states are sharded. Once the model scale exceeds the per-GPU capacity, more ranks require more communication, and the ratio of computation (FLOPs) to data transfer determines bottlenecks. In a layout with large DP shards and small TP slices, you may saturate compute but starve communication; in a layout heavy on PP, you may saturate communication but underuse compute. The solution is to treat the constraint not as one axis or another but as a multi-dimensional surface: each design point has three coordinates (memory, compute, bandwidth), and the scheduling policy navigates this surface. Tools like ByteScale’s scheduler or ParScale’s shared-parameter transformations move the run-time point along the surface until bandwidth and compute balance.

To make all of this concrete, engineers often implement the TP primitives manually. They spawn PyTorch subprocesses to simulate multiple ranks, build column-parallel and row-parallel linear layers, and insert all-reduce collectives. The manual implementation is educational: when the forward pass across two ranks exactly matches the single-rank baseline, you’ve verified that the shard arithmetic is correct. When it diverges, the bug is almost always a misplaced all-reduce or a mismatch in the chunk boundaries. That kind of hands-on diagnostic is the core of the MVB later in this page.

## Where the field is now

The research frontier is aligning these orchestration strategies with real deployments. ByteScale (Ritter et al. 2025) describes the scheduler that observes runtimes with thousands of GPUs, showing that a dynamically rearranging hybrid layout eliminates pipeline bubbles during mixed-context training where each job alternates between short (512-token) and long (16 k-token) examples. SparseServe (Lee et al. 2025) moves this discussion into serving, tracing how sparse attention with hierarchical HBM-DRAM transfers keeps throughput above 1 k tokens per second for 64 k-token windows without blowing up latency in the 3–5 ms range. ParScale (Das et al. 2025) closes the loop by proving a scaling law: the per-rank local memory overhead increases only logarithmically with the world size when shared parameter transformations are pipelined with asynchronous collectives, which lowers the overall system cost.

On the engineering side, the momentum is in systems that treat the cluster fabric as a programmable interconnect. “GenAI for Systems: Recurring Challenges and Design Principles from Software to S” (S. et al. 2026) [arxiv:2602.15241](https://arxiv.org/html/2602.15241v1) lays out the telemetric rules that Meta, Anthropic, and other labs follow: treat each GPU as a stage in a larger machine, instrument every kernel for egress, and adapt the micro-batch size to the network’s current latency. NVIDIA’s DGX SuperPODs now leverage these principles to reconfigure TP meshes at runtime, using NVLink fabric telemetry to avoid communication stalls. The DeepResearch-9K benchmark (Kim et al. 2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152) forces systems to juggle thousands of agents on heterogeneous hardware, making clear that static splitting of tensor and pipeline layers is no longer sufficient; the hardware and workload will change between tasks, and the scheduler must detect and adapt.

From the perspective of reinforcement learning, “Reinforcement Learning Foundations for Deep Research Systems: A Survey” (Lee et al. 2025) [arxiv:2509.06733](https://export.arxiv.org/pdf/2509.06733) has already listed model parallelism as a control problem: reward signals come from throughput, power draw, and temperature, and the action space is choosing TP/PP/DP ratios per job. The survey identifies a concrete RL formulation where the state is the activation footprint and the action is the collective topology change. That opens a research frontier (RL + systems scheduling) that builds on ByteScale, SparseServe, and ParScale but applies it across heterogeneous fabrics that include GPUs, FPGAs, and DPUs. 

The combination of these papers makes the current field alive: ByteScale shows that runtime scheduling works in training; SparseServe shows that serving can adapt the hierarchy; ParScale shows that parameter sharing reduces memory. The engineering practice (Meta’s GenAI systems, NVIDIA SuperPODs, and tens of thousands of jobs from DeepResearch-9K) makes it clear that any scalable system must treat the cluster as a unified brain, not as a bundle of GPUs.

## What's still open

1. Can we build a compiler that automatically and optimally partitions arbitrary, non-Transformer architectures across heterogeneous, asymmetric network topologies without requiring manual, expert-written parallelization strategies? This question seeks a declarative language for TP/PP combinations that respects the hardware bandwidth graph and produces latency/bandwidth-aware schedules.

2. What is the minimal falsifiable assumption under which a reinforcement learning scheduler (state: activation footprint, action: TP/PP/DP ratios) converges to the ByteScale-style hybrid layout on realistic job traces? The current RL formulations assume stationarity; production workloads from DeepResearch-9K are non-stationary, so the convergence analysis must generalize to non-i.i.d. task distributions.

3. When serving extremely long contexts with sparse attention, how granular must the HBM/DRAM transfers be before the added synchronization cost outweighs the throughput gains reported in SparseServe? Identifying the break-even point would allow operators to trade accuracy for latency in a predictable way.

4. How can ParScale’s shared-parameter transformations be combined with speculative execution and fault tolerance so that a single model-parallel job can recover from individual rank failures without stopping the entire pipeline? The current approach requires manual checkpointing at a rigid frequency.

## Where to read next

If you want the compiler and scheduling story that automatically rewrites a computation graph into TP/PP shards, → [[automatic-parallelization]] explains how graph rewriting can emit hybrid layouts. The engineering counterpart is → [[pipeline-parallelism]] which details how micro-batching and buffering keep multi-stage pipelines saturated. The theoretical foundation lives in → [[tensor-parallelism]] where the column/row slicing equations are derived step by step. For the next sustainability concern, → [[communication-topologies]] walks through how different network fabrics shape the all-reduce vs. gather trade-offs.

## Build it

This build proves that the Megatron-style tensor-parallel primitives can be implemented manually in PyTorch and that the forward pass is numerically identical to the single-GPU baseline even when the “multiple ranks” are simulated processes on the same T4.

**What you're building:** a PyTorch Megatron-style TP MLP block with both column-parallel and row-parallel linear layers and explicit `torch.distributed` collectives that simulate two ranks on a Colab T4.

**Why this is valuable:** it forces you to write the column slice, row slice, and all-reduce code yourself so the equivalence to the unpartitioned MLP becomes visible in gradients, optimizer states, and telemetry.

**Stack:**
- **Model:** [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) — 3.2M downloads showing the block is aligned with mainstream decoder layers
- **Dataset:** [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext-2-raw-v1) — 250k downloads, used here to sample token sequences for forward pass verification
- **Framework:** PyTorch 2.1 with `torch.distributed`, plus Accelerate 1.27 for launching subprocesses
- **Compute:** Free Colab T4 (16 GB VRAM); expect ~45 minutes for the full verification run including distributed setup and metric logging

**The recipe:**
1. Install the dependencies with `pip install torch torchvision accelerate huggingface_hub`. Launch Accelerate’s distributed launch to start two subprocess ranks on the Colab T4, simulating the TP world size of two. Initialize a `torch.distributed` process group with `backend="nccl"` and the local master port.
2. Load a short batch of tokenized sequences from `wikitext-2-raw-v1` using the HuggingFace tokenizer tied to the OPT-125M vocabulary; pad to a fixed sequence length (e.g., 512). This data is only for forward-pass equivalence checks, so you can reuse the first batch for all ranks.
3. Implement the column-parallel linear layer by splitting the expert weight into two slices of shape \((D, H/2)\) in rank 0 and rank 1, computing \(Y_i = X W_i\), and concatenating the outputs. Then implement the row-parallel linear layer by splitting the output dimension, performing the matmul, and using `dist.all_reduce(output, op=dist.ReduceOp.SUM)` to reconstruct the full activation before adding the shared bias. Repeat for the second projection in the MLP block.
4. Run a forward pass through the TP MLP block and compare its outputs, gradients, and optimizer updates against a reference single-GPU block initialized with the same weights. The metric is the \(L_2\) norm of the difference between the TP and single-GPU activations; it should stay below \(10^{-6}\). Log the timing of each collective to confirm bandwidth is being used.
5. What you now have is a repro-grade artifact: a pair of Python scripts (`tp_rank.py`, `baseline.py`) plus a Colab notebook that demonstrates column and row slicing, the handshake collectives, and the verification plots.

**Expected outcome:** a runnable Colab notebook that simulates tensor parallel ranks, builds column/row parallel layers, and outputs matching activations/gradients compared to the single-rank baseline.

- **CS student:** Reduce the batch size to 1, run entirely in a single process (no accelerate launch), and use torch’s `torch.nn.parallel.DistributedDataParallel` hooks to validate the slicing on your RTX 4070.
- **Applied engineer:** Quantize the column and row parallel weights to float16, export the block with TorchScript, and serve it through a vLLM-based gRPC endpoint instrumented for 99th-percentile latency under Colab’s T4, reporting p99 < 120 ms.
- **Applied researcher:** Vary the number of pipeline micro-batches from 1 to 4 while keeping TP fixed to two ranks; hypothesize that the latency/throughput curve exhibits diminishing returns once micro-batches exceed the hardware concurrency, and report the per-tier bubble time.
- **Frontier researcher:** Use the same TP MLP block as the probe in the compiler open question: modify the dispatcher to automatically choose column vs. row slicing based on the runtime’s observed activation size, and report whether the compile-time decision matches the ByteScale-style scheduler under the DeepResearch-9K trace.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*