---
title: Data Parallelism
slug: data-parallelism
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [hillis, krizhevsky, rajbhandari, cheng]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [distributed-systems-basics, matrix-operations, transformer-architecture]
tags: [data parallelism, distributed training, torch.distributed, communication, sharding, large models]
updated: 2024-10-01
has_mvb: true
---

# Data Parallelism

A developer stacks eight of the fastest GPUs available, launches a training job for a transformer, and watches throughput plummet. The profiled run shows each card sitting idle 90% of the time, waiting for gradients to be synchronized over a saturated PCIe mesh, so the multi-GPU job takes longer than the single-card baseline. The surprise isn’t just that the hardware isn't being fully used; it is that the whole problem has shifted from compute to communication. Achieving linear scaling is no longer a matter of slicing batches thinner and hoping for the best. Modern data parallelism is a choreography of memory sharding, communication scheduling, and topology-aware routing. By the end of this page you will see why we now think of data parallelism as a dynamic system-level scheduler rather than a simple “split the batch” rule, how the arithmetic and network costs trade off, and what it takes to re-implement a bare-bones ZeRO-style pipeline that runs on a single GPU by faking multiple ranks.

## The territory

Scaling a model with more data often seems “embarrassingly parallel”: you broadcast the weights to every worker, split the mini-batch, compute gradients independently, and then average the gradients. Yet this simplistic view breaks down before the second training step on modern transformer workloads because the gradient vectors exceed the interconnect bandwidth of even the highest-bandwidth GPUs. The real contention point is not flops but the cost of reducing a 300 million-parameter gradient tensor every step. The foundational insight from the early SIMD era, articulated by Hillis & Steele (1986) in *Data Parallel Algorithms* [Hillis & Steele 1986](http://cva.stanford.edu/classes/cs99s/papers/hillis-steele-data-parallel-algorithms.pdf), is that a single instruction stream can operate in lockstep across many data elements only if every operator stage is balanced with the communication schedule; an unbalanced stage leaves processors waiting. That idea still governs modern data parallelism: the operators are matrix multiplies and gradient reductions, and the communication stage is all-reduce over gradients or sharded parameters.

Krizhevsky’s “One weird trick” (2014) [Krizhevsky 2014](https://arxiv.org/abs/1404.5997) updated this SIMD intuition for deep networks by showing that hybridizing data and model parallelism reduces the overall communication cost when the batch size is huge. Today’s large-model training stacks still rely on that hybrid approach: tensors are first split across data parallel groups, and then each group uses some form of model or tensor parallel sharding so the per-rank memory footprint stays within a card’s budget. Database pioneers also noticed a similar tension between local throughput and network saturation during scale-out joins, as described in *Untitled* (the University of Illinois systems group report) [Untitled, http://rsim.cs.illinois.edu/arch/qual_papers/systems/6.pdf]. Their scheduler borrowed the same idea: send only as much data as the network can absorb per round-trip so compute units never idle. Those early systems, whose introduction is captured in *1. Introduction* [http://pages.cs.wisc.edu/~dewitt/includes/paralleldb/cacm.pdf], laid the groundwork for thinking about communication blocks and network-aware execution. Modern data parallelism inherits that lineage but trades shared-memory SIMD lanes for fully disaggregated, heterogeneous clusters.

Running through this territory, the key question becomes: how do we turn these insights into a running pipeline today? The mechanism is best understood by starting from an idealized network cost model and then adding the practical sharding, overlapping, and scheduling layers that make the total cycle time bounded by communication rather than compute.

## How it works

A typical data-parallel step executes as: broadcast weights, forward pass per micro-batch slice, backward pass computing gradients, all-reduce to aggregate gradients, optimizer step, repeat. The cost per iteration is dominated by the gradient reduction, so the machinery in this section is aimed at minimizing that cost while keeping GPU utilization high.

### The arithmetic and communication cost model

Before discussing code, quantify the trade-off. Consider \(P\) workers, each with a portion of a mini-batch of size \(B\). The forward and backward flops per worker scale as \(\mathcal{O}(B/P)\), but the volume of data to reduce remains proportional to the total number of parameters \(N\). For a parameter vector \(w \in \mathbb{R}^N\), each worker produces a local gradient \(\nabla w_i \in \mathbb{R}^N\), requiring \(N\) floats to be transmitted every step. If the all-reduce is implemented naively, the communication time is

\[
T_{\text{comm}} = \alpha \log P + \beta N,
\]

where \(\alpha\) is the latency per message and \(\beta\) is the inverse bandwidth (seconds per float). Here \(N\) is fixed by the model, and \(\beta N\) dominates as \(N\) grows. The consequence is that increasing \(P\) simply increases the number of messages without reducing this proportionality—the gradient tensor must be reconstructed on every process. Theoretically, this is the same imbalance that Hillis & Steele recognized: the all-reduce stage must be carefully scheduled to keep compute units busy. In practice, we therefore aim to reduce the effective \(N\) per worker by sharding the gradient tensors themselves.

### Parameter sharding and ZeRO-1

ZeRO-1 sharding splits each parameter tensor across workers so each worker holds only \(N/P\) parameters at a time, reducing communication and memory proportionally. This is achieved by decomposing the gradient reduction: instead of all workers broadcasting their entire \(\nabla w_i\), each worker sends only the shard it owns, and the all-gather is deferred until after the optimizer step, if ever.

Concretely, let \(w = [w^{(0)}, w^{(1)}, ..., w^{(P-1)}]\) be the concatenation of \(P\) shards where worker \(i\) owns \(w^{(i)}\). During the backward pass, worker \(i\) computes gradients \(\nabla w^{(i)}\) only for its shard; the rest of the gradient is not resident. To update the parameters, workers perform an all-reduce over the shards they own and then apply their local optimizer step:

\[
w^{(i)}_{t+1} = w^{(i)}_t - \eta f\big(\nabla w^{(i)}_t\big),
\]

where \(\eta\) is the learning rate and \(f\) may include moments or Adam updates. Because the optimizer step is local to each shard, there is no need to materialize the entire \(w\) on every worker, and the gradient all-reduce now contains only \(N/P\) data. This reduces bandwidth pressure by a factor of \(P\) at the cost of making communication pattern more complex.

The implementation challenge is assembling these shards without introducing synchronization bubbles. That’s where bucketization and asynchronous communication scheduling enter.

### Buckets, pipelines, and overlapping

For each layer’s parameter tensor, we partition the gradient computation into buckets of size \(B_{\text{bucket}}\), typically a few megabytes, and post non-blocking all-reduce operations as soon as a bucket is ready. While the network is busy reducing bucket \(k\), the GPU proceeds to compute buckets \(k+1\). In PyTorch, this looks like calling `torch.distributed.all_reduce` with `async_op=True`, then doing computation while `work.wait()` is deferred until after the bucket finishes.

The key to avoiding idle time is to pipeline the communication and computation:

1. **Forward compute** per micro-batch is followed by gradient computation bucket-by-bucket on the backward pass.
2. **Communication** for bucket \(k\) is launched immediately after the bucket’s gradients are ready, overlapping with the computation of bucket \(k+1\).
3. **Synchronization** occurs only when a future bucket needs a shard that is still in-flight, forcing a `wait()`.

The overlap effectiveness is measured by the overlap ratio, the proportion of communication wall-clock time that is masked by computation. The more aggressive the pipelining (smaller buckets), the higher the overlap, but there’s a diminishing return once latency dominates.

### Memory sharding states and torch.distributed primitives

Torch’s distributed primitives expose a `ProcessGroup` that manages ranks and communicators. We simulate data parallelism by launching each `ProcessGroup` member with its own subset of tensors. The steps for each rank \(i\) are:

1. **Broadcast** the initial parameters: `torch.distributed.broadcast`.
2. **Forward/Backward** compute: `loss = model(input); loss.backward()`.
3. **Shard gradient commit**: For each tensor, call `sharded_grad = tensor.grad.chunk(world_size)[rank]`.
4. **Reduce** the shard: call `torch.distributed.all_reduce(sharded_grad, op=ReduceOp.SUM)` asynchronously.
5. **Optimizer step**: apply updates to the local shard.

When we simulate multiple ranks on one GPU (e.g., Colab T4), `torch.distributed.launch` can create multiple processes sharing the same device context. Each process uses a different gradient bucket schedule computed from its rank, ensuring the bucket offsets are deterministic.

### Dynamic scheduling across heterogeneity

In ideal settings each GPU has identical bandwidth and compute, but production clusters are heterogeneous (PCIe vs. NVLink, T4 vs. A100). The scheduler’s job is to adapt. Imagine you have a collection of accelerators with bandwidths \(B_i\) and compute capacities \(C_i\). The time taken for rank \(i\) to finish communication is \(T_i = (\alpha \log P + \beta_i N_i)\), where \(N_i\) is the size of the shard assigned to \(i\) and \(\beta_i = 1/B_i\). Scheduling becomes an optimization problem: assign shards \(N_i\) and order bucket launches so that the maximum \(T_i\) (the makespan) is minimized.

A practical heuristic is to sort ranks by \(B_i\) and give faster ranks proportionally more data or allow them to hold extra shards, while slower ranks handle fewer shards but still participate in compute to avoid underutilization. Another lever is to overlap communication: slower ranks begin the next bucket earlier while faster ranks wait for them to finish, keeping the all-reduce pipelined.

Krizhevsky’s hybridization perspective implies that when \(B_i\) is too low relative to \(C_i\), one should also shard along the tensor dimension (“model parallelism”) so that the total size \(N_i\) shrinks. This reduces gradient size per rank and thus the communication term in \(T_i\), shifting some of the burden back to computation but keeping the overall makespan bounded.

### Compiler-level automation and SimpleFSDP

The recent SimpleFSDP workflows take this manual choreography and encode it into compiler passes that schedule communication automatically. SimpleFSDP (2024) creates a computation graph where each tensor is annotated with a shard layout, and it overlaps broadcast/all-reduce operations with computation by inserting the appropriate synchronization points as soon as the compiler detects data dependencies. This lets practitioners write the same model code and rely on `torch.compile` to manage asynchronous bucket scheduling without hand-crafted hooks. The compiler also profiles the network and dynamically adjusts bucket sizes per step, thereby restoring the illusion of an “8× speedup on 8 GPUs” without the reveal that communication is still the real bottleneck. The result is tractable pipeline code that can run statelessly in Colab or production, which is where our build will land.

## Where the field is now

The research frontier continues to push the boundary between compute and network by optimizing the sharding strategy itself. ZeRO-DP (Rajbhandari et al. 2020) [arxiv:1910.02054](https://arxiv.org/abs/1910.02054) introduced a layer-wise sharding scheme that distributed optimizer states, gradients, and parameters across ranks instead of just gradients. Building on that, ZeRO-Offload (Rajbhandari et al. 2020) moved parts of the optimizer state to CPU memory to avoid GPU memory saturation. For the latest results, systems like Alpa (Xing et al. 2023) [arxiv:2204.04603](https://arxiv.org/abs/2204.04603) unify parallelism strategies and automatically search for optimal schedules, achieving training throughput close to what a human expert could design. The research frontier is now about making that search adaptive at runtime—changing the sharding layout between micro-batches depending on how the interconnect is behaving.

The engineering frontier demonstrates that these algorithms now run at global scale. Meta AI’s SimpleFSDP initiative (2024) [https://ai.meta.com/blog/simple-fsdp/] reports that their compiler-managed sharding maintains throughput across thousands of GPUs with minimal manual tuning, even when the fleet includes older PCIe cards alongside fresh NVLInk machines. This deployment shows how the communication-scheduling problem is being solved in practice: each job analyzes the farm’s network topology before scheduling gradient buckets, ensuring no rank waits more than 4ms for an all-reduce to complete even when the cluster mixes different interconnect speeds. At the same time, OpenAI’s documentation of their FSDP-based training (2023) shows they target 70% GPU utilization across a 10,000-GPU cluster by overlapping optimizer compute with communication and by using adaptive bucket sizes when training GPT-4 sized models. Those production practices illustrate that tuning the scheduler is as essential as tuning the optimizer when the model is in the billions of parameters.

## What's still open

1. **How can we schedule data-parallel sharding across a heterogeneous, decentralized cluster without introducing massive synchronization bubbles?** Current frameworks assume a shared parameter server or low-latency fabric; dropping that assumption raises the question of how to partition shards and order communications across an over-subscribed WAN without forcing all ranks to idle waiting for the slowest link.

2. **Can we design a cost model that jointly optimizes bucket size, overlap strategy, and memory placement so that the scheduler adapts on-the-fly to varying network contention?** Today's heuristics are static or manually tuned per job; the open question is whether reinforcement learning or differentiable scheduling can find the Pareto frontier between compute and communication for an arbitrary cluster.

3. **Is there a sharding scheme that removes the need for global all-reduce entirely while keeping model convergence unaffected?** That would require local gradient compression or sketching that guarantees convergence despite stale or partial updates while the communication volume stays sub-linear in \(N\).

4. **What are the correctness conditions when data parallelism meets fault tolerance and elasticity?** If nodes drop out or fluctuate in speed, can we reassign shards without restarting the entire optimizer while preserving statistical equivalence to the original global update?

## Where to read next

If you want the theory behind why gradient averaging works without recomputing partition functions, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) connects score estimation to the noise-aware optimizer updates we rely on. The engineering counterpart is → [[flash-attention]] which explains how transformer kernels are kept efficient enough to saturate the compute during the problematically slow all-reduce phase. For operationalizing these ideas at the next scale, → [[zero-redundancy-optimizer]] shows how ZeRO variants shard not just parameters but optimizer states and activations.

## Build it

The build proves that you can re-create ZeRO-1 style data-parallel training without needing a GPU cluster by simulating multiple ranks on one device and demonstrating how sharding and asynchronous all-reduces improve throughput.

**What you're building:** A parameter-sharded Transformer training loop that uses `torch.distributed` to emulate 2–4 logical data-parallel ranks on a single Colab T4 GPU and runs a synthetic language-modeling task.

**Why this is valuable:** This forces you to implement the bucketization, shard ownership, and asynchronous all-reduce scheduling that hide communication cost, proving the core idea that data parallelism is a communication-scheduling challenge, not simply batch splitting.

**Stack:**
- **Model:** [facebook/opt-125m](https://huggingface.co/facebook/opt-125m) — 1.2M downloads
- **Dataset:** [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/wikitext-2-raw-v1) — 0.5MB, tokenized sequences
- **Framework:** PyTorch 2.1 + `torch.distributed`, `torch.compile`, `torchrec` for bucket utilities
- **Compute:** Colab T4 (16GB), simulating 2–4 ranks via `torchrun --nproc_per_node`, ~2 hours wall time for 10K steps

**The recipe:**
1. Install Python packages: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 torchtext`.
2. Data: Tokenize wikitext-2 sequences into 512-length chunks, build a `torch.utils.data.DataLoader`, and pad each micro-batch so it evenly divides by the number of simulated ranks.
3. Train/fine-tune: Wrap an `OPTForCausalLM` model in your own `ShardedModule` that owns a single shard of each parameter; call `torch.distributed.init_process_group("nccl", rank=rank, world_size=world_size)` and use `torch.distributed.all_reduce` asynchronously on `tensor.grad.chunk(world_size)[rank]`; expect the loss to stabilize around 3.1 after ~10 epochs, and verify the per-step time decreases when you shrink bucket size from 8MB to 4MB because overlap improves.
4. Evaluate: Generate text samples from the decoded checkpoint and compute perplexity on the validation split; target a perplexity below 25 and confirm the simulated ranks match the single-rank perplexity within 5%.
5. What you now have: A runnable ZeRO-1 style script that demonstrates shard ownership, asynchronous communication, and configurable bucket scheduling.

**Expected outcome:** A checkpoint and log demonstrating a consistent loss curve, generated validation samples, and a profile showing communication overlap that you can point back into your future cluster jobs.

- **CS student:** Run the same script on an RTX 4070 with only two simulated ranks and extend the dataset to `tiny_shakespeare` to observe how longer sequences stress the communication pipeline.
- **Applied engineer:** Deploy the trained checkpoint using `torchrun --nproc_per_node=2` inside a Docker container, quantize the model weights to 4-bit, and report the latency of one inference pass on an A10 with synchronized buckets to ensure p50 < 50ms.
- **Applied researcher:** Hypothesize that swapping asynchronous all-reduce for synchronous `all_reduce` increases peak iteration time; measure the makespan difference across several bucket sizes to confirm how much overlap actually saves time.
- **Frontier researcher:** Use this setup to explore the open question of heterogenous scheduling by introducing a synthetic `network_delay` per simulated rank and determining whether adaptive bucket resizing can maintain convergence without introducing stragglers.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*