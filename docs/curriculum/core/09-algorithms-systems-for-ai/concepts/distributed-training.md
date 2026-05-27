---
title: Distributed Training
slug: distributed-training
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [dean, krizhevsky, sergeev, ranzato]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [data-parallelism, pytorch-basics, gpu-architecture]
tags: [distributed-systems, collective-ops, pytorch, ddp, communication, parameter-server]
updated: 2025-04-26
has_mvb: true
---

# Distributed Training

Pushing a prototype from eight to sixty-four GPUs often uncovers a paradox: instead of a 4× throughput gain, the iteration time inflates because the GPUs spend most of their cycles waiting for gradients that never arrive. Most distributed-training failure stories have nothing to do with floating-point operations per second (FLOPS, the count of arithmetic operations executed every second); they arise when a machine’s compute units are ready but the network fabric is still delivering the gradient packets they rely on. This page treats distributed training as the communication-bound co-design problem it is. It lays out how engineers translate the statistical requirement that every worker agree on a gradient into network protocols, why those protocols are the real bottleneck, what the modern toolchain does to hide or eliminate that bottleneck, and how to run a concrete PyTorch DistributedDataParallel experiment that makes the datapaths visible.

## The territory

Scaling deep learning across tens, hundreds, or thousands of accelerators splits into three design axes: how the model is sliced (data parallelism, tensor parallelism, pipeline parallelism), how gradients or parameters are aggregated, and how the network fabric is scheduled so the compute units never idle. Distributed training sits at the intersection of those axes, translating statistical desiderata (all workers must agree on a descent direction) into systems requirements (synchronize 3.4 GB of gradient bytes in under a millisecond) and back again. What changes from the single-GPU regime is not the math but the control flow: a single optimizer step becomes a micro-protocol that coordinates CPUs, GPUs, NICs, and racks.

Historically the story began with centralized parameter servers. DistBelief introduced a server that held the master copy of every weight; workers read gradients, pushed updates back, and progressed independently without waiting for peers (Dean et al. 2012) [https://www.cs.toronto.edu/~ranzato/publications/DistBeliefNIPS2012_withAppendix.pdf]. That design treated communication as an inevitable expense and hid latency with asynchrony, but it also exposed the communication tax—every new worker increased the total synchronization volume. The field then gravitated toward collective algorithms that explicitly minimize the number of bytes traversing the interconnect, which is why Horovod (Sergeev & Del Balso 2018) [https://arxiv.org/abs/1802.05799] and Krizhevsky’s “one weird trick” (Krizhevsky 2014) [https://arxiv.org/pdf/1404.5997] remain reference points. These two families—parameter servers plus asynchronous SGD on one side, collectives plus compression on the other—define the territory. The stage is set to understand how the stack balances compute and communication: that story begins with the cost model for a synchronized training loop.

## How it works

Every step’s wall-clock time for a data-parallel model with \(p\) workers is
\[
T_{\text{step}} = T_{\text{comp}} + T_{\text{comm}}.
\]
Here \(T_{\text{comp}}\) is the time each device spends computing gradients for its local mini-batch and performing the optimizer update, and \(T_{\text{comm}}\) is the time spent exchanging those gradients or updated parameters over the network fabric. When \(p\) increases, \(T_{\text{comp}}\) stays roughly constant (the batch per worker shrinks), but \(T_{\text{comm}}\) tends to grow unless the network protocol keeps the per-worker bandwidth cost and latency cost in check. The distributed-training challenge reduces to keeping \(T_{\text{comm}} \ll T_{\text{comp}}\) even as \(p\) grows.

### Parameter servers and asynchronous SGD

DistBelief’s parameter server kept a centralized copy of every weight. Each worker read the latest parameters, computed gradients on its shard of data, then pushed the gradients back. Because the server serialized access, the communication cost seen by any worker was
\[
T_{\text{comm}}^{\text{ps}} \approx \alpha + \beta n,
\]
where \(\alpha\) is the latency to initiate a message, \(\beta\) is the inverse bandwidth, and \(n\) is the number of bytes in the worker’s gradient block. This cost does not shrink with \(p\), so scaling to hundreds of workers amplifies the ratio of network bytes to useful work unless the system splits the parameter matrix across multiple servers (sharding) and aggregates updates hierarchically. Asynchronous SGD allowed workers to move ahead without waiting; Tsitsiklis et al. (1986) [https://www.mit.edu/~jnt/Papers/J014-86-asyn-grad.pdf] showed convergence still held under bounded staleness if the learning rate decayed gracefully.

The trade-off came through loose consistency. Every optimizer step had to tolerate gradients that might be several iterations stale, which meant extra noise persisted even if the compute nodes were fast. In practice, parameter servers pushed the communication latency into the optimizer loop, making hyperparameter tuning more fragile. This tension is precisely why modern workloads started returning to synchronous protocols that guarantee every worker sees the same gradients before stepping. The remaining question became how to shape those synchronous collectives so \(T_{\text{comm}}\) stays lean.

### Ring AllReduce, bucketization, and alpha–beta dynamics

Synchronous collective algorithms require every worker to participate in aggregation. A centralized gather-and-broadcast costs \(\mathcal{O}(p)\) messages at the master node, flooding one machine with bandwidth requirements. Horovod’s Ring AllReduce (Sergeev & Del Balso 2018) organizes the workers into a logical ring and circulates the gradients, avoiding the master bottleneck. The time per aggregation becomes
\[
T_{\text{allreduce}} = \alpha (p - 1) + \beta \frac{n}{p},
\]
because each worker performs \(p - 1\) peer-to-peer transfers (the first term) and sends/receives \(n/p\) bytes through the ring (the second term). The total volume is \(2n\) because every byte traverses the ring twice. When the network topology is uniform, the scheme reaches bandwidth optimality.

Krizhevsky’s “one weird trick” anticipated this by showing that fusing gradients into large buckets sized to match the NIC’s maximum transmission unit (MTU) saturates the interconnect instead of sending thousands of micro-messages. The bucket size parameter directly enters the cost model: if each bucket contains \(b\) bytes, then the latency term becomes \(\alpha (p - 1)\) per bucket, so using fewer, larger buckets reduces the apparent latency cost by lowering the number of collective calls. PyTorch’s DistributedDataParallel (DDP) now assigns gradients to buckets around the bucket_size_mb setting, overlaps bucket communication with backward computation, and triggers NCCL’s fused kernels. This overlapping means the aggregate latency \(\alpha\)-term is amortized over the bucket, while the bandwidth cost \(\beta (n/p)\) is metered by NCCL’s pipelined reduce-scatter and all-gather.

Taking a step back, the cost model reveals why synchronous collectives replaced parameter servers: asynchronous SGD hides latency by letting workers drift in parameter space, but synchronous algorithms—and their latency-amortizing buckets—contain \(\alpha\) in a predictable way, so system designers can reason about scaling across nodes.

### Sharding, ZeRO, and Fully Sharded Data Parallel (FSDP)

As models grew beyond 10 billion parameters, even the memory needed for optimizer states dominated device memory. ZeRO (Rajbhandari et al. 2020) [https://arxiv.org/abs/1910.02054] splits the optimizer state, gradients, and parameters across devices so that each GPU stores only \(1/p\) of the total tensor. Zero Stage 3, the fully sharded optimizer, interleaves reduce-scatter and all-gather operations whose communication cost is
\[
T_{\text{zero3}} \approx \alpha (p - 1) + \beta\left(2 \frac{n}{p}\right),
\]
because every tensor participates in a reduce-scatter (sending \(n/p\) bytes per worker) and an all-gather (another \(n/p\) bytes). Compared to unsharded collectives where each worker handles \(n\) bytes, ZeRO’s sharding shrinks the bandwidth term \(\beta\) by \(p\) and keeps the latency term bounded by the number of stages in the bucketed collective.

FSDP (Dettmers et al. 2021) [https://arxiv.org/abs/2104.04473] builds on this idea inside PyTorch: it shards each module’s parameters, gradients, and optimizer momenta, and it overlaps communication with computation. The key insight is that sharding reduces the ``\(n/p\)'' part in the cost model without altering \(\alpha\). FSDP now matches the latency and bandwidth usage expected in ZeRO Stage 3 and provably outperforms non-sharded DDP when \(n\) grows large. Collectively, ZeRO and FSDP demonstrate that one can tackle the bandwidth term at scale by slicing tensors instead of adding more network wires.

### Hierarchical all-reduce, tensor parallelism, and pipeline parallelism

For modern training runs, dimension-wise parallelism multiplies the communication complexity. Tensor parallelism slices each matrix multiply so that the activation tensors themselves are replicated across devices, requiring additional all-reduces for each weight update. Pipeline parallelism breaks the model’s layers into stages; each stage processes micro-batches asynchronously while gradients propagate backward. GPipe (Huang et al. 2018) [https://arxiv.org/abs/1811.06965] introduced micro-batching and a 1F1B schedule that keeps compute units busy even when pipeline bubbles exist.

The combined cost now includes pipeline-induced idle time \(T_{\text{pipeline}}\) and tensor parallel all-reduces. If tensor parallelism partitions the activation into \(q\) shards, then the bandwidth term becomes \(\beta (n/p) \cdot q\), because each shard must exchange partial results with the others. Designers therefore add hierarchical all-reduce strategies that treat intra-node communication (accelerators inside a single host) differently from inter-node communication, reducing the effective \(\alpha\) seen per gradient by performing multiple smaller collectives and then re-aggregating. A practical cost estimate for such a multi-level reduction splits \(T_{\text{comm}}\) into local and global components, both of which feed into the same \(\alpha\) and \(\beta\) terms but with smaller \(p\) and \(n\) per level.

### Gradient compression, quantization, and asynchronous compensation

Another lever for shrinking \(T_{\text{comm}}\) is lowering \(n\). Gradient compression techniques—top-k sparsification, quantization, or dynamic thresholding—reduce the number of bytes transmitted at the cost of adding \(T_{\text{comp}}^{\text{compress}}\), the CPU/GPU time to encode and decode compressed tensors. These techniques change the bandwidth term to \(\beta (rn/p)\), where \(r\) is the compression factor (e.g., \(r=0.1\) for a 10× reduction). Designers must balance the added computation with the saved network time, and many systems use error-feedback to compensate for the information lost during compression.

Taken together, the cost model explains why modern distributed training pipelines rarely rely on a single primitive. Parameter servers dominated the early era when \(n\) was manageable and latency was masked by asynchrony. Ring AllReduce—and its fused bucketization—governed the first wave of synchronous scaling. Sharding (ZeRO/FSDP), tensor/pipeline parallelism, and compression formed the toolkit for training trillion-parameter models. Every optimization resolves back to the same relationship between \(\alpha\) and \(\beta\), and the success of a deployment depends on whether it keeps \(\beta n/p\) and \(\alpha (p - 1)\) within the compute budget.

## Where the field is now

### Where this concept appears

Distributed training is the glue between algorithm design and systems engineering across the artificial-intelligence curriculum. This page anchors the 09-algorithms-systems-for-ai subject’s treatment of large-scale model training, connecting downstream to [[data-parallelism]] (which drills into batch partitioning) and [[tensor-parallelism]] (which explains how to slice weight updates before they hit the network). The concept also appears whenever the curriculum addresses HPC infrastructure for training LLMs, RL agents, or large diffusion models because all those arcs depend on the communication-optimization techniques explained above.

### Research frontier

Research efforts today push beyond ZeRO Stage 3. ZeRO Infinity (Rajbhandari et al. 2023) [https://arxiv.org/abs/2304.07308] extends ZeRO to heterogeneous memory hierarchies—parameter, GPU, CPU, and SSD—by treating each tier as a shard and designing a scheduler that minimizes total data motion while honoring the same \(\alpha\)/\(\beta\) calculus. Megatron-LM’s tensor–pipeline parallelism (Shoeybi et al. 2019) [https://arxiv.org/abs/1909.08053] blended pipeline scheduling with tensor partitioning and inspired subsequent work on automatic stage balancing. More recent papers explore automatic communication compression tuned to gradient distributions and algorithms that co-design the optimizer with the communication pattern. These frontier works aim to extend the cost model to heterogeneous fabrics, making \(T_{\text{comm}}\) predictable even when \(p\) spans GPUs, TPUs, and bespoke accelerators.

### Engineering and production frontier

OpenAI’s GPT-4 technical report (2023) [https://openai.com/research/gpt-4] describes training runs across thousands of A100 GPUs using a heavily optimized data-parallel pipeline built on PyTorch DDP, ZeRO sharding, and Horovod-style collectives. The same core ideas run inside Meta’s LLaMA and Llama 3 training, where FSDP and ZeRO sharding enable 2× larger context windows without adding racks of additional nodes. These production deployments highlight the practical questions that product managers care about: how much does an extra \(\alpha\) hit cost for a new region, how many idle GPUs does the scheduler tolerate, and what is the ROI of squeezing \(\beta\) by using third-party compression hardware? The field is still measuring these trade-offs, but the best-in-class stacks are those that treat the communication model with the same rigor as the optimizer.

## What's still open

Can a scheduler automatically decide when to switch between ring-based collectives, hierarchical all-reduce, and asynchronous parameter servers as network topology, GPU count, and model size change, while still providing convergence guarantees? Will aggressive gradient compression and lossy quantization sustain convergence when the network layer drops entire packets or introduces burst latency, or is a new theory of gradient noise necessary to account for those failures? How can pipeline parallelism be fused with tensor sharding so that the combined system keeps \(T_{\text{comm}}\) under control even when the compute-to-communication ratio varies wildly across layers?

## Where to read next

If the engineering side is appealing, → [[data-parallelism]] explains how to parcel the dataset, → [[tensor-parallelism]] shows how tensors themselves can be split, and → [[pipeline-parallelism]] drills into micro-batching across sequential layers; the mathematical underpinnings of the communication model live in → [[sgd-theory]] and the hardware story is captured by → [[gpu-architecture]].

### Connected topics

Connected topics include [[collective-ops]] for the full taxonomy of NCCL, MPI, and gloo collectives, while [[optimizer-design]] explores how communication-aware optimizers (like AdamW with gradient clipping) close the loop between statistics and systems.

## Build it

**What you're building:** A PyTorch DDP + ZeRO Stage 3 micro-lab that trains `distilbert-base-uncased` on `glue/sst2` across two GPUs, making gradient synchronization visible in logs.

**Why this is valuable:** This hands-on experiment turns the abstract \(\alpha\)/\(\beta\) model into tangible observables, showing how bucket size, gradient sharding, and NCCL primitives interact in a real training job.

**Stack:**
- **Model:** [distilbert-base-uncased](https://huggingface.co/distilbert-base-uncased) — 100M parameters, optimized for easy fine-tuning.
- **Dataset:** [glue/sst2](https://huggingface.co/datasets/glue/viewer/sst2/train) — binary sentiment classification with 67k sentences.
- **Framework:** PyTorch (2.1) + DeepSpeed (0.9) with NCCL backend;