---
title: Data Parallelism
slug: data-parallelism
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [hillis, krizhevsky, paszke, rajbhandari]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [collective-communication, gradient-descent, distributed-training-basics]
tags: [scaling, distributed-training, systems, communication, pytorch]
updated: 2024-11-27
has_mvb: true
---

# Data Parallelism

Imagine 100 speed-readers scattered across a university library, each devouring a different stack of research papers. Their plan is to reconvene and share summaries with one coordinator who types all of the notes into a single document. After an hour of reading, most of them are stuck standing idle in line to hand their pages over to the coordinator because the line moves slower than their eyes. That is what modern GPU clusters feel like when training a neural network without careful data parallelism: the accelerators complete their math quickly, then stare into the network while gradients shuffle around the ring. By the time synchronization finishes, the processors are ready for the next batch but have been babysitting the interconnect instead of doing useful work. Read this page and you will see that the meat of data parallelism is not copying models onto more devices, but racing to overlap communication with computation so the network never stalls. Understanding that race immediately tells you where to measure latency, why gradients need bucketing, and how to write your own ring all-reduce loop. The build at the end proves you can do it on a free Colab GPU and exposes every synchronization primitive you need.

## The territory

Data parallelism lives inside the broader question of how to scale a fixed neural network across more hardware without redesigning the model itself. The knobs are deceptively simple: replicate the same parameters across \(M\) workers, feed each worker a disjoint slice of the minibatch, and reconcile the gradients so every replica takes the same optimizer step. The tension arises because the data stays split but the parameters must stay synchronized, meaning each batch of gradients must circulate through the interconnect before the next forward pass. Hillis and Steele (1986) established that mapping the same operation across different data points—rather than slicing the instruction stream—is the only way to scale arithmetic-intensive kernels efficiently, and the same principle applies to today’s convolutional layers running on many GPUs [Hillis et al. 1986](http://cva.stanford.edu/classes/cs99s/papers/hillis-steele-data-parallel-algorithms.pdf). The problem’s concrete implementation changes, but the goal remains: every worker should spend as much wall-clock time as possible doing computation rather than waiting for a collective.

The systems community broke this coordination problem into two pieces decades ago. DeWitt and Gray (1992) argue in “1. Introduction” that a distributed database increases throughput not by speeding up CPUs but by careful partitioning of data and minimizing synchronization across nodes [DeWitt & Gray 1992](https://pages.cs.wisc.edu/~dewitt/includes/paralleldb/cacm.pdf). In distributed training the database is the gradient tensor, and the synchronization is a reduction operation across the ring fabric. The new twist in AI is the interconnect: PCIe switches, NVLink bridges, and InfiniBand fabrics now determine whether the reduced gradient arrives before the worker’s forward pass has already begun. That is why the mechanism for data parallelism should be read as a communication-bound optimization problem. Faster networks or more aggressive overlapping reduce the "line at the coordinator" so that the cluster’s throughput grows with the number of readers instead of plateauing.

How does it actually work? The next section traces the mathematics of gradient aggregation, the ring-allreduce algorithm that implements it, the bucketization that lets us overlap with computation, and the tooling that glues all of this together inside frameworks like PyTorch.

## How it works

The objective of data parallelism is still the same as single-device training: minimize the loss \(\mathcal{L}(\theta)\) with respect to parameters \(\theta\). What changes is the way we compute gradients. If \(\mathcal{B}_i\) is the minibatch processed by worker \(i \in \{1,\dots,M\}\), and \(\ell\) is the per-sample loss, then each worker computes its local gradient
\[
g_i = \frac{1}{|\mathcal{B}_i|} \sum_{x \in \mathcal{B}_i} \nabla_\theta \ell(\theta; x),
\]
where \(\mathcal{B}_i\) contains the examples currently assigned to worker \(i\) and \(g_i\) is a vector the size of \(\theta\). The parameter update used by synchronous optimizers is
\[
\theta \leftarrow \theta - \eta \cdot \frac{1}{M} \sum_{i=1}^{M} g_i,
\]
where \(\eta\) is the learning rate. The core of data parallelism is therefore the reduction \(\frac{1}{M} \sum_{i=1}^M g_i\) across \(M\) workers. Without an efficient collective, the network fabric becomes the “coordinator desk” that slows down every processor.

### Ring all-reduce as the basic collective

Ring all-reduce breaks down the gradient vector into \(B\) buckets and circulates them through the \(M\) workers in a ring topology, thereby limiting each worker’s bandwidth requirement to a fraction of the full vector while keeping the messages contiguous and cache-friendly. The algorithm alternates between two phases. In the reduction phase, worker \(i\) sends bucket \(b\) to worker \((i+1)\bmod M\) while receiving bucket \(b-1\) from worker \((i-1)\bmod M\) and accumulating it with its local copy. After \(M-1\) rounds every bucket contains the sum \(\sum_{i=1}^M g_i^{(b)}\), where \(g_i^{(b)}\) is the bucket \(b\) of worker \(i\). The second phase is the broadcast: the accumulated buckets are propagated around the ring again so each worker ends up with the full summed gradient. In code, this is implemented with `torch.distributed.send` and `torch.distributed.recv` or the higher-level `torch.distributed.all_reduce` once the buckets have been arranged.

Communication in a naive implementation is blocking—`all_reduce` waits until the network has finished transferring the bucket. The innovation that keeps the GPUs occupied is to interleave these transfers with computation. Instead of waiting for a bucket to finish, the worker launches a non-blocking broadcast for bucket \(b\) while immediately starting the backward pass computations for bucket \(b+1\). The key to overlapping is to ensure that the backward pass processes buckets in ascending order so that the communication for bucket \(b\) is issued as soon as the corresponding gradients become available. This is naturally aligned with the autograd graph, which computes gradients for layers in reverse order. The ring all-reduce’s per-bucket window becomes the "line at the coordinator": as soon as the bucket is filled, the worker pushes it into the ring and continues computing the next bucket’s gradients.

### Gradient bucketing and its arithmetic

Workloads like convolutional neural networks produce gradients that differ in size by orders of magnitude across layers. Krizhevsky (2014) recognized that convolutional layers are best handled with data parallelism while fully connected layers benefit from model parallelism, and he introduced automatic layer partitioning so that large convolutional kernels send fewer bytes across the network [Krizhevsky 2014](https://arxiv.org/pdf/1404.5997). The same insight applies to gradient bucketing: small tensors cannot saturate the link, so we accumulate multiple gradient tensors into a single contiguous bucket before issuing the collective. A bucket can be thought of as a contiguous view into the flattened parameter gradient vector, which is arranged as
\[
G = [g^{(1)}, g^{(2)}, \dots, g^{(B)}]
\]
where each \(g^{(b)}\) is the concatenated gradient of one or more parameters. The bucketization procedure ensures that each message is large enough to amortize the startup cost of the collective while still being small enough that multiple buckets can be processed concurrently. In practice, PyTorch allows us to specify `bucket_cap_mb` or manually orchestrate bucket boundaries to match the size of the underlying NIC.

The result is latency hiding: while bucket \(b\)’s collective is traversing the ring, the GPU computes the backward pass for bucket \(b+1\). Because the backward pass follows the dependences in the computation graph, the GPU never idles unless the network takes longer to complete a collective than the time to compute the next bucket, which is the scenario we want to avoid. Algorithms such as NCCL’s asynchronous progress threads or PyTorch’s C++ `ProcessGroupNCCL` provide background progress for these operations, but the fundamental tension remains the same.

### Manual control with torch.distributed

Understanding how to schedule communication manually gives the intuition to trust higher-level wrappers. In PyTorch the building blocks are `torch.distributed.launch`, `torch.distributed.init_process_group`, and the per-process autograd hooks that trigger gradient reductions. As soon as the backward pass produces the gradient tensor \(\nabla_\theta \ell\) for layer \(L\), a hook can bucket it and call `dist.all_reduce_(bucket, async_op=True)` to start the collective in a non-blocking way. Worker \(i\) posts its asynchronous operation and returns to compute the next layer’s gradients without waiting. The check for completion happens later with `op.wait()`, either in a custom synchronization point or when the bucket is reused.

This manual control is why the build uses `torch.multiprocessing.spawn` and explicit bucket management: it forces the learner to see the bucket filling, the collective starting, and the gradient returning. Once that low-level view is understood, frameworks such as PyTorch’s Fully Sharded Data Parallel (FSDP) or DeepSpeed’s ZeRO can be seen simply as automation that does the same bucketing and overlapping for you. "Untitled" (2001) from the University of Illinois systems group showed that exposing those controls to the runtime lets scheduler innovations—like early reservation of NIC bandwidth—prevent stragglers from stalling the entire system [Untitled 2001](http://rsim.cs.illinois.edu/arch/qual_papers/systems/6.pdf). The runtime’s job is to keep the ring full without letting any worker run out of work.

### Compilers and automatic overlap

The lean from manual to automatic happens in PyTorch 2.2’s SimpleFSDP, which combines bucket fusion with `torch.compile` so that the compiler can see the entire backward pass and schedule communications without manual hooks [PyTorch 2.2 blog](https://pytorch.org/blog/torch-2-2/). Because `torch.compile` produces an intermediate representation of the graph, SimpleFSDP can insert asynchronous collectives at compile-time, making the bucket size decisions and launch order part of the optimization pass. The compiler also reasons about memory, which prevents the buckets from growing so large that they cause out-of-memory errors, a critical safety net when training models with billions of parameters.

Walking through these layers—mathematical reduction, ring all-reduce, manual bucketization, and compiler automation—gives a full view of the mechanism. The build section now codifies those ideas: you will implement a ring all-reduce on Colab, intentionally bucket gradients, and measure how much compute you save by overlapping communication.

## Where the field is now

The research frontier in data parallelism today is embodied by PyTorch’s SimpleFSDP (2024). Rather than relying on hooks, SimpleFSDP uses `torch.compile` to capture the backward pass’s IR, locate the gradient tensors as they become available, and automatically issue non-blocking collectives with bucket fusion. Because the compiler sees the entire graph, it can reorder kernels so that the communication latency is hidden behind independent compute, effectively turning the scheduler into a just-in-time orchestrator. SimpleFSDP demonstrates that the problem of overlapping communication with computation can be solved at the IR level, leaving researchers free to focus on mixed precision, activation checkpointing, and numerical stability.

On the engineering front, OpenAI’s GPT-4 training infrastructure shows what happens when the interconnect dominates everything else. Their blog describing the GPT-4 training stack (2024) emphasizes custom multi-ring all-reduce topologies running on NVIDIA A100 clusters, a scheduler that launches collectives immediately after the bucket is computed, and a diagnostics stack that monitors gradient staleness at 1 ms granularity [OpenAI GPT-4 research](https://openai.com/research/gpt-4). The system also mixes data and tensor parallelism across layers to match each layer’s communication-to-compute ratio, echoing the hardware-aware split that Krizhevsky introduced for convolutional networks. These practical deployments show that data parallelism is now an engineering question of fabric-tuning, not just algorithm design.

## What's still open

Can we design a runtime that adaptively shards highly variable, mixed-length sequences—ranging from 256K to 2M tokens—so that data parallel collectives are scheduled on-the-fly to eliminate compute bubbles without triggering out-of-memory spikes?  

Is there a provably optimal trade-off between bucket size and model size that accounts for both NIC contention and per-layer compute cost, or are current heuristics the best we can do given hardware non-determinism?  

What pieces of the gradient reduction can be executed speculatively on partial tensors, similar to speculative execution in CPUs, so that collectives have warm links even when one worker finishes its backward pass early?  

## Where to read next

If you want the low-level primitives behind these collectives, → [[collective-communication]] dives into ring all-reduce, tree reductions, and the cost models that pick one over the other; the engineering counterpart is → [[nccl-overview]] showing how hardware vendors expose bandwidth counters and asynchronous progress for real deployments; if you want more on how compiler-level scheduling can automate this, → [[simple-fsdp]] explains how PyTorch’s new compiler rewrites backward passes for simultaneous bucket fusion and communication.

## Build it

This build tests whether you can keep the ring full yourself: by writing a bare-metal PyTorch Distributed Data Parallel loop that buckets gradients, launches asynchronous ring all-reduce, and overlaps it with backward computation on a free Colab GPU, you will directly observe how communication latency affects throughput.

**What you're building:** a PyTorch DDP trainer that runs a CNN on MNIST while manually implementing gradient bucketing and ring all-reduce so the backward pass never stalls the accelerator.

**Why this is valuable:** it exposes the precise places where gradients are reduced, how bucket size influences the trade-off between NIC latency and GPU compute, and why frameworks like SimpleFSDP automate this for scale.

**Stack:**
- **Model:** `pytorch/tutorials:mnist_main` (PyTorch official example) — download count: ~1.2M
- **Dataset:** `huggingface/mnist` — well-known, small enough for Colab while still training a CNN
- **Framework:** `torch==2.2` with `torchvision`, `torch.distributed`, and `torch.multiprocessing`
- **Compute:** Free Colab with T4 (16 GB VRAM); expect ~30 minutes for 10 epochs with manual barrier logging

**The recipe:**
1. Install and initialize: `pip install torch==2.2+cu118 torchvision torchmetrics`, launch `mp.spawn` with 2 replicas, and call `init_process_group("nccl")` inside each worker.
2. Data: use the HuggingFace MNIST loader, shard the dataset manually by slicing the dataset index with `torch.utils.data.distributed.DistributedSampler`, and normalize images to \([0,1]\); ensure `drop_last=True` so buckets stay constant size.
3. Train/fine-tune: after the forward pass, collect gradients per layer using `register_full_backward_hook`, flatten them into a contiguous tensor, split into 4 buckets with `torch.chunk`, launch `dist.all_reduce(bucket, async_op=True)` for each bucket, and immediately proceed to the next backward layer before calling `op.wait()` in the optimizer step; log `bucket_size` and sync time each iteration to observe overlap.
4. Evaluate: compute accuracy on the validation split after every epoch; expect ≥97% accuracy and watch the logged "communication delay" drop below 20 ms once bucket overlap stabilizes.
5. What you now have: a checkpointed CNN, a dashboard of communication vs compute timing, and a Colab notebook where you can vary bucket size and see the wall-clock effect.

**Expected outcome:** a runnable Colab notebook that reports the per-epoch accuracy and a plot showing how communication delay is hidden as buckets overlap with backward computation.

- **CS student:** Run the same notebook on a local RTX 4070 with 2 GPUs, reducing the number of buckets to 2 so memory fits, then compare the logged communication time against the Colab run.
- **Applied engineer:** Extend the notebook to quantize gradients to FP16 before the ring all-reduce, launch the `torch.compile`-optimized SimpleFSDP wrapper on top of it, and measure that the p95 sync latency stays under 25 ms while still hitting 97% accuracy.
- **Applied researcher:** Use the notebook to test the hypothesis “bucket size proportional to layer FLOPs gives lower communication stalls than a fixed bucket size”; run 3 bucket-size policies, chart the overlap ratio, and confirm whether the policy with FLOP proportionality wins.
- **Frontier researcher:** Probe the open problem by varying input lengths from 256K to 2M tokens (simulate variable batch sizes) and implement a scheduler that resizes buckets on the fly; falsify the scheduler if any worker still stalls for more than 2× the average compute time.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*