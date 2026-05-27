---
title: Communication Collectives
slug: communication-collectives
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [hindman, fikes, tarasov, simplefsdp, commbench]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [collective-communication, distributed-training, memory-systems]
tags: [collective-communication, allreduce, hardware-co-design, memory-bound, distributed-training, simulation]
updated: 2025-04-01
has_mvb: true
---

# Communication Collectives

Imagine walking into a hollow data center where every NVIDIA GH200 or H100 is ticking-but-idle, its lights glowing but its FLOPS unused because the gradient Allreduce at the heart of every training step is waiting for a slow, monolithic data transfer. Hundreds of GPUs are paying a silent tax of latency while an unseen scheduler blocks them until `torch.distributed.all_reduce` finally finishes. The question this page answers is: how do you stop the tax by designing collectives that respect memory bandwidth, topology, and the compiler’s view of work so compute and communication truly overlap? By the end, you will see how collectives are memory-bound algorithms shaped by the hardware graph, how pipelined chunking cuts latency on real clusters, and how a PyTorch simulation can let you experiment with those knobs without needing a full supercomputer.

## The territory

Collectives sit at the intersection of distributed algorithms, system architecture, and compiler planning. The original view treated them as simple network transfers—broadcasts, reduces, allreduces, and scatters that the MPI standard shipped with static schedules. As soon as operators tried to run those algorithms across thousands of GPUs, the missing piece became obvious: the communication substrate has a richer structure than the flat network model assumes. Hardware schedulers want to treat GPUs, NICs, and RDMA queues as shareable resources, which is exactly the mindset of Mesos (Hindman et al. 2011) [https://people.csail.mit.edu/matei/papers/2011/nsdi_mesos.pdf], where fine-grained resource sharing became a planning problem. Collectives inherit that planning complexity when building a schedule that touches every NIC and every shared PCIe bridge.

At the same time, scheduling in this space is not a loose heuristic—it is as systematic as STRIPS planning. Fikes and Nilsson (1971, repr. 1972) [https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/strips.pdf] marry the idea of precondition/effect search with resource constraints; the same idea applies to picking which GPUs feed which NICs at each step of a collective, where preconditions include “NIC queue ready” and effects include “partial sum available.” Those early planning algorithms still inspire today’s runtime schedulers that adapt to contention and topology. Before MPI had smart collective algorithms, the MPI-1 collective operations relied on the static tree structures defined in Thakur et al. (1999) [https://arxiv.org/pdf/cs.LG/9908014], and that work is the bridge between naive network thinking and the memory-copy-aware algorithms we need: it mapped message size, number of ranks, and tree shape to latency, letting MPI implementations choose the fastest schedule.

Collective communication no longer just copies buffers. For medium-to-large messages, memory-copy and host-device transfer dominate the clock ticks, not raw network bandwidth, as PICO (Tarasov et al. 2024) [https://arxiv.org/abs/2508.16809] quantifies. The field is moving toward compiler-driven, topology-aware execution where one planner reasons about the communication graph, a dataflow graph, and memory overheads simultaneously. How does it actually work in detail?

## How it works

Collectives operate on \(p\) ranks that each own a buffer of \(n\) bytes. The canonical latency model is
\[
T(n, p) = \alpha \log p + \beta n,
\]
where \(\alpha\) is the per-step, per-rank startup cost (handshake latency), \(\beta\) is the per-byte transfer cost, and the logarithmic factor reflects how tree-based algorithms reduce depth. The simplicity of this formula hides two crucial facts: first, \(\beta\) is itself dominated by memory copies—staging data through host memory, copying into GPU memory, and queuing DMA transfers—and second, the effective \(\alpha\) depends on how many NICs, PCIe links, and CUDA streams each rank must coordinate. That is why Tarasov et al. (2024) in PICO can say “library heuristics lose 30–40% of peak” not because networks are slow but because the scheduler fails to optimize those hidden copies.

### Hardware topology and memory-copy inflation

A collective does not traverse a flat network but a hierarchical graph: each GPU sits on a PCIe/Swiftlink or NVLink bridge, which connects to a PCIe root complex, which connects to a host CPU and NIC, which finally connects to other nodes via Infiniband or Ethernet. Each edge has its own bandwidth \(b_e\) and latency \(\ell_e\). When a rank performs Allreduce, the cost of stirring \(n\) bytes includes copying from device memory to host and back, then injecting into the NIC. If \(m\) is the number of memory copies needed for the schedule, then the effective per-byte cost becomes
\[
\beta = \sum_{i=1}^{m} \frac{1}{b_{copy,i}} + \frac{1}{\min_j b_{link,j}},
\]
where \(b_{copy,i}\) are the bandwidths of each copy \(i\) (GPU DMA, host memcpy, etc.) and \(b_{link,j}\) are the link bandwidths the chunk traverses. PICO shows that for the medium-to-large messages common in large-batch training, the sum of the copy terms dominates; GPUs are waiting on host-to-device copies while the NIC sits idle. Without accounting for this multiplicative copy factor, the library’s “all-to-all” heuristics pick tree shapes that over-subscribe the NIC while leaving the PCIe copy as the bottleneck.

In hierarchical topologies, the solution is to decompose the collective into layers: intra-node, intra-rack, inter-rack. Tarasov et al. profile these layers and show that optimizing within a layer (e.g., using NVLink bandwidth for intra-node and only using the NIC for aggregated chunks) can reclaim 40% of compute. CommBench’s ICS 2024 study (CommBench Authors et al. 2024) emphasizes that each bladder of NICs and CPUs needs its own tuned scheduler. For example, if a rack has 4 nodes with dual NICs and each node has two GPU cliques, an efficient Allreduce first sums within each clique, then across NICs, and finally merges across racks using an algorithm like NCCL’s hierarchical tree. When this layering is made explicit, the critical path is the slowest layer’s \(\alpha\), and the rest can be overlapped.

### Compiler-guided overlapping and chunking

Default collective operators expose only blocking calls. SimpleFSDP (SimpleFSDP Authors et al. 2024) flips this by compiling the computation graph (via `torch.compile` and abstractions like `torch.distributed._dtensor`) so that the graph planner inserts explicit allreduce IR nodes annotated with chunk size, target streams, and expected completion times. The planner rewrites a model’s backward pass using cost models similar to:
\[
T_{\text{chunk}} = \alpha + \beta \cdot \frac{n}{k},
\]
where \(k\) is the number of chunks a rank splits its gradient into. By pipelining these chunks, every rank can start sending the first chunk while still computing gradients for the last layers. The compiler leverages the DAG of dependencies: it only launches communication kernels when the chunk’s gradients are ready, and it reuses idle CUDA streams to simultaneously copy chunk \(i\) to the NIC while chunk \(i-1\) is in flight on the link.

To illustrate, imagine each GPU computes gradients for layers \(1\) through \(L\). The compiler adds metadata that chunk \(i\) corresponds to gradients for layer \(i\) and schedules:
1. Launch kernel that computes gradients up to layer \(i\).
2. As soon as chunk \(i-1\) is ready, start an asynchronous copy to the NIC on CUDA stream \(s_2\).
3. While chunk \(i-2\) is traversing the NIC, begin kernel for chunk \(i\).

This overlap reduces the effective latency from \(\alpha\) (per chunk) to \(\max(\alpha, \text{computation time})\), making the Allreduce less of a barrier. SimpleFSDP’s compiler also reorders collective nodes so that small chunks that complete faster are sent before large ones, which mitigates stragglers on heterogeneous links.

### Simulation and mock CUDA streams

To reason about these schedules without requiring a full cluster, you can simulate an Allreduce using PyTorch tensors and explicit CUDA streams inside Colab. The ring Allreduce you will build later divides the buffer of size \(n\) into \(k\) chunks and performs \(2(p-1)\) steps where each chunk is both sent and received once per rank. The total amount of data traversed on each link is \(2n(p-1)/p\). By dividing into chunk size \(n/k\), the time per chunk becomes \(\beta \cdot (n/k)\) plus the per-step \(\alpha\), and pipelining means the wall-clock finish time is approximately
\[
T_{\text{pipelined}} \approx \alpha k + \beta n + \text{overlap},
\]
where overlap accounts for the fact that while chunk \(i\) is on the link, the GPU can be computing chunk \(i+1\). Without chunking, \(\alpha\) applies once per whole buffer, and the NIC sits idle while the GPU copies the next buffer. The simulation uses mock CUDA streams to interleave asynchronous host/device copies with CPU timers, giving you a controllable environment to compare naive vs. pipelined schedules.

### When the plan meets the network

CommBench confirms that off-the-shelf libraries like NCCL, RCCL, or OneCCL still need topology-aware tuning to achieve these overlaps. They benchmarked training jobs on multi-GPU, multi-NIC hierarchical networks and showed that libraries which pick a single algorithm for all participants fail to saturate the PCIe link even when the NIC is busy. Their finding is that the scheduler must understand node locality, NIC bandwidth, and host memory speed simultaneously—only then can you choose between ring, hierarchical tree, or tree-based Allreduce for the given message size and hardware. When a compiler like SimpleFSDP exposes these choices, it becomes possible to generate runtime schedules that adapt as the job runs, responding to congestion or hardware degradation.

## Where the field is now

The research frontier is currently occupied by profiling and synthesis work that ties collective performance to memory-bound behavior. PICO (Tarasov et al. 2024) provides the latest instrumentation, showing that modern libraries lose 30–40% of peak allreduce throughput due to cache line thrashing and redundant host-device copies, and it releases a benchmark suite to stress-test custom schedulers. That suite feeds directly into SimpleFSDP’s compilation pipeline, which automatically buckets gradient tensors, schedules them on CUDA graphs, and inserts `torch.distributed` calls that respect both loop-carried dependencies and hardware queues. SimpleFSDP reports that on 64 H100 GPUs, the compiler-managed collectives hit within 2% of a custom NCCL kernel, proving that a compiler can near-match handcrafted performance while also providing controls for chunk size and stream mapping.

On the engineering frontier, NVIDIA’s NCCL blog posts (developer.nvidia.com/blog/nccl-2-8-release/) still describe how the library picks between ring, tree, and tree-ring algorithms based on message size and topology, but the newer releases also expose hooks for topology-aware tuning libraries to inject custom schedules. AWS’s distributed training blog (aws.amazon.com/blogs/machine-learning/accelerating-large-scale-training-with-nccl/) reports that their SageMaker clusters pair NCCL with Elastic Fabric Adapter (EFA) NICs and monitor both PCIe utilization and RDMA queue depth in real time, allowing them to adapt chunk sizes during training. These engineering deployments demonstrate that the future of collectives is not just faster networks but smarter schedulers that see streams, NICs, and memory copies as a single optimization problem.

## What's still open

Can a runtime compiler synthesize mathematically optimal, topology-aware collective algorithms on the fly, given noisy inputs about congestion and hardware degradation, without resorting to brute-force search? In particular, what cost model lets a scheduler trade off ring vs. hierarchical trees when link bandwidths are time-varying, and how quickly can it recompile the schedule before the next Allreduce starts?

How do we extend compiler-managed collectives like SimpleFSDP to heterogeneous racks where some GPUs are fully-connected NVLink cliques and others are PCIe-only? The current chunk sizes work when gradients are uniform, but quantization, sparsity, and adaptive optimization introduce non-uniform gradient shapes. Can the compiler balance the chunking so that each link carries proportional work and no rank becomes a straggler?

What is the right abstraction for memory-copy awareness in the collective graph? Current schedulers treat copies as costs, but a principled extension would incorporate the state of host caches, DMA engine occupancy, and asynchronous stream contention. What schedule semantics would let a planner reason about cache-miss penalties and overlap them with NIC transmission?

## Where to read next

If you care about the probability distributions that underlie why chunked allreduces still converge despite delayed gradients, → *collective gradient consistency* <!-- [[collective-gradient-consistency]] --> explains how the mathematics of delayed aggregation preserves correctness. The engineering counterpart is → *nvlink topologies* <!-- [[nvlink-topologies]] -->, which maps real GPU topologies to the PCIe/NVLink graphs that the schedulers see. For a broader arc that turns this understanding into a full training stack, → *hierarchical distributed training arc* <!-- [[hierarchical-distributed-training-arc]] --> walks the concepts that plug into the MVB you just read.

## Build it

This build proves that a properly chunked Ring-Allreduce mimicking CUDA stream overlaps can be implemented on a single Colab GPU and that the latency gap between naive and pipelined schedules is measurable even without a multinode cluster.

**What you're building:** a PyTorch simulation of chunked Ring-Allreduce with mock CUDA streams comparing naive, monolithic transfers to pipeline-friendly chunking.

**Why this is valuable:** the simulation exposes the memory-copy dominated cost terms highlighted by PICO and lets you observe how chunk size, stream order, and overlap affect latency before touching a real cluster.

**Stack:**
- **Model:** `facebook/convnext-tiny` — ~5.4M parameters, used to size gradient buffers referenced by the allreduce pipeline.
- **Dataset:** `huggingface/datasets::wikitext-2-v1` — short sequences supply token embeddings whose gradients drive the simulated collective.
- **Framework:** PyTorch 2.1 with `torch.compile` and CUDA stream APIs.
- **Compute:** Single Colab T4 (16 GB VRAM) or any RTX 4070 laptop — data download + simulation runs in ~45 minutes.

**The recipe:**
1. `pip install torch torchvision datasets git+https://github.com/pytorch/torchdynamo` to ensure `torch.compile` and asynchronous streams are available, then download the `wikitext-2-v1` split and preprocess it into batches of 4-token sequences.
2. Load `convnext_tiny` from Torchvision, run a forward pass to materialize gradients sized roughly like the full model (the gradient buffer serves as the collective payload), and split the gradient tensor into \(k\) contiguous chunks of \(n/k\) elements.
3. Implement two Allreduce simulations: (a) Naive: for each rank (virtualized as a Python list) copy the entire buffer over `torch.cuda.Stream()` objects sequentially, mimicking synchronous host-device transfer. (b) Pipelined: schedule each chunk’s copy onto its own stream, issue asynchronous `torch.cuda.current_stream().record_event()` calls, and simulate NIC latency by awaiting `torch.cuda.Event().query()` with synthetic wait times. Use `torch.cuda.Stream()` to overlap chunk \(i+1\)’s copy with chunk \(i\)’s simulated transfer.
4. Measure wall-clock time for each algorithm across \(p=4\) simulated ranks and vary chunk counts \(k \in \{1,4,8,16\}\). Log throughput (bytes/sec) and latency (seconds per Allreduce) to a CSV; expect pipelining to outpace naive once \(k>1\) because the NIC and GPU copies overlap.
5. Plot the latency curves showing where pipelining overtakes naive at scale and annotate the “memory-copy dominated” regime where Tarasov et al.’s cost model (aggregate copy bandwidth \(b_{copy}\)) predicts the bottleneck.

**Expected outcome:** a Colab notebook with latency-throughput curves comparing naive vs. chunked Ring-Allreduce, plus a saved CSV of per-chunk timing that demonstrates pipelining’s benefit.

- **CS student:** On free Colab (T4/RTX 4070), reduce \(k\) to \(2\) and disable `torch.compile` so the focus is on the latency measurement without compiler scheduling; report the timing difference between direct `torch.cuda.synchronize()` calls.
- **Applied engineer:** Wrap the pipelined simulation in a FastAPI endpoint and quantize chunk buffers to `float16`, then serve it via vLLM/TGI with a latency target of ≤90 ms p95 for a synthetic gradient request; this shows how real collectives shape inference pipelines.
- **Applied researcher:** Formulate the hypothesis “Increasing chunk count reduces latency until CUDA stream contention reverses the gain.” Run a sweep \(k \in \{1,2,4,8,16\}\), plot latency vs. chunk count, and falsify if latency stops decreasing before \(k=8\).
- **Frontier researcher:** Extend the simulation with a topology-aware scheduler that chooses between ring and hierarchical tree schedules based on synthetic NIC congestion signals; treat the open question “Can runtime synthesis recompile collectives fast enough to react to congestion?” and measure the recompilation cost vs. the latency saved.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*