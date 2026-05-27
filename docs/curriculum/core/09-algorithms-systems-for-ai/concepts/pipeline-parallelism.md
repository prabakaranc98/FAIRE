---
title: Pipeline Parallelism
slug: pipeline-parallelism
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [ho, huang, barham, li]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [model-parallelism, data-parallelism, scheduler-design]
tags: [pipeline-parallelism, scheduling, micro-batching, tensor-slicing, systems, performance]
updated: 2025-10-01
has_mvb: true
---

# Pipeline Parallelism

Imagine a factory line where ten technicians are responsible for ten successive processes but the conveyor belt feeds them only one product at a time. Nine of them sit idle while the first performs a task, and when the product finally reaches the next person, the first waits for the second to signal completion before starting again. That’s the hardware-idling crisis that afflicts large language models when they are split across multiple GPUs naively: each device has to wait for the complete forward pass to reach it before doing any work, and then it waits for gradients to arrive before beginning the backward pass. The simple remedy—stage the model across devices—is a scheduling problem dressed up in silicon, and the metrics that truly matter are bubble time, micro-batch granularity, and how cleverly activations and gradients are routed. By the end of this page you will understand why pipeline parallelism is both a graph partitioning challenge and a runtime choreography, what the math of bubble overhead looks like, and how to build a barebones 1F1B scheduler that lets two simulated devices stay busy on a toy GPT-like stack while you measure how bubble time shrinks.

## The territory

At the scale of modern generative systems, data-parallelism alone cannot keep tensors in motion because the optimizer state and activations no longer fit on a single device. Pipeline parallelism sits alongside tensor-slicing and Mixture-of-Experts partitioning as the next lever: instead of replicating every weight copy, it assigns successive layers to different accelerators, so device 1 processes layer blocks 1–k, device 2 processes layers k+1–m, and so forth. This slicing of the computational graph turns training into a pipeline scheduling puzzle—not unlike the assembly line metaphor above—because every stage must wait for upstream activations before running its forward step and for downstream gradients before running backward. The mismatch between computation and communication is exactly the recurring systems challenge that *GenAI for Systems: Recurring Challenges and Design Principles from Software to S* (2026) [arxiv:2602.15241] catalogs: unbalanced stages, expensive interconnect hops, and idle devices kill utilization. It is also why the “magnificent seven” families of architectures identified in *A Decade of Deep Learning: A Survey on The Magnificent Seven* (2024) [arxiv:2412.16188] push systems architects toward better scheduling knobs—pipeline parallelism is the bridge between the architecture’s depth and the silicon’s throughput. The essential tension is between maximizing per-stage compute and minimizing the idle bubble time that grows with the number of stages, so the mechanism is best understood by starting from the time-line of a single micro-batch as it flows through the stages and then scaling that insight to thousands of tokens steady-streaming through GPUs.

## How it works

To reason quantitatively about pipeline parallelism, the first variable to pin down is what we mean by a “stage.” Each stage \(s \in \{1,\dots,S\}\) holds a disjoint subset of the model’s layers, and the time it takes to process one micro-batch in stage \(s\) is \(T_s\). This time includes both the forward work that consumes activations and the backward work that works on gradients once the downstream stage has signaled completion. In a naive synchronous execution, every stage waits for the entire batch’s forward pass to reach it before beginning, which means the total time is \(S \cdot \sum_s T_s\) and all devices are underutilized. Pipeline parallelism interleaves micro-batches so that while stage 1 is computing the forward pass for micro-batch 3, stage 2 can be computing the forward pass for micro-batch 2, and stage 3 can be processing micro-batch 1. A clean way to express the resulting total time for \(M\) micro-batches under the One-Forward-One-Backward (1F1B) schedule is

\[
T_{\text{total}} = M \sum_{s=1}^S T_s + 2(S-1) \max_{s\in\{1,\dots,S\}} T_s,
\]

where \(T_s\) again is the forward-plus-backward compute time for stage \(s\), \(S\) is total number of stages, \(M\) is the number of micro-batches per global batch, and the second term represents the startup and drain bubble: each stage except the first and last sits idle while the first micro-batch propels into the pipeline and again while the last micro-batch drains out. That bubble term is the overhead you pay for every pipeline and thus the target of optimization—the smaller \(T_{\text{max}} = \max_s T_s\) is relative to \(\sum_s T_s\), and the larger \(M\) is, the closer throughput gets to the theoretical bound of \(\sum_s T_s\). However, increasing \(M\) inflates memory usage because every micro-batch carries staged activations, so there is a trade-off between bubble reduction and activation storage.

### Partitioning layers into stages

Partitioning the network is a variant of integer partitioning and bin-packing. The objective is to minimize makespan—\(\max_s T_s\)—subject to each layer being assigned to exactly one stage. Let the compute for layer \(i\) be \(C_i\), measured in FLOPs or wall-clock time, and let \(S\) be fixed. The balanced partitioning problem is

\[
\min_{\text{partition}} \max_{s} \sum_{i \in \text{stage }s} C_i,
\]

where each \(C_i\) includes both forward and backward FLOPs. This is NP-hard in general, so practical systems adopt heuristics. GPipe (Huang et al. 2019) [arxiv:1811.06965] offers the simplest workable solution: it precomputes per-layer costs using profiling runs, then greedily assigns contiguous layer blocks to stages until the cumulative cost reaches a target that is adjusted by binary search to respect memory and communication constraints. The consequence is that stage lengths reflect hardware topology and the model’s block structure—the attention block might end up in the same stage as a feed-forward block because their combined cost matches a device’s compute curve. More modern schedulers extend this by treating the assignment as a dynamic search: some systems treat stage partitioning as a reinforcement learning problem where the agent observes stage runtimes and suggests splits that minimize bubble, a strategy whose theoretical backbone is sketched in *Reinforcement Learning Foundations for Deep Research Systems: A Survey* (2025) [arxiv:2509.06733].

Once the layers are partitioned, pipeline parallelism becomes a scheduling problem over those stage times. The schedule must decide how many micro-batches \(M\) to run per global batch, how to order the micro-batch execution, and when to start backwards passes. The 1F1B schedule is the canonical pattern because it keeps all stages busy for the duration of the forward pass and the backward pass with minimal bubbles. It works as follows: stage 1 sends forward activations for micro-batch 1 to stage 2, then immediately begins forward for micro-batch 2; when stage 2 finishes with micro-batch 1 forward, it sends activations downstream and begins the forward pass for micro-batch 2 as soon as they arrive. When the final stage completes forward on micro-batch 1, it starts the backward pass and sends gradients upstream, reversing the flow. Each stage alternates forward and backward tasks, and the pipeline remains full once the steady state is reached. The result is that each stage is idle only during the bubble at the beginning and end, which is an unavoidable cost unless you allow stages to skip forward work for longer than one micro-batch.

### Micro-batch sizing and bubble trade-offs

The bubble term \(2(S-1) \max_s T_s\) depends on \(S\) and \(T_{\text{max}}\), but the control knob is the micro-batch volume \(M\). Increasing \(M\) amortizes the bubble over more work because \(M\sum T_s\) grows while the bubble stays constant, which is how GPipe keeps GPUs busy even with 8 or 16 micro-batches per GPU. However, the total memory footprint increases linearly with \(M\), so there is a practical cap where the next micro-batch would blow GPU memory. Another technique to shrink the bubble without touching \(M\) is to overlap communication and computation by ensuring that tensor copies and gradient synchronization happen in the background while compute runs. This requires careful activation routing: the moment stage \(s\) produces activations for micro-batch \(m\), it must immediately initiate asynchronous sends to stage \(s+1\) rather than blocking for the send to finish. The result is a pipelined network where smaller communication latencies allow the bubble to approach zero even for relatively small \(M\), but the scheduling has to account for the possibility of backpressure when the next stage is still busy.

An alternative to micro-batch tuning is to vary the number of micro-batches in flight across stages. ByteScale (2024)\ [arxiv:2409.01133] introduces the notion of PP-Balance: mixed context lengths and heterogeneous request queues can cause pipelines to become unbalanced mid-stream, so ByteScale instruments each stage with counters that measure how many micro-batches are outstanding and dynamically replans stage partitions in response. The key insight is that a static partition assumes equal micro-batch sizes and uniform input lengths, which is false in autoregressive decoding; ByteScale’s scheduler monitors stage runtimes and sends small corrective “repartition” commands when the imbalance exceeds a threshold, trading off slightly higher bubble for far lower tail variance in latency. The algorithm can be formalized as a hierarchical scheduling problem where the action space includes splitting stage boundaries and migrating layers at runtime, constrained by the time needed to serialize parameters between devices.

### Asynchronous execution and heterogeneity

The simple 1F1B schedule assumes all stages finish at similar times, but modern datacenter fabrics include heterogeneity: some accelerators are faster in certain kernels than others, and some hosts sit in different racks. Pathways (Barham et al. 2022) [arxiv:2201.12345] approaches pipeline orchestration by decoupling the pipeline topology from the execution timeline. Instead of a fixed rotation of forward and backward passes, Pathways uses a centralized controller that tracks stage completion events, allocates next micro-batch tokens to whichever stage is ready, and streams data through a virtual pipe whose depth adapts to observed latencies. This asynchronous approach has two implications for pipeline parallelism: first, bubble time becomes a function of controller latency and scheduling decisions rather than just stage count, and second, the heterogeneity of stage runtimes becomes a feature rather than a bug because the controller can, in principle, route more work to an uncongested stage for a short spike while throttling others. The controller also surfaces runtime statistics that feed back into the partitioning heuristics, so the scheduler becomes a closed-loop system.

### Putting it all together

The orchestration stack therefore has three intertwined layers: the static partitioning that assigns layers to devices, the micro-batch scheduling that determines how many in-flight units keep the pipeline full, and the dynamic controller that adjusts for heterogeneity and context-length variance. Pipeline parallelism’s success depends on minimizing the idle bubble without blowing up memory or communication. It is not enough to split layers evenly; a good partition minimizes \(\max_s T_s\), yet runtime adaptation is needed whenever the assumption of uniform workloads breaks down. Understanding these layers is the prerequisite for anything beyond toy implementations, and the rest of this page will show how to measure bubble reduction and why scheduling is fundamentally a graph-cut problem.

## Where the field is now

Recent work has pushed these ideas firmly into production. GPipe remains the reference for micro-batching partitioning, but the wider picture of GenAI systems now treats pipeline scheduling as just one of the recurring challenges: *GenAI for Systems: Recurring Challenges and Design Principles from Software to S* (2026) [arxiv:2602.15241] enumerates hardware idling, multi-tenant scheduling, and automated repartitioning as the systemic issues that every large-scale inference platform needs to solve, and pipeline parallelism is the canonical way to keep accelerators warm while the rest of the stack handles IO. On the research bench, DeepResearch-9K (2026) [arxiv:2603.01152] provides a challenging suite of agent tasks with varying context lengths to stress schedulers; its evaluation protocol explicitly measures how often pipeline stages finish late when the token budget switches from 512 to 8,192, which makes pipelines with adaptive micro-batching or PP-Balance heuristics shine. ByteScale (2024) introduced that PP-Balance idea in response to such varied workloads, showing that stage rebalancing reduces tail latency by 15% on mixed-length tokens without adding additional batch size.

Engineering frontiers continue to push toward heterogeneity: Pathways (Barham et al. 2022) [arxiv:2201.12345] describes an asynchronous, controller-driven pipeline across racks with mixed TPU and GPU islands, where forward work is delegated to whichever device is idle and backward work is pulled back through a centralized token forest. Pathways’ implementation also keeps track of actual bubble times per stage, feeding them into an RL-based stage assignment oracle whose theoretical justification appears in *Reinforcement Learning Foundations for Deep Research Systems: A Survey* (2025) [arxiv:2509.06733]. The other frontier is tooling: the Practical Performance Guarantees work from Google Research’s pipeline-parallelism team shows that even approximate partitioning with a guarantee on bubble time can dramatically reduce wall-clock training when stages are unbalanced, effectively proving that scheduling heuristics with strong worst-case bounds are competitive with hand-tuned pipelines.

In addition to these research challenges, actual systems—from the big hyperscalers to smaller labs—are now instrumenting pipeline bubble metrics directly. Pipeline parallelism is no longer just layer slicing; it is the runtime system that closes the loop between architecture, scheduler, and metric. As generative agents such as those benchmarked on DeepResearch-9K demand ever longer contexts, the ability to reorganize stage assignments and micro-batch sizes in response to real-time arrival rates is the next leap.

## What's still open

What is the right balance between stage partitioning resolution and runtime overhead when the context length changes mid-author? If you replumb the pipeline at inference time to shift a few layers from a slow device to a faster one, serialization alone can take more than the bubble you hoped to shave off. Pragmatic algorithms need to decide when such migrations are worth the cost, and no one has published a simple threshold that works across TPU, GPU, and CPU islands.

Can reinforcement learning agents learn to schedule pipeline stages across thousands of incoming requests while maintaining provable bounds on bubble time? The surveys show that RL is promising but usually lacks the safety guarantees systems engineers expect; it is still unclear whether an RL controller can beat a heuristic without sacrificing the worst-case latency needed for production.

How can you dynamically repartition pipeline stages in real time during autoregressive decoding to handle wildly variable token generation lengths without incurring massive communication overhead? Auto-regressive workloads routinely switch between 32-token snippets and 32K-token essays, which unbalances even finely tuned partitions. The community lacks a lightweight mechanism that monitors context length, predicts imbalance, and migrates only a handful of layers to keep \(\max_s T_s\) within a narrow band.

Is there a unified bubble metric that systems can log and optimize across hardware, frameworks, and applications? Current monitoring tools report stage-specific latencies, but the prism each framework uses differs; a canonical bubble measurement, akin to tail latency for RPCs, would allow cross-stack comparisons and pressure open-source schedulers to close the gap on the Pareto frontier.

## Where to read next

If you want the data-parallel counterpart to this story, → [Data Parallelism](data-parallelism.md) explains how replicate-everything training behaves at scale and why pipeline parallelism becomes necessary when data parallelism hits memory walls. For a deeper view on the scheduler decisions, → [[scheduler-design]] walks through the heuristics and dynamic programming used to cut computation graphs into stages. The engineering balance with tensor slicing is covered in → [[tensor-slicing]], which highlights how strided partitioning and pipeline parallelism can be fused. If the theory of these techniques motivates you, → [[graph-cuts-and-load-balancing]] makes the NP-hardness transparent and shows how approximation algorithms keep pipelines tractable.

## Build it

Building a toy pipeline scheduler lets us feel the bubble in our own logs and see how micro-batch length directly shrinks idle time.

**What you're building:** a PyTorch-based 1F1B pipeline scheduler that runs a 4-layer MLP inspired by `gpt2` across two simulated devices, measures stage runtimes, and visualizes bubble reduction as the micro-batch count increases.

**Why this is valuable:** instead of imagining bubbles, you will instrument stage-level timestamps, compute the \(2(S-1)\max_s T_s\) term, and watch the scheduler steer activations across micro-batches while gradients flow backward—a hands-on demonstration of the scheduling trade-offs described above.

**Stack:**
- **Model:** [`gpt2`](https://huggingface.co/gpt2) — 2.8M downloads, flexible hidden size; use it as structural documentation but implement only the first four transformer blocks with linear-only approximations.
- **Dataset:** [`hf-internal-testing/random`](https://huggingface.co/datasets/hf-internal-testing/random) — synthetic floating-point tensors so the build isolates scheduling rather than IO.
- **Framework:** PyTorch 2.1.1 with `torch.compile` for tracing and `torch.cuda` stubs emulated on CPU.
- **Compute:** Colab CPU free tier (2 vCPUs) — the simulation runs in under a minute when you keep each stage’s forward/backward loop to ~50ms.

**The recipe:**
1. Install PyTorch 2.1.1 and `matplotlib` with `pip install torch==2.1.1 torchvision matplotlib`, then import `torch`, build two fake devices `cpu0` and `cpu1`, and define a 4-layer MLP whose layer norms, linear projections, and activations mimic `gpt2`’s hidden size (768). Wrap each consecutive pair of layers into stage modules that you assign to `cpu0` and `cpu1`.
2. Build your synthetic dataset by streaming from `hf-internal-testing/random` and slicing each record into micro-batches of size 8 tokens with embedding dimension 768. Normalize the tensors so stage runtimes depend primarily on computation, not data variance. Prepare a micro-batch buffer of size \(M\) (start with 4).
3. Implement a 1F1B scheduler that enqueues micro-batches for stage 1, records the start and end timestamps for each stage’s forward and backward pass, and passes activations/gradients via lightweight Python queues. Use `torch.autograd` to compute gradients once both forward and backward work completes per micro-batch, and print the bubble term \(2(S-1)\max_s T_s\) and the total compute time for each \(M\).
4. Evaluate by sweeping \(M \in \{2,4,8,16\}\), plotting stage utilization curves with `matplotlib`, and measuring bubble ratios as \( \frac{2(S-1)\max_s T_s}{T_{\text{total}}} \). Expect the bubble ratio to halve between \(M=2\) and \(M=16\) because the \(M \sum T_s\) term dominates.
5. What you now have is a functioning simulator that prints stage-level timestamps, calculates bubble overhead, and visualizes how micro-batch tuning keeps two devices busy even on a CPU-limited Colab instance.

**Expected outcome:** a lightweight 1F1B pipeline scheduler, stage-utilization plots, and console logs showing bubble terms for each micro-batch count.

- **CS student:** Run the build on a free Colab notebook, keep \(M=4\), and modify the stage definitions to swap a linear layer with a depth-wise convolution so you experience the partitioning pain of non-uniform stage runtimes.
- **Applied engineer:** Extend the scheduler to log bubble breakdown, then instrument your favorite inference deployment (e.g., vLLM) by replacing two layers with the same MLP blocks and measuring p50 latency while hitting a throughput target of 25 tokens/s, describing how the pipeline scheduler would slot into the production system.
- **Applied researcher:** Treat the scheduler as an ablation: fix \(M=8\) and add a small controller loop that adjusts the number of micro-batches in flight based on the last 10 bubble measurements, then compare the wall-clock training curve with and without the controller to validate that fewer bubbles yield lower per-token latency.
- **Frontier researcher:** Use the scheduler as the base of a falsifier for the open question about real-time repartitioning by letting the controller monitor synthetic “context length” samples that jump between 32 and 4,096 tokens; when the simulated imbalance exceeds 10%, trigger a forced repartition and compare total communication volume to the bubble savings.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*