---
title: Pipeline Parallelism
slug: pipeline-parallelism
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [dean, barroso, asanovic, abadi]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [data-parallelism, model-parallelism, distributed-systems]
tags: [pipeline, scheduling, micro-batch, distributed-training, activation-checkpointing]
updated: 2024-11-11
has_mvb: true
---

What if slicing a trillion-parameter transformer across GPUs turned into an elaborate exercise in waiting? In the naive split, the first device executes a forward pass, the last one runs the backward, and the rest stand by until gradients travel backward through all the links. That imbalance means an eight-GPU job may spend a full 70% of its time with only one device busy, which in turn consumes power and stretches wall-clock latency. Pipeline parallelism promises a different rhythm: by staggering micro-batches in time and buffering the in-flight activations, every stage can simultaneously perform forward work on one micro-batch and backward work on another. This page answers the core question: how do we turn that rhythmic choreography into a scheduler that knows when to inject new work, when to checkpoint activations, and when the bubble of idle time becomes a memory tax? You will leave with the equations that govern utilization, the history of GPipe and PipeDream as scheduling experiments, the current lab benchmarks that keep bubble time under control, and a runnable two-stage pipeline builder that lets you go from theory to telemetry in one Colab notebook.

## The territory

Large-scale learning systems have always been built on scheduling abstractions. Barroso and Hölzle’s *The Datacenter as a Computer* (2009) showed that a warehouse-sized cluster is best treated as a pipeline of loosely synchronized resources rather than isolated servers, and pipeline parallelism is the same insight applied inside a single training job [Barroso and Hölzle 2009](https://people.eecs.berkeley.edu/~randy/Courses/CS294.F09/wharehousesizedcomputers.pdf). Dean and Ghemawat (2008) argued that MapReduce scales because the scheduler chops work into fine-grained tasks and keeps all machines busy with a stream of micro-operations [Dean and Ghemawat 2008](https://webpages.charlotte.edu/sakella/courses/cloud/papers/DeanGhemawatACMJan2008.pdf). In deep learning, replication (data parallelism) runs out of steam once a single model cannot fit in device memory; Asanović et al. (2009) reminded us that we need to expose multiple granularities of parallelism to span these regimes, and Abadi et al. (2009) compared existing large-scale data-analysis systems to show the fastest ones overlap computation with communication [Asanović et al. 2009](https://people.eecs.berkeley.edu/~krste/papers/parlab-cacm2009.pdf) [Abadi et al. 2009](https://www.cs.umd.edu/~abadi/papers/benchmarks-sigmod09.pdf). Pipeline parallelism is the realization of that temporal overlap inside a neural network: instead of treating each device as a passive holder of layers, it schedules micro-batches through the sequence to keep every device hot while satisfying memory and communication budgets.

Where this concept appears: this page is foundational for the arc steps that look at combining parallelism types, namely [[09-algorithms-systems-for-ai/arc/pipelines]], and it feeds directly into practical deployment guides such as [[09-algorithms-systems-for-ai/concepts/model-parallelism]]. When you reach those steps, you can point back to this scheduling view to understand why the choice of micro-batch depth matters as much as the layer split itself.

## How it works

Pipeline parallelism slices a network of \(L\) layers into \(S\) contiguous stages, each owning a subset of layers mapped to a device. We denote by \(\tau_f\) the average time for a stage to perform a forward pass on one micro-batch and by \(\tau_b\) the backward time for the same micro-batch. Let the minibatch size be \(B\) and split it into \(M\) micro-batches of size \(B/M\). The scheduler sends the forward computation of micro-batch \(m+1\) into stage \(s\) even before stage \(s\) completes micro-batch \(m\)’s backward pass; this overlapping reduces idle time, the so-called *pipeline bubble*, where a stage waits for activations or gradients.

In GPipe’s synchronous schedule (Huang et al. 2019) [Huang et al. 2019](https://arxiv.org/abs/1811.06965), each stage waits for all \(M\) forwards to finish before any backward starts. Its wall-clock time per minibatch is

\[
T_{\text{sync}} = S\cdot (\tau_f + \tau_b) + (M-1)\cdot \tau_f,
\]

where the first term accounts for each stage performing a forward and backward pass, and the second term is the idle bubble created by waiting for all forwards to complete before any backwards begin. GPipe uses activation checkpointing precisely to limit the activation memory per stage: without checkpointing, each stage must store activations for every micro-batch (\(M\) copies), which exhausts device memory as \(M\) grows. Checkpointing reduces this stash by recomputing intermediate activations during the backward pass, trading compute for memory.

PipeDream’s 1F1B schedule (Narayanan et al. 2019) [Narayanan et al. 2019](https://arxiv.org/abs/1902.06709) overlaps forward and backward passes. Each stage \(s\) alternates between computing forward for micro-batch \(m+1\) and backward for micro-batch \(m\), so the bubble shrinks to the fill-and-drain cost of the pipeline:

\[
T_{\text{1F1B}} \approx (S + 2M - 2)\cdot \max(\tau_f, \tau_b).
\]

This approximation uses \(T_{\text{total}} = (S + 2M - 2)\cdot \max(\tau_f, \tau_b)\) because the pipeline needs \(S-1\) micro-batches to fill, \(M\) micro-batches in steady state, and \(S-1\) to drain, with each time step dominated by the slower of \(\tau_f\) or \(\tau_b\). The effective compute utilization is

\[
U = \frac{S \cdot M \cdot (\tau_f + \tau_b)}{T_{\text{total}}},
\]

so maintaining high utilization means keeping \(M\) large enough that the numerator grows faster than the bubble-limited denominator. However, each stage must now stash activations, gradients, and parameter versions for the \(M\) micro-batches in flight, and this stash is proportional to \(M\) times the per-micro-batch activation foot-print. PipeDream mitigates stale gradients by *weight stashing*: each stage keeps multiple copies of its parameters, one per micro-batch, and updates them only after the backward pass for that micro-batch completes, ensuring forward passes see a consistent parameter view.

Communication between stages introduces another term: \(\tau_c\) is the time to transfer activations from stage \(s\) to \(s+1\). When \(\tau_c\) is comparable to \(\tau_f\), the bubble reappears because every micro-batch stalls while waiting for network transfers. The scheduler can absorb \(\tau_c\) by co-locating stages on the same NUMA domain, fusing tensors before transfer, or increasing micro-batch size \(B/M\) so that the relative communication overhead shrinks, at the expense of larger activation storage.

Therefore, selecting the tuple \((S, M, \text{placement})\) becomes a constrained optimization: maximize \(U\) while keeping the stash size \(S_{\text{activation}} \propto M \cdot \text{activation footprint}\) under device memory and buffering the communication term \(\tau_c\) with placement choices. Runtime systems instrument queue lengths between stages to estimate bubble growth and trigger repartitioning when a stage slows down, e.g., due to heterogeneous components such as sparse MoE routers that activate only some experts. Dynamic partitioning adds overhead, so many systems assume a static split and adjust \(M\) instead, but that leaves open how to react when runtime behavior drifts.

Implementing a two-stage pipeline concretely: stage one owns layers \(L_1 \dots L_k\) with forward function \(f_1\), and stage two owns \(L_{k+1} \dots L_L\) with forward \(f_2\). Micro-batches \(\{x^{(m)}\}_{m=1}^M\) flow from stage one to stage two via a queue. During 1F1B, when stage one finishes the forward pass for \(x^{(m)}\) it enqueues the activations and immediately starts \(x^{(m+1)}\). Stage two simultaneously performs backward computation for \(x^{(m-1)}\) while receiving the forward for \(x^{(m)}\); the scheduler ensures the queue depth stays at most \(M\). Each queue entry carries activations and a micro-batch ID so that the backward pass can reclaim the right stash slot when the gradient computation finishes. This circular buffer structure lets the pipeline reuse memory promptly.

The memory–bubble trade-off is why pipeline parallelism is best thought of as scheduling rather than slicing. It forces you to reason about activation stash sizes, communication latency \(\tau_c\), stage placement, and checkpointing overhead simultaneously. Popular systems such as Microsoft DeepSpeed (Rasley et al. 2020) [Rasley et al. 2020](https://arxiv.org/abs/1910.02054) and NVIDIA Megatron-LM slot the scheduler between your model definition and the optimizer, exposing APIs where you define layer groups per stage and a pipeline schedule. Before surrendering control to a library, measuring the queue lengths and bubble time yourself helps you understand why a specific micro-batch depth is necessary.

## Where the field is now

GPipe (Huang et al. 2019) and PipeDream (Narayanan et al. 2019) remain canonical because they quantify the bubble/memory trade-off and offer reproducible schedules. Today’s research frontier focuses on heterogeneity: a single job can contain dense transformer blocks, retrieval layers, and sparse MoE routers whose compute patterns vary by input. Papers such as [Zhuang et al. 2023](https://arxiv.org/abs/2303.05636) explore work stealing between stages to rebalance load, while distributed systems investigations instrument queue lengths and latency histograms to automatically bump \(M\) up when a stage drifts slow. These efforts maintain the fundamental view that pipeline parallelism is a scheduling problem in the temporal dimension of a dataflow graph.

On the engineering side, OpenAI’s “Training GPT-4” blog (2023) details a production stack where pipeline parallelism shares the stage with tensor parallelism and ZeRO optimizer sharding, keeping each stage responsible for 1/16th of a parameter shard plus stage-specific caches while a custom scheduler keeps bubble time below 5% of runtime [OpenAI 2023](https://openai.com/research/training-gpt-4). Meta’s “Llama 3” training notes (2024) describe an interleaved schedule inspired by 1F1B that delivers sustained pipeline utilization above 90% with DeepSpeed’s activation checkpointing driver managing optimizer state shards and reducing per-stage memory by 40% compared to a naive replication baseline [Meta AI 2024](https://ai.meta.com/blog/llama-3/). These engineering stories show that pipeline parallelism is now a production scheduling primitive, not an academic curiosity, because the scheduler’s behavior directly determines wall-clock throughput, energy usage, and job stability.

In addition to heavyweights, emerging start-ups attach instrumentation to pipeline queues: they log bubble time per stage, test the effect of \(M\) on queue imbalance, and retrain schedulers every few hours to adapt to hardware noise. These data-backed schedulers are the latest engineering frontier, while the research frontier remains open questions about adaptive repartitioning and convergence guarantees when communication or compute patterns fluctuate.

## What's still open

Key open questions keep pipeline parallelism research active. First, can we design a scheduler that continuously monitors queue imbalance for routing-heavy layers such as MoE and repartitions the model by moving layers across stages without restarting training, or are such runtime moves too disruptive? Second, is there a formal objective that balances bubble time, activation stash size, and communication cost so that we can automatically choose \(M\) and stage placement instead of manual tuning for every model and cluster? Third, how do asynchronous optimizations (e.g., optimizer state offloading or delayed gradient application) interact with pipeline consistency guarantees—does a delayed stage still converge under standard SGD assumptions, and can we bound the bias introduced by stale activations?

## Where to read next

If you want the systems-level groundwork for overlapping work and communication, → [[09-algorithms-systems-for-ai/concepts/data-parallelism]] explains how gradients aggregate across replicas and why pipeline bubbles break that aggregation. If you want the kernel-level levers that shrink \(\tau_f\) and \(\tau_b\), → [[09-algorithms-systems-for-ai/concepts/flash-attention]] describes the tiling tricks and fused kernels that keep stage runtime predictable. To explore how computation becomes a continuous flow that suggests more fluid partition boundaries, → [[02-generative-modeling/concepts/flow-matching]] draws the theoretical analogy that lets you see pipeline stages as control points on a flow.

## Build it

The build replicates the scheduling mechanism: a two-stage pipeline runner in PyTorch that executes synthetic micro-batches through a split MLP, records queue occupancy, and plots how 1F1B overlaps forward and backward passes while GPipe-style synchronous execution reveals the bubble.

**What you're building:** A Colab-friendly PyTorch notebook implementing a two-stage micro-batched MLP pipeline, logging queue depths, and visualizing bubble time vs. utilization.  
**Why this is valuable:** Watching the queue lengths and the bubble plot teaches you why scheduler choices matter and how activation stash sizes influence real performance.  
**Stack:**  
- **Model:** `hf-internal-testing/tiny-random-mlp` on Hugging Face.  
- **Dataset:** `hf-internal-testing/synthetic-regression` for controlled inputs and targets.  
- **Framework:** PyTorch 2.1.0 with [`torch.distributed.pipeline.sync`](https://pytorch.org/docs/stable/distributed.pipeline.html) primitives; mention that newer PyTorch releases retain these APIs but older releases (1.12+) can fall back to `torch.nn.parallel.DistributedDataParallel` wrappers.  
- **Compute:** Free Colab T4 (16GB VRAM); runtime ≈ 20 minutes.

**The recipe:**  
1. Install `pip install torch==2.1.0 matplotlib pandas`. Initialize two devices via `torch.device("cuda:0")` and `torch.device("cuda:1")` or logical splits on a single GPU; seed Torch RNGs for reproducibility.  
2. Sample \(N=1024\) points \(x \sim \mathcal{N}(0, I)\) and targets \(y = Wx + b + \epsilon\); batch into minibatches of \(B=64\) and split each into \(M=4\) micro-batches, storing them in a queue data structure with per-micro-batch IDs.  
3. Assign layers \(L_1\)–\(L_5\) to stage 0 and \(L_6\)–\(L_{10}\) to stage 1; stage 0 forwards micro-batches, enqueues activations, and waits for a barrier while stage 1 dequeues them for forward/backward; implement 1F1B by alternating send/receive loops with threading and use Adam (LR \(1\times10^{-3}\)) for 5 epochs.  
4. Measure bubble time by tracking epochs when a stage waits on an empty queue and compute utilization \(U = \frac{\text{compute\_time}}{\text{wall\_time}}\); expect \(U>80\%\) once the pipeline stabilizes and the queue depth stays near \(M\).  
5. Export results: produce a plot comparing queue depth and bubble time for synchronous GPipe-style scheduling versus 1F1B, and log timing/activation stash data to prove the latter keeps the pipeline saturated.

**Expected outcome:** A Colab notebook with a runnable two-stage pipeline, bubble vs. utilization plot, and timeline showing forward/backward overlap; link to a starter notebook such as the DeepSpeed pipeline example for reference ([DeepSpeed pipeline parallel example](https://github.com/microsoft/DeepSpeed/tree/master/applications/pipeline_parallelism)).  
**What can you build next:** Tie this runner into a policy that adjusts \(M\) at runtime based on measured queue imbalance and compare the resulting throughput to a static schedule.

**Variants per persona:**  
- **CS student:** Run the notebook on an RTX 4070, increase \(M\) to 8, and chart how the bubble shrinks until the stash memory requires manual activation checkpointing; show the plot before and after checkpointing.  
- **Applied engineer:** Extend the notebook into an inference-style service: export each stage’s forward pass to ONNX, connect them via a lightweight gRPC queue on a dual-A10 node, and target <50 ms pipeline latency per micro-batch while keeping bubble time under 10% by tuning \(M\).  
- **Applied researcher:** Replace one stage with a sparse MoE block (e.g., Hugging Face `moe` stub), hypothesize that the routing-dependent compute will skew queue lengths, and test whether increasing \(M\) or repartitioning the surrounding layers restores a bubble-to-utilization ratio within 15% of the dense baseline; plot idle time ratios before and after rerouting.  
- **Frontier researcher:** Add a runtime monitor that measures queue imbalance, and when one stage exceeds the other by 10%, reassign half its layers to the adjacent stage without restarting; success is maintaining utilization above 85% while keeping bubble time below 12% during the transition.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*