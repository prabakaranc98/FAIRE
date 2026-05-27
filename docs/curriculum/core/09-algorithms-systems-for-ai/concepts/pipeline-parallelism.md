---
title: Pipeline Parallelism  
slug: pipeline-parallelism  
layer: core  
subject: 09-algorithms-systems-for-ai  
page_type: concept  
state: drafted  
authors_anchored: [dean, asanovic, abadi, barroso, huang, narayanan]  
feeds_de_pillar: []  
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher, curious-learner, theory-student]  
prereqs: [data-parallelism, model-parallelism, gradient-checkpointing]  
tags: [parallelism, model-scaling, scheduling, pipeline, training-systems]  
updated: 2025-02-01  
has_mvb: true  
---

Imagine a car assembly line where safety rules allow only one worker in the entire factory at a time. The chassis waits while the frame worker finishes, then the engine fitter waits while the paint booth finishes, and so on. The rack of GPUs that should be sprinting through a large language model suffers the same absurdity when every forward and backward pass runs serially on a single device segment—thousands of petaflops of silicon are held hostage by a single sequential sweep. Pipeline parallelism is the answer to that absurd rule: it invites a continuous stream of micro-batches and stage-to-stage overlap so every accelerator in the rack is simultaneously hammering, buffering, or communicating, rather than waiting for the previous forward-backward cycle to finish.

## The territory

Training foundation models confronts a bifurcation: accelerators have finite memory, and once the network exceeds that footprint, you cannot even instantiate a copy within a single GPU. Data parallelism duplicates the entire network on every device, so its scaling limit bangs directly into that memory ceiling. The only alternative is to slice the network itself, so each accelerator owns a subset of layers and the data stream flows through that assembly line in pieces. This mirrors the operational shift Dean and Ghemawat (2008) [https://webpages.charlotte.edu/sakella/courses/cloud/papers/DeanGhemawatACMJan2008.pdf] described for MapReduce: decompose work into fine-grained tasks and orchestrate communication along a structured pipeline. Similarly, Asanović et al.’s “A View of the Parallel Computing Landscape” (2009) [https://people.eecs.berkeley.edu/~krste/papers/parlab-cacm2009.pdf] argues for exposing multiple parallel granularities; pipeline parallelism exposes both spatial (stage partitioning) and temporal (micro-batch scheduling) slices that the GPU cluster scheduler can manipulate. Barroso and Hölzle’s “The Datacenter as a Computer” (2009) [https://people.eecs.berkeley.edu/~randy/Courses/CS294.F09/wharehousesizedcomputers.pdf] already treated a rack as a latency-sensitive pipeline of services instead of a bag of isolated servers, and pipeline parallelism does the same for the forward/backward wavefront marching through layers. Abadi et al.’s comparison of large-scale data analysis systems (2009) [https://www.cs.umd.edu/~abadi/papers/benchmarks-sigmod09.pdf] shows that without careful staging and buffering, a single slow stage drags every other stage to its pace. In this context, pipeline parallelism recasts the training problem as a scheduling problem where the measure of success is how little idle “bubble” time each accelerator accumulates. How does the choreography actually work?

## How it works

Pipeline parallelism begins with a simple choice: assign contiguous layer blocks to different accelerators, so that activations flow in a directed path and the gradient flow in the reverse. Each stage \(s\) owns \(L_s\) layers; the total depth \(L = \sum_s L_s\) may exceed a single GPU’s memory, but each \(L_s\) must fit within that GPU’s capacity. During training we split a global mini-batch of size \(B\) into \(M\) micro-batches of size \(b = B/M\). Every micro-batch \(m \in \{1, \dots, M\}\) travels through the pipeline, and the updates across all micro-batches accumulate before we step the optimizer. This micro-batching is the core of GPipe (Huang et al. 2019) [https://arxiv.org/abs/1811.06965], which uses gradient accumulation to mimic a large batch while keeping bubble time bounded.

### Stage timing and the bubble

Throughput depends on how well the pipeline sustains a steady stream of work. Let \(T_{s}^{\text{comp}}\) be the time stage \(s\) spends computing a forward or backward pass for one micro-batch and \(T^{\text{comm}}_{s \rightarrow s+1}\) be the time spent transmitting activations or gradients to the next stage. When the pipeline first fills, the first few micro-batches are idling downstream stages until their turn arrives, creating a bubble. The total time for processing \(M\) micro-batches with \(S\) stages, assuming perfect overlap of computation and communication, is approximately
\[ T_{\text{total}} \approx T_{\text{fill}} + T_{\text{steady}} + T_{\text{drain}}, \]
where \(T_{\text{steady}} \approx M \cdot \max_s \left( T_{s}^{\text{comp}} + T^{\text{comm}}_{s \rightarrow s+1} \right)\) reflects the slowest stage, \(T_{\text{fill}} \approx \sum_s T_{s}^{\text{comp}}\) is the cost of filling the pipeline, and \(T_{\text{drain}} \approx \sum_s T_{s}^{\text{comp}}\) is the cost of emptying it. The bubble per stage is the fraction of time a stage waits for its peers, and minimizing it means balancing \(T_{s}^{\text{comp}}\) across stages and overlapping transfers where possible.

When stage durations differ by more than the communication latency, throughput collapses because faster stages must wait for slower ones, and the bubble becomes the dominant term. PipeDream (Narayanan et al. 2019) [https://arxiv.org/abs/1806.03377] mitigated this by introducing the 1F1B (one-forward-one-backward) schedule: once the first micro-batch reaches stage 2, stage 1 immediately starts processing another micro-batch, alternating a forward pass and backward pass in lockstep so every stage is nearly always busy. The 1F1B schedule keeps each stage servicing a pair of micro-batches, and PipeDream also allows asynchronous weight updates so that the expensive backward pass does not block the next forward.

### Micro-batch buffering and activation stashing

Each micro-batch must leave behind its activations for the backward pass; that stash is stored either in stage-local memory or in checkpointed form. If we store all activations for each micro-batch, the stash size grows with \(M\). Pipeline implementations therefore checkpoint every \(k\) layers or re-compute parts of the forward pass during backward as needed. GPipe’s gradient accumulation handles this by keeping a stash for each micro-batch until its backward pass arrives and then discarding it, while PipeDream reuses a rolling ring buffer per stage. Activation checkpointing reduces storage at the cost of extra computation, trading stencil memory for compute, and the stage-local stash budget must be roughly
\[ S_{\text{stash}} \geq M \cdot b \cdot \sum_{l \in L_s} \text{activation_size}(l) \]
where \(\text{activation_size}(l)\) includes any precision reduction (e.g., fp16). That stash budget is why larger micro-batches increase bubble: more micro-batches in flight simply require more buffered activations.

### Communication overlap and staggered transfers

To hide communication costs, pipeline parallelism overlaps activation transfers with computation. Each stage launches a non-blocking send of activations to the next stage while computing on the current micro-batch, and it starts receiving gradients while performing the next backward pass. The overlap is only perfect when the communication bandwidth matches the computation rhythm, so stage placement must also consider PCIe or NVLink topology. These overlapped streams are crucial on heterogeneous clusters where not every accelerator is identical: a stage may wait for the slower partner’s send to complete, creating uneven bubble.

### Dynamic balancing and PP-Balance

Static partitioning cannot cope with token-by-token variation (e.g., speculative decoding, long context windows) — a stage executing a sparse attention block on a long sequence is suddenly slower. ByteScale (2025) introduced the PP-Balance controller that monitors stage latency, inflight micro-batch counts, and gradient queue depth, and then dynamically reassigns layers or reduces the micro-batch count of overloaded stages in real time. The controller maintains a latency vector \(\mathbf{L} = [L_1, \dots, L_S]\) representing the average duration of each stage’s compute-communicate cycle, and it applies a correction \(\Delta L_s\) by either migrating layers via rematerialization or splitting heavy layers across multiple devices. By keeping \(\max_s L_s\) as close as possible to the mean, ByteScale drives down the bubble without sacrificing the upstream data-parallel replica count.

### Failure modes and heuristics

Imbalanced stage partitions, too few micro-batches, or overzealous activation stashing cause pipelines to stall. If \(M < 2S\), the pipeline is underfilled, and the bubble from \(T_{\text{fill}} + T_{\text{drain}}\) dominates, so throughput is below what data parallelism would achieve. If \(M\) is too high, stage-local stash memory overwhelms the GPU, leading to OOM when storing activations for an entire micro-batch queue. The heuristics used in production schedulers therefore sweep \(M\) from \(2S\) up to the maximum that fits the stash, while the scheduler monitors timestamped completion times to adjust \(L_s\) dynamically. The real leverage point is not the compute per stage but the bubble, defined as idle time that a stage experiences because upstream or downstream work is slower. Minimizing the bubble—by tuning micro-batch count, overlapping communications, and rebalancing layers—is the central mechanism that keeps the assembly line moving.

## Where the field is now

Modern systems combine pipeline parallelism with tensor, sequence, and data parallelism in a hybrid layout to amortize both memory and compute constraints. GPipe’s micro-batching and gradient accumulation remain a backbone, but production runs now layer PipeDream’s 1F1B schedule on top of tensor-sliced kernels inside each stage. Research continues to push stages toward data-driven balancing: ByteScale (2025) reported running PP-Balance on 12,000 GPUs with mixed context lengths, showing that dynamically remapping layers reduced bubble time by 18% compared to a static partition and improved overall throughput by 25% on their production LLM benchmarks. On the engineering side, OpenAI’s GPT-4 training blog (OpenAI 2023) [https://openai.com/research/gpt-4] describes using pipeline parallelism across thousands of A100 chips, achieving GPU utilization above 90% during the bulk of the run by combining micro-batches with 1F1B scheduling and activation recomputation. These deployments underscore that the research frontier (dynamic scheduling and heterogeneity-aware balancing) and the engineering frontier (multi-thousand-GPU pipelines with precise latency budgets) are the same needle—both demand that bubble time be the primary metric that a cluster scheduler optimizes.

## What's still open

1. **Can a zero-bubble pipeline schedule be computed online when micro-batch workloads are non-deterministic?** Speculative decoding and dynamic sparse attention produce token-by-token latency spikes, so any static partition will incur a bubble. The question is whether a scheduler can predict or observe these latencies fast enough to reshuffle layers or micro-batch counts without halting the pipeline.
2. **What is the correct trade-off between stash memory and rematerialization in a low-latency inference pipeline?** Keeping activations allows fast backward passes, but on inference-heavy workloads, every stored activation is a memory tax. A quantitative model that trades stash size for recomputation cost and pipeline bubble is missing.
3. **How can pipeline parallelism co-exist with fully sharded data-parallel federated updates when stage failure is frequent?** When a stage drops out, the entire bubble inflates; recovery needs a deterministic rollback or speculative copy, but the current approaches either halt training or accept stale gradients.
4. **Is it possible to integrate a latency-aware pipeline scheduler with differentiable architecture search so that layer splits and micro-batch scheduling co-evolve?** Current pipelines decide stage boundaries before training, so the scheduler cannot react to learned characteristics such as activation density.

## Where to read next

If you want the architectural sibling of pipeline layout, → [Model Parallelism](model-parallelism.md) explains how tensor and expert parallelism divide computation across the same stage boundary. To understand where pipeline buffering saves memory, → [[gradient-checkpointing]] walks through rematerialization and stash budgeting. The scheduler-based view lives in → [[scheduling-heuristics]], which lays out how job managers turn latency measurements into placement decisions.

## Build it

This build proves that even on a single Colab T4, you can simulate a production-style pipeline: the script explicitly stages layers, issues micro-batches, stashes activations, and orchestrates simulated sends/receives with CUDA streams so you measure bubble time and see how 1F1B keeps each stage busy.

**What you're building:** A PyTorch script that trains a 4-layer MLP with a 2-stage pipeline simulator, logging bubble time as synthetic regression data flows through micro-batches.

**Why this is valuable:** The script turns the abstract bubble-time discussion into real numbers, forcing you to implement micro-batch queues, activation stashing, and the backward-pass routing that makes pipeline parallelism memory-feasible.

**Stack:**
- **Model:** Custom 4-layer MLP (PyTorch sequential block with LayerNorm/residual and dropout).
- **Dataset:** Synthetic regression set created via `torch.randn(10000, 64)` inputs and linear targets.
- **Framework:** PyTorch 2.1 with torch.cuda.Stream + torch.cuda.Event for ordering.
- **Compute:** 1×Colab T4 (16 GB VRAM) — the training loop runs in ~45 minutes for 1000 micro-batch steps.

**The recipe:**
1. Install PyTorch and logging helpers: `pip install torch==2.1.0 tensorboard torchmetrics`.
2. Generate synthetic data on-the-fly, chunk it into micro-batches of 8 samples, and assign the first two layers to stage 0 and the last two to stage 1; each stage gets its own CUDA stream and optimizer.
3. Implement the forward pass by enqueuing micro-batches, sending activations via `torch.cuda.Event.record` and `stream.wait_stream`, and stash the outputs in a FIFO per micro-batch; record stage start/end times to compute bubble.
4. During backward, run the 1F1B schedule: stage 1 starts its backward once it receives gradients, stage 0 immediately processes the next forward, and gradients flow backward through the buffer before being averaged across the micro-batch slice.
5. Evaluate bubble time by plotting the ratio of idle duration to compute duration per stage and ensure the simulated pipeline reaches >80% overlap before checkpointing the model state dict.

**Expected outcome:** A checkpointed pipeline-training script with logged bubble-time metrics and a visual plot showing how micro-batching plus 1F1B reduces idle time.

- **CS student:** Run the same script on an RTX 4070 by increasing micro-batch size to 16 and verifying that the bubble time falls below 15% when \(M = 4S\).
- **Applied engineer:** Wrap the pipeline script into a TorchServe backend, quantize the model to int8, and target <120 ms p95 latency for each micro-batch by profiling stage overlap with NVIDIA Nsight.
- **Applied researcher:** Hypothesize that doubling the micro-batch count while halving per-stage depth reduces the bubble more than splitting the same layers across more GPUs; test by comparing bubble time and final loss across the two configurations.
- **Theory student:** Derive the bubble time formula (\(T_{\text{bubble}} = T_{\text{total}} - M \cdot \max_s (T_{s}^{\text{comp}} + T^{\text{comm}}_{s \rightarrow s+1})\)) and verify numerically that the simulated timings match the prediction within 5%.
- **Frontier researcher:** Probe the open question of zero-bubble adaptive partitioning by extending the script to monitor stage latency variance and, when variance exceeds 20%, migrate one layer between stages with activation rematerialization; your falsification criterion is that adaptive migration reduces bubble time by at least 10% without increasing training loss variance.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*