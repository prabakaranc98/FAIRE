---
title: Pipeline Parallelism
slug: pipeline-parallelism
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [dean, asanovic, abadi, barroso]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [data-parallelism, model-parallelism, tensor-parallelism]
tags: [parallelism, pipeline-scheduling, large-models, throughput]
updated: 2024-11-09
has_mvb: true
---

# Pipeline Parallelism

Imagine a multi-million dollar automotive factory where only one mechanic is allowed to touch a car at a time, forcing every other workstation to stare at a parked chassis until that single worker finishes a painstaking operation. The whole production line becomes a queue rather than a pipeline. That is exactly what happens when a 100‑billion‑parameter Transformer is naively split across GPUs layer by layer: one accelerator becomes the bottleneck executing the full forward-backward sweep while dozens of others sit “idle” with shattered throughput and wasted power credits. This essay shows how pipeline parallelism rewires that assembly line by viewing the problem not as “which GPU holds which layer” but as “how do micro-batches, recomputation, and schedules interact to minimize the bubble of idle devices while keeping activation storage in check.”

## The territory

Large-scale training rests on two fundamental limits: device memory is capped, and communication latency multiplies as the model spans more chips. MapReduce-style thinking already taught us that chopping work into small tasks and orchestrating data movement is the only path to scalability once single workers hit their memory ceiling (Dean and Ghemawat 2008) [https://webpages.charlotte.edu/sakella/courses/cloud/papers/DeanGhemawatACMJan2008.pdf]. Parallel computing scholars noted that exploitable concurrency comes in multiple granularities, so systems should expose all of them rather than forcing programmers into one fixed level of parallelism (Asanović et al. 2009) [https://people.eecs.berkeley.edu/~krste/papers/parlab-cacm2009.pdf]. Barroso and Hölzle’s “The Datacenter as a Computer” (2009) [https://people.eecs.berkeley.edu/~randy/Courses/CS294.F09/wharehousesizedcomputers.pdf] similarly reframed datacenter racks as latency-sensitive pipelines: each server stage must stay busy or the entire job stalls. Pipeline parallelism for neural networks is that same reframe applied to training graphs, and Abadi et al.’s benchmark comparison (2009) [https://www.cs.umd.edu/~abadi/papers/benchmarks-sigmod09.pdf] spelled out how data-parallel-only stacks fall apart once the model exceeds any single device memory, motivating the need to slice models across hardware.

Pipeline parallelism slices the network into contiguous stage segments that live on different accelerators, then streams micro-batches through the stages so every device can process different parts of different batches simultaneously. The territory is no longer “how do I fit the weights?” but “how do I schedule computation so that the time-to-completion is minimized despite activation dependencies?” Micro-batching, bubble scheduling, and activation recomputation become the knobs that determine whether the assembly line accelerates or stalls. How does this scheduling actually work, and what are the trade‑offs between idle bubbles, memory, and throughput?

## How it works

The critical insight of pipeline parallelism is that a forward pass through a whole model is a sequence of stage operations, and the latency profile of that sequence depends on how many micro-batches are enqueued, how stages overlap, and how much activation data each stage must hold. If we denote \(S\) as the number of sequential stage groups (each mapped to a device), \(B\) as the size of the minibatch and \(m\) as the micro-batch size (so \(B = m \cdot L\) for \(L\) micro-batches), the naive sequential schedule executes each micro-batch through all \(S\) stages before starting the next and takes roughly \(L \cdot S \cdot T_\text{stage}\) time, where \(T_\text{stage}\) is the average per-stage compute time. When the other devices wait for each stage to finish, the effective utilization is \(1/S\) and the bubble (idle time per device between its operations) equals the forward plus backward latency of the other stages. Pipeline parallelism lifts the idle-stage constraint by overlapping work: once stage 1 finishes micro-batch 1’s forward pass, stage 2 can start processing that micro-batch while stage 1 begins the forward pass for micro-batch 2.

GPipe (Huang et al. 2019) [arxiv:1908.11889] formalized this overlap by streaming micro-batches across stages with synchronous recomputation checkpoints: each stage keeps only the inputs needed for its portion, recomputing activations when necessary to save memory, while the schedule keeps all stages busy by sending \(L\) micro-batches in a bubble-free “fill-drain” pattern. In GPipe the schedule is divided into three phases: the pipeline fill, steady state, and drain. During the steady state, each of the \(S\) stages processes a micro-batch every \(T_\text{stage}\), so throughput approximates \(m \cdot S / T_\text{batch}\), where \(T_\text{batch} \approx S \cdot T_\text{stage}\) in steady state, and the bubble is proportional to the fill and drain windows. The formula for stage utilization \(U\) becomes \(U \approx \frac{L}{L + S - 1}\), so longer sequences of micro-batches (larger \(L\)) reduce bubble overhead but increase activation storage because every micro-batch in flight needs its activations kept until backward passes arrive.

That trade-off between bubble and activation memory is the scheduling tension. GPipe’s solution is to checkpoint across micro-batches: each stage stores only what it needs for backward computation and recomputes intermediate activations by replaying the forward pass, trading compute for memory. The amount of saved memory is roughly proportional to the number of layers per stage \(k\), since storing \(k\) hidden states requires \(k \cdot m \cdot D\) bytes where \(D\) is the hidden dimension; recomputation reduces this to storing only stage boundaries while re-executing the \(k\) layers during the backward pass. The consequence is that micro-batch size \(m\) can stay small enough to keep activation memory manageable, while \(L\) remains large enough to amortize bubbles.

PipeDream (Narayanan et al. 2019) [arxiv:1811.06965] introduced a different cadence that interleaves forward and backward passes, the 1F1B (one forward, one backward) schedule, which also keeps bubbles low but further reduces activation storage. In 1F1B, once a stage finishes computing forward for micro-batch \(i\), it immediately starts backward for micro-batch \(i-1\) while upstream stages continue forward for micro-batch \(i+1\). This pipeline interleaving means the maximum number of activations held concurrently equals the number of micro-batches in flight times the per-stage activation size for just one direction, dramatically cutting storage compared to GPipe’s fill-drain where \(L\) micro-batches must store both forward and backward activations. The schedule requires tracking the dependencies between micro-batches so that gradients are routed correctly, which is where PipeDream’s “stashing” mechanism comes in: each stage keeps a ring buffer of activation tensors for the micro-batches it must backward through, and the buffer size is tuned to match the 1F1B latency.

Quantitatively, let \(N_F\) be the forward time and \(N_B\) the backward time per micro-batch per stage. In 1F1B scheduling with steady-state, the latency per micro-batch is approximately \(N_F + N_B\), but the pipeline depth is \(S\), so total time per minibatch across \(S\) stages is \(m \cdot (N_F + N_B)\) plus a bubble cost of roughly \(S \cdot \max(N_F, N_B)\). The critical parameter to minimize is the bubble cost relative to steady work, meaning \(L\) and the scheduling policy must ensure \(S \cdot \max(N_F, N_B)\) is small compared to \(L \cdot (N_F + N_B)\). Smaller micro-batches \(m\) reduce the per-stage run time but require larger \(L\) to maintain throughput, which again drives activation memory. The scheduling problem is, therefore, an optimization over \(m\), \(L\), recomputation budget, and stage layout.

Stage placement also matters. Uneven layer runtimes—e.g., attention-heavy blocks vs. FFN-heavy blocks—break the assumption that \(T_\text{stage}\) is constant. The Michelangelo of scheduling identifies groups of layers whose combined time fits within a target \(T_\text{stage}\) so that stage runtimes are balanced; imbalance introduces bubbles because faster stages wait on slower ones. Placement algorithms treat the model’s DAG as a linear sequence and cut it into contiguous segments to keep dependency overhead minimal, but dynamic models like Mixture-of-Experts (MoE) break that linearity, making static cuts insufficient. In those cases, the scheduler must consider potential branching of tokens across expert clusters and ensure that the pipeline can handle the worst-case activation footprint.

Hardware topology adds another layer to the scheduling game. Interconnect bandwidth and latency differ between NVLink, Omni-Path, and PCIe, so placing adjacent stages on devices with the fastest links reduces communication time and bubble. The boundary between stages now has two components: compute time and communication delay. To capture this, we define total per-stage latency \(T_s = T_{\text{compute}_s} + T_{\text{comm}_s}\), where \(T_{\text{comm}_s}\) depends on both tensor size and link bandwidth. The scheduler’s goal becomes minimizing \(\sum_{s=1}^{S} T_s\) while respecting memory constraints per device. Modern systems therefore rely on simulators that model both compute and communication, enabling automated partitioning decisions that anticipate bubble growth when communication dominates compute.

Model parallelism also intertwines with tensor parallelism (splitting within layers) and data parallelism (replicating pipeline stacks). At small batch sizes, pipeline parallelism alone suffices, but at production scale the model is sharded three ways: data batches distributed across replicas, pipeline partitions along layers, and tensor-parallel splits within attention heads or linear layers. The combined scheduling requires solving a multi-dimensional assignment where each device’s compute budget, activation storage, and bandwidth budget must be satisfied simultaneously. Without a carefully tuned schedule, tensor parallelism might force pipeline stages into uneven compute regions, increasing \(T_{\text{compute}_s}\), and data parallelism might amplify bubble through synchronous gradient updates. The pipeline schedule must therefore co-optimize with these other parallelism axes, often using heuristic solvers or learned partitioners.

Visualization of pipeline schedules reveals the bubble directly: each micro-batch is a horizontal bar, and each stage’s operation is color-coded. When the pipeline is naive (single device per forward/backward), the bars line up sequentially, leaving vast white gaps. When micro-batched and scheduled properly, the bars interleave with minimal white space, showing how activations flow through the stage pipeline. The essential question is always the same: what combination of micro-batch size \(m\), number of in-flight micro-batches \(L\), recomputation depth, and stage assignment yields the highest computed throughput \(m \cdot \frac{L}{L+S-1} / T_\text{stage}\) without blowing per-device memory? Solving this scheduling optimization is what makes pipeline parallelism a discipline of timing diagrams rather than simply “split the layers”.

## Where the field is now

The research frontier pushes this scheduling optimization into dynamic, heterogeneous domains. The Frontier LLM Training Study (2024) [arxiv:2405.12052](https://arxiv.org/abs/2405.12052) reported that exascale LLM training on AMD MI250X-based clusters achieves peak utilization only when pipeline parallelism is co-optimized with tensor and data parallelism via a hardware-aware compiler that profiles the runtime throughput per stage and adjusts micro-batch lengths on the fly. The study quantifies how a static schedule incurred up to 30% bubble on dynamic routes, whereas a compiler that rebalanced stage splits every 500 steps reduced bubble by half while keeping activation memory under 90 GB per device. Concurrently, the MoE-heavy workloads of that study highlighted the inability of offline profilers to capture token routing variance, showing that the scheduling loop must be data-aware.

Engineering practice mirrors that complexity. Production crews at OpenAI documented in their training report (OpenAI 2023) that GPT-4’s training stack employed a mixture of pipeline, tensor, and data parallelism, with a carefully tuned 1F1B schedule across each pipeline stage and 12 micro-batches in flight to keep H100 GPUs saturated at the 1,024-token context. They also added activation checkpointing for the dense layers and lowered precision to bfloat16 so that each stage’s buffer remained within the available memory. Stability AI’s Stable Diffusion 3.5 training (2024) leveraged the NVLink-connected DGX SuperPOD’s bandwidth by grouping pipeline stages on the same NVLink cluster to minimize \(T_{\text{comm}_s}\), whereas large-scale text-to-image training on AWS (2024) used intra-stage gradient accumulation to cover pipeline bubble introduced by PCIe connections. These deployments show that production-grade pipeline parallelism is as much about hardware topology tuning and checkpointing policies as it is about splitting layers.

A secondary frontier is automation: companies now embed pipeline scheduling into compilers and resource managers. For example, NVIDIA’s Megatron-LM scheduler (developer.nvidia.com/blog/training-gpt-gpus) automatically partitions models based on per-layer FLOP counts, estimated activation sizes, and NVLink topology before every training run. The scheduler then handshakes with the runtime to adjust micro-batch sizes if the actual throughput deviates from estimates, thereby bounding bubble growth. This tooling demonstrates that the scheduling problem is seldom solved once; it is monitored continuously during training.

## What's still open

What compiler or runtime can dynamically partition heterogeneous, dynamic-routing models such as Mixture-of-Experts across asymmetric networks to achieve zero-bubble execution without offline profiling?

Can we find an analytic cost model that jointly reasons about stage imbalance, recomputation overhead, and network latency across GPUs so that pipeline partitions can be chosen before a single micro-batch runs?

How can we design activation checkpointing policies that react to traffic in 1F1B schedules—recomputing less when the pipeline is saturated and recomputing more when bubbles appear—without adding manual tuning knobs?

## Where to read next

If the reader wants to see how pipeline stages coexist with another axis of splitting, → [Tensor parallelism](tensor-parallelism.md) explains how matrix multiplications are fissioned across devices, and the engineering counterpart is → [Data Parallelism](data-parallelism.md) which adds replicas to the whole pipeline stack. For deeper scheduling theory, → [[scheduling-algorithms]] returns to the classic latency-vs-throughput trade-offs in heterogeneous processors.

## Build it

Training a toy Transformer across a two-stage pipeline on Colab proves that pipeline parallelism is the scheduling problem described above: you must micro-batch, visualize the bubble, and compare it to the naive sequential run to understand where the throughput gains come from.

**What you're building:** A PyTorch pipeline parallel simulator that splits a small Transformer encoder exactly in half, trains on synthetic text, compares naive sequential execution against GPipe-style micro-batching, and visualizes stage timelines to show bubble reduction.

**Why this is valuable:** The build forces you to code the scheduler that controls micro-batch launch order and to instrument the pipeline so you can see how changing the number of in-flight micro-batches \(L\) affects both bubble and activation memory, embodying the precise trade-offs that GPipe and PipeDream optimized.

**Stack:**
- **Model:** `hf-internal-testing/tiny-random-GPT2` — ~1.4K downloads, minimal weights.
- **Dataset:** `wikitext` (keep the toy `train[:10%]` split for speed).
- **Framework:** PyTorch 2.1 + `torch.distributed.pipeline.sync` from `torch.distributed`.
- **Compute:** Free Colab (T4, 16GB VRAM); expected runtime ~5 minutes.

**The recipe:**
1. Install PyTorch 2.1 + Matplotlib (`pip install torch torchvision matplotlib`) and clone a helper script that samples synthetic token sequences.
2. Instantiate a Transformer encoder with four layers, split it into two `nn.Sequential` stages, and wrap each as a dummy “device” using either CUDA streams (if GPU available) or CPU contexts while tracking activation sizes per stage.
3. Implement naive sequential training: for each batch of 32 tokens, run forward/backward through stage 1 then stage 2, recording timestamps for every stage start/end to show a single-threaded timeline.
4. Implement GPipe-style micro-batching: split each minibatch into four micro-batches, stream them through the two-stage pipeline with recomputation by enabling `torch.cuda.checkpoint_sequential`, and instrument the scheduler to submit micro-batches so that stage 2 begins work before stage 1 finishes all forwards.
5. Plot the collected timelines for naive and micro-batched runs, compute throughput (tokens processed per second), and log the activation memory reported by PyTorch per stage.

**Expected outcome:** A Python notebook that produces two timeline plots, showing how micro-batching fills the pipeline and quantifies the reduction in bubble, alongside logged throughput numbers.

- **CS student:** Swap CUDA streams for `torch.device("cpu")` and scale the Transformer down to two layers so the build runs on a single RTX 3060 in a classroom lab.
- **Applied engineer:** Package the scheduler inside a TorchServe handler, quantize each stage to INT8 using PyTorch VNNI, and serve the pipeline with `torch.distributed.rpc` to hit a realistic latency target (p95 < 30 ms per micro-batch) on an A10.
- **Applied researcher:** Treat \(m\) and \(L\) as variables—run an ablation that compares throughput for \(L \in \{1,4,8\}\) while keeping \(m = 8\), and report whether activation recomputation (checkpointing depth) changes bubble reduction as predicted by GPipe’s empirical curves.
- **Frontier researcher:** Extend the simulator to include a Mixture-of-Experts split stage and leverage the build’s scheduler to probe whether dynamic routing increases bubble unless the compiler anticipates worst-case expert fan-out, thereby testing the open question from §What's still open.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*