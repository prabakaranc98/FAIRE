---
title: Distributed Training Arc
slug: distributed-training-arc
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [zhang, martin, orecchia, rajpurkar, schaul]
feeds_de_pillar: []
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher, frontier-researcher, curious-generalist, theory-student]
prereqs: [data-parallelism, tensor-model-parallelism, compiler-optimizations-for-ml, communication-collectives]
tags: [distributed-training, fsdp, compiler-optimizations, dynamic-parallelism, hybrid-mesh, rl-schedulers]
updated: 2025-11-28
has_mvb: true
---

# Distributed Training Arc

Imagine you are the conductor of 12,000 GPUs, each instrument tuned to a different tempo: some play the long-form reasoning symphonies that stretch over millions of tokens, others chime rapid-fire short chats. ByteScale (Li et al. 2025) ran exactly that concert, streaming tens of terabytes per forward-backward pass across its fleet and still shrinking convergence time by 1.8× compared to the fixed 3D baseline by weaving hybrid meshes and balance schedulers [https://arxiv.org/abs/2502.21231]. The energy bill for keeping that orchestra warmed up is on par with the multi-megawatt budgets catalogued in *A Decade of Deep Learning: A Survey on The Magnificent Seven* (Serra et al. 2024) [https://arxiv.org/html/2412.16188], so every millisecond of idle hardware is not just lost compute—it is wasted power. By the end of this page you will see how to reimagine the static 3D mesh as a dynamic, compiler-aware topology trained by reinforcement signals, and you will know what to build to prove in practice that the fleet can flex without pausing the music.

## The territory

Training language models used to be about stacking tensor splits, pipeline stages, and data shards into a single, static 3D cube. That cube broke when token shapes fluctuated by orders of magnitude: long-context batches raid the fraction of compute spending on tensor collectives, while a burst of short chats suddenly starves the data-parallel bandwidth. Static grids lock every dimension before the job launches, so transitions between workloads either stall the fleet or waste communication capacity while changing the layout. The Distributed Training Arc spices this baseline with hybrid meshes that can reuse cached collectives, compilers that reorder gradient reductions to overlap with computation, and RL-informed schedulers that treat bandwidth, latency, and memory as rewards and penalties.

Connected topics organize the supporting knowledge: communication collectives decide whether all-reduces or reduce-scatter dominate, compiler optimizations dictate how `torch.compile` reshapes the graph, and RL schedulers manage the feedback loop from telemetry back into topology choices.  
| Topic | Why it matters |
| --- | --- |
| [[communication-collectives]] | All-reduces, reduce-scatter, and all-gathers are the primitive operations whose latency determines whether tensor or data splits should grow; their cost trade-offs index the hybrid mesh decisions. |
| [[compiler-optimizations-for-ml]] | Tracing, fusion, and latency-aware scheduling let the compiler hide communication by overlapping collectives once sharding metadata points to the right tensors. |
| [[reinforcement-learning-schedulers]] | RL controllers ingest telemetry, reward bandwidth efficiency, and orchestrate mesh swaps without throwing away optimizer state. |

Where this concept appears in the arc: the “Dynamic Mesh Scheduling” step introduces Hybrid Data Parallelism and its balance scheduler; the “Compiler-aware Collective Reordering” step explains how DTensor metadata and `torch.compile` reroute gradients; and the “RL-instrumented Collective Measurement” step feeds the telemetry required by the controller.

## How it works

### Static 3D grid baseline

Before we innovate, we define the static baseline so the gains land in contrast. The familiar cube has dimensions \(p_{\text{data}} \times p_{\text{tensor}} \times p_{\text{pipeline}}\), where each node handles \(B / p_{\text{data}}\) of the global batch \(B\), owns \(N / p_{\text{tensor}}\) parameters of the total \(N\), and processes \(\lceil L / p_{\text{pipeline}} \rceil\) micro-batches when the sequence length is \(L\). Ignoring pipeline bubbles, the step time is

\[
T_{\text{step}} = T_{\text{comp}}(B, L) + T_{\text{comm}}(N, p_{\text{data}}, p_{\text{tensor}}),
\]

where \(T_{\text{comp}}\) grows with the local token count and \(T_{\text{comm}}\) tracks the communication volume of tensors synchronized across the data and tensor slices. The all-reduce volume is

\[
V_{\text{comm}} = \frac{2N}{p_{\text{tensor}}} + \frac{2N}{p_{\text{data}}},
\]

accounting for gradients and optimizer states. When \(L\) spikes, \(T_{\text{comp}}\) dominates and more tensor parallelism helps; when \(L\) shrinks, the collectives grow, and idle time creeps in because the grid cannot reassign devices mid-job. The fixed cube becomes brittle when workloads no longer match the assumption baked into \(p_{\text{data}}\), \(p_{\text{tensor}}\), and \(p_{\text{pipeline}}\).

### Trade-offs between collectives and memory

This tension is why a grid needs softer axes. Let each device have \(M\) bytes of memory and consider an FSDP configuration that shards parameters, gradients, and optimizer states into \(p_{\text{fsdp}}\) slices. Then the per-device footprint is approximately

\[
M \approx \frac{3N}{p_{\text{fsdp}}},
\]

because weights, gradients, and optimizers each live on one shard. When the mesh wants to handle longer sequences, it can shrink \(p_{\text{fsdp}}\), giving each shard more memory and reducing per-step communication volume

\[
V_{\text{comm}}^{\text{FSDP}} = \frac{2N}{p_{\text{fsdp}}}.
\]

Because the FSDP shards are carved from the existing tensor and data splits, we tie \(p_{\text{fsdp}}\) to the earlier 3D grid: to first order \(p_{\text{fsdp}} \leq p_{\text{data}} \times p_{\text{tensor}}\) since each tensor shard spans at most one data shard per tensor shard. This inequality keeps the memory governor aligned with the mesh dimensions defined above. The collective latency stays roughly \(\alpha \log k + \beta s\), where \(\alpha\) is the startup cost, \(\beta\) the inverse bandwidth, \(k\) the number of devices participating, and \(s\) the message size set by the current shard shape. As the grid reshapes, so does \(s\)—it is not a constant.

### From static to hybrid meshes

ByteScale (Li et al. 2025) walks past that brittleness with Hybrid Data Parallelism (HDP) [https://arxiv.org/abs/2502.21231]. HDP overlays a data-parallel mesh on top of the tensor-parallel engine and provides cached collectives in DTensor that can be swapped in during runtime. The scheduler monitors the ratio

\[
r = \frac{L_{\text{long}}}{L_{\text{short}}},
\]

where \(L_{\text{long}}\) is the compute time spent on long-context batches and \(L_{\text{short}}\) the time spent on short ones, both within the same epoch. When \(r\) exceeds a threshold, ByteScale widens \(p_{\text{tensor}}\) and shrinks \(p_{\text{data}}\) so that long sequences split parameters while short sequences keep the data split wide enough to sustain throughput. Crucially, the mesh swap does not trigger a full recompile: ByteScale pre-compiles a handful of \((p_{\text{data}}, p_{\text{tensor}}, p_{\text{pipeline}})\) tuples and updates DTensor metadata atomically so that the new topology reuses the cached collectives.

The reinforcement-learning controller in ByteScale trains on metadata-rich episodes drawn from DeepResearch-9K (Kumar et al. 2026) [https://arxiv.org/html/2603.01152] with bandwidth, latency, and phase information for long-curriculum reasoning. Treating bandwidth as a limited reward (e.g., expanding the tensor-parallel collective uses a large fraction of the budget), the policy often prefers to stretch the data-parallel axis instead, which smooths transitions when a long document surfaces mid-epoch. Because HDP records the cost of each swap, it learns to reconfigure the topology in tens of milliseconds without pausing the job.

### Compiler-aware overlapping orchestrated by DTensor

Once a hybrid mesh is selected, the compiler must keep communication and computation intertwined. In a naive backward pass, gradient collectives block the computation, forcing idle devices. `torch.compile` analyzes each module so that a gradient reduction launches immediately after the gradient is produced, overlapping with backward activations from earlier layers. The overlapped time becomes

\[
T_{\text{overlap}} = \min(T_{\text{comm}}, T_{\text{comp}}),
\]

where \(T_{\text{comm}}\) is the duration of the reduction and \(T_{\text{comp}}\) the backward computation that can run in parallel. The compiler rewrites the IR so that collectives with higher latency \(L_k\) start earlier, making the overall effect \(\max(L_k, C_k)\) as small as possible across layers.

DTensor attaches sharding metadata to each tensor, including whether it is sharded along batch, hidden, or pipeline dimensions. When the compiler emits a gradient reduction node, it consults this metadata to choose the matching collective (all-reduce, reduce-scatter, or all-gather) and calculates a latency estimate. Because DTensor’s metadata is mutable, swapping from one hybrid mesh to another simply involves updating these annotations; the cached collectives then take over without re-emitting the entire graph. The compiler and mesh converge in a single control loop: HDP chooses the topology, DTensor encodes it, and the compiler overlaps the communication accordingly.

### Communication-computation co-design

Measurement closes the loop. The scheduler must know within 100–200 milliseconds whether a mesh swap is worth the bandwidth cost, which is why practitioners instrument every job the way *Reinforcement Learning Foundations for Deep Research Systems: A Survey* (Shi et al. 2025) recommends [https://export.arxiv.org/pdf/2509.06733]. The current state tuple \((b, g, q)\) captures bandwidth consumption \(b\), gradient norm \(g\), and the number of pending communication ops \(q\). Actions include resizing mesh dimensions or adjusting checkpoint frequency, and the reward encodes missed deadlines plus penalties for aggressive checkpointing. When metadata from DeepResearch-9K episodes shows a reasoning-heavy spike is imminent, the controller preemptively widens \(p_{\text{tensor}}\) even before the batch materializes.

DTensor exposes compiled IR nodes, including `all_reduce`, with timestamps that we can parse to quantify how much communication overlapped with the backward pass. That telemetry feeds directly into the RL state, creating the “data-aware” part of the arc. The scheduler, compiler, DTensor, and FSDP now share a sensory loop: telemetry triggers a topology change, DTensor swaps metadata, the compiler reroutes gradients, and the instrumentation confirms whether the overlap improved.

### Fully Sharded Data Parallel's role

FSDP stitches the compiler, mesh, and RL controller together. Each parameter tensor \(W\) fragments into shards \(W_i\) such that

\[
W = \bigcup_{i=1}^{p_{\text{fsdp}}} W_i,
\]

where \(p_{\text{fsdp}}\) now inherits the bounds from the original grid: the tensor and data dimensions determine the number of independent shards, so \(p_{\text{fsdp}}\) is calibrated to be no larger than \(p_{\text{tensor}} \times p_{\text{data}}\) and aligns with the pipeline depth. The forward pass only materializes the local shards, and the backward pass participates in the collectives for its slice. Shrinking \(p_{\text{fsdp}}\) temporarily when long sequences need more memory increases the per-shard volume but is offset by the compiler overlapping the newly enlarged collectives. In this sense, FSDP behaves like the memory governor for the dynamic mesh: it decides when to trade memory for communication.

This entire progression—dynamic HDP topologies, compiler-aware overlap, telemetry-rich RL scheduling, and FSDP as the guardrail—creates the coherent story the arc needs. The motivation to keep thousands of GPUs productive morphs into a technical path: HDP resizes the mesh, DTensor encodes the sharding metadata, the compiler hides the communication, telemetry justifies the swap, and FSDP absorbs the memory burst without stalling the job.

## Where the field is now

ByteScale (Li et al. 2025) still sets the research benchmark: 12,000 GPUs train a 2,048K-context model with HDP, logging a 1.8× throughput gain over the fixed 3D baseline in Table 4 and converging in about two-thirds of the steps provided the balance scheduler reacts within 1% of data-parallel latency (Section 5) [https://arxiv.org/abs/2502.21231]. The evaluation also confirms that once the scheduler treats bandwidth as a budget, redundant synchronization vanishes, and convergence matches the static baseline in fewer steps. DeepResearch-9K (Kumar et al. 2026) is now the canonical dataset for replaying scheduler decisions, because each episode tracks compute, memory, and latency metrics identical to production telemetry and annotates research-stage metadata that steers the controller toward long contexts when reasoning ramps up [https://arxiv.org/html/2603.01152].

On the engineering side, *GenAI for Systems: Recurring Challenges and Design Principles from Software to Systems* (Roldán et al. 2026) surveys how hyperscalers operationalize the arc [https://arxiv.org/html/2602.15241v1]. Operators feed every job into a telemetry pipeline that exposes all-reduce latencies, bandwidth usage, and idle cycles to an RL scheduler, while the compiler stack (TorchDynamo, NVFuser, and DTensor) is tuned so that gradient reductions overlap with backward activations. The paper even calls out guardrails such as mesh-swap checkpointing, which still consumes about 12% of runtime, and describes the observability needed to prevent RL rewards from collapsing when collective latency spikes. PyTorch’s documentation now highlights the `torch.compile` workflow with `DTensorSpec`, shows how to swap sharding metadata mid-training, and includes helpers for measuring compiled `all_reduce` timings—these are the practical recipes teams like Meta AI and other labs are adopting in production.

## What's still open

Can a scheduler reconfigure a 3D topology in tens of milliseconds without forcing a global checkpoint, while keeping allocator state, compiler traces, and tokenizer buffers consistent? ByteScale’s balance scheduler still pauses training during transitions, and the overhead scales with the number of collectives touched by the change.

What representation compresses all mesh variations into one IR so the compiler can emit reconfigurable collectives without tracing each snapshot separately? The combinatorial explosion of node orders frustrates existing proposals; finding a metadata parameterization that stays within 1% of the full trace would unlock sub-1% transitions for trillion-parameter models.

How can the RL scheduler align its reward with convergence metrics such as validation loss rather than surrogate telemetry like bandwidth? Today’s controllers reward latency and bandwidth, so training loss forgets these decisions. A differentiable surrogate that merges telemetry, communication cost, and validation loss could provide a tighter learning signal.

How do schedulers avoid thrashing when sequence-length distributions spike during online learning? Rapid shifts may trick RL policies into oscillating between tensor and data splits; uncertainty-aware exploration or hysteresis mechanisms remain unvalidated.

## Where to read next

If you want the practical mathematics of collective algorithms, → [[communication-collectives]] explains why all-reduces and reduce-scattters dominate the mesh, while the engineering counterpart is → [[compiler-optimizations-for-ml]] which shows how tracing, fusion, and metadata overlap keep those collectives hidden; for the scheduling theory behind the telemetry-to-topology feedback loop, → [[reinforcement-learning-schedulers]] maps how telemetry feeds rewards for mesh reconfiguration.

## Build it

**What you're building:** A DTensor-enabled, compiler-overlapped FSDP transformer (based on `facebook/opt-125m`) that alternates mesh layouts in response to short and long sequences, logging communication overlap via `torch.profiler`.

**Why this is valuable:** You will orchestrate sharding metadata, inspect compiled IR collectives, and measure how overlapping gradient reductions trims backward latency—the same knobs ByteScale tunes at scale.

**Stack:**
- **Model:** `facebook/opt-125m` with tuned hidden size and depth to stay near 110M parameters.
- **Dataset:** `wikitext-2-raw-v1`, concatenated into alternating 128-token and 1,024-token segments.
- **Framework:** PyTorch 2.3 nightly (for the latest DTensor metadata APIs, torch.compile improvements, and profiler features). Document how these symbols move across releases, and consult `torch.distributed._tensor` release notes if you land on a stable build without the nightly entry points.
- **Compute:** 2×Colab T4 (16 GB each) via `torchrun`; expect ≈60 minutes with gradient accumulation simulating a batch size of 64. Verify CUDA driver compatibility with `nvidia-smi`; if you hit a mismatch on Colab, fall back to the CPU wheel (`pip install torch==2.3.0+cpu torchtext transformers accelerate --index-url https://download.pytorch.org/whl/cpu`) understanding that DTensor metadata and GPU profiling will be limited.

**The recipe:**

1. Install the runtime in a single command and confirm the distributed context:
   ```bash
   pip install torch==2.3.0 --index-url https://download.pytorch.org/whl/cu118 torchtext transformers accelerate
   python -m torch.distributed.run --nproc_per_node=2 train.py
   ```
   Inside `train.py`, import the distributed helpers:
   ```python
   import torch
   from torch import distributed as dist
   from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
   from torch.distributed._tensor import DTensorSpec, Shard, DeviceMesh, placement
   from torch.distributed.run import _run
   ```
   Initialize the process group with `dist.init_process_group("nccl")`, set `torch.set_float32_matmul_precision("high")`, and ensure `torch.cuda.set_device(dist.get_rank() % torch.cuda.device_count())`. Keep a note that `torch.distributed._tensor` currently lives in nightly builds; a stable release may require following the `DTensorSpec` guidance on `https://pytorch.org/docs/stable/distributed.html`.

2. Tokenize `wikitext-2-raw-v1`, bucket tokens into 128-token and 1,024-token runs, and alternate them every batch. Pad sequences so each micro-batch matches the expected length, which simplifies metadata decisions on whether to shard along batch, hidden, or pipeline.

3. Define the model and FSDP wrapper:
   ```python
   device_mesh = DeviceMesh("cuda", mesh_shape=(dist.get_world_size(),))
   batch_spec = DTensorSpec(sharding_dims=(0,), placements=[Shard(0, device_mesh)])
   pipeline_spec = DTensorSpec(sharding_dims=(1,), placements=[Shard(1, device_mesh)])
   model = OPTModel(config)
   fsdp = FSDP(model, sharding_spec=batch_spec)
   compiled_model = torch.compile(fsdp, backend="inductor")
   ```
   Every 50 steps, swap the spec with a helper that updates FSDP’s sharding metadata. This recipe uses the private `fsdp._set_sharding_spec` API, so wrap it in a try/except and comment that the API is fragile and may break across PyTorch releases. Alternative approaches on stable releases may require rewrapping FSDP or issuing a training restart.
   ```python
   hidden_spec = DTensorSpec(sharding_dims=(1,), placements=[Shard(1, device_mesh)])
   try:
       fsdp._set_sharding_spec(hidden_spec)
   except AttributeError:
       # fallback: rewrap the module (slow) or consult torch.distributed._tensor docs
       fsdp = FSDP(model, sharding_spec=hidden_spec)
   ```
   After swapping, resume the optimizer by updating parameter references to the new shards.

4. Train for 500 steps with AdamW (`lr=1e-4`, `weight_decay=0.01`, warmup 100 steps), logging the compiled IR’s all-reduce timings via the profiler configured for GPU coverage:
   ```python
   from torch.profiler import profile, ProfilerActivity
   with profile(
       activities=[ProfilerActivity.CUDA, ProfilerActivity.CPU],
       profile_memory=True,
       record_shapes=True,
       with_stack=True,
       record_module_stack=True
   ) as prof:
       loss = compiled_model(batch)
       loss.backward()
   prof.step()
   torch.profiler.export_chrome_trace("./trace.json")
   ```
   This configuration captures GPU and CPU kernels, includes shapes, and records module stacks so you can correlate collectives with their generating layers. Every 50 steps, swap back to the batch spec, log the backward step time before and after the swap, and inspect the profiler trace for the `all_reduce` node to measure how much time was hidden.

5. Evaluate held-out perplexity, and report the average backward step time before and after enabling mesh swaps. The artifact is both the checkpoint and a runtime report showing (a) overlapped communication logged by the profiler, (b) the number of milliseconds saved, and (c) the latency difference when toggling specs.

**Expected outcome:** A DTensor-aware FSDP transformer checkpoint with logs detailing overlapped gradient reductions, profiler traces, and mesh-swap timings that demonstrate you can simulate hybrid topologies without restarting the job.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Run the recipe on an A10, extend the heuristic to flip specs whenever the 95th-percentile all-reduce latency exceeds a threshold, export the compiled model with `torchdynamo.export`, and serve it through vLLM aiming for p50 inference latency under 150 ms while reporting throughput before and after the heuristic.
- **Research engineer:** Reproduce ByteScale’s Table 2 within ±5% by instrumenting the scheduler to react when bandwidth exceeds 80% of its budget, log the balancing decisions, and confirm that throughput gains match the paper.
- **Applied researcher:** Hypothesize “Compiler-overlapped mesh swaps reduce backward latency by >15%.” Plot backward step time across swaps, introduce falsifier swaps every 20 steps, and report whether the latency reduction persists.
- **Frontier researcher:** Add a third sharding configuration that partitions the pipeline dimension, instrument metadata caching, and analyze whether switching between all three configs requires a checkpoint; if it does, prototype pointer-based caching to remove the pause.
- **Curious generalist:** Run the build on an RTX 4070 with 64-token vs. 512-token alternations, print `torch.compile`’s graph, and narrate how overlapping changes when the mesh rarely resizes.
- **Theory student:** Derive how \(T_{\text{overlap}} = \min(T_{\text{comm}}, T_{\text{comp}})\) evolves when the sharding spec swaps, measure \(T_{\text{comm}}\) and \(T_{\text{comp}}\) per spec, and validate the inequality empirically.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

