---
title: Pipeline Parallelism  
slug: pipeline-parallelism  
layer: core  
subject: 09-algorithms-systems-for-ai  
page_type: concept  
state: drafted  
authors_anchored: [dean, asanovic, abadi, barroso]  
feeds_de_pillar: []  
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher, curious-learner, theory-student]  
prereqs: [data-parallelism, model-parallelism]  
tags: [parallelism, model-scaling, training-systems]  
updated: 2024-11-09  
has_mvb: true  
---

Three quarters of a rack can deliver 200 TFLOPS, yet a 100-billion-parameter language model still stalls because the weights no longer fit inside a single accelerator and the rest of the cluster sits idle waiting for one ugly forward-backward sweep to finish. When data parallelism is the only tool, that redundancy makes it impossible to even start training once the model exceeds any single device’s memory cap. Pipeline parallelism redeploys the cluster as an assembly line: each accelerator hosts a contiguous segment of the network, and micro-batches stream down the line so the whole rack stays busy. This note follows that choreography, showing how stage layout, bubble time, and stash budgets keep every accelerator working without sacrificing gradient fidelity, and why modern datacenter programs have built this assembly line into their production scheduler.

## The territory

Large-scale training faces two immutable facts: device memory is fixed, and communication latency explodes as the model spans more chips. Data parallelism duplicates a full copy of the network on each accelerator, so its first constraint is the memory ceiling. Once the network grows beyond that, the options are either to shrink the model or to split it. Pipeline parallelism takes the second route by slicing the layers into sequential stages and letting each accelerator own only the layers it can store, while the dataset traverses the stages in small groups. This reframes the training challenge as a scheduling problem rather than a memory one, much like Barroso and Hölzle’s “The Datacenter as a Computer” (2009) [https://people.eecs.berkeley.edu/~randy/Courses/CS294.F09/wharehousesizedcomputers.pdf] treated a rack of blades as a latency-sensitive pipeline instead of isolated servers. Dean and Ghemawat’s 2008 MapReduce primer [https://webpages.charlotte.edu/sakella/courses/cloud/papers/DeanGhemawatACMJan2008.pdf] similarly showed that breaking work into fine-grained tasks and orchestrating communication is the only path to scalability. Asanović et al.’s “A View of the Parallel Computing Landscape” (2009) [https://people.eecs.berkeley.edu/~krste/papers/parlab-cacm2009.pdf] argued that parallel systems must expose multiple granularities, and pipeline parallelism exposes the right one for deep learning by combining spatial partitioning with temporal pipelining. The comparisons between data analytics frameworks in Abadi et al.’s 2009 SIGMOD survey [https://www.cs.umd.edu/~abadi/papers/benchmarks-sigmod09.pdf] reinforce the same lesson: the biggest wins come from ordered communication and fine-grained scheduling, exactly the levers pipelining applies to the forward/backward pass. This section ends where the next begins—how do you actually keep the micro-batches flowing without drowning in bubbles or memory?

## How it works

Pipeline parallelism divides the network into \(S\) sequential stages \(\{L_1, \dots, L_S\}\), each hosted on its own accelerator, and trains using \(B\) micro-batches of size \(m\) that enter the pipeline in order. At micro-batch \(b\) and stage \(s\), the forward pass computes activations \(a_{s,b}\) and the backward pass later produces gradients \(\delta_{s,b}\). To keep the backward pass exact without recomputing the forward work, each stage stashes its \(a_{s,b}\) until \(\delta_{s,b}\) is ready to proceed. The latency per micro-batch is

\[
\mathcal{L} = \sum_{s=1}^{S} \left( \tau^{\text{fwd}}_{s} + \tau^{\text{bwd}}_{s} \right) + \sum_{s=1}^{S-1} \tau^{\text{comm}}_{s \leftrightarrow s+1},
\]

where \(\tau^{\text{fwd}}_{s}\) is the time stage \(s\) spends executing the forward pass on a single micro-batch, \(\tau^{\text{bwd}}_{s}\) is the backward time, and \(\tau^{\text{comm}}_{s \leftrightarrow s+1}\) is the communication time for activations/gradients to pass between stages \(s\) and \(s+1\). The first sum is useful compute; the second is the overhead that pipeline engineering must hide. Micro-batch size \(m\) compresses \(\tau^{\text{comm}}\) by reducing the shape of the tensors, while the number of micro-batches \(B\) controls how fast the pipeline fills and how many bubbles (idle steps) occur. A fully saturated pipeline runs for \(B+S-1\) steps per iteration, so the steady-state throughput is approximately \(\frac{B \cdot m}{\mathcal{L}}\). Increasing \(B\) fills more bubbles but also increases stash memory linearly, as each stage must remember its activations for \(B\) micro-batches:

\[
M_{\text{stashes}} = \sum_{s=1}^{S} B \cdot \text{size}(a_{s}),
\]

where \(\text{size}(a_{s})\) is the byte size of the activations that stage \(s\) must cache for a single micro-batch. Mixed-precision stashing (e.g., storing \(a_{s,b}\) in bfloat16) reduces this cost at the price of potential rounding error across the stage boundary. In practice, the engineer balances \(B\), \(m\), and stash compression to keep \(\tau^{\text{comm}}\) and bubble overhead tolerable without blowing up memory.

### Partitioning, scheduling, and balance

Assigning layers to the \(S\) stages is an optimization over compute time, memory, and communication. Toolkits like TorchGPipe and DeepSpeed take a layer graph, group layers into \(P_1, \dots, P_S\), and assign them to stages, but the naive equal-parameter split can still leave \(\tau^{\text{fwd}}_{s} + \tau^{\text{bwd}}_{s}\) unbalanced. Engineers profile per-layer FLOPs and activation sizes before committing to a partition so that each stage’s total compute time is roughly the same; otherwise, the slowest stage becomes the bottleneck and the pipeline runs at the speed of the slowest \(\tau_s\). The partitioner must also consider inter-stage communication: placing layers with large activation maps on the same stage reduces \(\tau^{\text{comm}}_{s \leftrightarrow s+1}\), while splitting them across stages increases \(\mathcal{L}\). Finally, when hardware is heterogeneous—GPUs with different FLOPS or memory—the partitioner needs to allocate more or less work per device, which turns the problem into optimizing the stage-wise latency profile \(\{\tau_s\}\). The same partitioner typically exposes two knobs: the number of stages \(S\), which is the spatial split, and the number of micro-batches \(B\), which is the temporal depth.

Pipeline parallelism naturally composes with other parallelisms. Within a stage, tensor parallelism shards matrix multiplies (e.g., Megatron-LM’s tensor parallelism), while across stages you can add data parallelism by replicating the entire pipeline and synchronizing with standard All-Reduce after each optimizer step. This layered approach multiplies the effective world size to \(S \times \text{data-parallel-size}\), so gradient synchronization becomes hierarchical: a stage first runs local reductions to resolve tensor-parallel shards, then the data-parallel group runs its own sync after the pipeline completes its per-stage gradients \(\delta_{s,b}\).

### Mathematical foundations

The pipeline throughput metric is more precise when written as

\[
\text{throughput} \approx \frac{B \cdot m \cdot S}{(B + S - 1) \cdot \mathcal{L}},
\]

where \(\mathcal{L}\) is defined as above and the numerator \(B \cdot m \cdot S\) counts the total micro-batch work across all stages per iteration, while the denominator captures the actual number of pipeline steps the rack executes. As \(B \to \infty\), the denominator approximates \(B \cdot \mathcal{L}\), so throughput converges to \(\frac{m \cdot S}{\mathcal{L}}\), which equals \(m\) times the number of stages divided by the per-micro-batch latency. The bubble fraction is \(\frac{2(S-1)}{B+S-1}\); it shrinks as \(B\) increases but the stash memory \(M_{\text{stashes}}\) scales linearly with \(B\). This is why empirical tuning first sets \(m\) to match memory budgets (hence controlling \(\tau^{\text{comm}}\)) and then raises \(B\) until the bubble fraction is acceptable, checking that \(M_{\text{stashes}} \leq \text{available RAM}\) at every stage. These equations also explain why asynchronous schedules (where different stages are allowed to start processing the next micro-batch before older ones finish) increase the effective \(B\) without actually increasing the memory budget: they overlap communication and compute to reduce the perceived \(\mathcal{L}\) seen by the steady-state.

### Activation stashing, recomputation, and failure modes

Each stage stashes activations \(a_{s,b}\) because recomputing the forward pass would repeat the work for every micro-batch, slowing training by roughly a factor of two. The stash sits either in GPU memory or on CPU; storing it in CPU RAM adds transfer latency but alleviates GPU memory pressure, while spilling to NVMe is even slower. Checkpointing can replace stashes entirely: rather than caching \(a_{s,b}\), the stage replays the forward pass when the backward pass requires it. This trades compute time (increasing \(\tau^{\text{fwd}}_{s}\) effectively) for memory, and TorchGpipe exposes `checkpoint()` hooks for this style. Mixed strategies combine stashes for the first few layers with checkpointing in later layers to minimize total latency. Hardware failure or network stalls are handled by saving a small stash-level log so a stage can re-request activations from its peer rather than recompute the entire pipeline segment. These mechanisms define the cost/benefit triangle: increase \(B\) to reduce bubble fraction, but monitor \(M_{\text{stashes}}\) and \(\mathcal{L}\); reduce \(\tau^{\text{comm}}\) with smaller \(m\) but accept more synchronization events; trade stash memory for recomputation via checkpointing to keep the pipeline running on low-memory devices.

## Where the field is now

Pipeline parallelism research now splits along two axes: reducing bubbles and supporting asynchronous schedules. PipeDream (Harlap et al. 2018) [https://arxiv.org/abs/1804.01662] introduced weight stashing, enabling micro-batches to move ahead even when earlier ones were still being updated, and proved that asynchronous execution could keep the pipeline busy with tolerable gradient staleness. GPipe (Huang et al. 2019) [https://arxiv.org/abs/1811.06965] showed that synchronous partitioning with automatic re-batching could train a 1.5-billion-parameter Transformer without extra gradient noise by tightly balancing partition sizes and micro-batch counts. A more recent thread follows PipeDream-2’s idea of decoupling stage progress (2022) [https://arxiv.org/abs/2207.02819], letting slower stages buffer additional micro-batches without stalling the others, and mixing precision when stashing activations to keep tape sizes low while preserving accuracy.

On the engineering horizon, system builders have already baked pipeline parallelism into production. Megatron-LM (Shoeybi et al. 2019) [https://arxiv.org/abs/1909.08053] described training an 8.3-billion-parameter Transformer across 512 V100 GPUs with both tensor and pipeline parallelism, proving that balanced schedules and stage choices scaled linearly in practice. Meta’s OPT-175B release (Zhang et al. 2022) [https://arxiv.org/abs/2205.01068] described a DeepSpeed-backed run on 4,096 A100 GPUs that combined 2D tensor-model parallelism with pipeline stages so each chip only needed to hold a fragment of the network. Google’s PaLM 540B (Chowdhery et al. 2022) [https://arxiv.org/abs/2204.02311] trained on 6,144 TPU v4 cores with pipeline parallelism ensuring each TPU pod stage stayed fully utilized even while the datum streamed past. This convergence—research tuning bubble latency and production orchestrating thousands of chips—forms a story: pipeline parallelism matured from a small scheduling experiment into a required stage in every billion-parameter training pipeline.

## What's still open

Can we find a fully automatic partitioner that minimizes the maximum per-stage latency \(\max_{s} (\tau^{\text{fwd}}_{s} + \tau^{\text{bwd}}_{s})\) and communication \(\tau^{\text{comm}}_{s \leftrightarrow s+1}\) across heterogeneous accelerators, with an evaluation metric such as the reduction in the max stage-to-stage skew below 5 ms compared to uniform splits?

How can activation stashing be guaranteed bounded memory usage for sparse or Mixture-of-Experts models when each micro-batch touches a different subset of experts, and can we measure success by reducing the worst-case per-stage stash footprint variance to under 10 % while keeping update variance within standard deviation 0.01?

Is there a rigorous, cross-stage rounding-error model for mixed-precision or INT8 stashes with a measurable impact on pipeline stability, for example by bounding the growth in \(\delta_{s,b}\) variance per stage below 2× over fp32 so that we can choose the lowest precision without empirical trial and error?

## Where to read next

If you want the production story, → [[../concepts/data-parallelism.md]] explains how pipeline parallelism nests inside data-parallel replicas while the scheduler still balances inter-stage gradients. For tighter memory control, → [[../concepts/activation-checkpointing.md]] shows how recomputation relieves stash pressure when \(M_{\text{stashes}}\) is the binding constraint, and → [[../concepts/model-parallelism.md]] lays out the tensor-party alternatives that can sit inside each stage. Connected topics such as tensor-model parallelism, hierarchical All-Reduce, and communication-aware schedulers all appear around the large-model-training arc, which you can explore further through the [[../arcs/large-model-training-systems.md]] link.

## Build it

What you're building: a pipeline-parallel PyTorch training run that fits a four-layer MLP on MNIST using micro-batching, activation stashing, and checkpointing so each of two GPUs only holds half of the model parameters.

Why this is valuable: it provides hands-on experience scheduling micro-batches, measuring bubble overhead, tracking stash budgets, and comparing stash memory vs. recomputation before you move to billion-parameter Transformers.

Stack:
- **Model:** Four-layer MLP with ReLU activations, trained from scratch (no pre-trained weights).
- **Dataset:** [mnist](https://huggingface.co/datasets/mnist) — standard train/test splits and the canonical 10-class classification task.
- **Framework:** PyTorch 2.1 + [torchgpipe](https://pypi.org/project/torchgpipe/) for pipeline abstractions and activation checkpointing.
- **Compute:** 2×RTX 4090 (24 GB) or Google Colab Pro+ with two A5000 GPUs; expect ~30 minutes for 20 epochs at 94–98% accuracy.

The recipe:
1. **Install & configure:** Run `pip install torch==2.1.0 torchgpipe==0.3.0 torchvision datasets` inside a `cuda`-enabled environment, then use `torch.cuda.set_device` to pin two GPUs and register them in a `torchgpipe.devices.split_devices([0, 1])` call so the `Pipe` object owns both devices.  
2. **Construct the pipeline:** Define the MLP as four `nn.Sequential` blocks (two layers per stage) and wrap them with `torchgpipe.Pipe(seq, devices=split_devices, chunks=4)` to create \(S=2\) stages and \(B=4\) micro-batches.  
   ```python
   from torchgpipe import Pipe
   block = lambda in_dim, out_dim: nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
   seq = nn.Sequential(
       block(28*28, 512), block(512, 512),
       block(512, 256), block(256, 10)
   )
   pipe = Pipe(seq, devices=[0, 1], chunks=4, checkpoint='always')
   ```
   This sets micro-batch size \(m=32\) (since MNIST batch size 128 split into 4 chunks) and enables fp16 checkpointing to control \(M_{\text{stashes}}\).  
3. **Run training:** Use Adam (lr=1e-3, betas=(0.9, 0.999)) and gradient accumulation across \(B=4\) micro-batches by calling `optimizer.step()` every 4 forward passes; the `Pipe` automatically exposes stage timing through `pipe.profiler`. Track `bubble_time = pipe.profiler.last_step['idle_time'] / pipe.profiler.last_step['total_time']` to compute the bubble fraction defined by \(2(S-1)/(B+S-1)\).
4. **Evaluate:** After each epoch, evaluate on the MNIST test set (10k samples) with `pipe.eval()`; the checkpointed pipeline should reach 96–98% accuracy with ±0.5% variance if bubble overhead stays below 10% and the learning rate is \(1 \times 10^{-3}\).  
5. **Artifact:** Save the pipeline checkpoint `torch.save(pipe.state_dict(), "pipeline_mlp.pth")` along with a CSV of per-stage latency, stash memory usage, and bubble fraction so you can tune \(B\), \(m\), and checkpointing next time.

Expected outcome: a live pipeline-parallel checkpoint that demonstrates bubble-aware scheduling, stash budgets, stage timing logs, and test accuracy, giving you the confidence to extend the same pattern to larger models.

Variants per persona:
| Persona | Variant |
| --- | --- |
| **CS student** | Swap the MLP for a toy Transformer encoder so you can inspect how attention layers change \(\tau^{\text{comm}}\) and stash sizes; log the per-stage FLOPs to justify the partition. |
| **Applied engineer** | Wrap the pipeline run in stateful inference/export hooks and measure bubble time under synthetic load, exporting the per-stage timing dashboard for ops handoff to a scheduler team. |
| **Applied researcher** | Disable activation stashing and rerun with recomputation every other layer to plot the trade-off curve between stash memory and \(\mathcal{L}\), reporting the point where bubbles start dominating throughput. |
| **Frontier researcher** | Extend the recipe to heterogeneous devices (e.g., A5000 + IPU) and implement a search that minimizes the maximum per-stage latency \(\max_s (\tau_s)\) subject to a total budget, evaluating success by how much you reduce peak-latency imbalance versus the uniform split baseline. |
| **Curious learner** | Visualize the pipeline bubble fraction and per-stage bandwidth in a Jupyter notebook, narrating how micro-batches keep each accelerator busy without needing to understand every optimizer detail. |
| **Theory student** | Re-derive the throughput and bubble equations in the “Mathematical foundations” section, then verify them empirically by logging \(\mathcal{L}\), \(B\), and \(m\) for each run to confirm the predicted bubble fraction matches observations. |

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*