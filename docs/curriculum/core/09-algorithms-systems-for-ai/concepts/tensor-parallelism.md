---
title: Tensor parallelism
slug: tensor-parallelism
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [shoeybi, rajbhandari, goyal, brown]
feeds_de_pillar: []
mvb_personas: [cs-student, curious-learner, theory-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [distributed-training, transformer-architecture, attention-mechanisms]
tags: [tensor-parallelism, distributed-training, megatron, communication-bound, torch-distributed]
updated: 2025-05-10
has_mvb: true
---

# Tensor parallelism

A single weight matrix from a 175B-parameter transformer can be thicker than any GPU’s memory shelf—trying to load it is like pushing a wall of books through a doorway that is only one person wide. The cluster sits idle while your matrix multiplication stalls, not because the FLOPs aren’t there, but because the tensor refuses to be split unless you teach the GPUs to cooperate on the linear algebra itself. Tensor parallelism rewrites individual matrix multiplies so each device stores a slice of a weight tensor and the math is stitched together through carefully scheduled communication. Along the way, you’ll learn why the strategy is best described as communication-bound spatial partitioning, how a column or row partition is implemented in PyTorch-level linear layers, and how those primitives feed into a Megatron-style attention block that you can run on free Colab by emulating the communication on a single device. If you need to refresh distributed training, attention, or transformer primitives, start with the [[distributed-training]], [[attention-mechanisms]], and [[transformer-architecture]] overviews before returning here.

## The territory

The scaling story for large language models splits into two battles: first, how do you increase parameter count without running out of memory on each accelerator, and second, how do you keep latency and throughput acceptable once the parameter count spikes? Data parallelism conquers the second by copying the entire model, but it fails the first as soon as a single matrix multiply from a feed-forward or attention projection exceeds one GPU’s RAM. Pipeline parallelism slices layers across devices, and ZeRO-style optimizer sharding slices optimizer state, yet neither lets one matrix multiply live on two GPUs at once.

Tensor parallelism answers the question “how can two GPUs cooperatively compute the same \(y = x W^\top\) without ever materializing the full \(W\) or \(y\)?” It sits inside a single layer’s spatial domain instead of across the batch (data parallelism) or optimizer state (ZeRO). That’s also why tensor parallelism must live beside the other axes: pipeline handles the order of layers, ZeRO handles optimizer shards, data parallelism handles mini-batches, and tensor parallelism handles the dense algebra inside each projection. The key tension is that the answer to “how big are my slices?” is determined not by compute but by the network fabric—overlap the communication and you hide the cost, but if you split too finely without communication overlap you pay latency instead of gaining throughput. How does it actually work?

## How it works

Tensor parallelism reshapes a dense linear layer so that each GPU holds a slice of the weight tensor and only computes the corresponding portion of the matrix product. Consider a dense projection \(y = x W^\top\), where \(x \in \mathbb{R}^{B \times D_{\text{in}}}\) is the batch of input activations, \(W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}\) is the weight matrix, and \(y \in \mathbb{R}^{B \times D_{\text{out}}}\) is the output pre-activation. The two canonical choices are column partitioning (along \(D_{\text{out}}\)) and row partitioning (along \(D_{\text{in}}\)). Each choice produces a different communication pattern and a different shape of the local compute.

In column parallelism, \(W = [W_1; W_2; \dots; W_m]\) is sliced along the output dimension, so GPU \(i\) stores \(W_i \in \mathbb{R}^{\frac{D_{\text{out}}}{m} \times D_{\text{in}}}\). Each GPU computes a partial output \(y_i = x W_i^\top \in \mathbb{R}^{B \times \frac{D_{\text{out}}}{m}}\), and the full \(y\) emerges by concatenating the partial outputs. Column partitioning keeps \(x\) intact, so the forward path only needs a collective to gather the \(y_i\) slices: each GPU runs a local dense multiply and then participates in an `AllGather` on the \(D_{\text{out}}\) axis. The backward path mirrors this split: gradients for \(W_i\) remain local, but the gradients for \(x\) require an `AllReduce` across \(y\) because every slice saw the full \(x\).

Row parallelism slices \(W\) along the input dimension, splitting \(x\) accordingly. Let \(x = [x_1, x_2, \dots, x_m]\) with each \(x_i \in \mathbb{R}^{B \times \frac{D_{\text{in}}}{m}}\), and let \(W = [W_1, W_2, \dots, W_m]\) with \(W_i \in \mathbb{R}^{D_{\text{out}} \times \frac{D_{\text{in}}}{m}}\). Each GPU computes \(y_i = x_i W_i^\top\) locally, and the final result is \(y = \sum_{i=1}^{m} y_i\). To assemble that sum, the GPUs run an `AllReduce` over the output activations. Row parallelism therefore burdens both forward and backward passes with reduction along the output axis, yet it spreads the input-side memory across devices.

The runtime cost of both partitions is shaped by compute \(T_{\text{comp}}\) and communication \(T_{\text{comm}}\). After partitioning into \(m\) GPUs, each GPU’s compute is proportional to its slice:

\[
T_{\text{comp}} = \frac{2 \times B \times D_{\text{in}} \times D_{\text{out}}}{m \times \text{FLOPS}_{\text{GPU}}}
\]

where \(B\) is the batch size, \(D_{\text{in}}\) and \(D_{\text{out}}\) are the projection dimensions, \(m\) is the partition count, and \(\text{FLOPS}_{\text{GPU}}\) is the per-device FLOPs capability. The communication cost for an \(s\)-byte exchange is

\[
T_{\text{comm}} = \alpha + \beta \times s,
\]

where \(\alpha\) is the interconnect latency and \(\beta\) is the per-byte inverse bandwidth on that fabric. Column parallelism exchanges \(\mathcal{O}(B \times D_{\text{out}})\) bytes in both the forward `AllGather` and the backward `AllReduce`; row parallelism exchanges \(\mathcal{O}(B \times D_{\text{out}})\) bytes either during the reduction of \(y_i\) or during the activation redistribution. The ratio \(R(m) = T_{\text{comm}} / T_{\text{comp}}\) therefore depends explicitly on the projection widths: increasing \(m\) shrinks \(T_{\text{comp}}\) but leaves \(T_{\text{comm}}\) roughly unchanged because the communicated activations still involve \(D_{\text{out}}\). Tensor parallelism becomes communication bound when \(R(m)\) climbs above one, and that is the gate where additional GPUs stop improving throughput.

### Layer implementation in practice

Frameworks such as Megatron-LM wrap these sliced multiplies inside `ColumnParallelLinear` and `RowParallelLinear` modules. The column-parallel forward pass calculates the local output \(y_i\) and then issues `torch.distributed.all_gather` (often with `group=tp_group`) to reconstruct \(y\). The backward pass lets PyTorch’s autograd accumulate \(W_i\) gradients locally while performing an `AllReduce` on \(\nabla_x\). Row-parallel layers first redistribute \(x\) to match the weight slices—often by `all_reduce` with `tp_group`—then compute each local contribution \(y_i = x_i W_i^\top\), and finally `all_reduce`-sum the \(y_i\) before passing it down the pipeline. The implementation must also select the correct parallel mode (model, data, pipeline) so that collectives stay confined to the participating GPUs and do not accidentally cross over into the global communicator.

### Communication and compute overlap

To keep \(R(m)\) below one, modern stacks overlap communication with computation. Async tensor parallelism uses non-blocking collectives so that the GPU can launch the next matmul while the current gradient reduction is still in flight. Practically, each GPU issues `torch.distributed.all_reduce(..., async_op=True)` on \(\nabla_W\) or \(\nabla_x\) and lets the background stream handle the reduction while the forward pass for the next layer proceeds. The gradient must not be consumed before the reduction finishes, so the optimizer waits on the handle just before the step. Overlapping also requires pacing: aggressive scheduling of asynchronous collectives can swamp the fabric, so heuristics from Tokasaurus-style Async-TP (Juravsky et al. 2025) monitor outstanding communication counts and prevent the next kernel from launching until the current collectives fall below a threshold. (Juravsky et al. 2025) track CPU-side gradient buffers because they may live on the host when the GPU is busy with the next kernel, and they throttle the CPU so that host-GPU transfers do not add another bottleneck.

### Other parallelisms and failure modes

Tensor parallelism rarely runs alone. Megatron-LM’s preferred pattern nests tensor parallel layers inside pipeline stages and wraps the entire pipeline in ZeRO-style optimizer sharding, with a final data-parallel replica for throughput. Each axis introduces its own schedule: pipeline stages decide when to sync activations, tensor parallel groups decide how many GPUs to enlist, and ZeRO decides how much optimizer state to move off-device. Weak scaling reports (Optimizing Distributed Training on Frontier 2024) show that linear scaling persists only if the tensor-parallel group sizes remain small enough that \(R(m)\) stays below one, otherwise the fabric becomes the bottleneck and the near-perfect scaling dissolves.

Common failure modes include mismatched group sizes—if a layer expects four GPUs but PyTorch only gives it two, the `all_gather` or `all_reduce` call simply hangs. Another failure occurs when the slices shrink so much that communication latency dominates, which can be diagnosed by profiling the \(T_{\text{comm}} / T_{\text{comp}}\) ratio per layer using Nsight Systems (`nsys profile ...` around the Python script) or `torch.profiler` with `schedule=torch.profiler.schedule(wait=1, warmup=1, active=3)` and logging the time spent inside collectives. Finally, overlapping asynchronous reductions without proper `.wait()` calls corrupts the gradients; the optimizer step must always wait on the handles before applying updates.

### Example: Megatron-style tensor-parallel attention block

A Megatron-style attention block slices all query, key, value, and output projection matrices along the same axis to keep the tensor parallel group consistent. For column parallelism, each GPU stores a slice \(W_i \in \mathbb{R}^{\frac{D_{\text{model}}}{m} \times D_{\text{model}}}\), computes local projections such as \(Q_i = x W_i^\top\), and participates in a collective to gather the full \(Q\) before applying softmax. Row parallelism slices \(x\) and the projection weights so that each device computes its portion of \(Q, K, V\) and then reduces the attention scores after the dot product. The communication-bound nature of the tensor parallel slice is clearest here: the softmax cannot run until the gather finishes, so any delay in the `AllGather` directly delays the kernel that consumes \(Q\). This is why Megatron-LM interleaves asynchronous collectives with the next layer’s matmul—only by overlapping can the fabric keep up with the FLOPs, and this connection between communication and kernel order is the synthesis that ties the partition math back to the attention block.

## Where the field is now

The field now treats tensor parallelism as both a research space and an engineering dial. From a research standpoint, the survey *A Decade of Deep Learning: A Survey on The Magnificent Seven* (2024) [https://arxiv.org/html/2412.16188] frames tensor parallelism as one of the seven “magnificent” scaling primitives and highlights its tight relationship to communication lower bounds, making it essential reading for anyone who wants to reason about why overlapping collectives matters. The survey also collects benchmark numbers showing that the best tensor-parallel implementations keep \(R(m)\) near 0.5 even as model sizes cross 100B parameters.

Two recent papers push current frontiers. ByteScale (2024) reorganizes tensor-data hybrids for 2M-token contexts by letting tensor-parallel scopes reuse data-parallel reductions, which cuts the number of `AllReduce` calls by half and keeps throughput above 10 tokens/ms on 1024-context micro-batches (ByteScale 2024). Tokasaurus (Juravsky et al. 2025) studies Async-TP on heterogeneous CPU-GPU inference farms and shows that pacing communication on CPUs lowers end-to-end latency by 18% on 32-GPU clusters while maintaining numerical fidelity. Together these works show that research now converges on making collectives adaptive—either by reusing reductions or by pacing them based on host occupancy.

On the engineering side, GenAI for Systems: Recurring Challenges and Design Principles from Software to S (2026) [https://arxiv.org/html/2602.15241v1] catalogs how tensor parallelism has become a software-design decision. The report details how systems at OpenAI, Anthropic, and Stability AI expose tensor-partition choices to schedulers, telemetry, and observability tools, because communication patterns now dominate latency budgets even on the fastest NVLink fabrics. Production stacks such as Megatron-DeepSpeed expose API hooks to toggle different tensor-parallel group sizes at runtime and pair them with communication-aware autoscalers that reduce the degree of partitioning when interconnect bandwidth dips. DeepResearch-9K: A Challenging Benchmark Dataset of Deep-Research Agent (2026) [https://arxiv.org/html/2603.01152] stresses these systems with 9,000 long-form research assistant transcripts, forcing training batches to scale in both sequence length and batch size and making the communication cost of each attention block an observable metric in telemetry dashboards.

This research-engineering convergence means tensor parallelism is no longer an optional optimization; it is now a required axis in the stack. The key question for every cluster is the same: how much of the global weight matrix do we replicate, how much do we partition, and at what point does communication start to dictate the shape of each GPU’s compute window?

## What's still open

The remaining questions boil down to adaptivity under heterogeneity. First, how can tensor parallel partitioning be scheduled dynamically across heterogeneous hardware (GPUs, TPUs, and CPU nodes) so that each matrix multiply chooses the optimal slice count without incurring scheduling overhead that negates the communication savings? Second, can a general-purpose tensor-parallel runtime borrow ideas from reinforcement learning systems to learn these partition assignments online, as proposed in *Reinforcement Learning Foundations for Deep Research Systems: A Survey* (2025) [https://export.arxiv.org/pdf/2509.06733], instead of relying on hand-tuned heuristics tuned to a single fabric? Third, can we predict reductions far enough ahead of time to overlap communication on non-NVLink clusters, perhaps with a compiler pass that buffers gradient updates just long enough to hide latency without blowing up memory with extra copies? And finally, when context windows grow to millions of tokens, do the existing column/row slices keep optimizer consistency, or do we need a new pipeline/tensor hybrid that prevents stale gradients after dozens of additional reduction steps? Each of these questions maps directly to a paper-length exploration: they are about measurably falsifiable hypotheses rather than vague “more work needed” statements.

## Where to read next

If you want hands-on arc material, → *step 03 tensor parallelism* <!-- [[step-03-tensor-parallelism]] --> implements Megatron-style column and row splits on two GPUs and explains how they combine with optimizer sharding. If latency reduction is your goal, → *step 05 async tp* <!-- [[step-05-async-tp]] --> walks the Async-TP pipeline that hides `AllReduce` latency during decoding. If you want the theoretical context for the communication lower bounds that make \(R(m)\) the decision metric, → [[parallel-algorithmics]] frames tensor parallelism inside the broader class of distributed linear algebra lower bounds. These links keep tensor parallelism connected to the rest of this arc so that the next build you read about deepens the same operational picture.

## Build it

The build proves tensor parallelism on a single Colab T4 by slicing a linear layer into column and row shards, wiring up the corresponding collectives, and verifying that the reconstructed activations match a dense reference while logging communication times.

**What you're building:** Megatron-LM-style `ColumnParallelLinear` and `RowParallelLinear` layers wired into a toy MLP block, running inside a simulated tensor-parallel group on one GPU but still exercising the communication primitives explicitly.

**Why this is valuable:** Executing the collectives by hand on a cheap GPU makes the “communication-bound” regime tangible: you can see the ratio of `AllReduce`/`AllGather` time to matmul time, test overlap heuristics, and verify that the final activations align with the dense baseline.

**Stack:**
- **Model:** `facebook/llama-7b` (314k+ downloads) — load just the linear layer weights or a distilled checkpoint to stay within 8GB.
- **Dataset:** `flower_photos` (https://huggingface.co/datasets/flower_photos) filtered to 32×32 RGB crops to generate dummy \(x\) tensors.
- **Framework:** `torch==2.2` with `torch.distributed`; use the `gloo` backend when running on a single Colab T4 and drop back to `nccl` when you move to a multi-GPU machine.
- **Compute:** single Colab T4 (16GB VRAM) — the loop (batch size 8, 2 micro-batches) completes in about an hour and prints per-iteration communication/computation ratios.

**The recipe:**
1. Install the stack with `pip install torch==2.2 torchvision==0.17` and launch `torchrun --nnodes=1 --nproc_per_node=2` inside Colab; set `MASTER_ADDR=127.0.0.1`, `MASTER_PORT=29500`, and export `CUDA_VISIBLE_DEVICES=0` so each rank thinks it has `local_rank=0` while the `gloo` backend handles the single physical GPU.
2. Preprocess `flower_photos` by resizing images to 32×32 and stacking them into a tensor of shape \((B, D_{\text{in}})\) with \(D_{\text{in}} = 32 \times 32 \times 3\); broadcast the tensor across both processes by calling `torch.distributed.broadcast` before entering the forward loop so that both ranks see identical \(x\).
3. Implement `ColumnParallelLinear` by slicing \(W \in \mathbb{R}^{D_{\text{out}} \times D_{\text{in}}}\) along \(D_{\text{out}}\), computing \(y_i = x W_i^\top\) locally, and reassembling the output with non-blocking `torch.distributed.all_gather` before applying the bias; on the backward pass, issue `torch.distributed.all_reduce` on \(\nabla_x\) before the optimizer step (Adam, lr=1e-4, weight_decay=0).
4. Implement `RowParallelLinear` by slicing \(W\) along \(D_{\text{in}}\), scattering \(x_i\) slices using `torch.distributed.reduce_scatter`, computing each \(y_i = x_i W_i^\top\), and reducing the outputs with `all_reduce`; validate by comparing the sum of \(y_i\) to `torch.matmul(x, W.t())` with `torch.allclose`.
5. Compose the two layers into an MLP, run 500 iterations, log wall-clock times before and after each collective, and graph `comm_time / total_time` per iteration; confirm that the chunked activations equal the dense reference and that `AllReduce` times drop when you enable asynchronous overlap.

**Expected outcome:** a checkpointed toy MLP that logs communication vs. compute ratios, demonstrates matching activations with the dense baseline, and leaves you with a reproducible setup to experiment with overlap heuristics.

**What can you build next:** extend the toy block into a mini-Megatron-style attention head, add a synthetic token stream, and measure how latency increases when you widen the tensor-parallel group versus when you increase the batch size.

**Variants per persona:**
- **CS student:** Run the recipe entirely on Colab with batch size 4, make the collectives synchronous (`all_reduce(...).wait()`), and plot the ratio of communication to computation across iterations to see the point where \(R(m)\) exceeds one.
- **Curious learner:** Instrument the code to print human-readable messages when each collective starts and finishes, so you can tell a story about what the GPUs are waiting on and why tensor parallelism looks like “shared work, shared wait.”
- **Theory student:** Derive the expression \(R(m) = \frac{\alpha + \beta B D_{\text{out}}}{\frac{2BD_{\text{in}}D_{\text{out}}}{m \times \text{FLOPS}_{\text{GPU}}}}\), plug in the actual numbers from your run, and write a short proof sketch showing how increasing \(m\) amplifies the denominator but leaves the numerator almost unchanged.
- **Applied engineer:** Export the shardable layers as TorchScript, quantize the weights to fp16, serve them as part of a vLLM-like pipeline on an A10 instance, and aim for end-to-end latency under 60 ms per token while keeping NCCL collectives intact and logging `comm_latency / token`.
- **Applied researcher:** Replace the synchronous `AllGather` in the column-parallel layer with an asynchronous call that overlaps with the next layer’s matmul, and test whether the total step time drops by at least 10% on the toy data by logging before/after timestamps.
- **Frontier researcher:** Implement a runtime scheduler that adjusts the tensor slice count each iteration based on runtime bandwidth estimates and show that a static split is no longer optimal by measuring throughput drift when the estimated link bandwidth falls below 80% of peak.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*