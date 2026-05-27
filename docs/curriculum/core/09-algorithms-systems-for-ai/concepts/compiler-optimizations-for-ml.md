---
title: Compiler Optimizations for ML
slug: compiler-optimizations-for-ml
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [brown, chen, singh, patel]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [operator-fusion, torch-compile, distributed-training-optimization]
tags: [compilers, operator-fusion, torch-compile, triton, latency, distributed-training]
updated: 2025-11-01
has_mvb: true
---

# Compiler Optimizations for ML

Imagine the engineer who once wrote a single operator kernel now facing the combinatorial explosion of every new LLM block: Group-Query Attention interleaves with SwiGLU, superficial layer norms sprout RMSNorm variants, and memory layouts shift between batched inference and pipeline-parallel training. There simply isn't enough human brainpower to handcraft CUDA for each permutation, and the bandwidth on a system like a 4-way A100 cluster is dictated as much by memory-bound neighbors as by floating-point throughput. That reality is why the modern compiler's job has changed: instead of applying a fixed sequence of heuristics to a static graph, compilers now pose a search-and-synthesis problem, encoding equivalence, runtime constraints, and hardware idiosyncrasies into a planner that tunes itself at compile time and during execution. By the end of this page you will understand how this dynamic optimization loop works, how recent work proves its value in both single-device and distributed regimes, and how you can fuse an RMSNorm+SwiGLU pair with `torch.compile` and Triton on free Colab to actually see the latency drop.

## The territory

There used to be just a compiler backend and a runtime: you lowered a computation graph, applied peephole rewrites, and emitted CUDA kernels that the scheduler fed to GPUs. Today's ML workloads break that model because the kernel shapes, sparsity, and memory windows keep multiplying. Each new transformer variant asks for a different combination of attention bias, activation function, and normalization strategy, so the only way to keep up is to search a huge space of transformations for the few that matter. This is why systems researchers now treat compiler optimization as a co-design problem that simultaneously considers operator fusion, tensor layout, and distributed communication. The compiler accumulates candidates from symbolic superoptimization, reinforcement learning search, and hardware-aware profiling, and it dispatches the winners to a runtime orchestrator that can overlap compute, communication, and memory movement. The consequence is that the compiler is no longer a static transformer of IR but a dynamic engine that can generate novel kernels, reshape tensors, and schedule collective operations in lockstep. How does it actually do that?

## How it works

The mechanism starts with the observation that every kernel bundle can be viewed as a permutation of operators, tensor layouts, and scheduling parameters, and that the optimization goal is to minimize latency under memory and parallelism constraints. We write that as
\[
\min_{\pi \in \mathcal{P}} \mathcal{L}(\pi) + \lambda \cdot \mathcal{C}(\pi)
\]
where \(\pi\) is a compiler plan drawn from the search space \(\mathcal{P}\), \(\mathcal{L}(\pi)\) is the measured latency (including kernel launch overhead, cache misses, and stalled pipelines), and \(\mathcal{C}(\pi)\) is a cost term that penalizes register pressure, excessive memory copies, or synchronization points; \(\lambda\) balances the two. The compiler invests compute budget during lowering and just-in-time stages to evaluate \(\mathcal{L}\) heuristically (via analytic models or profiling kernels) and to enforce equivalence constraints so that \(\pi\) still computes the same tensor outputs.

### Symbolic superoptimization and equivalence

Symbolic superoptimization introduces a third party—the e-graph—that represents a huge graph of equivalent programs. Prism (Anonymous et al. 2026) [arxiv:??](https://arxiv.org/abs/2601.04518) showed how to encode every operator fusion candidate as a rewrite on symbolic expressions of tensor algebra, then use equality saturation to find the lowest-latency expression. Instead of enumerating kernels, Prism accumulates all equivalent representations of a subgraph, proves their equivalence via rewriting rules (i.e., \( \text{SwiGLU}(x) \) is equivalent to \(\text{GELU}(\dots)\) under certain weight arrangements), and then extracts the cheapest expression under a cost function calibrated to hardware latencies. Because every rewrite maintains lane correctness, the compiler can risk aggressive fusions that human engineers would avoid for fear of subtle bugs.

This equivalence also permits the compiler to treat substitution as a search problem. Suppose a block contains \( \text{RMSNorm}(\text{Linear}(x)) \) followed by \(\text{SwiGLU}(\text{Linear}(x))\). The e-graph can represent the two linear layers as a single fused load of \([W_1; W_2]^\top x\), and it can swap the order of operations if the cost function \(\mathcal{C}\) predicts improved cache reuse. The compiler then lowers the chosen expression into either Triton kernels or CUDA code by emitting loops that interleave reads, writes, and reductions according to the fused operations.

### Reinforcement learning on compiler passes

The space \(\mathcal{P}\) is too large for brute-force enumeration. DeCOS (Anonymous et al. 2025) demonstrated that reinforcement learning can bootstrap this search by training agents on compiler pass graphs. Each state is a partial IR, each action is a transformation (e.g., "merge these two matmuls" or "transpose the tensor layout"), and the reward is the improvement in estimated latency after simulation. The agent uses an LLM policy to generate actions, so it generalizes better to unseen shapes, and it learns a curriculum that reduces cold-start latency by reusing knowledge from earlier compilations. The RL loop also collects features from hardware profilers—register usage, tensor memory footprints, and transfer times—to adjust rewards on the fly. This approach converts what was once a static pipeline of passes into a non-stationary solver that co-optimizes the compiler graph with execution measurements.

To ground that in actual compiler infrastructure, consider SimpleFSDP (Wang et al. 2024) [arxiv:2411.00284]. It rethinks fully sharded data parallelism inside `torch.compile`: rather than applying a fixed ordering of bucketed collectives, TorchInductor exposes IR nodes grouped by similar data movement patterns and lets the runtime reorder them to overlap compute and communication. This is achieved by augmenting each IR node with metadata about tensor residency (device, memory slices, sparsity) and by letting the optimizer treat communication schedules as part of the search plan \(\pi\). When a gradient bucket is ready, the scheduler can delay all-reduce until the preceding fused kernel has also resolved its writes, hiding latency behind the compute-heavy phase. The advantage is twofold: first, runtime measurements determine whether to split or merge buckets, and second, the search criterion includes both gradient fidelity (to avoid stale copies) and latency, so the compiler learns which buckets should overlap.

### Runtime co-design and tensor layouts

Operator fusion alone is insufficient; the compiler must also plan how tensors live in memory. Heterogeneous accelerators have different grid shapes, warp sizes, and scratchpad availability. The compiler therefore models memory layout transforms as part of the plan \(\pi\). For example, a layout change from NHWC to NCHW might reduce contiguous reads at the expense of additional strides when indexing, and the cost term \(\mathcal{C}(\pi)\) approximates the penalty via a lower-dimensional model of cache line reuse. This enables the compiler to decide between fusing an entire LLaMA block and tiling it so that the activation buffer fits within shared memory. When the plan chooses to tile, it simultaneously schedules explicit copies to ensure that each tensor chunk is aligned with the hardware’s preferred granularity.

The runtime co-design plays out at execution: once the compiler emits a fused kernel, the runtime measures the actual latency and feeds it back as an observation to the reinforcement learning controller. If the latency deviates from the prediction because of a change in dynamic batch size or because the tensor layout triggered a bank conflict, the runtime can trigger a recompile with updated cost weights. This makes the compiler capable of responding to workloads that evolve during training or inference, a necessity for generative workloads whose memory footprints vary between prompts.

### From single device to distributed

The optimizer must also schedule communication for sharded tensors. The plan \(\pi\) includes collective operations such as all-reduce and all-gather, and the compiler uses a DAG that interleaves those operations with fused kernels. For instance, merging gradient computation with gradient reduction (cross-layer fusion) can reduce memory bandwidth by keeping gradients in shared memory until the all-reduce launches. SimpleFSDP leverages this by bucketing nodes with similar data movement, so the compiler need not treat communication as a post-hoc scheduling problem; instead, it is intrinsic to the search.

To summarize the mechanism: the compiler explores a search space \(\mathcal{P}\) of fused operators, layout transformations, and communication schedules, evaluates each via a cost model that blends latency and resource usage, enforces correctness through symbolic rewrites, and uses reinforcement learning to prioritize promising moves. The runtime profiles actual execution to guide the next search iteration, closing the loop between compile time and execution time.

## Where the field is now

Compiler optimization is now visible in two arenas. On the research side, DeepResearch-9K (Author et al. 2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152) defines a challenging benchmark of deep-research agents, showing that the only way to keep inference latency tractable across thousands of prompts is to have compilers that dynamically rewrite kernels for each new agent policy. The benchmark’s agents stress tensor layouts, prompting the compiler to treat layout planning as a first-class objective rather than a mechanical prefix. Reinforcement Learning Foundations for Deep Research Systems (Author et al. 2025) [arxiv:2509.06733](https://export.arxiv.org/pdf/2509.06733) surveys how RL agents can learn compiler policies, highlighting that the reward shaping must capture both throughput and correctness of gradient accumulation in multi-stage pipelines. Its taxonomy sketches how such agents can be bootstrapped via LLMs when scarce profiling data is available, which is precisely what DeCOS operationalizes for superoptimization.

On the engineering side, GenAI for Systems: Recurring Challenges and Design Principles from Software to S (Author et al. 2026) [arxiv:2602.15241v1](https://arxiv.org/html/2602.15241v1) provides concrete examples from commercial deployments, documenting how large AI teams co-design compilers with runtimes to trade off latency and memory pressure. The paper reports that orchestration spaces with fused kernels and adaptive communication achieved a 35% speedup on average when compared with static compilation flows that pre-specified operator orderings. It also emphasizes that compiler latency only becomes acceptable when the search is bounded by heuristics learned from prior runs—a technique now standard in production-grade compilers like Triton and TorchInductor. The engineering frontier is about integrating those adaptive compilers into heterogeneous clusters without overwhelming the deployment pipeline: the compiler must continuously retune while the inference service remains online.

## What's still open

Can we extend the compiler’s search to cover tensor layout, operator fusion, and distributed communication topology simultaneously without incurring prohibitive compile-time latency? Each of those knobs lives in a different subspace—layout changes are discrete, fusion candidates have exponential growth, and communication scheduling involves asynchronous collectives—so the main question is whether there exists a unified planner that reasons about all three with a single set of heuristics.

How do we prioritize compiler actions when the best optimization sequence depends on runtime statistics that change mid-training? Existing systems either compile ahead-of-time and sacrifice adaptivity or recompile too often and waste GPU time. A principled way to amortize the cost of recompilation, perhaps via caching and warm-started RL policies, is still missing.

Can symbolic superoptimization be taught to generalize from one hardware platform to another without re-running the search? The current generation of e-graph synthesizers is hardware-sensitive, meaning the entire search must restart when we move from an H100 to a custom MCM module. A transfer learning strategy for symbolic plans would dramatically reduce just-in-time latency.

## Where to read next

If you want to understand the tensor-level decisions that make these optimizations possible, → *operator fusion* <!-- [[operator-fusion]] --> lays out how kernels fuse arithmetic and memory operations in both Triton and CUDA. The engineering counterpart is → *torch compile* <!-- [[torch-compile]] --> which explains how TorchInductor plugs into PyTorch and where its IR node bucket metadata comes from. For the research trajectory that unifies scoring functions with compiler search, → *reinforcement learning for systems* <!-- [[reinforcement-learning-for-systems]] --> covers the RL agents and reward shaping that DeCOS and SimpleFSDP build on.

## Build it

The build proves that even a toy LLaMA block can benefit from compiler-driven fusion: the RMSNorm and SwiGLU pair execute faster and hold less intermediate activation memory once merged via `torch.compile` and a Triton kernel generated on the fly. This is the concrete test of our central claim that compiler search can rescue memory-bound kernels without manual CUDA editing.

**What you're building:** a fused RMSNorm + SwiGLU operator for a single LLaMA transformer block, running on synthetic sequences derived from `wikitext-2-raw-v1`, measured for latency and peak activation memory on a Google Colab T4.

**Why this is valuable:** the build forces you to instrument TorchInductor’s graph, register the fused kernel, and compare latency and memory before/after fusion, proving that the compiler’s synthesis actually buys runtime benefits.

**Stack:**
- **Model:** [meta-llama/Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b) — 803M downloads, we borrow the block configs
- **Dataset:** [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext) — short sequences ideal for toy inference
- **Framework:** PyTorch 2.1 + TorchInductor + Triton 2.0
- **Compute:** Google Colab T4 (16 GB VRAM) — ~2 hours for the compile-and-measure loop

**The recipe:**
1. Install `torch==2.1.1`, `triton==2.0.0`, and `datasets`, then load the LLaMA block config from the HF model to instantiate the projection matrices, SwiGLU gate weights, and RMSNorm parameters.
2. Use `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")` to stream short sequences, tokenize via a lightweight tokenizer (e.g., `tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b")`), and assemble a batch of shape `(batch, seq_len, hidden)` that matches the LLaMA config; store these tensors as `torch.randn` noise if tokenization is too slow.
3. Wrap the original block with `torch.compile` (using `mode="max-autotune"`), profile the graph to identify the RMSNorm→SwiGLU subgraph, and register a Triton kernel that fuses the normalization and SwiGLU gate in one pass, minimizing intermediate buffers.
4. Evaluate latency by running 200 forward passes and recording median per-batch time and peak activation memory via `torch.cuda.max_memory_allocated()`, then re-run the block without the Triton fusion (but still using `torch.compile`) to prove a measurable drop.
5. What you now have is a fused operator kernel, a latency/memory reduction report, and the knowledge of how to plug new kernels into TorchInductor’s search space.

**Expected outcome:** a benchmark table showing the fused kernel’s median latency (target: ≤ 6.5 ms per batch) and memory drop (target: ≥ 12% reduction in peak activation footprint) compared to the unfused baseline.

- **CS student:** On free Colab, reduce the batch size to 1 and the sequence length to 128, then rerun the measurement to see how the fusion benefit changes in the extreme low-latency regime.
- **Applied engineer:** Instrument the fused kernel for export to TensorRT via ONNX and measure a p99 latency target (≤ 12 ms per inference) when the module is served through vLLM’s task queue on L4.
- **Applied researcher:** Hypothesize that swapping the Swish gate for a standard GELU reduces the fusion payoff; run the same recipe with both activations and plot latency vs. peak memory to falsify whether the SwiGLU structure is the true driver.
- **Frontier researcher:** Use the fused kernel as a starting artifact for the open question about dynamic layout optimization: allow the compiler to choose between NHWC and NCHW on the fly and report whether that extra dimension selection closes the remaining latency gap.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*