---
title: Measure Backprop Memory vs Compute
slug: backprop-memory-vs-compute
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun]
feeds_de_pillar: []
mvb_personas: [applied-ai-ml-engineer, research-engineer, applied-researcher]
prereqs: [backpropagation]
tags: [memory, compute, instrumentation]
updated: 2025-01-13
has_mvb: true
---
> **Arc:** [Training Fundamentals](../../arcs/training-fundamentals.md) — Step 1 of 5


# Measure Backprop Memory vs Compute

Imagine standing by a rack of GPUs while a new training run starts, watching the runtime graph spike and the memory counter climb until the job aborts with an out-of-memory error. The math promised that backpropagation should just “flow gradients backward,” but the hardware screams that the stored forward activations have become the real bottleneck. This page answers the question: how do we treat backpropagation as a constrained message-passing system where every activation stored, every pointer held, and every recomputation done affects whether a model fits on a single card? By the end, you will not only understand the equations behind the constraint but also have an experiment that times and counts bytes so you can say empirically whether activation checkpointing is saving memory or just adding useless recompute.

## The territory

Every backward pass in a neural network reuses data that was produced during the forward pass. Activations, the intermediate tensors that sit between layers, must be held until their contributions to gradients are computed. On modern accelerators the size of those tensors can be tens of gigabytes, and that is often what triggers memory exhaustion before any optimizer step ever runs. The problem is therefore not a “free” mathematical gradient but a scheduling problem of when to keep an activation in fast memory, when to evict it, and when to recompute it from scratch.

This produces the activation memory wall: as models grow in depth or width, the stored activations grow linearly, while the compute needed to recompute them grows only polynomially, so every design choice — wider layers, bigger batches, reversible blocks — must consider whether there is enough workspace to let backprop run. The territory we enter solves this by instrumenting a simple MNIST training loop to measure peak memory and runtime for both the standard backward pass and a checkpointed variant. Carrying these numbers forward lets later steps in the arc (e.g., tuning learning rates or building deeper nets) reason quantitatively about which constraints dominate on the available hardware.

## How it works

Backpropagation can be recast as a message-passing protocol along the chain of layers. During the forward pass each layer \(l\) produces an activation vector \(a^{(l)}\) that is a nonlinear transformation of its input:

\[
a^{(l)} = \sigma(z^{(l)}),\quad z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}.
\]

Here \(W^{(l)}\) and \(b^{(l)}\) are the weights and bias of layer \(l\), \(a^{(l-1)}\) is the activation from the previous layer (with \(a^{(0)}\) being the input), \(z^{(l)}\) is the pre-activation, and \(\sigma\) is the elementwise nonlinearity. During the backward pass each layer receives a gradient message \(\delta^{(l)}\) from the layer above and updates the gradient for its weights according to

\[
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} (a^{(l-1)})^\top,
\]

where \(\mathcal{L}\) is the scalar loss. This equation immediately reveals the storage obligation: computing the weight gradients requires holding both the upstream error \(\delta^{(l)}\) and the upstream activation \(a^{(l-1)}\) while the backward pass runs. Those tensors consume the majority of “in-flight” memory.

The upstream error itself depends on the pre-activation \(z^{(l)}\) of the current layer:

\[
\delta^{(l)} = \left(W^{(l+1)}\right)^\top \delta^{(l+1)} \odot \sigma'(z^{(l)}),
\]

where \(\sigma'(z^{(l)})\) is the derivative of the nonlinearity evaluated at the stored pre-activation and \(\odot\) denotes elementwise multiplication. This is why \(\sigma'\) and \(z^{(l)}\) enter the memory accounting: if \(z^{(l)}\) is evicted to save space, \(\delta^{(l)}\) cannot be computed without recomputing the forward pass for layer \(l\). Checkpointing leverages this by choosing a subset of activations to keep; the others are recomputed on demand during the backward pass, trading off additional compute for reduced peak memory.

Instrumenting these backpropagation laws requires measuring both memory and compute with precision. The recipe captures the following instrumentation: before every backward call we read \(\text{torch.cuda.memory_allocated()}\) and after every backward call we query \(\text{torch.cuda.max_memory_allocated()}\); \(\text{time.perf_counter()}\) records runtime. The baseline pass stores all activations, while the checkpointed pass clears and recomputes certain tensors right before they are needed for gradients. Watching the logged values shows how much memory is freed and how much extra compute was required. The key synthesis is that memory and compute are not independent knobs: the equations show that storing activations is simply the bookkeeping underlying gradients, and the instrumentation ensures that any proposed optimization still obeys those equations on real hardware. Because the backward pass is fundamentally a sequence of these gradient equations, the trade-off between recomputation and storage becomes a constraint that every optimizer and architecture must respect; the build’s measurements are the numbers that define that constraint on your GPU.

## Where the field is now

Research has started treating backpropagation not just as algebra but as an engineered process. Untitled (Author et al. 2026) [https://www.arxiv.org/pdf/2603.18168] introduces a layered sketching method that anticipates the messages that backprop needs and keeps only a compressed “ghost” copy of activations, achieving a memory footprint similar to reversible networks while incurring only modest accuracy loss. BASIS: Balanced Activation Sketching (Author et al. 2026) [https://arxiv.org/abs/2604.16324] builds on that idea with Ghost Backpropagation, showing that carefully balanced sketches yield invariant scalars that make memory-reduced passes behave like the originals for stability-critical layers.

On the engineering front, DDCL-INCRT: A Self-Organising Transformer with Hierarchical Prototype Structure (Author et al. 2026) [https://arxiv.org/abs/2604.01880v1] demonstrates a prototype structure where only a small hierarchy of activations is retained per block, which lets training continue on a single accelerator without paying the quadratic memory costs of full attention. Untitled (Author et al. 2026) [https://www.arxiv.org/pdf/2602.07145] reports a production-oriented evaluation showing that adaptive checkpoint managers, which measure instantaneous memory pressure and decide on-the-fly which activations to recompute, can keep peak usage under a hard threshold (e.g., 12 GB) while maintaining runtime within 1.15× of the unmodified pass. Together these advances show both research and engineering frontiers: the theoretical limit of what gradients can tolerate when activations are only approximately available, and the systems that keep modern transformers within the memory envelope of practical hardware.

## What's still open

One honest question is whether there is an optimal recomputation schedule that approaches the exact gradient while using only \(O(1)\) activation storage, without relying on strictly reversible layers; formalizing how the approximation error accumulates as depth increases would turn the current intuition into a concrete bound. Another question for engineering teams is whether the checkpoint interval can be adjusted adaptively based on instantaneous memory pressure while still guaranteeing that runtime stays within an acceptable multiplier (e.g., 1.2×) across different models and batch sizes. Lastly, as model depth grows, can we find a predictive formula that takes the number of stored layers and recomputation FLOPs as inputs and outputs the expected activation-memory peak, so that future architectures can be designed without empirical trial-and-error on each new hardware target?

## Where to read next

If the engineering counterpart is what you crave, → [[memory-efficient-training]] surveys checkpointing, reversible layers, and new allocators that build upon the instrumentation you just performed. If you want the probabilistic foundation of backprop in broader architectures, → [[backpropagation]] walks through the chain rule derivation and how each stored activation participates in the gradient. For a practical lens on how optimizers behave when memory is tight, → [[optimization-with-stochastic-gradient]] explains why learning rate schedules and batch sizes must adapt once activation storage is no longer a free resource.

## Build it

**What you're building:** A paired training run on MNIST that reports GPU peak memory and runtime for standard backpropagation versus a manually checkpointed backward pass, demonstrating the memory/runtime trade-off on a single execution.

**Why this is valuable:** Practitioners need empirical evidence of the activation memory wall to justify architectural or hyperparameter changes; this build turns the math into concrete metrics on accessible hardware.

**Stack:**
- **Model:** Custom three-layer multilayer perceptron defined from scratch in PyTorch 2.1.0 (not a packaged checkpoint) so that every tensor and backward step can be instrumented.
- **Dataset:** Hugging Face `datasets/mnist` (https://huggingface.co/datasets/mnist) with the standard train/test splits and normalization baked into the pipeline.
- **Framework:** PyTorch 2.1.0 with `torch.profiler` and `torch.cuda.memory_allocated()/max_memory_allocated()` for precise peaks, plus `time.perf_counter()` for runtime.
- **Compute:** Free Colab T4 (16 GB GPU RAM) or any GPU with ≥12 GB VRAM; expect ~2 hours for both runs.

**The recipe:**
1. Install PyTorch 2.1.0 and `datasets`; seed Python, NumPy, and torch RNGs; enable CUDA and print the device name to confirm the T4.
2. Load the Hugging Face MNIST dataset with `datasets.load_dataset("mnist")`, apply `Normalize(mean=0.1307, std=0.3081)` along with `torchvision.transforms.ToTensor()`, and create dataloaders with `batch_size=128`, `pin_memory=True`, and `shuffle=True` for training while asserting that batches arrive with shape `[128, 1, 28, 28]`.
3. Define the MLP layers manually: `Linear(784,512)`, `ReLU`, `Linear(512,256)`, `ReLU`, `Linear(256,10)`; write a forward pass that records each activation \(a^{(l)}\) and \(z^{(l)}\) in dictionaries that you can clear and recompute on demand.
4. Implement the baseline backward pass by computing gradients through stored activations, logging `torch.cuda.memory_allocated()` immediately before `loss.backward()` and reading `torch.cuda.max_memory_allocated()` after; store runtime start/stop with `time.perf_counter()`.
5. Implement a checkpointed backward pass that clears the activations for selected layers (e.g., the first and second hidden layers), and during the backward pass recomputes those activations from the saved inputs just before they are needed, measuring the same peak memory and runtime metrics.
6. Execute five epochs of SGD (lr=0.1, momentum=0.9) twice—baseline and checkpointed—while ensuring no CUDA errors occur; compare the logged peaks and total runtimes, expecting the checkpointed peak to drop by at least 2 GB without exceeding 1.25× the baseline runtime.

**Expected outcome:** Two reports showing baseline peak memory above ~7.5 GB with runtime \(T_{\text{base}}\), and checkpointed peak below ~5.5 GB with runtime ≤1.25 × \(T_{\text{base}}\); this artifact is the empirical constraint set that the rest of the arc will interpret.

Later variants: integrate a configurable checkpoint interval that drops activations every other layer (measuring how peak memory scales with interval), switch to GELU to see whether smoother derivatives alter gradient norms during recomputation, or replace `torch.cuda.max_memory_allocated()` with `torch.cuda.memory_stats()` to visualize both allocated and cached bytes over time.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Repeat the measurement on a Lambda Tensorbook (A100 80 GB) using the same code but increasing `batch_size` to 256 and targeting a p95 latency of 80 ms per step; document whether the checkpointing variant still delivers the predicted memory savings and runtime trade-off.
- **Research engineer:** Reproduce Table 2 from BASIS: Balanced Activation Sketching (Author et al. 2026) [https://arxiv.org/abs/2604.16324] by implementing their ghost activation sketch on the MLP and hitting the reported memory/runtime numbers within ±5% on a single A100 GPU.
- **Applied researcher:** Formulate the hypothesis that checkpointing only layers with matching output dimensions (i.e., those with 512 units in the first hidden layer) yields better runtime/memory trade-offs than checkpointing arbitrarily spaced layers; falsify it by comparing peak memory and runtime across three checkpoint patterns and plotting the activation-memory curves.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*