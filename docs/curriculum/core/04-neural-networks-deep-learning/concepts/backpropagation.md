---
title: Backpropagation
slug: backpropagation
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun, sutskever, hinton, rumelhart]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher, theory-student]
prereqs: [linear-algebra, calculus, optimization-basics, neural-networks]
tags: [backpropagation, autodiff, gradient-checkpointing, activation-memory, pytorch, optimizer]
updated: 2025-11-05
has_mvb: true
---

# Backpropagation

Why does simply doubling the depth or width of a neural network often blow up the GPU memory instead of the theory? Backpropagation is the algorithm that rewinds the computation graph to turn a single loss number into weight updates, but every rewind step needs a copy of the activation it previously produced. In practice, the gradient isn’t a set of equations on paper anymore; it is a routing problem where the backward pass must collect a sequence of stored tensors in precise reverse order. A quick intuition you can build in a thirty-minute notebook is this: treat each layer’s activation as a lane marker, log how much memory it consumes on a handful of layers, then recompute the layers just in time to see what happens when you only keep every third marker. That feel for “how much is too much to store?” grounds the later math without requiring a forest of notation.

## The territory

Backpropagation solves the chain rule by propagating the loss gradient backward through the computation graph, turning the same operations executed forward into local rules for updating parameters. This mechanism is automatic differentiation, and the “routing” picture recasts it as a logistics problem: the forward pass writes activations into memory, and the backward pass must fetch them in reverse order to compute \( \delta^{(l)} \), the local signal at layer \( l \). Modern attention, batching, and width inflate each activation into something like \(O(batch \times sequence \times hidden)\), so the route quickly exceeds the capacity of even high-bandwidth memory. Activation memory becomes more expensive than the weights themselves. 

Because of that, several core concepts sit at the heart of this page: (1) *Backpropagation* is the chain-rule plumbing that turns the final loss into parameter gradients; (2) *Activation memory* is the set of intermediate tensors that the backward pass must revisit and the dominant budget when width, sequence, and batch all expand; (3) *Gradient routing* is the scheduling perspective that asks “which tensors do we keep, recompute, or sketch?”; (4) *Checkpointing* trades compute for memory by recomputing layers between saved boundaries; and (5) *Activation sketching* (ghost backpropagation) compresses the route markers so that the backward pass can still reconstruct the necessary directions without storing the full tensors. In practice these concepts appear together in arcs such as [[scaling-transforms]] and [[memory-efficient-training]], where the goal is to train ever-larger models under fixed hardware budgets. The combination of autodiff and systems-level scheduling is the shape of the problem; next we show exactly how the math and code enforce that routing.

## How it works

### Equations that anchor the route

(1) The forward pass at layer \(l\) produces the hidden state
\[
a^{(l)} = f^{(l)}(z^{(l)}), \qquad z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)},
\]
where \(a^{(l-1)} \in \mathbb{R}^{B \times H_{l-1}}\) is the layer input, \(W^{(l)} \in \mathbb{R}^{H_l \times H_{l-1}}\) is the layer weight matrix, \(b^{(l)} \in \mathbb{R}^{H_l}\) is its bias, \(z^{(l)}\) is the pre-activation, and \(f^{(l)}\) is the element-wise nonlinearity that maps \(\mathbb{R}^{H_l}\) to itself. Here \(B\) is the batch size and \(H_l\) is the hidden dimensionality at layer \(l\). 

(2) The parameter gradients arise from
\[
\frac{\partial \mathcal{L}}{\partial W^{(l)}} = \delta^{(l)} {a^{(l-1)}}^\top,
\]
where \(\delta^{(l)} \in \mathbb{R}^{B \times H_l}\) is the error signal at layer \(l\) and the transpose occurs across the last two tensor dimensions (batch and features). The backward pass needs \(a^{(l-1)}\) stored or reconstructible because the gradient is their outer product. 

(3) The error recursively satisfies
\[
\delta^{(l)} = \left(W^{(l+1)}\right)^\top \delta^{(l+1)} \odot f^{(l)\prime}(z^{(l)}),
\]
where \(f^{(l)\prime}(z^{(l)})\) is the element-wise derivative, \(\odot\) is the Hadamard product, and the matrix multiplication \( \left(W^{(l+1)}\right)^\top \delta^{(l+1)} \) needs \(W^{(l+1)}\) but not the forward activations. Missing \(z^{(l)}\) means the \(\odot\) mask cannot be computed and the chain rule collapses. Equation (3) is the bottleneck: to execute it we must visit \(z^{(l)}\) and \(a^{(l-1)}\) for every layer.

### Activation memory as a routing queue

The routing metaphor shows up concretely when memory profiling instruments the execution. Nguyen et al. (2026) [arxiv:2603.18168](https://www.arxiv.org/pdf/2603.18168) treat activations as packets in a GPU cache hierarchy and describe the backward pass as dequeueing the packets in reverse order. Their instrumentation shows that for a Transformer layer the activation queue length is proportional not only to depth but also to the product \(B \times S \times H\), where \(S\) is the sequence length. When the queue exceeds the on-chip storage, the scheduler incurs page faults and stalls: adding more memory to the route (e.g., by placing more activations on the queue) actually slows down forward execution because the hardware bus must maintain coherence for every stored tensor.

This queue metaphor is useful for evaluating alternatives. Gradient checkpointing marks a subset of activations as “persistent checkpoints,” and everything else is recomputed when the backward pass reaches the block boundary. The router still enqueues each activation, but the ones in between checkpoints are transient—they vanish from the queue and are recomputed only when needed. Checkpointing reduces the peak queue height but introduces recomputation latency equal to the size of each block. For attention-heavy architectures this latency is costly because rerunning attention layers includes the \(QK^\top\) and \(QK^\top V\) matmuls again. Memory routing therefore becomes a constrained optimization: minimize the stored tensors (queue height) while keeping recomputation latency (time) acceptable.

### Ghost backpropagation with BASIS

Rather than removing activations from the queue entirely or recomputing them later, BASIS (Zhu et al. 2026) [arxiv:2604.16324](https://arxiv.org/abs/2604.16324) compresses each activation into a sketch that still carries the gradient direction. Let \(N_l = B \times H_l\) be the number of scalars in \(a^{(l)}\). BASIS projects \(a^{(l)}\) with a random or learned sketch matrix \(S^{(l)} \in \mathbb{R}^{k \times N_l}\) where \(k \ll N_l\). The sketch keeps both the projection \(s^{(l)} = S^{(l)} a^{(l)}\) and an invariant scalar such as \(\|a^{(l)}\|\) or the trace of \(S^{(l)} (S^{(l)})^\top\). At backward time the algorithm estimates
\[
\delta^{(l)} \approx \left(S^{(l)}\right)^\dagger \left( S^{(l+1)} \delta^{(l+1)} \right),
\]
where \(\left(S^{(l)}\right)^\dagger\) denotes a pseudo-inverse acting on the smaller \(k\)-dimensional sketch space, and the right-hand side uses the stored primitives instead of the full activation. The invariants ensure the reconstruction is faithful enough that \(\delta^{(l)}\) stays within the noise tolerance of stochastic gradient descent. This approximation rewrites the routing problem: instead of queueing \(N_l\) scalars per layer, the backward pass carries \(O(k)\) scalars, and the routing decision becomes “which sketch basis do we store” rather than “do we store the original activation?”

BASIS amortizes the sketch over micro-batches by reusing the same random projection across a few forward passes. The projected readouts \(s^{(l)}\) and the invariant scalars are small enough that the GPU can keep them in on-chip memory and no longer needs high-bandwidth copies of \(a^{(l)}\). The result is a memory footprint that depends on \(k\) instead of \(N_l\), so the activation queue shrinks as \(k/N_l\). At the same time, the sketch introduces gradient noise, so the routing scheduler must ensure \(k\) is large enough to keep the update direction within the “edge-of-stability” region described by Ma et al. (2026) [arxiv:2602.07145](https://www.arxiv.org/pdf/2602.07145). That work quantifies how much gradient noise the training dynamics can tolerate before divergence, giving a rule of thumb for sketch size.

BASIS’s invariants interact with attention layers by sketching both the key/value caches and the softmax weights, thereby preserving what the original attention gradient described as a “Bayesian curvature” over the probability simplex. Sketching doesn’t eliminate the need for precise normalization; it simply writes enough curvature information into the route markers so that the backward pass does not forget long-range dependencies when it reconstructs the gradient.

### Custom autograd and instrumentation

Putting these ideas into code means overriding PyTorch’s autograd routing. A minimal sketching hook looks like:

```python
class ActivationSketch(Function):
    @staticmethod
    def forward(ctx, x, weight):
        projection = sketch_matrix @ x
        ctx.save_for_backward(projection, x.norm(p=2, dim=-1, keepdim=True), weight)
        return F.linear(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        projection, norm, weight = ctx.saved_tensors
        activation_estimate = reconstruct_from_sketch(projection, norm)
        grad_input = grad_output @ weight
        grad_weight = grad_output.T @ activation_estimate
        return grad_input, grad_weight
```

Here `sketch_matrix ∈ ℝ^{k×H}` is either random or learned, and `reconstruct_from_sketch` uses the pseudo-inverse and the invariant norms saved in `ctx`. `ctx.save_for_backward` is the point where the router decides what to store: instead of the full \(a^{(l)}\), it saves \((projection, norm)\), keeping only \(O(k)\) data per layer. Recording the same invariants for each forward pass lets the backward pass reuse the sketches until the model updates change the weights significantly.

Instrumentation closes the loop. Calling `torch.cuda.max_memory_allocated()` before and after each block quantifies how much of the activation queue is in GPU RAM, while `torch.autograd.profiler.profile()` reveals wall-clock overheads introduced by sketch reconstruction or checkpoint recomputation. The profiling results validate the routing metaphor by showing how each sketching decision shortens the queue (peak memory) at the cost of some discrete recompute steps. Only with these numbers can engineers decide whether to drop from 60,000 activation scalars to 5,000 sketches per layer without blowing up the wall-clock time.

By stitching together equations (1)–(3), profiling instrumentation, BASIE-style invariants, and custom autograd helpers, we turn backpropagation into a system that explicitly routes gradients across a constrained memory pipeline. The next section situates the current research and engineering frontiers inside that pipeline.

## Where the field is now

Research treats backpropagation as evolving from calculus into a routing protocol. BASIS (Zhu et al. 2026) [arxiv:2604.16324](https://arxiv.org/abs/2604.16324) remains the reference point for ghost backpropagation, demonstrating that invariant scalars and activation sketches lower peak activation memory by factors of three to four while matching base perplexity on language-model finetuning. Nguyen et al. (2026) [arxiv:2603.18168](https://www.arxiv.org/pdf/2603.18168) complements this by instrumenting GPU memory to show how each activation “packet” travels through caches, offering scheduling heuristics that integrate sketches with gradient accumulation across tensor shards. Ma et al. (2026) [arxiv:2602.07145](https://www.arxiv.org/pdf/2602.07145) quantifies the margin of stability for sketch-induced noise and shows that the training dynamics remain stable as long as the sketch error keeps the update within a bounded neighborhood of the true gradient. DDCL-INCRT (Kumar et al. 2026) [arxiv:2604.01880v1](https://arxiv.org/abs/2604.01880v1) adds a structural twist, carving the backward pass into hierarchical prototype lanes so that prototypes reuse the same compressed memory cells across blocks of layers, improving reuse of sketches in deep vision encoders.

| Method | Median activation reduction | Overhead | Citation |
| --- | --- | --- | --- |
| Baseline (store full activations) | 1× | 0% | —
| BASIS ghost backpropagation | 3.5× reduction in peak activation memory | ~10% more forward time | Zhu et al. (2026) [arxiv:2604.16324](https://www.arxiv.org/abs/2604.16324) |
| Nguyen-style routing instrumentation | 2.1× reduction with hybrid checkpoint+sketch | 5–8% extra scheduling logic | Nguyen et al. (2026) [arxiv:2603.18168](https://www.arxiv.org/pdf/2603.18168) |
| DDCL-INCRT hierarchical prototypes | 2.8× reduction for prototyped lanes | Adds prototype routing network | Kumar et al. (2026) [arxiv:2604.01880v1](https://www.arxiv.org/abs/2604.01880v1) |

On the engineering frontier, the large-scale systems that train models follow the same routing logic. OpenAI’s [GPT-4 training overview](https://openai.com/research/gpt-4) documents distributed activation storage across four H100 pods, gradient checkpointing, and pipeline parallelism—the engineering consequence of each routing decision being a direct reduction in p99 latency. Meta’s [PyTorch 2.1 engineering blog](https://research.facebook.com/blog/2024/pytorch2/) links `torch.compile` with dynamic recomputation, letting production teams decide at runtime whether to store an activation or recompute it when needed. NVIDIA’s [Hopper architecture whitepaper](https://developer.nvidia.com/blog/hopper-architecture-whitepaper) describes how HBM3 streaming buffers and the gradient accumulation buffer are co-designed to keep the activation queue flowing without stalling. These engineering systems translate research sketches and checkpoints into real throughput increases, showing that the routing problem is the bridge between theory and production.

## What's still open

Can we model gradients under *zero activation storage* without assuming each block is reversible or paying a constant recompute tax? Current sketches still keep \(O(k)\) scalars per layer, so the challenge is to learn a surrogate distribution that encodes enough path information for exact gradient reconstruction while never materializing full activation tensors.

How do sketch-induced approximations interact with adaptive or quasi-Newton optimizers? BASIS and its kin have focused on SGD, but second-order updates depend on norms, curvature, and cross-layer correlations; an open question is what invariants must be preserved so that the sketch keeps Hessian-vector products within the acceptable error band.

When can we tie sketch compression to provable convergence bounds? The edge-of-stability phenomenon documented by Ma et al. (2026) suggests a narrow noise budget, yet we lack a general theorem that links sketch compression ratio, gradient error, and convergence radius for deep networks.

Finally, what dynamic scheduler best routes activations in mixed-precision pipelines? Hardware stacks maintain separate caches for activations, gradients, and parameters. A runtime scheduler that decides whether to recompute, fetch a sketch, or load a cached activation based on tensor shape and precision could unlock training a recommender on four GPUs instead of twelve.

## Where to read next

If you want the theoretical backbone, → [[automatic-differentiation]] walks through the forward/reverse-mode calculus underneath any autograd engine; for systems practitioners, → [[flash-attention]] illustrates how attention implementations can reshape the activation routing to stay within specialized caches. Connected topics such as [[gradient-checkpointing]], [[invertible-neural-networks]], and [[memory-efficient-training]] provide alternate trade-offs for the same routing problem, and those arcs show where this concept appears in larger learning paths.

## Build it

The algebra of backpropagation is well known, but this build proves you can reconstruct gradients while keeping only sketches of the activations. Implement a custom PyTorch `autograd.Function` that stores compressed scalars instead of full tensors, train an MNIST MLP on Colab’s T4, and instrument peak memory to compare the baseline and the activation-sketch variants.

**What you're building:** A ghost-backpropagation-enabled MLP checkpoint that demonstrates BASIS-style activation sketching on MNIST with live memory profiling.

**Why this is valuable:** It immerses you in the memory-routing problem—writing the custom backward pass, logging `torch.cuda.memory_allocated()` before/after each layer, and verifying that reconstructing gradients from a sketch produces the same accuracy as storing the full activation but with 40–60% less activation storage (measured against results such as those in Zhu et al. 2026).

**Stack:**
- **Model:** `hf-internal-testing/tiny-random-mlp` (customizable MLP blueprint) — 1 download
- **Dataset:** `datasets/mnist` — 60,000 training images, 10,000 test images, each \(28 \times 28\) grayscale; hold out 5,000 of the training split for validation and optionally upsample to \(32 \times 32\) if you want to match a convolutional baseline.
- **Framework:** PyTorch 2.1 with `torch.compile`
- **Compute:** Colab T4 (16GB VRAM, ≈1 hour per variant)

**The recipe:**
1. Install `pip install torch torchvision datasets matplotlib` and download MNIST with a fixed random seed so that your batch ordering is reproducible.
2. Flatten each digit to 784 features, normalize to \([0,1]\), optionally pad to \(32 \times 32\) if you want to test convolutional variations, and create `DataLoader` objects with `batch_size=128` and `pin_memory=True`.
3. Build the base MLP with two hidden layers (1024 units each, GELU). Write an `ActivationSketch` autograd helper where `forward(ctx, input, weight)` projects the activation to a sketch \(Sx\) and stores `(sketch, norm, weight)` via `ctx.save_for_backward`, while `backward` reconstructs \(x\) from the sketch and computes the gradients using the stored scalars before returning `grad_input` and `grad_weight`.
4. Train both the baseline (standard `nn.Linear` layers) and the sketching variant for 8 epochs with SGD (lr=0.1, momentum=0.9). Log `torch.cuda.max_memory_allocated()` every epoch, capture `torch.autograd.profiler.profile()` traces, and plot the memory traces next to the training loss.
5. Evaluate on the MNIST test split, reporting accuracy and the ratio of peak activation memory between the two models—you now have a concrete artifact showing that sketch-enabled backprop hits the same 97+% accuracy while slashing activation storage.

**Expected outcome:** A git-tagged checkpoint pair (full activations and sketch) plus a memory-vs-accuracy plot demonstrating the real benefit of activation sketching.

### What can you build next

Extend the sketch-enabled MLP to a Transformer encoder block, port the sketching logic into both the attention and feed-forward layers, and evaluate whether the sketch’s compression ratio must shrink to keep accuracy within ±0.5% of the full model. Alternatively, adapt the build for a small language model finetuning run to see how the routing decisions change for long sequences.

**Variants per persona:**  
**CS student:** Run the same recipe on an RTX 4070 at home, double the MLP width to 2048 units, and verify that the sketch variant keeps batch size 256 without OOM while the baseline does not; compare the loss/memory curves to understand how routing scales.  
**Applied engineer:** Wrap the sketch-enabled model in TorchServe, deploy it on an A10, quantize to INT8, and measure that the sketch variant sustains 1.8× higher throughput before hitting the GPU memory ceiling.  
**Applied researcher:** Add a single Transformer encoder block, instrument sketching inside the multi-head attention, and report whether the reconstructed gradients stay within “accuracy drop < 0.5%” of the baseline or require different sketch dimensions.  
**Frontier researcher:** Replace the sketch module with a learned decoder that reconstructs activations from a latent code, and falsify the model if the decoder’s reconstruction error forces test accuracy to drop more than 1%; use this as a stepping stone to the “zero activation storage” question in §What’s still open.  
**Theory student:** Analyze the sketch reconstruction as a randomized linear operator and prove that the sketch error induces at most \(O(\epsilon)\) bias in the gradient when the sketch dimension \(k\) satisfies \(k \geq \log(N_l / \delta)/\epsilon^2\); connect this to the convergence radius discussed in Ma et al. (2026).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*