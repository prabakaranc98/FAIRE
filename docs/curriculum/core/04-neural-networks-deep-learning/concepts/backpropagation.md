---
title: Backpropagation
slug: backpropagation
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [rumelhart, werbos, lecun, hinton, widrow]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher, curious-learner, theory-student]
prereqs: [gradient-descent, neural-networks-intro, calculus-chain-rule]
tags: [backpropagation, autodiff, dynamic-programming, gradient-based-learning, neural-training]
updated: 2025-01-15
has_mvb: true
---

# Backpropagation

How can a single scalar loss signal at the very end of a computation tell every earlier layer how to adjust itself, even when millions of parameters lie between the input and the output? Humans do this by passing blame backwards—every stage on the production line hears whether its contribution helped or hurt—while neural networks rely on backpropagation to do the same. The algorithm reverses execution, reuses cached activations, and performs a constant amount of work per layer or timestep rather than re-running the entire network from scratch; that reuse is what turns the chain rule into a dynamic-programming engine that makes modern deep and sequential training possible.

## The territory

Backpropagation sits at the heart of gradient-based training because the scalar loss lives only at the very end of the composition of many affine maps and nonlinearities. The challenge is structural: most parameters appear in intermediate layers, so computing \(\partial L / \partial W\) naïvely would require replaying the entire network for each parameter. Once the forward activations and pre-activations are cached, though, the recomputation vanishes, and the chain ruling through stored partials becomes a linear backward sweep. Werbos (1990) [https://axon.cs.byu.edu/~martinez/classes/678/Papers/Werbos_BPTT.pdf] made the temporal version of this concrete by “unrolling” recurrent dynamics, proving that the same reverse pass works when time becomes depth. LeCun (1988) [https://new.math.uiuc.edu/MathMLseminar/seminarPapers/LeCunBackprop1988.pdf] and Plaut, Nowlan, and Hinton (1986) [https://www.cnbc.cmu.edu/~plaut/papers/pdf/PlautNowlanHinton86TR.backprop.pdf] later recast the procedure as constrained optimization, introducing auxiliary activations and multipliers so that minimizing the augmented Lagrangian is equivalent to executing the backward pass. Widrow (1992) [https://www-isl.stanford.edu/~widrow/papers/c1992backpropagationand.pdf] translated that same math to analog circuitry, showing that every weight update depends only on the two nodes it connects when each synapse locally stores one forward activation and the backward delta.

### Where this concept appears

Backpropagation appears wherever a long loss–parameter chain demands dynamic programming. It is the training kernel in the classification and regression arcs, the backward sweep inside transformer stacks such as the [[04-neural-networks-deep-learning/transformers]] arc, the optimizer driver for the [[04-neural-networks-deep-learning/recurrent-networks]] arc, and the constraint-aware training loop described in [[04-neural-networks-deep-learning/hardware-aware-training]]; each of those arcs relies on backpropagation’s cached activations, whether the computation runs on GPUs, inference chips, or energy-constrained analog accelerators. Having laid out the problem and where it shows up, the next section explains exactly how the backwards recurrence recovers every gradient while walking only once through the cached states.

## How it works

### Forward pass, storage, and notation

Every neural net can be written as a sequence of layers \(1 \leq l \leq L\), where layer \(l\) maps input activation \(x^{(l-1)} \in \mathbb{R}^{d_{l-1}}\) to output activation \(x^{(l)} \in \mathbb{R}^{d_l}\) via a weight matrix \(W^{(l)} \in \mathbb{R}^{d_l \times d_{l-1}}\), bias \(b^{(l)} \in \mathbb{R}^{d_l}\), and nonlinearity \(f^{(l)}\):
\[
z^{(l)} = W^{(l)} x^{(l-1)} + b^{(l)}, \qquad x^{(l)} = f^{(l)}(z^{(l)}),
\]
where \(z^{(l)} \in \mathbb{R}^{d_l}\) is the pre-activation whose components each depend on the layer’s parameters and the previous activation \(x^{(l-1)}\). The scalar loss \(L\) depends on the final activation \(x^{(L)}\) and optionally a label \(y\), \(L = \ell(x^{(L)}, y)\). Every \(z^{(l)}\) and \(x^{(l)}\) is cached during the forward pass because the backward pass needs local partials such as \(\partial x^{(l)} / \partial z^{(l)} = \operatorname{diag}(f^{(l)\prime}(z^{(l)}))\). Without caching, computing \(\partial L / \partial W^{(l)}\) would require replaying layers \(l\) through \(L\) for each parameter, leading to an exponential blow-up; caching makes it a linear, constant-factor pass that visits each layer once in backward order, which is why the chain rule becomes dynamic programming.

### Reverse pass and gradient equations

The backward sweep propagates the error signal \(\delta^{(l)} \coloneqq \partial L / \partial z^{(l)}\), and the recurrence is
\[
\delta^{(l)} = (W^{(l+1)})^\top \delta^{(l+1)} \odot f^{(l)\prime}(z^{(l)}),
\]
with \(\delta^{(L)} = \nabla_x \ell(x^{(L)}, y)\), \((W^{(l+1)})^\top\) the transpose of a \(d_{l+1} \times d_l\) weight matrix, and \(f^{(l)\prime}(z^{(l)})\) the elementwise derivative vector at the cached pre-activation \(z^{(l)}\). Each layer multiplies the next layer’s delta by the transpose and by the local derivative, reusing the cached values to compute \(\delta^{(l)}\) without re-running layers \(l+1,\dots,L\). With \(\delta^{(l)}\) in hand, the gradients of the parameters are
\[
\frac{\partial L}{\partial W^{(l)}} = \delta^{(l)} (x^{(l-1)})^\top, \qquad \frac{\partial L}{\partial b^{(l)}} = \delta^{(l)},
\]
where \(x^{(l-1)} \in \mathbb{R}^{d_{l-1}}\) is the cached input to layer \(l\). These formulas illustrate the dynamic-programming intuition: the backward sweep only recomputes each \(\delta^{(l)}\) once, every cached \(x^{(l-1)}\) is used exactly once to compute its incoming gradients, and the cost is proportional to the forward pass.

Digital implementations require that the backward operator mirror the forward weights so that the transpose \((W^{(l+1)})^\top\) is available. Neuromorphic and analog proposals cannot always afford to store both the forward matrix and its transpose, which is the weight transport problem. Feedback alignment, as discussed by Lillicrap et al. (2016) [https://arxiv.org/abs/1509.06461], replaces the transpose with a learned backward matrix; even though this matrix does not equal the forward weights, the error signal \(\delta^{(l)}\) defined above can still emerge because the backward weights learn to align with the forward gradients. To keep this heuristic connected to the dynamic-programming narrative, observe that both the \(\delta^{(l)}\) and the Lagrange multipliers \(\lambda^{(l)}\) that we introduce below represent the same error signal, so any variant—symmetric or not—must still respect the cached activations and the local multiplier to remain consistent.

### Backpropagation as constrained optimization

LeCun (1988) [https://new.math.uiuc.edu/MathMLseminar/seminarPapers/LeCunBackprop1988.pdf] treated each forward equation \(z^{(l)} = W^{(l)} x^{(l-1)} + b^{(l)}\) as an equality constraint and paired it with a Lagrange multiplier \(\lambda^{(l)} \in \mathbb{R}^{d_l}\). The augmented Lagrangian is
\[
\mathcal{L} = \ell(x^{(L)}, y) + \sum_{l=1}^L \lambda^{(l)\top} \left(z^{(l)} - W^{(l)} x^{(l-1)} - b^{(l)}\right),
\]
where each \(\lambda^{(l)}\) has the same dimensionality as \(z^{(l)}\) and the sum runs over layers \(l=1\) to \(L\). Stationarity with respect to \(z^{(l)}\) yields \(\lambda^{(l)} = \partial \ell / \partial z^{(l)} = \delta^{(l)}\), and differentiating with respect to \(W^{(l)}\) and \(b^{(l)}\) reproduces the gradient formulas above. This construction explains why constrained training recipes—safety constraints, hardware energy caps, weight sharing—can still rely on backpropagation: the multipliers enforce consistency between forward dynamics and constraints, and the backward pass updates multipliers and parameters together so the cached activations continue to carry the necessary local information. Plaut, Nowlan, and Hinton (1986) [https://www.cnbc.cmu.edu/~plaut/papers/pdf/PlautNowlanHinton86TR.backprop.pdf] demonstrated this explicitly for shared constraints, showing that error signals traverse shared components exactly as they do individual weights when the multipliers track shared structure.

### Backpropagation through time and memory trade-offs

Sequential models such as recurrent neural networks produce outputs depending on a history of inputs. Werbos (1990) [https://axon.cs.byu.edu/~martinez/classes/678/Papers/Werbos_BPTT.pdf] demonstrated that unfolding the recurrence across timesteps turns the temporal problem into a deep feedforward network: each timestep \(t\) contributes a hidden state \(h_t\) cached during the forward sweep, the loss at final time \(T\) depends on every \(h_t\), and the backward pass propagates deltas from \(\delta^{(T)}\) down to \(\delta^{(1)}\) exactly as in the feedforward case. Caching every \(h_t\) costs memory proportional to \(T\), so truncated BPTT, gradient checkpointing, and reversible architectures trade storage for recomputation while still taking exactly one backward pass over the stored states. Those trade-offs preserve the dynamic-programming core: the cached activations remain the only frontier where recomputation is allowed, so the backward sweep is still linear in the number of (layer, timestep) pairs.

### Backpropagation as the engine of gradient-based learning

Widrow (1992) [https://www-isl.stanford.edu/~widrow/papers/c1992backpropagationand.pdf] mapped backpropagation onto analog multipliers and capacitors, showing that the update \(\Delta W^{(l)} = -\eta \delta^{(l)} (x^{(l-1)})^\top\) depends only on the cached input activation \(x^{(l-1)}\) and the propagated delta \(\delta^{(l)}\), with learning rate \(\eta\). This reinforces the dynamic-programming story: global loss is distilled into local context, every weight update reads two cached values, and the backward sweep—the “reuse of cached activations”—is sufficient to compute the gradient without additional global state.

### Failure modes and practical heuristics

Failure modes arise when cached activations or multipliers become unreliable. Deep networks amplify or shrink deltas when weight matrices have extreme singular values, so careful initialization (Xavier, He) and batch normalization stabilize the variance that travels backward. The weight transport problem resurfaces here because physical systems may not store exact transposes; feedback alignment still works because it learns separate backward weights that produce useful gradients, but that heuristic is only meaningful once one accepts that the multipliers \(\lambda^{(l)}\) and the deltas \(\delta^{(l)}\) are the same error signal. Thinking in terms of multipliers explains why any heuristic must deliver a consistent backward signal that uses cached activations: if \(\lambda^{(l)}\) drifts away from the true \(\nabla_{z^{(l)}} \ell\), the updates no longer respect the Lagrangian stationary conditions, and the dynamic-programming efficiency collapses. This is why designing architectures with local, asymmetric updates remains challenging—the caches are the only place that contains the information needed for consistency, and once they are corrupted, every downstream gradient loses its footing.

## Where the field is now

Recent work still revolves around the dynamic-programming core, but explores how the cached activations interact with generalization and scale. One research strand studies spectral bias and gradient flow. Rahaman et al. (2019) [https://arxiv.org/abs/1901.09491] show that neural networks preferentially fit low-frequency components, so the cached activations carry smooth gradients that evolve slowly across layers; Chizat and Bach (2018) [https://arxiv.org/abs/1712.04771] analyze alternating sparse regimes and gradient flows that revisit cached states, documenting how the stored activations continue to provide structure to guarantee convergence even when smoothness assumptions break down. These analyses keep pointing back to the same dynamic-programming intuition: the backward sweep remains a linear pass no matter how wildly the forward activations behave, because every delta only depends on what was cached directly below it.

### In production

Engineering continues to push backpropagation to trillion-parameter regimes by distributing the caches and gradients while keeping the backward pass linear. DeepSpeed’s ZeRO stage 3 (Rasley et al. 2020) [https://arxiv.org/abs/2004.08799] and the ZeRO sharding implementation in Rajbhandari et al. (2020) [https://arxiv.org/abs/1910.02054] shard parameters, gradients, and optimizer states across ranks so that backward updates never require a single accelerator to hold the entire model or its cached activations; this keeps the backward pass computationally linear even as the caches are scattered. Hugging Face Accelerate’s distributed launch scripts [https://huggingface.co/docs/accelerate/index] and DeepSpeed’s ZeRO optimizer ensure that the gradient accumulation stage reuses the same cached activations as if they lived on one GPU, which keeps the dynamic-programming guarantees intact even when activation and optimizer footprints are squeezed to tens of gigabytes per rank. These production deployments show that backpropagation still delivers both scientific insight and deployment value, bridging the same caching principles from toy networks to huge transformers.

## What's still open

Can local learning rules that avoid symmetric weight transport match backpropagation’s sample efficiency on benchmarks such as ImageNet or LLaMA pre-training while running on analog or neuromorphic chips? The question translates to: can one design multipliers \(\lambda^{(l)}\) that stay within 1% of the baseline accuracy without accessing the exact transpose of \(W^{(l+1)}\)?  
Do adaptive gradient caching strategies—the network dynamically choosing which intermediate activations to store and which to recompute—offer a provable trade-off between memory savings and convergence speed, or does the benefit collapse after a certain depth or nonlinearity class?  
Is there a principled integration of hard constraints (resource budgets, safety invariants) into the Lagrangian formulation so that training enforces them automatically, without painstaking penalty tuning?

## Where to read next

If you want the probabilistic foundation, → [[04-neural-networks-deep-learning/score-matching]] explains how backpropagation gradients approximate the data score function. The engineering counterpart is → [[04-neural-networks-deep-learning/gradient-checkpointing]] for reshaping cached activations when memory becomes the bottleneck, and the systems story continues in → [[04-neural-networks-deep-learning/flow-matching]] where continuous-time dynamics mimic the backward pass over noising trajectories. Connected topics include → [[04-neural-networks-deep-learning/autodiff]] for the computation-graph plumbing, → [[04-neural-networks-deep-learning/optimizer-design]] for how gradients feed parameter updates, and → [[04-neural-networks-deep-learning/hardware-aware-training]] for the constraints that originally motivated the Lagrangian formulation.

## What can you build next

Scale the manual engine from this page into the curriculum’s heavier arcs: → [[04-neural-networks-deep-learning/gradient-checkpointing]] shows how to recompute caches selectively and trace the runtime/accuracy trade-off, → [[04-neural-networks-deep-learning/optimizer-design]] matches the backward gradients to different optimizers, and → [[04-neural-networks-deep-learning/hardware-aware-training]] revisits the Lagrangian story with explicit energy or safety constraints so you learn how cached activations play in constrained deployments.

## Build it

**Artifact:** a lightweight Hugging Face-hosted MLP with manual autograd that trains a simple Iris split, exposing every cached activation and delta.  
**Motivation:** it highlights backpropagation’s dynamic-programming nature while producing a complete model checkpoint you can inspect, perturb, or extend to heuristics like feedback alignment.  

**Stack:**
- **Model:** [`hf-internal-testing/tiny-random-mlp`](https://huggingface.co/hf-internal-testing/tiny-random-mlp) as the reference architecture for exportable tensors and weights.  
- **Dataset:** [`iris` on Hugging Face Datasets](https://huggingface.co/datasets/iris) normalized to zero mean and unit variance so caches stay order-one.  
- **Framework:** Python 3.12 with NumPy 1.26 and scikit-learn 1.4 inside Google Colab (no PyTorch/TensorFlow).  
- **Compute:** single Google Colab T4 (16 GB VRAM) or comparable 10 GB GPU, ~15-minute setup then ~15 minutes of training on the 120/30 train/test split.

**The recipe:**
1. Install `pip install numpy scikit-learn matplotlib datasets` and start a Colab notebook. Implement a `Tensor` class that stores `.data`, `.grad`, `.creator`, `requires_grad`, and a `.backward()` method that performs an explicit stack-based topological traversal: push the loss tensor, pop creators in reverse topological order, accumulate gradients, and avoid recursion limits by using an explicit stack while marking visited tensors.  
2. Load `datasets.load_dataset("iris")`, normalize each of the four numeric features to zero mean and unit variance (using a fixed `np.random.default_rng(seed=42)`), and split deterministically into 120 training samples and 30 test samples so accuracy expectations are reproducible.  
3. Define a two-layer MLP (32 hidden units, tanh activation) via the custom tensor arithmetic. Use manual SGD with learning rate 0.1, batch size 16, and 80 epochs; after each forward pass compute `.backward()` to propagate gradients through the cached activations and update weights immediately so the backward sweep stays tied to new caches.  
4. Evaluate by computing classification accuracy and loss on the static test split after every epoch; expect ~88% accuracy once the training loss dips below 0.1, which reflects the limited dataset size and the deterministic split. Plot the loss curve to show the downstream effect of cached activations.  
5. What you now have: a reusable `Tensor` class, a working `.backward()` pass relying on stack-based traversal, recorded plots of loss and accuracy, and the ability to inspect every cached activation paired with its propagated delta.

**Expected outcome:** a reusable Colab notebook that exports the manual MLP checkpoint, visualizes the model’s decision regions, and lets you inspect how each cached activation contributes to the backward sweep and resulting gradients.

**Variants per persona:**
- **CS student:** halve the hidden size to 16, add dropout before the head, and keep the dataset loading, logging, and forward/backward passes explicit so the notebook fits within a single RTX 4070 session.  
- **Applied engineer:** wrap the notebook into a Dockerized FastAPI service that exposes the manual MLP checkpoints, logs the cached activations for the last prediction, and records the backward sweep that produced the gradients in production-like telemetry.  
- **Applied researcher:** implement two runs—one that caches every activation and one that checkpointing the hidden layer by recomputing it during backward passes—and measure whether the recomputation delays hurt convergence, documenting which layer depths lose accuracy.  
- **Curious learner:** augment the visualization with animated heatmaps that show the cached activations before and after the backward pass for a sample batch, reinforcing the dynamic-programming picture.  
- **Theory student:** derive each partial derivative in the notebook symbolically, compare the analytic gradients to the ones computed by `.backward()` for a subset of parameters, and explain any scaling factors that appear.  
- **Frontier researcher:** replace the backward matrix multiplies with learned feedback matrices updated via a local Hebbian rule and measure whether the resulting model matches the baseline test accuracy within 1 percentage point while never using the exact transpose of \(W^{(l+1)}\).

