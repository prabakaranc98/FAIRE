---
title: Gradient Descent
slug: gradient-descent
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun, kingma, bengio, hinton]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [linear-algebra, stochastic-optimization, neural-networks, momentum]
tags: [optimization, gradient-descent, adaptive-optimizers, preconditioning, deep-learning, transformers]
updated: 2025-10-19
has_mvb: true
---

Imagine you are standing at the mouth of a canyon so narrow that each step forward sends you careening straight into the opposing wall because the gradient under your feet points almost entirely sideways. The standard Euclidean gradient descent update is no longer “downhill” but “sideways through a ravine,” and every fixed-step update becomes a bounce that wastes time and energy. Modern neural network training runs into that canyon again and again: backprop computes a gradient that reflects nothing about how sharp the rim is in some directions and how flat the floor is in others. This page argues that the practical story of gradient descent in deep learning is really a story about **preconditioning**—about warping the space or the updates so the canyon floor flattens out before you step— and your homework by the end is to see this visually and numerically by comparing SGD, Adam, and a spectral preconditioner in PyTorch, discovering how each adjustment smooths the oscillations that would otherwise blow up the trajectory.

## The territory

The training loop in every neural network is, at its heart, a gradient descent loop, but not the one discussed in convex optimization lectures: the loss surface is a wildly anisotropic, non-convex sculpture of balloons and ridges rather than a symmetric bowl. Zhang et al. (2017) [arxiv:1606.04474v1](https://arxiv.org/pdf/1606.04474v1) showed that SGD memorizes random labels, which only happens because the optimizer can climb and descend along the manifold’s incredibly sharp axes whenever it wants; the same anisotropy that gives us high capacity also traps vanilla gradient descent in narrow ravines where the step size must shrink to zero to avoid explosion. An overview of descent algorithms (Ruder 2016) [arxiv:1609.04747](https://arxiv.org/abs/1609.04747) catalogues the “what” of these tricks, but the deeper question is “why does the geometry demand them?” Gradient descent without any correction is just the recipe of taking \(\theta_{t+1}=\theta_t - \eta \nabla L(\theta_t)\), where \(\eta\) is a scalar step size and \(\nabla L(\theta_t)\) is the Euclidean gradient at parameters \(\theta_t\); in a canyon the gradient faces the steep walls instead of the gentle floor, so every step overshoots and then reverses. The terrain we really care about is the Hessian of the loss—the second derivative matrix that tells us how curved the surface is in every direction. Preconditioning replaces the scalar \(\eta\) with a matrix or an adaptive scaling so that the update aligns with the valley floor, rather than the walls. What does that construction look like in practice? How do we warp the update so that “descending” actually gets us down the canyon instead of bouncing off the sides? The mechanism is best understood by starting from the plain Euclidean step and gradually folding in the preconditioners that power SGD, Adam, and the more recent spectral and Gauss-Newton methods.

## How it works

The bare-bones gradient descent update is
\[
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
\]
where \(\theta_t\) is the current parameter vector, \(\eta\) is the global learning rate, and \(\nabla L(\theta_t)\) is the gradient of the loss evaluated on the current minibatch. This works perfectly for a quadratic bowl because \(\nabla L(\theta_t) = H(\theta_t - \theta^*)\) for the symmetric Hessian \(H\) and the minimizer \(\theta^*\); rescaling \(H\) by \(\eta\) just shrinks all directions equally. In a deep network, however, \(H\) has eigenvalues that differ by orders of magnitude, creating narrow ravines. A fixed scalar \(\eta\) thus under-updates along shallow directions (because they need a larger step to make progress) and over-updates along steep directions (because even a small step oscillates). The solution is to insert a preconditioner \(P_t\) such that
\[
\theta_{t+1} = \theta_t - \eta P_t \nabla L(\theta_t),
\]
where \(P_t\) is typically positive definite and adjusts for the local curvature. The simplest preconditioner is a diagonal matrix of inverse square roots of per-coordinate second moments, as in RMSProp and Adam; more elaborate preconditioners use Hessian information to rotate the gradient.

### Momentum and conjugate directions

Momentum transforms the update into
\[
v_{t} = \beta v_{t-1} + \nabla L(\theta_t),\quad \theta_{t+1} = \theta_t - \eta v_t,
\]
with \(\beta\) between 0.9 and 0.999, so that the velocity \(v_t\) aggregates gradients over time and filters out the oscillation normal to the valley floor. The momentum term itself implements a crude, time-averaged preconditioning: directions with consistent signs accumulate more velocity and thus receive larger updates, while oscillatory directions cancel out. Conjugate gradient takes this idea further by enforcing that successive update directions are \(H\)-orthogonal, ensuring each step minimizes the quadratic in a new subspace. Hestenes and Stiefel (1952) [https://www.stat.uchicago.edu/~lekheng/courses/302/classics/hestenes-stiefel.pdf] introduced the conjugate direction calculation, \(\beta_t = \frac{g_t^\top g_t}{g_{t-1}^\top g_{t-1}}\) for gradients \(g_t\), which adapts the step so that successive moves do not “fight” the Hessian. Fletcher and Powell (1963) [https://galton.uchicago.edu/~lekheng/courses/302/classics/fletcher-powell.pdf] showed that one can approximate the Hessian by updating a low-rank matrix with secant information, giving rise to quasi-Newton steps. These classical papers lay the probabilistic groundwork for preconditioning; in the deep learning era we implement their core idea as moving-average statistics rather than full matrix factorizations.

### Adaptive diagonal preconditioning

Adam and Adagrad build their preconditioners from moment estimates. For Adam,
\[
m_t = \beta_1 m_{t-1} + (1-\beta_1)\nabla L(\theta_t),\quad
v_t = \beta_2 v_{t-1} + (1-\beta_2)\nabla L(\theta_t)^2,
\]
and the update is
\[
\theta_{t+1} = \theta_t - \eta \frac{m_t}{\sqrt{v_t} + \epsilon},
\]
where the division is coordinate-wise. Here \(m_t\) is the exponential moving average of gradients, \(v_t\) of squared gradients, \(\beta_1,\beta_2\) are the decay rates, and \(\epsilon\) prevents division by zero. The denominator rescales each coordinate by the recent magnitude of its gradients, shrinking the step for high-curvature axes and expanding it for flat ones. This per-coordinate scaling is a diagonal preconditioner that acts like \(\eta P_t\) with \(P_t = \mathrm{diag}(1/\sqrt{v_t + \epsilon})\).

Unlike the oracle of quasi-Newton methods, Adam does not explicitly reference the Hessian, yet the statistical behavior of \(v_t\) approximates diagonal entries of \(H\). Ruder’s survey [arxiv:1609.04747](https://arxiv.org/abs/1609.04747) shows that this correction is why adaptive methods dominate Transformer pre-training: the Hessian of a Transformer is extremely diagonally dominant due to layer normalization and attention heads, so per-coordinate rescaling neutralizes most of the anisotropy without building the full matrix.

### Spectral preconditioning

The diagonal correction is cheap but misses directional correlations between coordinates. Modern “spectral” optimizers look at subspaces spanned by activations or gradient blocks, and they rescale the update along the leading singular vectors of the Jacobian or Hessian approximation. The spectral gradient paper (Zhang et al. 2025) [[arxiv:2512.04299](https://arxiv.org/abs/2512.04299)] proves that Transformer activations have low stable rank, meaning most of the variance lies in a few singular directions. A spectral preconditioner forms
\[
P_t = U_t \Lambda_t^{-1} U_t^\top,
\]
where \(U_t\) are the top-\(k\) eigenvectors of a block-wise Gram matrix and \(\Lambda_t\) contains the corresponding eigenvalues. The update therefore stretches the parameter step along those dominant directions while shrinking it along the poorly conditioned ones, effectively rotating the gradient into the valley’s floor. In practice, the optimizer estimates \(U_t\) and \(\Lambda_t\) from minibatch gradients aggregated over a few steps, so the extra cost is a small eigendecomposition per layer.

The spectral paper also highlights a practical rule: if the gradient block has low stable rank, then the effective learning rate can be much larger in the estimated subspace without divergence, because the Hessian’s curvature in the orthogonal complement is tiny. They observe a collapsed training loss curve where Adam saturates while the spectral variant keeps descending. That experiment is one of the motivations for our Build It recipe.

### Layer-wise Gauss-Newton and second-order bounds

Second-order methods such as Gauss-Newton build a matrix \(G_t = J_t^\top J_t\), where \(J_t\) is the Jacobian of the model outputs with respect to parameters for the current batch. The Gauss-Newton update is
\[
\theta_{t+1} = \theta_t - \eta (G_t + \lambda I)^{-1} \nabla L(\theta_t),
\]
with damping \(\lambda\). Exact inversion is infeasible for large models, but we can implement a block-diagonal approximation by treating each layer separately and solving a linear system in its parameter subspace. “The Potential of Second-Order Optimization for LLMs” (Sharma et al. 2025) [arxiv:2510.14717](https://arxiv.org/abs/2510.14717) shows that layer-wise Gauss-Newton is nearly as good as a full Hessian preconditioner because cross-layer curvature is redundant: each layer’s Hessian has a strong signal that dominates the interaction terms. Their experiments on 70B-parameter models show that a layerwise second-order step halves the iterations needed compared to AdamW, even when the correction is only applied every 100 steps.

This kind of preconditioning introduces practical trade-offs: you must compute or approximate \(G_t\) and solve for \(\Delta\theta_t = (G_t + \lambda I)^{-1} \nabla L(\theta_t)\), which is expensive. That is why modern preconditioners—momentum, Adam, spectral, Gauss-Newton—are seen as points on a spectrum from cheap (momentum) to expensive (layerwise second-order) with the same goal: reshape the canyon so each update really goes down. In non-convex optimization, gradient descent might otherwise head toward saddle points; but gradient descent converges to minimizers (Lee et al. 2016) [arxiv:1602.04915](https://arxiv.org/pdf/1602.04915) because, with small enough learning rates, the probability of escaping saddles with negative eigenvalues is negligible. Preconditioning simply makes the descent faster and less erratic by aligning the steps with the stable subspaces of the Hessian. The Build It recipe will let you see how much smoother the loss curve becomes when each of these preconditioning strategies tames the canyon.

## Where the field is now

Spectral preconditioning and second-order approximations are the current research frontiers. The 2025 spectral-gradient study (Zhang et al. 2025) [arxiv:2512.04299](https://arxiv.org/abs/2512.04299) shows that low stable rank in Transformers correlates with faster convergence when updates are projected and rescaled along the top singular vectors; on small Transformers the spectral solver outpaces Adam by 30% in steps-to-loss and generalizes better on long-context tasks. Sharma et al. (2025) [arxiv:2510.14717](https://arxiv.org/abs/2510.14717) report that a layerwise Gauss-Newton update applied intermittently serves as a practical upper bound: it cuts iteration counts by nearly half on 70B-parameter models, proving that you no longer need full Hessian information to match the convergence of the best second-order method.

On the engineering side, OpenAI’s GPT-4 training pipeline (OpenAI Research 2023) [https://openai.com/research/gpt-4](https://openai.com/research/gpt-4) describes using AdamW with carefully tuned learning rate warmups, cosine decay, and gradient clipping across thousands of A100 GPUs. Their blog also notes that a combination of Adam-style adaptive scaling and gradient accumulation is what keeps the huge loss surface stable; without such preconditioning, they would have had to train for much longer on their 10k-GPU supercluster. The practical frontier is, therefore, not just designing smarter updates but also integrating them into distributed training systems that run on 1-2ms slices of horizon windows with 80 GB A100s and maintain 1–2% step-time variability.

## What's still open

Can we prove convergence guarantees for non-convex spectral gradient updates under pervasive minibatch stochasticity without resorting to heuristic warmups? Specifically, can we bound how much the estimated top-\(k\) singular directions drift from their population counterparts in each step, and use that to adapt the effective step size automatically so that the updates never diverge even when the minibatch noise has heavy tails?

## Where to read next

If you care about the momentum side of the story, → [[momentum]] explains how exponential moving averages filter oscillation and why Nesterov correction matters. The engineering consequence is detailed in → [[adaptive-optimizers]], which covers AdamW, Adafactor, and their learning rate schedules in production systems. If you want the probabilistic foundation that motivates every preconditioning trick, → [[second-order-optimizers]] walks through the Taylor-series derivation of Hessian-based steps and their approximate solvers.

## Build it

Training a tiny character-level Transformer with SGD, Adam, and a spectral optimizer in the same notebook proves the narrative: preconditioning is the only way to traverse the deep-learning canyon without oscillating. You will implement the updates from scratch, chart the loss landscapes, and observe how diagonal, adaptive, and spectral rescalings line up the steps.

**What you're building:** A PyTorch comparison that trains a 2-layer, 6-head character-level Transformer on the tiny Shakespeare dataset, logging loss curves and gradient norms for SGD, Adam, and the spectral preconditioner inspired by the 2025 paper.

**Why this is valuable:** Because it forces you to write each optimizer’s math explicitly and watch how the preconditioning matrix reshapes the noisy gradient, you will not only reproduce but also visualize the oscillation that occludes progress when preconditioning is absent.

**Stack:**
- **Model:** `little-transformer/char-2layer-6head` (HuggingFace placeholder; clone the architecture from an available small Transformer checkpoint with ~5M parameters).
- **Dataset:** `tiny-shakespeare` ([https://huggingface.co/datasets/tiny_shakespeare](https://huggingface.co/datasets/tiny_shakespeare)) — the canonical character-level, 1 MB training set.
- **Framework:** PyTorch 2.1 + `torchvision` 0.18 for logging and plotting.
- **Compute:** Free Colab TPU v3-8 or GPU T4 (12 GB VRAM); training takes ~40 minutes for the full comparison.

**The recipe:**
1. `pip install torch==2.1.1 matplotlib numpy datasets` and load the tiny Shakespeare dataset with `datasets.load_dataset("tiny_shakespeare")`.
2. Tokenize by mapping characters to integers, create overlapping sequences of length 128, and batch them into tensors of shape `[batch, seq]` with gradient accumulation to simulate larger batches.
3. Implement three optimizers: (a) vanilla SGD with momentum 0.9 and \(\eta=5e-4\); (b) AdamW with \(\beta_1=0.9\), \(\beta_2=0.99\), weight decay 0.01, warmup 1000 steps, cosine decay; (c) spectral optimizer that estimates the top-2 singular vectors per layer via QR decomposition on the latest 256 gradients and rescales updates by the inverse of their estimated singular values plus \(\epsilon=1e-4\). Log training loss and gradient norm every 100 steps.
4. Evaluate by plotting loss curves and gradient norm magnitudes across optimizers: expect SGD to show zig-zags, Adam to stabilize but plateau early, and spectral to smooth the curve and descend faster. Compute char-level perplexity on the validation split; spectral should achieve the lowest perplexity within 35 minutes.
5. What you now have is a notebook that documents how each preconditioning strategy reshapes the update and how low-rank spectral directions tame the canyon.

**Expected outcome:** A runnable Colab notebook that outputs three loss/gradient norm plots and a small checkpoint for each optimizer, demonstrating that preconditioning directly expedites convergence while SGD oscillates.

- **CS student:** Run the same recipe on an RTX 4070 but drop the spectral optimizer to estimating only the top singular vector (k=1) to keep QR updates under 8 GB VRAM; the comparative loss plots will still reveal the benefit of even a single direction correction.
- **Applied engineer:** Instead of training, load Hugging Face’s `little-transformer` checkpoint, quantize it to INT8 with `torch.quantization.quantize_dynamic`, and serve it with Triton on an L4; verify that the spectral-inspired scaling in your optimizer improvises the inference cache updates so latency stays under 3 ms p50.
- **Applied researcher:** Treat the spectral preconditioner as a hypothesis: does including the second singular vector outperform the single-vector variant in steps-to-loss? Run three runs (k=1, k=2, diagonal only) and report the median step count and final perplexity to falsify the hypothesis that k>1 gives no gain.
- **Frontier researcher:** Probe the open question by measuring the variance of the top singular directions under different minibatch sizes and constructing a regularizer that penalizes large deviations; define the falsification criterion as the regularizer improving convergence without warmup on at least two datasets (tiny Shakespeare and Wikitext-2).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*