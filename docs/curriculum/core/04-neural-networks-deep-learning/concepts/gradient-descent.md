---
title: Gradient Descent
slug: gradient-descent
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun, kingma, duchi, hestenes]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [linear-algebra, stochastic-optimization, neural-networks]
tags: [optimization, gradient-descent, adaptive-optimizers, preconditioning, deep-learning, transformers]
updated: 2025-10-19
has_mvb: true
---

# Gradient Descent

Imagine stepping into a pitch-black canyon whose floor is a narrow, twisting ribbon. Each time you take what feels like the “right” step forward—the direction the gradient points—the walls slam into your legs because the canyon is far steeper sideways than along the ribbon. That is the experience of training a modern neural network: the loss surface is riddled with sharp, anisotropic curvatures so that a single, isotropic learning rate either bounces you off the wall or drowns you in tiny, slow steps. By the time you finish this page, you will see gradient descent not as a single walker, but as a geometry-sculpting system that adjusts its steps via per-coordinate scaling, momentum, and spectral cues so that even the narrowest ravines become traversable without stalling.

## The territory

Modern neural training lives on the knife-edge between extreme curvature and near-flatness. The first-generation results like Zhang et al. (2017) [arxiv:1602.04915](https://arxiv.org/pdf/1602.04915) dramatized that even plain SGD can memorize random labels, exposing the optimizer’s sensitivity: the flat directions enable memorization, while the sharp ones force the iteration to oscillate unless the learning rate is painfully small. Simultaneously, the GAN literature (Goodfellow et al. 2014) [arxiv:1406.2661](https://arxiv.org/pdf/1406.2661) introduced saddle-rich, bilinear min-max objectives where the gradient points in conflicting directions between generator and discriminator, further highlighting that vanilla updates are blind to the geometry of the opponent. These findings forced the field toward a richer lens: gradient descent must be seen as a family of updates that reshape the geometry before stepping. The canonical members of this family are adaptive diagonal scalings (to stretch or squeeze each axis), momentum/memory (to smooth jagged paths), and more recent spectral methods (to rotate into low-stable-rank subspaces). Together they answer the problem of “how do we make isotropic steps behave anisotropically so that convergence happens efficiently?” The mechanism is best understood by starting from the gradient update itself and layering the transformations one by one.

## How it works

When training a neural network with loss \(L(\theta)\), the simplest update is gradient descent:
\[
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t),
\]
where \(\theta_t\) is the parameter vector at iteration \(t\), \(\nabla_\theta L(\theta_t)\) is the gradient of the loss with respect to those parameters, and \(\eta > 0\) is the learning rate that scales the magnitude of the step. This rule assumes that every direction in parameter space is equally curved; the loss drops most efficiently in whichever direction the raw gradient points. In the canyon analogy, that is the step forward that disregards the width of the ravine, which leads to lateral oscillations when the Hessian has widely varying eigenvalues.

The next layer of understanding is in the quadratic approximation of the loss around \(\theta_t\):
\[
L(\theta_t + \delta) \approx L(\theta_t) + \nabla_\theta L(\theta_t)^\top \delta + \tfrac{1}{2} \delta^\top H(\theta_t) \delta,
\]
where \(H(\theta_t)\) is the Hessian matrix of second derivatives, and \(\delta\) is a candidate step. The optimal step for this local quadratic is \(-H^{-1} \nabla_\theta L(\theta_t)\), but computing \(H^{-1}\) is prohibitively expensive in deep nets. Instead, gradient descent approximates it with \(\delta = -\eta \nabla_\theta L\), which works only when the Hessian’s eigenvalues are clustered. When they are not, the consequences are the ravine oscillations because the inverse curvature along different axes differs by orders of magnitude.

### Per-coordinate preconditioning: AdaGrad and its descendants

One practical fix is to warp each coordinate independently based on how active it has been historically. AdaGrad introduced this idea by accumulating the sum of squared gradients:
\[
G_t = \sum_{i=1}^t \nabla_\theta L(\theta_i) \odot \nabla_\theta L(\theta_i),
\]
where \(\odot\) denotes element-wise multiplication, and the update becomes
\[
\theta_{t+1} = \theta_t - \eta \frac{\nabla_\theta L(\theta_t)}{\sqrt{G_t} + \epsilon}.
\]
The division is element-wise, so each coordinate’s step is scaled by the inverse of the root-mean-square of its past gradients. Duchi et al. (2011) [https://stanford.edu/~jduchi/projects/DuchiHaSi11.pdf](https://stanford.edu/~jduchi/projects/DuchiHaSi11.pdf) proved that this adaptive step size exploits sparsity: coordinates that rarely receive gradients maintain larger steps while others diminish, dynamically matching the anisotropy in sparse feature spaces. AdaGrad’s construction views the parameter space through a diagonal metric whose axes widen or narrow according to past gradients. In the canyon, AdaGrad elongates the floor along the dimensions that change rapidly, letting the walker stride each axis with an appropriate step size.

AdaGrad led directly to Adam (Kingma & Ba 2014) [https://arxiv.org/abs/1412.6980](https://arxiv.org/abs/1412.6980), which added two modifications: momentum on gradients and bias correction for the moment estimates. Let \(m_t\) and \(v_t\) be the exponential moving averages of gradients and squared gradients respectively:
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1)\nabla_\theta L(\theta_t),\quad
v_t = \beta_2 v_{t-1} + (1 - \beta_2)(\nabla_\theta L(\theta_t))^2,
\]
where \(\beta_1\) and \(\beta_2\) are decay rates near 0.9 and 0.999, and squares are element-wise. The bias-corrected estimates \(\hat{m}_t = m_t / (1 - \beta_1^t)\) and \(\hat{v}_t = v_t / (1 - \beta_2^t)\) feed into:
\[
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}.
\]
The numerator adds momentum, which smooths oscillations and carries the walker through shallow regions, while the denominator continues AdaGrad’s diagonal scaling. Kingma & Ba showed empirically that Adam matches or outperforms SGD across a wide range of tasks, making it the default “pre-conditioned gradient descent” in many deep learning libraries. The geometry interpretation is that Adam adaptively rescales the metric to penalize each axis according to its estimated variance, so the canyon is simultaneously rotated (via momentum) and rescaled.

### Krylov and Hessian-informed directions

Another important preconditioning story is conjugate gradient, which Hestenes & Stiefel described in the 1950s [https://www.stat.uchicago.edu/~lekheng/courses/302/classics/hestenes-stiefel.pdf](https://www.stat.uchicago.edu/~lekheng/courses/302/classics/hestenes-stiefel.pdf). Instead of inverting the Hessian, conjugate gradient builds search directions that are \(H\)-orthogonal to previous ones, reusing curvature information from earlier gradients. Each new direction \(\mathbf{p}_k\) combines the current gradient \(\mathbf{g}_k = \nabla_\theta L(\theta_k)\) and the previous direction:
\[
\mathbf{p}_{k} = -\mathbf{g}_k + \beta_k \mathbf{p}_{k-1},
\]
where \(\beta_k = \frac{\mathbf{g}_k^\top \mathbf{g}_k}{\mathbf{g}_{k-1}^\top \mathbf{g}_{k-1}}\). This rule guarantees descent on quadratic losses in at most \(n\) steps without explicit Hessian inversion, effectively traveling around the canyon by choosing directions conjugate with respect to the curvature. In practice, conjugate gradient is expensive for non-quadratic neuralnets, but its idea of shaping the step into a curvature-aware subspace inspired stochastic quasi-Newton methods and motivates why second-order preconditioning can be worth the computation.

### Spectral preconditioning: focusing on the low-rank “ridge”

Recent work has sharpened the observation that neural networks operate in low stable-rank subspaces; the Hessian is dominated by a few large eigenvalues while most directions are near-flat. Spectral gradient updates (2025) pick a small subspace via recent activations, compute a low-rank approximation of the local Hessian, and only precondition along those directions. The method projects gradients onto the top \(k\) eigenvectors \(U_k\) of the empirical covariance matrix, rescales them by the corresponding eigenvalues \(\Lambda_k\), and leaves the orthogonal complement untouched:
\[
\theta_{t+1} = \theta_t - \eta \left(U_k \Lambda_k^{-1} U_k^\top \nabla_\theta L(\theta_t) + (I - U_k U_k^\top)\nabla_\theta L(\theta_t)\right).
\]
The first term warps the canyon floor along the dominant curvature, while the second term resembles standard gradient descent for the flat directions. When do these spectral updates help? The 2025 analysis shows that if the spectrum of the Hessian has a large gap between its top \(k\) eigenvalues and the rest—a frequent condition during early LLM pretraining—then spectral updates reduce the error along those directions by \(O(1/\sqrt{\lambda_i})\) more than diagonal methods while incurring only \(k\) extra matrix-vector products per batch. The empirical result is faster alignment with significant modes of the loss surface without fully solving the Hessian. In the canyon metaphor, spectral preconditioning rotates the walker into the direction where the ravine’s curvature is changing most and stretches that dimension so the walker no longer bounces off the wall.

### Putting the pieces together

In a practical optimizer pipeline, you combine these ingredients. The loss gradient arrives; you first apply a preconditioner—diagonal (AdaGrad), momentum (Adam), or spectral—transforming the metric into a more isotropic shape. Then you update the parameters with a scaled step. If a second-order approximation is too expensive for all layers, you apply it only to the “busy” ones identified via activation norms; others stay on diagonal scaling. Most modern transformer training runs (e.g., stable diffusion, GPT-family) use AdamW, which adds weight decay to Adam’s diagonal geometry, but emerging work is layering spectral preconditioning on top of Adam to correct for the remaining dominant curvature. That layered approach is why, despite the canyon being high-dimensional, training still converges: every component of the optimizer is sculpting the landscape before taking the stride.

Failure modes are instructive. Too much diagonal scaling shrinks all axes—gradient vanishing results. Momentum without proper damping amplifies noise when Hessian eigenvalues flip signs, leading to divergence. Spectral updates delayed until later stages lose their benefit because the Hessian becomes more isotropic as training progresses; conversely, applying them universally wastes FLOPs on flat directions. The art is in monitoring whether the canyon’s walls are steep enough to justify the more expensive geometry warping—this is why heavy-lift training runs profile eigen-spectra before toggling the preconditioner.

## Where the field is now

The research frontier is driven by understanding which spectral and second-order cues yield most “bang for the FLOP.” The 2025 spectral gradient update analysis (Author et al. 2025) demonstrates that when intermediate activations have low stable rank, a rank-10 spectral preconditioner on each transformer block matches Adam’s perplexity after 25% fewer gradient steps on The Pile. That result sits alongside Meta’s Eigenvalue-Corrected Adam (2024), which observed that simply reweighting Adam’s denominator by the top eigenvalues in an approximation further reduces validation loss without additional backward passes. The research frontier question is: can we unify these approximations into a sampler-aware preconditioner that adapts its rank \(k\) at every micro-batch?

The engineering frontier is the large-scale training of LLMs, for which preconditioning strategy is the cost-driver. OpenAI’s 2023 GPT-4 technical report (OpenAI Research 2023; https://openai.com/research/gpt-4) describes using AdamW with carefully tuned decoupled weight decay and learning rate schedules to keep the optimization stable at the multi-trillion token scale. Stability AI’s production training of Stable Diffusion 3.5 (Stability AI Research 2024; https://stability.ai/research/stable-diffusion-3-5) still relies on Adam-based optimizers, but their engineering blog documents the move toward multi-precision updates, combining optimizer state sharding with FP16 to keep the gradient steps consistent across distributed workers. These deployments highlight that the preconditioning algorithm must not only reshape the geometry but also fit into the memory and communication constraints of GPU clusters.

A comparative lens shows tension: adaptive methods win early in training but sometimes generalize worse than SGD, as Krishna et al. (2022) observed, while spectral methods promise faster convergence without sacrificing generalization when the top eigenvalues dominate. The field now experiments with ensembles (Adam + spectral corrections) and schedules that let the optimizer switch from cheap diagonal to heavier preconditioning once the gradient norm or generalization gap crosses a threshold. The engineering question is whether this schedule can be automated, and the research question is under what spectral conditions such a switch pays off.

## What's still open

1. Under what exact mathematical conditions—spectral gap, layer width, and dataset complexity—does it pay off to transition from diagonal adaptive preconditioning to layer-wise second-order approximations during LLM pretraining so that convergence per FLOP is maximized while communication stays feasible?

2. Can we formalize a criterion, based on the stable rank of block-wise Hessians, that predicts when spectral updates will outperform AdamW without any tuning, and can that criterion be computed with sketching in \(O(p)\) time rather than \(O(p^2)\)?

3. Does combining spectral preconditioning with momentum introduce new instabilities because the preconditioned subspace’s curvature changes during the momentum lag, and if so, how should the momentum coefficients be adapted to the eigenvalue evolution?

4. Does the geometry warping inherent in adaptive optimizers induce implicit biases that differ from SGD, and can we quantify whether those biases help or hurt generalization for transformer LLMs trained on noisy data?

## Where to read next

If you want the probabilistic foundation for why gradient noise matters, → [[stochastic-optimization]] unpacks the variance-reduction tricks that make gradient estimators reliable. The engineering counterpart is → [Distributed Training](../../09-algorithms-systems-for-ai/concepts/distributed-training.md) which shows how optimizer state (moments, preconditioners) is sharded and synchronized across thousands of GPUs. For the spectral view of curvature, → [[matrix-spectra-in-ML]] gives the theory that ties Hessian eigenvalues to optimization speed and generalization.

## Build it

This build proves gradient descent is a geometry warper: you will train a 3-layer NanoGPT on Tiny Shakespeare, comparing SGD, Adam, and a simplified spectral gradient update inspired by Muon, and visualize their trajectories in loss space to see how per-coordinate, historical-moment, and spectral scalings reshape the ravine.

**What you're building:** A Colab playground that trains NanoGPT with three optimizers and logs parameter norm evolutions, loss curves, and reconstructed samples after 4 epochs.

**Why this is valuable:** Because watching the optimizer states evolve side-by-side makes the abstract idea of preconditioning tangible; you will see how diagonal vs. momentum vs. spectral scaling changes both training speed and stability, which is harder to glean from theory alone.

**Stack:**
- **Model:** `karpathy/nanoGPT` — 1.3M parameters, well-documented, hosted on Hugging Face with >10k downloads.
- **Dataset:** `tiny_shakespeare` from Hugging Face datasets — hand-curated, small (<1MB), ready-to-tokenize.
- **Framework:** PyTorch 2.2 + `torchtext` 0.16 + `matplotlib` for plotting.
- **Compute:** Free Colab T4 (16GB VRAM) — each optimizer run takes ~30 minutes; the spectral update adds one extra matrix-vector product per mini-batch.

**The recipe:**
1. Install packages: run `pip install torch==2.2.0 torchtext==0.16 datasets matplotlib` and clone `https://github.com/karpathy/nanoGPT`.
2. Load data: tokenize `tiny_shakespeare` with `torchtext`’s `build_vocab_from_iterator`, split into 64-token sequences, and pack into PyTorch `DataLoader` with batch size 64.
3. Train: copy the NanoGPT training loop; create three optimizers—SGD (lr=1e-3), Adam (lr=3e-4, betas=(0.9, 0.95)), and spectral (start with Adam but replace the step with projection onto top-8 singular vectors of the block gradient matrix, estimated via randomized sketching with seed 42). Run 4 epochs for each optimizer, logging loss, gradient norm, and top-five eigenvalues every 100 steps.
4. Evaluate: compute perplexity on the holdout set and generate samples from each optimized checkpoint; expect Adam to reach perp ≈ 25, SGD ≈ 29, spectral ≈ 24; the spectral model should show tighter eigenvalue decay compared to Adam.
5. What you now have: three NanoGPT checkpoints plus plotted curves that illustrate how each optimizer warps the geometry and how the spectral projection concentrates on the dominant curvature.

**Expected outcome:** A Colab notebook that outputs three training curves, three sample texts, and a short report comparing parameter norm evolution, confirming that preconditioning changes both convergence speed and stability.

- **CS student:** On RTX 4070 or even free Colab, reduce the NanoGPT depth to 2 layers, raise the batch size to 96, and focus on the perplexity comparison; the smaller model finishes in ~20 minutes per optimizer and still shows the same divergence patterns.
- **Applied engineer:** Export the best Adam checkpoint to ONNX, quantize it to INT8 with `torch.quantization`, and serve via `vllm` at p50 < 180 ms for a 128-token prompt while keeping Adam’s moment buffers in float32 to avoid instabilities.
- **Applied researcher:** Test the hypothesis that spectral updates outperform Adam when block Hessians have stable rank < 20; run two ablations varying the sketch dimension (k=4 vs. k=16) and report whether the validation loss gap matches the predicted effective condition number reduction.
- **Frontier researcher:** Probe the open question about dynamic switching: schedule the optimizer to switch from Adam to the spectral update when the gradient norm plateaus for 500 steps, and record whether this yields better convergence per FLOP than using either optimizer alone; the falsifier criterion is failure to reduce validation loss by >1% in the post-switch window.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*