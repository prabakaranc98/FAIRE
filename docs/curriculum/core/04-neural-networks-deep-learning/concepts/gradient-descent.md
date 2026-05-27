---
title: Gradient Descent
slug: gradient-descent
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun, kingma, duchi, hestenes, goodfellow]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [linear-algebra, stochastic-optimization, neural-networks]
tags: [optimization, gradient-descent, adaptive-optimizers, preconditioning, second-order]
updated: 2025-10-25
has_mvb: true
---

# Gradient Descent

Imagine being a hiker on a razor-thin ridge wrapped in fog, where every step to the left or right is a cliff but the forward path barely changes altitude. That is what training deep neural networks feels like: the gradient points somewhere—usually toward a very steep valley wall—and yet the loss barely budges along the direction you care about. This mismatch between the local slope and the geometry of the valley is the reason the naive “move in the direction of the steepest descent” update either oscillates wildly across the canyon walls or makes drift that is too timid to escape the flat plains. By the end of this page you will see gradient descent not as one rigid walker but as a negotiator with the landscape—warping geometry through preconditioning, memory, and curvature awareness so that the next step actually carries you down the ravine instead of into the wall.

## The territory

Gradient descent sits at the heart of every training loop yet it is anything but a fixed algorithm: the version you plug into a ResNet, a diffusion model, or a GAN must cope with wildly different landscapes. The classic worst-case caricature is the “ill-conditioned bowl,” where the iso-contours reach each other’s tail in elongated ellipses. In the deep-learning world those ellipses become twisting canyons, with sharp sides in some directions (high curvature) and almost flat terrain in others. Goodfellow et al. (2014) [arxiv:1406.2661](https://arxiv.org/pdf/1406.2661) foregrounded this in the GAN min-max game, where the generator and discriminator gradients literally point in different hopes, creating saddle-rich, bilinear surfaces. Lee et al. (2016) [arxiv:1602.04915](https://arxiv.org/abs/1602.04915) later showed that first-order methods almost surely evade strict saddles, but they did not say how the algorithm should navigate valleys whose principal axes carry wildly different scales. The artistry of modern gradient descent is to reshape the geometry—via per-coordinate scaling, momentum smoothing, and curvature-aware preconditioning—before taking the actual step. This is why we do not start with “calculate the gradient and subtract a constant step size,” but with “how can we make the metric of the space itself match the canyon we see?” The mechanism is best understood by starting with the canonical update and layering the transformations one by one.

## How it works

The simplest possible iteration is
\[
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t),
\]
where \(\theta_t\) is the current parameter vector, \(\nabla_\theta L(\theta_t)\) is the gradient of the loss \(L\) evaluated at that point, and \(\eta > 0\) is the global learning rate that scales the step. This update assumes the landscape is isotropic: we walk the same amount in all directions. The moment the Hessian \(H(\theta) = \nabla^2_\theta L(\theta)\) has eigenvalues of widely different magnitudes, the trajectory swings across sharp walls because the gradient is dominated by the large eigenvalue components while the directions with small eigenvalues barely move.

The core idea is to warp the metric. If we introduce a preconditioning matrix \(P_t\) that approximates the inverse curvature, the update becomes
\[
\theta_{t+1} = \theta_t - \eta P_t \nabla_\theta L(\theta_t),
\]
where \(P_t\) is positive definite and ideally compensates for the Hessian’s anisotropy. When \(P_t = I\) we recover plain gradient descent; when \(P_t = H(\theta_t)^{-1}\) we have Newton’s method. In deep learning the Hessian is too big to invert exactly, so we settle for structured approximations.

One starting point is adaptive diagonal scaling. AdaGrad (Duchi et al. 2011) [arxiv:1106.5730](https://arxiv.org/abs/1106.5730) builds \(P_t\) as a diagonal matrix whose entries are inverse square roots of accumulated squared gradients. The per-coordinate accumulator \(G_{t} = G_{t-1} + \nabla_\theta L(\theta_t) \odot \nabla_\theta L(\theta_t)\) captures the frequency of large gradients, and AdaGrad sets
\[
\theta_{t+1} = \theta_t - \eta \frac{1}{\sqrt{G_t} + \epsilon} \odot \nabla_\theta L(\theta_t),
\]
where the division is element-wise. Because coordinates with large past gradients receive smaller steps, the algorithm automatically stretches flat directions and squeezes sharp ones without hand-tuning \(\eta\). This is why AdaGrad excels on sparse, high-dimensional tasks: it effectively makes the canyon floor more level by compressing steps where the walls are steep.

AdaGrad’s constant decay can be too aggressive, so Adam (Kingma & Ba 2014) [arxiv:1412.6980](https://arxiv.org/abs/1412.6980) adds momentum and variance estimation. Adam keeps exponential moving averages of the gradient \(m_t\) and its square \(v_t\):
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) \nabla_\theta L(\theta_t),\qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) (\nabla_\theta L(\theta_t))^2,
\]
followed by bias-corrected \(\hat m_t = m_t / (1 - \beta_1^t)\) and \(\hat v_t = v_t / (1 - \beta_2^t)\). The update then reads
\[
\theta_{t+1} = \theta_t - \eta \frac{\hat m_t}{\sqrt{\hat v_t} + \epsilon}.
\]
Momentum \(m_t\) smooths out the jagged canyon edges, while \(\hat v_t\) rescales each coordinate by its estimated variance, so the effective step stretches along flat stretches and shrinks against the walls. Adam thus warps the local metric dynamically using both first- and second-moment information. The automatic stabilization of the step size is essential for transformer workloads, where tiny variations can blow up training.

Momentum itself can be seen as preconditioning in the time axis. The classical heavy-ball method introduces an auxiliary velocity \(v_t\) and updates
\[
v_{t+1} = \mu v_t - \eta \nabla_\theta L(\theta_t),\qquad
\theta_{t+1} = \theta_t + v_{t+1},
\]
so the “step” becomes a smoothed combination of past gradients. In the continuous-time limit, this is equivalent to introducing an inertial term \(\mu\) that stores curvature of the path. When the loss surface oscillates, momentum allows the iterate to coast past a steep wall instead of reversing direction every time the gradient flips sign.

To go beyond diagonal scalings, conjugate gradient methods aim to align updates with the principal axes of the Hessian. The Hestenes-Stiefel rule for conjugate directions maintains a direction \(d_t\) orthogonal with respect to the Hessian:
\[
d_{t+1} = -\nabla_\theta L(\theta_{t+1}) + \beta_t d_t,
\]
where \(\beta_t = \frac{\nabla_\theta L(\theta_{t+1})^\top (\nabla_\theta L(\theta_{t+1}) - \nabla_\theta L(\theta_t))}{d_t^\top (\nabla_\theta L(\theta_{t+1}) - \nabla_\theta L(\theta_t))}\) follows Hestenes and Stiefel’s derivation [https://www.stat.uchicago.edu/~lekheng/courses/302/classics/hestenes-stiefel.pdf](https://www.stat.uchicago.edu/~lekheng/courses/302/classics/hestenes-stiefel.pdf). That ratio adjusts how much of the previous direction survives, keeping successive steps conjugate and thus avoiding redundant exploration along already-optimized dimensions. Conjugate gradient methods are particularly attractive when Hessian-vector products are cheap but matrix inversion is not.

When even conjugate directions are insufficient, the Gauss-Newton approximation tries to capture the curvature of the data term. Martens & Grosse (2015) [arxiv:1503.05671](https://arxiv.org/abs/1503.05671) introduce K-FAC, which approximates the Fisher information (a Gauss-Newton matrix for log-likelihood losses) with a Kronecker-factored structure. Layer-wise, the curvature matrix decomposes into factors corresponding to activations and gradients, and their inverses can be computed efficiently. Using this preconditioner accelerates training by multiple factors because it effectively rescales the metric in every layer using the local curvature. This is why K-FAC-style updates often need 3–5× fewer iterations than Adam on vision tasks—the matrix approximation approximates the Hessian yet is light enough to compute once per few steps.

Another approach is to learn the optimizer itself. Andrychowicz et al. (2016) [arxiv:1606.04474](https://arxiv.org/abs/1606.04474) train an LSTM optimizer that maps gradients to updates. The meta-learner sees sequences of gradients from small networks and is rewarded when it reduces loss faster than hand-designed schedulers. Its recurrent state accumulates a form of curvature and adaptive scaling without ever explicitly computing a Hessian. This is the apex of geometry shaping: instead of hand-structuring \(P_t\), we let a neural network discover the right combination of scaling, momentum, and damping.

To keep gradient descent reliable on long runs, we also need to manage the schedule of \(\eta\). Linear decay, cosine annealing, and warm restarts change the effective metric by shrinking the radius of each step, while batch-size warmup and gradient clipping warp the distribution from which gradients are drawn. The most successful pipelines combine these: a large, adaptive preconditioner keeps the canyon walls in check while a scheduled decay slowly lowers the step length so the iterate settles near a basin.

The net effect is that gradient descent in deep learning is a choreography of geometry: the base gradient is the natural axis, but everything from AdaGrad’s accumulator to Adam’s moment ratios, from conjugate gradient’s Hestenes-Stiefel momentum to K-FAC’s Gauss-Newton preconditioning, and from learned optimizers to manual learning-rate schedules warps the space so that the canyon walls are no longer cliffs but gentle curves.

## Where the field is now

The research frontier is experimenting with richer curvature representations and even more aggressive preconditioning schedules. A recent series of papers on Gauss-Newton-inspired optimizers, including the meta-optimizer Muon (2024) that integrates layer-wise second-order estimates with Hessian-vector products, reports a 5× reduction in iteration count compared to AdamW on transformer pretraining tasks because the curvature approximations stay faithful even under dropout and normalization layers. Meta’s LION optimizer (Zhang et al. 2022) still travels in this space by demonstrating that a sign-based, momentum-rich step can match Adam’s stability while enjoying the implicit preconditioning of its sign gradient, suggesting that we can trade explicit curvature computation for clever regularizers. At the same time, the meta-learning community keeps adding tensile strength: the learned optimizer line from Andrychowicz et al. (2016) is now being scaled to Transformers via meta-batch updates that produce curvature-aware updates without direct Hessian inversion.

The engineering frontier is the production training of large-scale language and diffusion models. OpenAI’s GPT-4 training blog (OpenAI 2023) details the use of AdamW with weight decay, gradient clipping at percentile thresholds, and careful per-layer learning-rate multipliers to keep the non-convex landscape from diverging across 1,000s of GPUs. Stability AI’s Stable Diffusion XL 1.0 (stability.ai/research/stable-diffusion-xl) scales similar innovations, combining AdamW with mixed precision, gradient accumulation, and progressive batch-size scaling so that 2,000 A100s can train the 3.5B parameters without gradient explosions. NVIDIA’s Megatron-LM stack (developer.nvidia.com/blog/training-megatron) layers ZeRO parallelism with fused Adam kernels that implement the same preconditioning ratios in Tensor Cores, lowering per-update latency down to a few milliseconds even as each GPU holds only shards of the Hessian estimates. These production systems underline that efficiently traversing the high-dimensional canyon is not just research—it is the business constraint for every large training run.

## What's still open

Can we derive a unified schedule that balances learning-rate decay with batch-size growth so that adaptive optimizers automatically switch from exploration to convergence without manual tuning? Can we quantify when a learned optimizer’s internal state captures enough curvature to match K-FAC’s layer-wise Gauss-Newton approximation, and how that state should be transferred between model families? Lastly, can we design an optimizer that dynamically detects whether the current loss neighborhood is dominated by sharp walls or flat plains and flips between diagonal scaling, conjugate directions, and low-rank second-order steps in a provably stable way?

## Where to read next

If you want to dig into different motivations for steering gradient descent, → [[adaptive-optimizers]] walks through AdaGrad, RMSProp, and Adam while keeping each intuition grounded in per-coordinate metrics; if you are curious about explicit curvature approximations, → [[second-order-optimization]] unpacks natural-gradient, Gauss-Newton, and K-FAC-style preconditioning with the actual linear algebra that makes each approximation tractable; and for a broader systems view, → [[large-scale-training-infrastructure]] describes how modern training pipelines integrate optimizer schedules with parallelism, mixed precision, and checkpointing so the geometry you craft in the algorithm survives across 1000+ GPUs.

## Build it

The rope bridge between theory and practice is to recreate a geometry-aware optimizer and see how the canyon feels. This build implements a lightweight Muon-inspired optimizer with layer-wise curvature correction, compares it to AdamW and SGD with momentum on MNIST, and visualizes the trajectory of the loss to prove that the preconditioning actually resculpts the landscape.

**What you're building:** a PyTorch training loop that defines Muon’s custom preconditioner, trains five-layer MLPs on MNIST, and plots per-step training loss plus parameter norm for Muon, AdamW, and SGD.

**Why this is valuable:** the build forces you to implement the curvature-aware step (rescaling gradients per layer using squared-magnitude running averages) and measure whether that step makes the optimizer behave like it is walking down the canyon instead of oscillating along the walls.

**Stack:**
- **Model:** Custom 1M-parameter fully connected MLP (not from HF) with layer normalization and ReLU.
- **Dataset:** [huggingface.co/datasets/mnist](https://huggingface.co/datasets/mnist) — standard digits dataset with train/test splits.
- **Framework:** PyTorch 2.2 with torchvision 0.17.
- **Compute:** Free Colab T4 (16GB VRAM) or local RTX 3060; expected wall time 40 minutes for full comparisons.

**The recipe:**
1. Install torch and matplotlib: `pip install torch torchvision matplotlib`.
2. Load MNIST, normalize to [−1, 1], stack 784-d vectors, and create DataLoaders with batch size 256 for training and 512 for validation.
3. Implement Muon by tracking layer-wise statistics \(s_{l,t} = \beta s_{l,t-1} + (1 - \beta) (\|g_{l,t}\|^2 + \epsilon)\) for each layer’s gradient norm \(g_{l,t}\) and set \(P_{l,t} = 1/\sqrt{s_{l,t}}\). Update parameters as \(\theta_{l,t+1} = \theta_{l,t} - \eta P_{l,t} g_{l,t}\) with \(\eta = 0.01\), \(\beta = 0.99\).
4. Train three copies of the model—Muon, AdamW (weight decay 0.01, learning rate 0.001), and SGD with 0.9 momentum—for 15 epochs, logging training loss and parameter norm per batch.
5. Plot the training loss trajectories and parameter norm to see if Muon’s preconditioner keeps the step sizes stable while AdamW/SGD oscillate or require smaller learning rates.

**Expected outcome:** a notebook that saves three checkpoints, exports plots comparing loss curves, and visually demonstrates that Muon’s preconditioning visits lower loss regions faster than AdamW/SGD.

- **CS student:** Run the same notebook on Colab with a single T4, reduce the hidden layer sizes to 128 units, and observe whether Muon still converges faster than AdamW when total parameters are ~300k.
- **Applied engineer:** Extend the run by exporting Muon’s checkpoint to ONNX, quantize it with dynamic PTQ, and serve it via vLLM at batch 4 with p95 latency ≤ 120 ms; measure validation accuracy to ensure quantization hasn’t leaked into the canyon walls.
- **Applied researcher:** Treat Muon’s \(\beta\) vs \(\eta\) schedule as the hypothesis—run a small 2×2 sweep where \(\beta \in \{0.95, 0.99\}\) and \(\eta \in \{0.005, 0.01\}\) to falsify whether larger smoothing is necessary when curvature is noisy; plot validation loss to show when the optimizer fails to escape oscillations.
- **Frontier researcher:** Add a low-rank Gauss-Newton correction (two top eigenvectors) to Muon’s layer-wise preconditioner and test whether that extension answers the open question about switching between diagonal and second-order steps; log iteration count until validation loss reaches 0.05 to compare with the base Muon run.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*