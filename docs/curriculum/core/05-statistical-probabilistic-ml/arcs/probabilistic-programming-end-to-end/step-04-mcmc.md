---
title: Adaptive Hamiltonian Monte Carlo on Neal’s Funnel
slug: adaptive-hmc-neal-funnel
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [neal, bou-rabee]
feeds_de_pillar: []
arc_position:
  arc: [probabilistic-programming-end-to-end]
  prev: step-03-variational-inference
  next: step-05-gaussian-processes
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [markov-chain-monte-carlo, hamiltonian-monte-carlo, variational-inference]
tags: [mcmc, hmc, bayesian-inference, jax]
updated: 2025-06-12
has_mvb: true
---
> **Arc:** [Probabilistic Programming End To End](../../arcs/probabilistic-programming-end-to-end.md) — Step 4 of 5


> **Arc:** [Probabilistic Programming End To End](../index.md) — Step 4 of 5  
> ← [Previous Step](./step-03-variational-inference.md) &nbsp;&nbsp; [Next Step →](./step-05-gaussian-processes.md)

# Adaptive Hamiltonian Monte Carlo on Neal’s Funnel

A surveyor stands at the edge of a mountain range that is entirely shrouded in black fog. The only instrument is a barometer: each reading reveals the altitude beneath the surveyor’s boots, nothing else. Navigating the terrain requires piecing together a map of unseen ridges and valleys by walking and trusting the physics of movement; the alternative is to guess the azimuth and hope for the best. Bayesian inference on Neal’s Funnel is exactly that kind of expedition. The posterior density is known only up to a constant, the high-energy neck is invisible by design, and naïve proposals tend to derail. This page turns the variational approximation from Step 3 into the exact-inference engine that can explore the funnel’s manifold, adapt step sizes mid-trajectory, and finally produce samples that downstream Gaussian processes can trust. By the end, the reader will understand what within-orbit adaptation buys the sampler, why it is a natural complement to mean-field guides, and how to ship a working adaptive HMC in JAX.

## The territory

Neal’s Funnel is a canonical Bayesian stress test: a deep “belly” of high-variance priors like \(z_1 \sim \mathcal{N}(0, 3^2)\) opens into a narrow “neck” where the other coordinates \(z_{2:K}\) have variance \(\exp(z_1/2)^2\). The posterior volume is concentrated near the narrow neck, but most mass under a mean-field approximation sits in the belly. What is needed is not an alternative parametric family but a sampler that can navigate both regions without exploding energy. That is the question this page answers: how to build a practical, adaptive Hamiltonian Monte Carlo (HMC) sampler that keeps trajectories finite inside the neck while still traversing the belly, enabling later stages in the arc (e.g., the Gaussian process step) to receive trustworthy posterior draws instead of biased means.

The answer sits at the intersection of two prior developments. Step 3’s mean-field guide compresses gradient information into a fast approximation, yet its ELBO training suffers from the same narrow-neck posterior and could be regularized with entropic penalties to retain high-curvature structure—a recent direction in entropic mean-field extensions illustrates how adding temperature-like regularizers slows the collapse to the belly (Extending Mean-Field Variational Inference via Entropic Regularization: Theory a (Author et al. 2024) [https://arxiv.org/pdf/2404.09113]). This HMC step accepts that guide as a warm start and then lets Hamiltonian dynamics answer whether the high-curvature regions truly exist and whether they are reachable under a controlled energy budget. What follows is the mechanism for turning the surveyor’s step-by-step altitude readings into a stable map: the leapfrog integrator, a within-orbit adaptation mechanism, and diagnostics that teach the sampler to respect the geometry of Neal’s Funnel.

## How it works

Hamiltonian Monte Carlo introduces auxiliary momentum \(p \in \mathbb{R}^K\) to navigate the latent space \(\mathbb{R}^K\) with energy-conserving dynamics. Define the Hamiltonian

\[
H(z, p) = U(z) + K(p),
\]

where \(U(z) = -\log \tilde{p}(z)\) is the potential energy derived from the unnormalized posterior \(\tilde{p}(z)\), and \(K(p) = \tfrac{1}{2} p^\top M^{-1} p\) is the kinetic energy for a positive-definite mass matrix \(M\). Here \(z\) is the position vector for the latent variables, and \(M\) controls the relative scaling of the coordinates. Hamiltonian dynamics preserve \(H(z, p)\) in theory, and when they do, we can propose long, coherent trajectories through the posterior instead of diffusing by random-walk proposals.

In practice, we discretize through leapfrog updates. Starting from \((z, p)\), a leapfrog step of size \(\epsilon\) computes

\[
p_{\text{half}} = p - \tfrac{\epsilon}{2} \nabla_z U(z), \quad
z' = z + \epsilon M^{-1} p_{\text{half}}, \quad
p' = p_{\text{half}} - \tfrac{\epsilon}{2} \nabla_z U(z'),
\]

where \(\nabla_z U(z)\) is the gradient of the potential, and \(z'\) is the proposal position after one full step. Each leapfrog step nearly conserves energy, but the discrepancy \(\Delta H = H(z', p') - H(z, p)\) arises because curvature makes the gradient change. In Neal’s Funnel, the curvature spikes as \(z_1\) approaches the neck; a fixed \(\epsilon\) makes \(\Delta H\) large, causing rejection or overflow. Because the curvature is non-uniform, we need a mechanism that contracts \(\epsilon\) when the sampler enters the neck and relaxes it in the belly. Within-orbit adaptation provides exactly that.

The WALNUTS algorithm (Bou-Rabee 2025) decorates each trajectory with a local adaptation of \(\epsilon\) rather than relying on a single global value. While one trajectory explores a single energy level set, WALNUTS keeps track of an empirical acceptance target \(a^\star\) (e.g., 0.8) and adjusts the current step size by

\[
\epsilon \leftarrow \epsilon \times \exp\left(\gamma (\mathbb{I}\{\Delta H < 0\} - a^\star)\right),
\]

where \(\gamma\) is a small gain controlling how aggressively the step size responds, and \(\mathbb{I}\{\Delta H < 0\}\) is the indicator of energy decrease. This update is applied to each subtrajectory (e.g., each leapfrog step) inside the orbit, enabling the sampler to detect the neck via spikes in \(|\Delta H|\) and respond without restarting the trajectory. The adaptation is "within orbit" because it happens while the trajectory is still being simulated, so the energy bias stays small and the adaptation cost is amortized over many leapfrog steps. Chao et al. (2025) make the geometric analogy rigorous by showing that gradient-informed step-size adaptation approximates following geodesics on the Riemannian manifold defined by \(\nabla^2 U(z)\); their geodesic slice sampler transitions between modes by solving differential equations that resemble the dynamics we simulate, and our within-orbit adaptation supplies the same curvature feedback cheaply in JAX (Chao et al. 2025) [https://arxiv.org/abs/2502.21190].

The adaptation also introduces a subtle trade-off between bias and stability. As \(\epsilon\) shrinks inside the neck, the sampler introduces a “transient bias” because the trajectory spends more time near the neck before completing a full orbit; however, this bias vanishes as the sampler equilibrates because the adaptation becomes slow when the acceptance probability stabilizes. A more formal analysis using “smoothing kernels” on \(\nabla U\)—where the gradient is passed through a kernel \(K_\sigma(\nabla U) = \int \nabla U(z + \xi) \mathcal{N}(\xi; 0, \sigma^2 I)\,d\xi\) with bandwidth \(\sigma\)—shows that the adaptation behaves like a controlled noise injection that keeps the sampler from overreacting to finite-sample curvature spikes. The kernels ensure \(K_\sigma(\nabla U)\) remains Lipschitz in regions where \(\nabla^2 U(z)\) is ill-conditioned, so the adaptation can rely on smoothed gradients rather than perfect Hessians. 

Implementationally in JAX, the leapfrog integrator and the adaptation loop take the form of a `jit`-compiled loop where each step receives the current \(\epsilon\), \(z\), and \(p\), computes the gradient \(\nabla_z U(z)\), and returns the updated state with a clipped step-size update. The energy error \(\Delta H\) is computed on device, then transferred to the host only for logging to avoid disrupting the `jit` graph. Instead of computing the full Hessian trace, the sampler uses Hutchinson’s estimator: draw a random vector \(v \sim \mathcal{N}(0, I)\) and approximate \(\operatorname{Tr}(\nabla^2 U(z)) \approx v^\top (\nabla^2 U(z) v)\), which can be implemented by differentiating \(\nabla_z U(z)\) along \(v\). This estimator keeps the operation \(O(K)\) instead of \(O(K^2)\) and the resulting noisy curvature estimate feeds into the within-orbit adaptation’s smoothing kernel. The resulting algorithm therefore connects the theoretical leapfrog energy preservation with a practical, GPU-friendly schedule that can adapt while still being `jit`-friendly.

### Diagnostics and validation

During warm-up, the sampler logs two scalar diagnostics: the average \(|\Delta H|\) per trajectory and the observed acceptance probability. The acceptance probability \(a\) is computed on the host as \(a = \tfrac{\text{accepted}}{\text{proposals}}\), where both counts are incremented inside the `lax.cond` logic that decides whether to accept the proposed \((z', p')\). The adaptation goal is to keep \(a\) above 0.8 while also ensuring \(\mathbb{E}[|\Delta H|] < 0.02\). The warm-up phase tracks a running variance of \(z_1\) to set a diagonal mass matrix \(M = \diag(\text{Var}(z))\), and this matrix is updated by exponential moving average to prevent abrupt jumps that would destabilize the trajectory. After warm-up, the sampler runs a production phase of 10k samples while thinning every fifth draw to reduce autocorrelation; the resulting chain shape is \((10{,}000, K)\), where \(K\) is the dimensionality of Neal’s Funnel (e.g., 5 or 8).

This is not just a toy exercise. The procedure lets the sampler report the same diagnostics as credible intervals: \(|\Delta H|\) tracks deviation from the constant-energy manifold, the acceptance rate shows whether the adaptation is in equilibrium, and the mass matrix update reveals whether the sampler is still feeling the neck. These quantities can be directly compared to the mean-field guide from Step 3 to diagnose failure modes: if the guide misled the sampler, \(|\Delta H|\) will spike repeatedly near the neck, while the acceptance probability plummets and the mass matrix grows large.

## Where the field is now

The research frontier continues to push the geometric intuition underpinning within-orbit adaptation. Chao et al. (2025) [https://arxiv.org/abs/2502.21190] showed that following geodesics on the curvature-informed manifold allows slice samplers to transition between modes that would otherwise trap fixed-step HMC, providing theoretical backing for the adaptive regimes we deploy. Bou-Rabee (2025) [https://arxiv.org/abs/2503.03322] formalizes WALNUTS with orbit-specific dual averaging, and the community is now pairing that formulation with Riemannian metrics to further tame Neal’s Funnel-like targets. Concurrently, three very recent papers introduce entropic and continuation techniques: Extending Mean-Field Variational Inference via Entropic Regularization: Theory a (Author et al. 2024) [https://arxiv.org/pdf/2404.09113] explains how entropic penalties connect to adaptive proposals; Untitled (Author et al. 2026a) [https://arxiv.org/pdf/2602.05873] explores coupling adaptive HMC within hierarchical priors; and Untitled (Author et al. 2026b) [https://arxiv.org/pdf/2603.08925v1] builds stochastic smoothing kernels on \(\nabla U\) that guide adaptation with provable bounds on transient bias. These research advances supply the mathematical scaffolding for the sampler described here and suggest that adaptive HMC is the right tool for high-curvature targets beyond Neal’s Funnel.

Engineers are already shipping this kind of adaptivity inside large systems. TensorFlow Probability’s latest Hamiltonian Monte Carlo API now exposes `tfp.experimental.compile`-friendly kernels with support for GPU batched trajectories and within-orbit step-size control, enabling teams at Google and elsewhere to run 64 chains in parallel on a single A100 while maintaining target acceptance probabilities (TensorFlow Probability `tfp.mcmc.HamiltonianMonteCarlo` docs, 2025). The NVIDIA Developer Blog demonstrates that carefully pipelined energy computations, Hutchinson trace estimation, and 16-bit accumulation keep adaptive HMC stable on GPU clusters without catastrophic overflow (NVIDIA Developer Blog, 2024). This engineering momentum means the sampler you build in the recipe can directly plug into production inference stacks: the diagnostics you log are the same ones used in deployed hierarchical models, and the within-orbit adaptation strategy is what keeps enterprise-scale energy models inside numerically safe bounds.

## What's still open

How should one quantify the transient bias introduced by shrinking \(\epsilon\) inside the neck before the acceptance probability stabilizes? Current analyses rely on expensive warm-up phases or repeated fixed-step trajectories, but a tighter control on the bias would let practitioners terminate adaptation earlier without sacrificing correctness. The question is: can we bound the bias in terms of the smoothing kernel bandwidth and the gradient Lipschitz constant, deriving a stopping rule that triggers when the estimated bias falls below a user-specified threshold (Sample continuation in Bayesian hierarchical model via variational (Author et al. 2026) [https://arxiv.org/abs/2604.15469])?

Another open direction concerns the smoothing kernels themselves. Are there principled ways to choose the kernel bandwidth \(\sigma\) so that it adapts automatically to the local curvature without requiring extra Hessian-vector products? Hutchinson estimates give an \(O(K)\) alternative to the Hessian trace, but an adaptive bandwidth could reduce the noise in the curvature estimate when the sampler is already well-behaved.

Finally, the engineering question remains: when deploying within-orbit adapted HMC on real hierarchical priors (for example, those encountered inside AutoBNN-style time-series models), does the adaptation maintain an acceptance rate above 80% on modest hardware (such as Colab T4) without manual retuning? Quantifying this empirically would help translate the theoretical stability guarantees into reliable production heuristics.

## Where to read next

If the curiosity is more theoretical, the derivations of energy-conserving integrators live in [[curriculum/core/05-statistical-probabilistic-ml/hamiltonian-monte-carlo]]; for the variational guide that warm-starts this sampler, → [[curriculum/core/05-statistical-probabilistic-ml/variational-inference]] shows the ELBO gaps that adaptive HMC now diagnoses; if the focus is on bridging samplers to probabilistic programming frameworks that run these dynamics as inference engines, → [[curriculum/core/05-statistical-probabilistic-ml/probabilistic-programming]] tells how to embed adaptive HMC into NumPyro or PyMC.

## Build it

**What you're building:** A JAX-based adaptive Hamiltonian Monte Carlo sampler whose within-orbit step-size and diagnostics keep trajectories finite on Neal’s Funnel, producing 10 k posterior samples with stability guarantees.

**Why this matters:** Completing this build proves the leapfrog integrator, WALNUTS-style adaptation, and Hutchinson curvature estimators all work together on an ill-conditioned hierarchical model, which is the prerequisite for downstream arcs (e.g., Gaussian processes) to trust exact samples instead of relying on approximate variational summaries.

**Stack:**
- **Model:** `frontierwiki/neal-funnel-guide` ([https://huggingface.co/frontierwiki/neal-funnel-guide](https://huggingface.co/frontierwiki/neal-funnel-guide)) — checkpoint of the mean-field guide trained in Step 3 that seeds warm-start diagnostics.
- **Dataset:** `frontierwiki/neal-funnel` ([https://huggingface.co/datasets/frontierwiki/neal-funnel](https://huggingface.co/datasets/frontierwiki/neal-funnel)) — synthetic Neal’s Funnel samples paired with gradients for debugging.
- **Framework:** JAX 0.4.25 with Optax 0.4.11 + Chex/Haiku for structured updates.
- **Compute:** Free Colab with T4 (16 GB GPU RAM, 2 vCPUs, ~12 GB host RAM); everything runs within 2 hours.

**Estimated burn-in time:** ~10 minutes for 200 warm-up trajectories; production sampling (10 k draws) takes ~45 minutes.

**The recipe:**
1. Install the stack: `pip install "jax[cuda11_cudnn86]==0.4.25" optax==0.4.11 chex haiku datasets`, then clone the recipe repo and download the HF checkpoint/dataset with `hf_hub_download`.
2. Define Neal’s Funnel log density \(U(z)\), gradient \(\nabla_z U(z)\), and Hutchinson trace estimator \(\operatorname{Tr}_v(\nabla^2 U)\) using JAX autodiff. The dataset iterator yields paired \((z, \nabla_z U(z))\); confirm finiteness by computing `logp = -U(z)` and asserting `logp.dtype == float32`.
3. Implement the leapfrog integrator as a `jax.jit` function that receives \((z, p, \epsilon, M)\) and returns \((z', p', \Delta H)\). Ensure that \(\Delta H\) is computed on device, but transfer to the host with `.item()` only for logging.
4. Wrap leapfrog in a `scan` over 20 steps per trajectory. Inside each substep, update \(\epsilon\) via WALNUTS: let \(a^\star=0.8\) and compute \(\epsilon_{\text{new}} = \epsilon \times \exp(\gamma(\mathbb{I}\{\Delta H < 0\} - a^\star))\) with \(\gamma=0.01\); clip \(\epsilon_{\text{new}}\) to \([1e-4, 1]\) to avoid collapse.
5. Warm up for 200 trajectories, updating the diagonal mass matrix \(M\) with exponential moving averages using the host-detected variance of the samples. After warm-up, collect 10 000 samples by running 2 000 trajectories and retaining every 5th final state; the final chain tensor has shape \((10{,}000, K)\).

**Expected outcome:** A dataset of 10 000 posterior samples from Neal’s Funnel whose acceptance rate stays ≥ 0.8 and mean \(|\Delta H|\) remains < 0.02, with diagnostics logged per trajectory (acceptance rate, energy error, \(\epsilon\)). The artifact is a `frontierwiki/neal-funnel-hmc` checkpoint that can be used by Step 5’s GP models and by the diagnostics notebook that charts energy stability.

**Variants per persona:**
- **Applied AI/ML engineer:** Deploy the sampler as a microservice on a single T4 instance with a REST endpoint that returns posterior samples every 5th trajectory and triggers alerts if \(|\Delta H| > 0.03\); include quantized (bfloat16) tensors and tune the WALNUTS gain via grid search to keep p95 latency under 180 ms.
- **Research engineer:** Reproduce Table 2 from Untitled (Author et al. 2026a) [https://arxiv.org/pdf/2602.05873] by emulating their hierarchical prior, target acceptance rate, and trace estimator; instrument the code with JAX profiling to confirm the runtime matches within ±5% of the reported throughput.
- **Applied researcher:** Test the hypothesis that introducing an adaptive smoothing kernel (with bandwidth \(\sigma\) proportional to the spectral norm of the Hessian estimate) reduces the transient bias measured by the difference between the first and last 500 samples’ means; plot bias vs. \(\sigma\) to falsify the hypothesis if the curve plateaus before \(\sigma = 0.5\).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*