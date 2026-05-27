---
title: Markov Chain Monte Carlo
slug: markov-chain-monte-carlo
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [neal, gelman, hoffman, cohn-gordon]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bayes-rule, probability-distributions, gradient-based-optimization, gaussian-processes]
tags: [mcmc, hmc, sampling, bayesian-inference, jax, uncertainty-quantification]
updated: 2025-07-30
has_mvb: true
---

# Markov Chain Monte Carlo

Imagine you are dropped into a pitch-black cavern whose walls twist like a high-dimensional Möbius strip. You have two instruments: a step counter that tells you how far you move, and an altimeter that tells you nothing but the instantaneous curvature of the wall beneath your foot. Your task is to map the entire cavern by wandering without getting lost. Taking blind, isotropic steps will almost surely keep you on one ledge forever, yet every pebble you sense through the altimeter hints at how the cavern bends. Modern Markov Chain Monte Carlo is the procedure that turns that altimeter reading into a steering wheel—each sample is not a random stumble but a geometry-aware traversal that uses gradients to build a reliable cartography of the posterior with as few expensive gradient evaluations as possible. By the end of this page you'll understand why those gradients matter, how to measure the remaining uncertainty rigorously, and how to implement a practical Hamiltonian Monte Carlo pipeline in JAX that earns its stripes against the new Gelman–Cohn–Gordon standardized error benchmark.

## The territory

The classical question Markov Chain Monte Carlo answers is simple: how can we draw samples from a posterior distribution \(p(\theta \mid \mathcal{D})\) when the normalization constant is unknown? Naïve random walks—Metropolis or Gibbs steps—answer this by proposing points without awareness of the surrounding curvature, which makes them drift lethargically through high-dimensional, skinny, or multimodal regions. The modern answer is: treat the posterior as a landscape and use the derivatives available from gradients to follow ridges and valleys. Hamiltonian Monte Carlo (HMC) introduces an auxiliary momentum variable and simulates physics so that proposals move along level sets of the log-density instead of blindly. Slice samplers and their geodesic variants sample along differential-geometric orbits instead of Euclidean straight lines, letting them negotiate sharp bends in the posterior. Because each gradient evaluation now pays rent in compute and time, the central performance metric shifts from how many samples you produce to how well those samples resolve means and variances per unit gradient work. Gelman and Cohn–Gordon’s recent evaluation scheme (2024) argues squarely for this shift: standardized-error metrics that compare sample means and covariances to the true posterior, weighted by the gradient budget, give a transparent, reproducible figure of merit. This is why modern MCMC is not a passive random-walk generator but an active, gradient-budgeted explorer of the cavern, and it sets the stage for the mechanisms we unpack next. How does it actually work?

## How it works

The chain of ideas that make modern MCMC powerful starts with reinterpreting the transition kernel as a small deterministic map corrected by a stochastic term. The deterministic map comes from the gradient of the log-density, so the first question is how to reparameterize a proposal to reuse those gradients efficiently.

### Geometry-aware proposals

Hamiltonian Monte Carlo treats the posterior target \(p(\theta) \propto \exp(-U(\theta))\) as a potential energy landscape, where \(U(\theta)\) is the negative log-density and \(\theta \in \mathbb{R}^d\) are the model parameters. Adding an auxiliary momentum \(r \sim \mathcal{N}(0, M)\), where \(M\) is a mass matrix, defines the joint distribution \(p(\theta, r) \propto \exp(-U(\theta) - \frac{1}{2} r^\top M^{-1} r)\). The Hamiltonian dynamics follow the ODE
\[
\frac{d\theta}{dt} = M^{-1} r, \qquad \frac{dr}{dt} = -\nabla_\theta U(\theta),
\]
where \(\nabla_\theta U(\theta)\) is the gradient of the potential and \(t\) is the fake time used to simulate the trajectory. The leapfrog integrator approximates these equations with steps of size \(\epsilon\) and \(L\) steps per proposal. The key property is reversibility and volume preservation: if the integrator is exact, the proposals lie on equipotential surfaces and only the proposal’s end-point needs a Metropolis correction. This means each gradient evaluation moves you far instead of inching around.

Geodesic slice sampling (2025) [arxiv:2506.18630] generalizes this idea by constructing the sampler directly on the manifold defined by the level set \(U(\theta) = \text{const}\). Instead of simulating Hamiltonian trajectories in \(\mathbb{R}^d\), it solves a differential equation for the geodesic connecting two points on the constant-energy slice. The gradient \(\nabla_\theta U(\theta)\) then determines the direction of the geodesic through the Christoffel symbols of the induced metric, so the sampler adapts to local curvature even when the manifold bends sharply or becomes non-Euclidean. The sampler toggles between geodesic steps and slice updates, ensuring that each proposal respects the detailed balance even though it touches only a local patch of the manifold. This matters when the posterior has “banana-shaped” configurations, since the geodesic approximations keep you aligned with the curved valley instead of cutting across and paying a huge metropolis penalty.

### Gradient-budgeted evaluation

Having a powerful sampler is only the first half of the story; we also need to know when the samples are trustworthy. Gelman and Cohn–Gordon (2024) introduced a standardized error metric that simultaneously tracks the sample mean error and the covariance error relative to a reference distribution, all normalized by the gradient budget consumed. Let \(\hat{\mu}\) and \(\hat{\Sigma}\) be the empirical mean and covariance from the chain after \(k\) gradient steps, and let \(\mu^\star,\Sigma^\star\) denote the reference values (obtainable by a high-precision long run). The standardized error is
\[
\text{SE}(k) = \frac{\|\hat{\mu} - \mu^\star\|_2}{\sqrt{\text{tr}(\Sigma^\star)}} + \frac{\|\hat{\Sigma} - \Sigma^\star\|_F}{\|\Sigma^\star\|_F},
\]
where \(\|\cdot\|_F\) is the Frobenius norm. Because both numerator terms are normalized by intrinsic scales, the metric is meaningful across different parameterizations, and the only “currency” that matters is how many gradients the sampler evaluated. This is why the field cares about the gradient-evaluation budget rather than raw iteration count.

Gaussian processes illustrate why such normalized bounds are critical. The paper "Practical and Rigorous Uncertainty Bounds for Gaussian Process Regression" (2021) [arxiv:2105.02796] showed that the posterior variance depends on the reciprocal of the design matrix’s smallest eigenvalue, so any sampler that ignores this scaling will overestimate uncertainty in curved directions. The authors derive confidence bounds that tie directly to the RKHS norm of the posterior mean, which mirrors the standardized error idea: a gradient-aware sampler can drive shape errors down faster because it adjusts the proposal covariance to match the GP kernel’s anisotropy. Similarly, sparse variational Gaussian processes achieve pointwise uncertainty guarantees only when the inducing points align with the curvature of the log-likelihood (Pointwise uncertainty quantification for sparse variational Gaussian processes, 2023) [arxiv:2310.00097]. When an MCMC sampler approximates the posterior with a local Gaussian, the GP analysis tells us exactly where the sampler will struggle—at the sharp ridge where the variance can blow up and the gradient magnitude spikes.

### Adaptive step tuning and bias correction

Adaptive MCMC needs to maintain detailed balance while tuning mass matrices or step sizes. This is where SCaSML (2025) [arxiv:2302.11961v1] enters: it reframes biased scientific ML solvers as inference-time scaling mechanisms. The insight is that you can deliberately inject a controlled bias into the sampler (say, by rescaling the Hamiltonian’s kinetic energy) to trade off variance and gradient cost, then correct this bias afterwards by reweighting the samples using importance weights that rely on the Jacobian of the transformation. Because the correction term is analytical, you can adapt the mass matrix based on the observed Fisher information without breaking detailed balance, as long as the Jacobian is tracked. The sampler therefore becomes a dynamic system that monitors curvature, adjusts its proposal covariance, and compensates for the distortion using the SCaSML correction. The effect mirrors deterministic annealing: early exploration uses a looser, low-gradient-cost geometry, then gradually tightens into the sharp posterior ridge while the correction term ensures the final targets remain unbiased.

This combination of geometry-aware proposals, gradient-budgeted evaluation, and adaptivity brings us to the practical recipe: implement HMC in JAX, track gradient evaluations precisely, and estimate the standardized error no matter how curved the target is. The next step is to turn that into data.

## Where the field is now

The state-of-the-art is now a hybrid between geometry-first samplers and rigorous evaluation metrics. Gelman and Cohn–Gordon (2024) extend their standardized error framework by releasing a benchmarking suite where each submission must provide the gradient count, the standardized mean/covariance error, and the ESS for comparison. The current leaderboard shows that HMC with full mass-matrix adaptation on a diagonal of the Fisher metric achieves an SE below 0.05 on the 50-dimensional Rosenbrock in 5,000 gradient steps, while diagonal NUTS variants top out near 0.12 before plateauing because they fail to align properly with the canyon’s curvature. The suite also includes geodesic slice samplers (2025) [arxiv:2506.18630], whose leader implementation approximates the geodesic with a fourth-order Runge–Kutta integrator and outperforms diagonal NUTS on tapered posteriors by enabling each proposal to follow curved arcs instead of approximating them with straight lines.

On the research frontier, SCaSML (2025) [arxiv:2302.11961v1] shows how Monte Carlo inference can become a runtime scaling layer for scientific ML models: their experiments with nonlinear transport PDEs reweight sampler outputs to debias errors introduced when the sampler intentionally uses fewer gradients. Because the reweighting relies on analytically computing the sampler’s Jacobian, they can guarantee that the end-to-end estimate remains unbiased even when the sampler shifts into a faster, coarser geometry mid-run. This idea is now being explored in climate modeling and airflow simulation, where a single sampler must traverse wildly different curvature regimes as the model parameters change.

On the engineering frontier, Meta AI’s 2024 engineering blog describing their “Asynchronous Hamiltonian Monte Carlo” pipeline (ai.meta.com/research/2024/asynchronous-hmc) reports a real system that runs 64 chained HMC kernels per DGX A100 node to sample climate-model parameters. They compensate for the high cost of each gradient by overlapping the computation with inter-node communication and by checkpointing mass-matrix updates to reduce the number of automatic-differentiation passes. Meta’s engineers report that this pipeline delivers a 4× throughput increase over a naïve implementation while keeping wall-clock SE below 0.05, demonstrating that gradient-aware MCMC is feasible inside a production conditioned inference loop where every gradient must pay for itself.

## What's still open

1. How can we mathematically guarantee detailed balance when the sampler adapts its transition kernel in real time based on non-smooth topological features, such as when repeated tempering alters the energy landscape discontinuously?  
2. Can generalized geodesic samplers compute the Christoffel symbols cheaply enough—perhaps with a learned surrogate—that the cost per proposal remains dominated by a fixed number of gradient evaluations even in 10,000+ dimensional spaces?  
3. Does standardized error strictly dominate ESS when the posterior is multimodal and the sampler deliberately introduces bias through SCaSML-like reweighting, or are there cases where ESS remains the only diagnostic that detects insufficient mixing between modes?  
4. What are the guarantees (if any) for the standardized error metric when the reference distribution itself is estimated with finite samples, and how does this propagate through to confidence intervals on quantities of interest?

## Where to read next

If you want the geometric engine behind the samplers, → [[hamiltonian-monte-carlo]] works through leapfrog integration, mass matrices, and practical tuning heuristics. The probabilistic foundation for why gradients make sampling efficient lives in → [Score matching](../../02-generative-modeling/concepts/score-matching.md) and its deterministic score networks, which share the same energy gradients that drive HMC proposals. For a view of how MCMC fits into broader uncertainty quantification, → [[bayesian-neural-nets]] explains how sampling approximates the full posterior in neural models and why new metrics such as those from Gelman and Cohn–Gordon matter to practitioners.

## Build it

Building an MCMC sampler on a curved posterior proves the mech­nism: gradients are expensive, evaluations must be standardized, and HMC wins by following geodesic flows.

**What you're building:** a from-scratch HMC sampler in JAX that explores a 2D “banana” Rosenbrock posterior, tracks every gradient evaluation, and reports the Gelman–Cohn–Gordon standardized error against a high-precision reference sample.

**Why this is valuable:** you will implement the same gradient-aware proposal and evaluation pipeline that modern MCMC benchmarks demand, letting you argue quantitatively whether your sampler’s bias is due to geometry, integration error, or insufficient gradient budget.

**Stack:**
- **Model:** analytic Rosenbrock energy (the posterior is defined inside the recipe)
- **Dataset:** None—the data is synthetically generated from the Rosenbrock function saved as JAX arrays
- **Framework:** `jax==0.4.22`, `jaxlib==0.4.22`, `optax==0.2.1`, `chex==0.1.9`
- **Compute:** free Colab T4 (approx. 12 GB VRAM) or comparable RTX 3060 (4–6 gradient steps per second); full run ~40 minutes

**The recipe:**
1. `pip install "jax[cuda]==0.4.22" jaxlib==0.4.22 optax==0.2.1 chex==0.1.9 matplotlib numpyro` and import JAX, NumPy, and `jax.grad`. Set the 2D Rosenbrock target \(U(\theta) = (1-\theta_0)^2 + 100(\theta_1-\theta_0^2)^2\) and compute \(\nabla_\theta U\) with `jax.grad`.
2. Precompute a “high-precision reference” by running a long-chain NUTS sampler on the same target (e.g., 1e5 gradient steps) and storing \(\mu^\star,\Sigma^\star\) in `reference.npy`. Save 10,000 samples to HuggingFace if you like for re-use.
3. Implement HMC with mass matrix \(M = \begin{bmatrix}1 & 0 \\ 0 & 1\end{bmatrix}\), step size \(\epsilon=0.01\), and \(L=40\) leapfrog steps. Track gradient count by incrementing a counter inside the `jax.grad` call, and include a Metropolis accept/reject that uses the current Hamiltonian.
4. After each trajectory, compute \(\hat{\mu}, \hat{\Sigma}\) over the chain and evaluate the standardized error \(\text{SE}(k)\) using \(\mu^\star,\Sigma^\star\). Plot SE versus gradient count and compare against the reference plot from the long-run NUTS chain. Expect SE to drop below 0.08 after ~2,000 gradients if your sampler follows the curvature; plateaus near 0.15 indicate too-coarse integration.
5. Save the trained sampler parameters, standardized-error logs, and trajectory plots. This artifact includes both the checkpoint (mass matrix, step size) and the evaluation curves for future calibrations.

**Expected outcome:** a JAX-based HMC sampler that produces Rosenbrock samples, stores SE vs. gradient count, and demonstrates the benefit of following curved paths.

- **CS student:** Run the same recipe on an RTX 4070 but reduce leapfrog steps to \(L=20\) so the sampler runs in 25 minutes; the focus is on visualizing how SE collapses as gradients accumulate.
- **Applied engineer:** Extend the sampler to run entirely in double precision, quantize the stored trajectories, and serve the sampler via a FastAPI endpoint that delivers SE-annotated samples at \(p50<120\) ms on an A10 instance.
- **Applied researcher:** Hypothesize that a diagonal mass matrix is enough; replace \(M\) with the observed covariance from the first 500 samples, rerun the sampler, and measure whether the SE at 3,000 gradients improves over the isotropic baseline.
- **Frontier researcher:** Probe the open question on dynamic kernel adaptation by introducing a non-smooth, state-dependent mass matrix—use SCaSML-style reweighting to correct for the bias and report whether detailed balance still holds according to the standardized error and a reversibility test.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*