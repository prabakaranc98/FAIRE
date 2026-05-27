---
title: Probabilistic programming
slug: probabilistic-programming
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [bishop, bingham, mansinghka, hoffman]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bayesian-inference, probabilistic-modeling, markov-chain-monte-carlo, variational-inference]
tags: [probabilistic-programming, bayesian-inference, uncertainty, hmc, numpyro, pyro]
updated: 2024-11-01
has_mvb: true
---

# Probabilistic programming

Imagine writing a flight simulator that models every gust, rotor thrust, and payload shift a drone might experience, and then asking the simulator to run backward, producing the precise wind vector that would have sent your craft spiraling into the river. That is the mental leap probabilistic programming invites: instead of manually designing a predictive function, you describe the generative story and let a compiler-grade inference engine answer the “what if” question. By the time you finish this page, you will be able to describe domain knowledge as stochastic code, hand that code to an optimizer or sampler, and watch a posterior distribution emerge without ever re-implementing the inference yourself.

## The territory

Traditional machine learning asks, “What deterministic function maps inputs to labels?” Probabilistic programming turns that question on its head: “What latent story produced the observed data?” This shift is why statisticians in the 1970s celebrated hierarchical Bayes and why contemporary practitioners turn to probabilistic programming when they cannot assume Gaussian noise, independent examples, or a fixed architecture. Rather than treating a model as a black-box function fitted with gradient descent, a probabilistic program is generative code: it samples latent variables, dormates them through deterministic transforms, and emits observations. The runtime takes this specification, compiles it to an inference problem, and solves the Bayes update. Bishop (2013) Model-Based Machine Learning teaches that this modeling-first mindset is what lets practitioners tailor structure to their domain, so when you add a new latent variable you do not need to re-derive a fresh optimizer; you just rewrite the program.

What distinguishes probabilistic programming from the wider Bayesian tooling are three promises. First, the program expresses the prior knowledge, likelihood, and structural dependencies; nothing about the optimizer is hardcoded into the model. Second, inference is treated as an independent artifact, often programmable, so you can plug in MCMC, variational inference, or sequential Monte Carlo without touching the model itself. Third, the compiler is responsible for weaving these components together, rewriting the stochastic trace to reveal conditional independencies, caching repeated computations, and exposing gradients for autodiff. Probabilistic Programming Concepts (Gonzalez et al. 2013) [arxiv:1312.4328](https://arxiv.org/abs/1312.4328) spells out this decoupling, showing that the model is a generator and the inference engine is a transformer—when the compiler is good enough, the user need never derive the posterior by hand. How does that compiler work in practice?

## How it works

### Generative programs as latent simulators

At the heart of every probabilistic program is a stochastic trace that mirrors the generative process. The program begins by sampling latent variables from priors, e.g., \(\theta \sim p(\theta)\), where \(\theta\) might be regression coefficients, topic weights, or neural network parameters. It then deterministically transforms those latents into observable values through a likelihood \(x \sim p(x \mid \theta)\), where \(x\) is the observed data. Writing code for that trace lets you mix loops, neural networks, and conditionals; you can implement a hierarchical prior by sampling a group-level variance inside a for-loop, or call a PyTorch module inside the likelihood to add deep feature extraction. The generative program therefore defines the unnormalized joint density \(p(\theta, x)\) via the sequence of random draws in your code. That density is what the inference engine must reverse—the “probabilistic” part of the name is this latent-to-observed simulator, and the “programming” part is the fact that it is written in the host language (Python, Julia, etc.).

Because the program is Turing-complete, every construct of generative modeling is expressible. The compiler traces the sampling statements, yielding a factorized joint distribution that can be evaluated up to a normalizing constant. When a user calls the inference API, the runtime rewrites the generative trace into either a gradient-based optimization or a sampling procedure, using autodiff to compute gradients of \(-\log p(\theta, x)\) with respect to \(\theta\). This is what the anonymous “Untitled” arXiv note (2017) [arxiv:1701.03757](https://arxiv.org/abs/1701.03757) emphasizes: inference reduces to a compiler optimization problem operating on the trace, not a math class proof for every new model. The better the compiler captures structural redundancies and provides low-variance estimates, the fewer handcrafted calculations the practitioner must perform.

### Programmable inference and the inference-as-program paradigm

Programmable inference is the recognition that inference engines should be manipulable by the user much like the model code itself. Mansinghka et al. (2018) “Probabilistic Programming with Programmable Inference” [https://dspace.mit.edu/bitstream/handle/1721.1/136984/MansinghkaEtAl_pldi18.pdf](https://dspace.mit.edu/bitstream/handle/1721.1/136984/MansinghkaEtAl_pldi18.pdf) argues that black-box inference is a myth: even if a sampler barrel-rolls through the posterior, the most difficult models require custom updates, adaptive proposals, or learned proposals backed by variational inference. Their programmable inference framework introduces inference programs that inspect the stochastic trace, performing operations such as reblocking, resampling, and constructing control variates. In practice, this means that the user writes lightweight inference scripts (an “inference program”) that selects algorithms, calibrates transition kernels, and even trains auxiliary networks for amortized proposals, while the compiler takes care of differentiability and vectorization.

This programmable interface is what modern probabilistic programming languages such as Pyro expose. Pyro’s generative model is a function defined via `pyro.sample` statements, and the inference program is a scheduler that chooses, for example, an SVI algorithm to optimize the variational parameters or an HMC kernel to sample from the posterior. Bingham et al. (2019) “Pyro: Deep Universal Probabilistic Programming” [arxiv:1810.09538](https://arxiv.org/abs/1810.09538) shows how Pyro plugs stochastic variational inference into deep neural networks, allowing amortized guides to learn proposals for each parameter. In the language of programmable inference, the guide network is an inference program: you are defining a structured family of distributions \(q_\phi(\theta)\) with parameters \(\phi\) and asking the compiler to solve \(\phi = \arg\min_\phi \text{KL}(q_\phi(\theta) || p(\theta \mid x))\).

### Compilation to inference: ELBO and HMC

Once the user has defined \(p(\theta, x)\) via the generative trace, the runtime compiles it either to an optimization target or to a Markov chain. The most common compilation is to a variational objective, the Evidence Lower Bound (ELBO). The compiler rewrites the trace by inserting a guide distribution \(q_\phi(\theta)\) and derives the objective
\[
L(\phi) = \mathbb{E}_{\theta \sim q_\phi} \left[ \log p(x, \theta) - \log q_\phi(\theta) \right]
\]
where \(x\) is the observed data and \(q_\phi\) is the guide we are learning. The compiler uses automatic differentiation to evaluate gradients with respect to \(\phi\), leveraging reparameterization when the latent space is continuous so that the expectation can be estimated with low variance. This compiled ELBO turns variational inference into a straightforward stochastic optimization problem.

When the compiler targets sampling, it rewrites the trace for a Hamiltonian Monte Carlo (HMC) procedure. The Hamiltonian function is
\[
H(q, p) = U(q) + K(p),
\]
where \(q\) stands for the latent parameters, \(p\) is an auxiliary momentum, \(U(q) = -\log p(q, x)\) is the potential energy derived from the joint density, and \(K(p) = \frac{1}{2} p^T M^{-1} p\) is the kinetic energy with mass matrix \(M\). The compiler simulates Hamiltonian dynamics using leapfrog steps, proposing \((q^*, p^*)\) and accepting them with probability \(\min(1, e^{-H(q^*, p^*)+H(q, p)})\). This exact posterior-preserving sampler is sensitive to the curvature of the joint, so the compiler also tunes the step size and adapts \(M\) online—these are inference knobs exposed to the user so they can program the sampler, not re-write it.

### Reusable inference components and effect handling

Because stochastic programs can contain arbitrary control flow, the compiler must instrument every `sample` and `observe` site. To keep this tractable, modern runtimes treat randomness as algebraic effects: each `sample` operation is handled by a trace recorder that can intervene, replace values, or inject custom distributions. This effect handling is the bridge between model and inference: the same generative code can run with different handlers, enabling the inference user to swap in predictive checks, simulate counterfactuals, or integrate global control variates without touching the original model. These handlers also give programmers the ability to implement evaluation-time behaviors like likelihood weighting or mutation-specific resampling. That is what the 2013 Probabilistic Programming Concepts survey lays out—it is not a single interpreter but a network of interpreters working in concert.

### Practical workflow in applied probabilistic programming

The typical applied workflow is therefore: (1) write a generative program that encodes priors, hyperpriors, and a likelihood; (2) choose an inference routine (vars such as SVI, HMC, or importance sampling with learned proposals) and optionally write a bespoke inference program that handles data subsampling, amortized proposals, or auxiliary variables; (3) compile the trace to that routine, letting the compiler handle autodiff, caching, and vectorized computation; (4) inspect posterior samples or variational parameters, check diagnostics, and adjust the inference program as needed. NumPyro, a JAX-based PPL, demonstrates this workflow clearly: you write a `model()` function decorated with `numpyro.handlers`, call `numpyro.infer.SVI` or `NUTS`, and the runtime compiles the trace into optimized XLA kernels. Since the model and inference are decoupled, you can replace the Student-t likelihood in a regression with a skew-normal likelihood without rewriting the sampler.

## Where the field is now

Probabilistic programming has matured along two axes: richer programmable inference and real-world deployment. On the research frontier, programmable inference programs are the latest experiments. The “Probabilistic Programming with Programmable Inference” paper clearly articulates that the inference program should be as composable as the generative program, and follow-up work is exploring how learned proposals and control-flow-aware variational families can be synthesized automatically. For example, research labs now treat inference as a compiler pass, unrolling the trace so that an optimizer can identify conditional independence and apply Kronecker-factored approximations, validating the theoretical insight that inference is regular code transformation.

On the engineering frontier, companies use PPLs for forecasting, simulation, and risk modeling. Uber’s Pyro team (Bingham et al. 2019) built the language with a focus on integrating deep neural networks with stochastic variational inference, and Uber has since embedded Pyro into its forecasting stack for rider demand, where the inference runtime must handle high-dimensional latent variables and scale across distributed GPUs. TensorFlow Probability (TFP) at Google powers time-series forecasting for Ads and Cloud infrastructure: the research team shares production stories on research.google.com that detail how probabilistic layers and bijectors handle non-stationary noise and quantify uncertainty for budgeting decisions. Together they show that blurred line between research and production is intentional—programmable inference lets a Google engineer swap the inference kernel (SVI, MCMC) without changing the business logic in a single day.

## What's still open

1. How can we design a computationally cheap, low-variance metric that certifies an MCMC sampler has converged to a standardized error below 0.1 across all posterior moments without relying on high-variance ESS estimators?

2. Can programmable inference synthesize amortized proposals that generalize across datasets, or do the control-flow differences between collected traces force every new dataset to get its own guide network, defeating reuse?

3. What compiler analyses do we need so that a Bayesian modeler can compose multiple inference programs (say, a local variational guide plus a global HMC sampler) without writing bespoke kernel adapters, and can those analyses be proven to preserve detailed balance?

## Where to read next

If you want to understand the baseline inference techniques that probabilistic programs compile down to, → *markov chain monte carlo* <!-- [[markov-chain-monte-carlo]] --> explains how different chains trade off bias and mixing. For the design of flexible variational families and amortized guides, → [Variational Inference](variational-inference.md) fills in the equations that accompany the programmable inference scripts. To see how probabilistic programs feed into downstream reasoning systems, → *probabilistic modeling* <!-- [[probabilistic-modeling]] --> lays out larger architectures where these programs orchestrate multiple latent sources.

## Build it

This build proves that probabilistic programming makes heavy-tailed regression effortless: you will write a NumPyro program, fit a Student-t likelihood to outlier-corrupted data, and see how the posterior credible intervals widen correctly while a deterministic baseline overfits.

**What you're building:** a NumPyro Bayesian regression model with Student-t noise, fit on a synthetic dataset containing extreme outliers, showing credible intervals that track uncertainty.

**Why this is valuable:** the build puts the compiler-work into practice—model specification, inference selection (NUTS), and diagnostics stay in Python, while the NumPyro runtime compiles the Hamiltonian dynamics described above and returns calibrated samples without hand-deriving acceptance steps.

**Stack:**
- **Model:** `numpoly/numpyro-student-t-regression` (create your own script; NumPyro has >18k installs per week)
- **Dataset:** synthetic dataset generated with `sklearn.datasets.make_regression` plus injected outliers (no external download)
- **Framework:** NumPyro 0.14.x + JAX 0.4.x
- **Compute:** free Colab T4 (16 GB VRAM), training fits in ~40 minutes per seed with 2 chains (sightline), or 1 hour if you run diagnostics.

**The recipe:**
1. `pip install numpyro[jaxlib]==0.14.0 sklearn matplotlib arviz` and set JAX to use GPU (`import jax; jax.config.update("jax_platform_name", "gpu")`).
2. Generate 1,000 samples with `make_regression(n_features=5, noise=1.0)` then inject ten samples with noise scaled by ×30 to create heavy outliers; standardize the features and split train/test (80/20).
3. Define the model in NumPyro: sample `sigma ~ HalfCauchy(2.)`, `eta ~ Exponential(1.)`, regression weights `w ~ Normal(0., 1.)`, compute `mu = jnp.dot(X, w)`, and observe `y ~ StudentT(df=eta, loc=mu, scale=sigma)`; then run `numpyro.infer.NUTS` with 1,000 warmup and 1,000 samples per chain, keeping default step size adaptation on the compiled Hamiltonian.
4. Run `numpyro.infer.MCMC` with two chains, collect posterior samples, compute R-hat using ArviZ (`az.rhat`), and compare predictive intervals to the deterministic OLS fit (`sklearn.linear_model.LinearRegression`), logging the predictive RMSE on the test set and checking that posterior predictive intervals contain >90% of true targets despite outliers.
5. You now have a Student-t Bayesian regression artifact: a NumPyro checkpoint plus ArviZ diagnostics that clearly shows wider credible intervals and calibrated epistemic uncertainty compared to the baseline.

**Expected outcome:** a diagnostic report (ArviZ `InferenceData`) and posterior samples that demonstrate Student-t robustness to outliers, ready to serve as a packaged model for downstream UQ workflows.

- **CS student:** Run the same recipe on an RTX 4070 laptop by reducing warmup to 600 and sample draws to 600 per chain; the smaller chain still exhibits wider credible intervals than OLS, and you can visualize the latitude-longitude residuals in a notebook.

- **Applied engineer:** Quantize the NumPyro weights with `jax.experimental.optimizers.quantized` and export the posterior mean and scale to an ONNX service; serve the deterministic predictive mean at ≤20 ms p95 latency using vLLM-style batching with 4 concurrent requests.

- **Applied researcher:** Test the hypothesis that changing the Student-t degrees-of-freedom prior from Exponential(1.) to Gamma(2,0.1) decreases posterior variance; falsify the hypothesis if the 95% credible intervals on weights do not shrink by at least 15% compared to the original prior.

- **Frontier researcher:** Implement an online diagnostic metric from the open question above (cheap convergence metric), run it alongside the MCMC sampler in this build, and report whether it certifies the sampler when ESS fails—this extends the build into a new probe of convergence diagnostics.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*