---
title: Bayesian optimization
slug: bayesian-optimization
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [bengio, williams, neal, hennig]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [gaussian-processes, acquisition-functions, probabilistic-inference]
tags: [bayesian-optimization, surrogate-models, multi-objective, constrained-optimization, uncertainty-quantification, active-learning]
updated: 2025-04-01
has_mvb: true
---

# Bayesian optimization

Imagine you had to design a next-generation battery material and every candidate costs \$10,000 and three weeks of lab time; you can only run 30 experiments before needing to deliver a prototype. Traditional grid or random search would burn through the budget without ever learning which combinations of precursors and sintering temperatures matter. That is the situation Bayesian optimization was created to avert: it turns each painfully expensive experiment into data that not only informs the current trade-offs but also predicts which experiment will be most informative next. By the end of this page you will see how to cast that high-stakes loop as probabilistic modeling plus acquisition, why modern constraints, risk metrics, and multi-objective objectives can be baked in, what the state of the art looks like, what questions remain, and how to build a working constrained, Pareto-aware loop on free compute.

## The territory

Bayesian optimization occupies the intersection of sequential decision-making, surrogate modeling, and active learning. When evaluations are expensive, noisy, or constrained, the only practical way to make progress is to replace the true objective \(f\) with a cheaper probabilistic surrogate \(g\) and then select points that balance improving our estimate of \(g\) with sampling the most informative region of the space. Classic workflows grew out of metallurgical design, hyper-parameter tuning, and expensive physical simulations, and they typically rely on Gaussian processes (GPs) as the surrogate because GPs provide not only a mean but also calibrated uncertainty. When the objectives are vector-valued, as in chemistry where cost and yield move in opposite directions, we need multi-objective acquisition rules that estimate expected improvements of the Pareto front without reducing everything to a single scalar. When constraints and risk terms such as CVaR matter, we must retrofit those into the acquisition query while honoring noisy feedback from the lab. These needs pull from adjacent families: variational inference anchors the surrogate’s posterior updates, multi-output GP kernels carry structure across objectives, and acquisition-level risk control borrows from reinforcement learning’s exploration/exploitation trade-offs. How does the math tie these pieces together? The mechanism is best understood by starting from the GP surrogate, the acquisition function, and their interplay with constraints, risk, and entropic regularization.

## How it works

The Bayesian optimization loop alternates between fitting a surrogate to past observations and selecting the next query by optimizing an acquisition function. The surrogate is typically a Gaussian process defined by a kernel \(k(x, x')\) that encodes smoothness or other prior knowledge across candidate inputs \(x \in \mathcal{X}\). After \(n\) evaluations at inputs \(\{x_i\}_{i=1}^n\) with noisy observations \(y_i = f(x_i) + \epsilon_i\) where \(\epsilon_i \sim \mathcal{N}(0, \sigma^2)\), the GP posterior produces a predictive mean \(\mu_n(x)\) and variance \(\sigma_n^2(x)\). The acquisition function \(a(x)\) is then a scalar that trades off exploitation (high \(\mu_n(x)\)) with exploration (high \(\sigma_n(x)\)); maximizing \(a(x)\) chooses the next \(x_{n+1}\). Classical choices include Expected Improvement (EI) and Upper Confidence Bound (UCB), but the story gets richer once we add multi-objective, constraints, or higher-order statistical structure.

### Acquisition functions for multi-objective, constrained problems

When we have \(m\) objectives \(f_1, \dots, f_m\), the goal becomes to discover the Pareto front \(P\) rather than a single optimum. The Expected Hypervolume Improvement (EHVI) acquisition function estimates, for each candidate \(x\), how much the hypervolume dominated by the Pareto front will grow if we observe \(f(x)\). Let \(\mathbf{r} \in \mathbb{R}^m\) be a reference point that is worse than the current Pareto approximations. The EHVI for \(x\) can be written as

\[
\mathrm{EHVI}(x) = \mathbb{E}_{\mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}_n(x), \Sigma_n(x))}\big[ \max\{0, \mathrm{HV}(\mathcal{P} \cup \{\mathbf{f}\}) - \mathrm{HV}(\mathcal{P})\}\big],
\]

where \(\boldsymbol{\mu}_n(x)\) and \(\Sigma_n(x)\) are the GP predictive mean vector and covariance matrix across objectives at \(x\), \(\mathcal{P}\) is the current Pareto set, and \(\mathrm{HV}(\cdot)\) computes the hypervolume dominated by a set. The expectation above averages over the multivariate normal capturing the surrogate’s uncertainty, ensuring we keep exploring regions that could expand the Pareto front. EHVI thereby outperforms naïve scalarization in low-data regimes by explicitly reasoning about the nondominated set, which is why the molecule-centric benchmark in *Bayesian Optimization for Molecules Should Be Pareto-Aware* (Raman et al. 2025) [arxiv:2507.13704](https://arxiv.org/abs/2507.13704) shows consistent gains when EHVI is used instead of linear objectives.

Constraints enter the acquisition in a similar probabilistic way. When a constraint \(c(x)\) must remain below zero, we maintain a GP posterior over \(c\) and integrate it into the acquisition as a feasibility probability. For example, a constrained EI may take the form

\[
a(x) = \mathrm{EI}(x) \cdot \mathbb{P}(c(x) \leq 0 \mid \mathcal{D}_n),
\]

where \(\mathcal{D}_n\) denotes the data gathered so far. When the constraints are noisy or safety-critical, we prefer to downweight \(x\) where the posterior probability of feasibility is low, following the recipe from *Constrained Bayesian Optimization with Noisy Experiments* (Facebook AI Research 2024) [https://ai.meta.com/research/constrained-bayesian-optimization-with-noisy-experiments](https://ai.meta.com/research/constrained-bayesian-optimization-with-noisy-experiments). Their framework also emphasizes asynchronous batch evaluations and robust estimates of the feasibility probability under heavy-tailed noise, which matters in real labs where measurement error often deviates from a Gaussian.

Risk-sensitive objectives such as Conditional Value at Risk (CVaR) further complicate acquisition because they depend on tails of the distribution. Instead of optimizing the mean, we can define the posterior predictive CVaR at level \(\alpha\) as

\[
\mathrm{CVaR}_\alpha(x) = \mathbb{E}[f(x) \mid f(x) \geq q_\alpha(x)],
\]

where \(q_\alpha(x)\) is the \(\alpha\)-quantile of the surrogate’s predictive distribution. The acquisition function then targets points that reduce the CVaR, ensuring the worst-case outcomes are manageable. *Bayesian Optimization for CVaR-based portfolio optimization* (Page et al. 2025) [arxiv:2503.17737](https://arxiv.org/abs/2503.17737) builds this idea into a constrained acquisition that analogously integrates risk measures into multi-objective selection, making it possible to keep both expected reward and tail loss under control.

### Entropic regularization and variational inference for surrogate modeling

Fitting the surrogate, especially when we move beyond Gaussian processes to, say, sparse, distributed, or deep surrogates, requires solving an inference problem. Variational inference provides a tractable approximation by minimizing the Kullback–Leibler divergence between the true posterior \(p(\theta \mid \mathcal{D}_n)\) and a parameterized mean-field distribution \(q_\phi(\theta)\). The objective is the Evidence Lower Bound (ELBO):

\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D}_n \mid \theta)] - \mathrm{KL}(q_\phi(\theta) \| p(\theta)),
\]

where \(\theta\) is the surrogate’s parameter vector. *Extending Mean-Field Variational Inference via Entropic Regularization: Theory* (Li et al. 2024) [arxiv:2404.09113](https://arxiv.org/pdf/2404.09113) observes that adding an entropy term to the ELBO faster disperses the surrogate’s belief when data are scarce and that this entropy serves the same purpose as exploration in the acquisition function. The regularized ELBO becomes

\[
\mathcal{L}_{\text{ent}}(\phi) = \mathcal{L}(\phi) + \lambda \mathbb{H}[q_\phi],
\]

where \(\mathbb{H}[q_\phi]\) is the Shannon entropy and \(\lambda > 0\) balances exploration in parameter space. In practice, the entropy regularization prevents the surrogate from collapsing to overconfident predictions that would suppress exploration, which is especially important when constrained or Pareto-aware acquisition functions rely on accurate uncertainty estimates.

Building on this, the 2026 preprints [arxiv:2602.05873](https://arxiv.org/pdf/2602.05873) and [arxiv:2603.08925v1](https://arxiv.org/pdf/2603.08925v1) examine how entropically regularized variational families and auxiliary flows behave when we continuously update the surrogate with new data. They emphasize that maintaining a temperature schedule on the entropy term ties directly to acquisition temperature: a hotter surrogate encourages broader exploration and a colder one more exploitation. The hierarchical nature of Bayesian optimization—where the acquisition’s expectation is itself an integral over the surrogate’s posterior—invites multi-level variational methods. The sample continuation ideas in [arxiv:2604.15469](https://arxiv.org/abs/2604.15469) show that propagating samples through intermediate surrogate updates can reduce variance when we evaluate acquisition gradients for high-dimensional inputs. These works collectively push the surrogate from a static GP toward a dynamic belief model that adapts both mean and uncertainty as new constraints and risk preferences arrive, making acquisition optimization more robust.

### Acquisition optimization, batch selection, and practical implementation

Finding the maximizer of \(a(x)\) is an inner optimization problem that itself requires care. Often \(a(x)\) is non-convex, multimodal, and expensive to evaluate because it contains expectations over the surrogate. Gradient-based optimizers with automatic differentiation (the default in BoTorch and GPyTorch) are the practical standard. Let \(x\) be parameterized through a vector \(z\) and \(x = \phi(z)\) where \(\phi\) handles any constraints via differentiable transforms (sigmoid for bounds, softmax for probabilities). We compute gradients \(\nabla_z a(\phi(z))\) via the reparameterization trick when \(a(x)\) involves expectations of Gaussian random variables:

\[
\mathbb{E}_{\mathbf{f} \sim \mathcal{N}(\boldsymbol{\mu}, \Sigma)}[g(\mathbf{f})] = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}[g(\boldsymbol{\mu} + L \epsilon)],
\]

where \(L\) is the Cholesky factor of \(\Sigma\). The reparameterization ensures gradients flow through both the mean \(\boldsymbol{\mu}\) and covariance \(\Sigma\), which is critical when acquisition functions like EHVI or CVaR involve the covariance across multiple objectives.

Batch selection (parallel evaluation of several candidates) is handled via fantasy models: we temporarily "hallucinate" observations at the selected points and update the surrogate to account for those futures, ensuring the later points in the batch do not collapse onto previously chosen ones. BoTorch supports this through joint acquisition functions, and GPyTorch’s lazy tensors keep the updated covariance efficient even for hundreds of hallucinatory points. The overall workflow is thus: fit the surrogate via variational inference (possibly with entropic regularization), sample hyper-latent variables to approximate the acquisition expectation, optimize \(a(x)\) via gradients, query the true oracle (lab) at the selected points, and update the surrogate with the new noisy measurements.

The interplay of entropic surrogates, risk-aware acquisition, and Pareto-aware criteria is why this toolkit remains relevant for expensive applied problems. The rich theoretical scaffolding from the 2024–2026 preprints ensures that the uncertainty estimates feeding into acquisition optimization are not brittle, the multi-objective EHVI machinery keeps the Pareto front moving forward, and constrained CVaR-aware acquisition functions keep worst-case scenarios in check even as noise overwhelms the few data points we can afford.

## Where the field is now

The research frontier is currently occupied by Pareto-aware and risk-aware acquisition designs that go beyond scalarization. The molecule optimization community demonstrated through *Bayesian Optimization for Molecules Should Be Pareto-Aware* (Raman et al. 2025) [arxiv:2507.13704](https://arxiv.org/abs/2507.13704) that EHVI yields better coverage of non-dominated chemical solutions than any fixed scalarization across six molecular objectives, hitting a Pareto hypervolume 18% larger in the 30-evaluation budget. Parallel to that, the portfolio optimization line, culminating in *Bayesian Optimization for CVaR-based portfolio optimization* (Page et al. 2025) [arxiv:2503.17737](https://arxiv.org/abs/2503.17737), embeds quantiles directly inside the acquisition and shows the risk-infused procedure reduces tail loss by 12 basis points versus vanilla EI without sacrificing mean return. On the safety front, Facebook AI’s *Constrained Bayesian Optimization with Noisy Experiments* (2024) provides the engineering pattern for building asynchronous, noise-aware constraint models, which Meta’s Ax platform now uses internally for thousands of production experiments in recommendation tuning. Ax combines BoTorch’s acquisition modules with database-backed logging to scale Bayesian loops to hundreds of experiments per day while maintaining feasibility guarantees.

The engineering frontier is dominated by platforms that orchestrate Bayesian optimization at scale. Google Research’s Vizier (research.google/pubs/pub45199) now performs millions of evaluations annually by queuing acquisition evaluations across TPU pods, caching surrogate updates, and integrating multi-fidelity sources such as smaller batch sizes; their recent blog reports over 95% of its critical hyper-parameter tuning jobs obey the feasibility filters derived from constrained acquisition functions. Meta AI’s Ax platform (ai.meta.com/research/ax-platform) extends these ideas further with a modular acquisition stack that supports EHVI, CVaR, and constrained extensions, allowing product teams to tune ads systems while honoring safety guardrails imposed by downstream teams. Other teams, including Microsoft Research’s open-source BoTorch contributions, continue to reduce the compute cost of acquisition optimization by improving the GPU kernels for multi-output Gaussian processes.

Thus, the state-of-the-art view is that Bayesian optimization now handles Pareto sets, risk constraints, and noisy feasibility conditions concurrently, with production-scale orchestration provided by Vizier and Ax.

## What's still open

1. Can we optimize over high-dimensional, discrete combinatorial spaces—such as graphs of molecular fragments—without relying on continuous relaxations or retraining the surrogate for every new topology? Current graph-embedding relaxations add bias, and surrogate retraining is costly, so a more sample-efficient surrogate that natively models combinatorial structure would unlock new domains.

2. How can acquisition functions simultaneously reason about risk metrics (CVaR, DRO), Pareto improvement, and procedural constraints without collapsing to overly conservative behavior? Existing approaches trade one requirement for another; the open question is whether a unifying acquisition or a game-theoretic optimization can balance all these axes with provable regret bounds.

3. What theoretical assumptions underlie the entropic regularization schedules in mean-field or hierarchical variational surrogates? The 2024–2026 preprints [arxiv:2404.09113](https://arxiv.org/pdf/2404.09113), [arxiv:2602.05873](https://arxiv.org/pdf/2602.05873), and [arxiv:2603.08925v1](https://arxiv.org/pdf/2603.08925v1) sketch the empirical benefits, but deriving convergence guarantees for acquisition optimization when the surrogate keeps its entropy high remains unsettled.

4. Can sample continuation strategies for hierarchical models (as in [arxiv:2604.15469](https://arxiv.org/abs/2604.15469)) be extended to quantify uncertainty propagation across batched acquisition evaluations, e.g., when a lab returns a noisy, batched result rather than independent scalars?

## Where to read next

If you want to ground the surrogate modeling story in probability theory, → [Gaussian processes](gaussian-processes.md) lays out the covariance kernels and exact inference steps that underpin modern BoTorch implementations. The engineering-level counterpart is → *probabilistic inference engineering* <!-- [[probabilistic-inference-engineering]] --> which explains how those GPs are distributed and quantized in production. For the multi-objective and constraint mechanics, → *pareto frontier optimization* <!-- [[pareto-frontier-optimization]] --> dissects EHVI, hypervolume, and efficient frontier tracking, while the more theoretical take on uncertainty quantification lives in → *variational inference entropy* <!-- [[variational-inference-entropy]] -->.

## Build it

This constrained, multi-objective loop proves that Bayesian optimization can juggle Pareto-sensitive objectives, probabilistic constraints, and explicit risk metrics while running on a single free Colab T4.

**What you're building:** a synthetic chemical optimization loop where BoTorch/GPyTorch models yield the Pareto-optimal yield-versus-cost trade-off under a CVaR-informed feasibility constraint, all orchestrated on a single RTX T4 (Colab).

**Why this is valuable:** it exercises the full stack—GP surrogate, EHVI acquisition, CVaR-aware constraint filtering, entropy-regularized variational updates, and multi-objective evaluation—on compute accessible to every practitioner.

**Stack:**
- **Model:** [davelotito/donut-base-sroie-bayesian-optimization](https://huggingface.co/davelotito/donut-base-sroie-bayesian-optimization) — 3.2k downloads; used here as an OCR-based cost-sheet parser to simulate realistic metadata that feeds into the surrogate.
- **Dataset:** synthetic dataset generated on the fly (Gaussian mixtures for yield, log-normal noise for costs, plus phantom constraint signals).
- **Framework:** BoTorch 0.8 + GPyTorch 2.2 + PyTorch 2.2 (with CUDA 12.0 backend).
- **Compute:** Google Colab T4 (16 GB VRAM, ~1 hour for the full build).

**The recipe:**
1. Install and import BoTorch, GPyTorch, PyTorch, and Hugging Face Transformers; load `davelotito/donut-base-sroie-bayesian-optimization` once to parse a mock cost sheet image into structured constraint inputs and cache the tokenizer for future runs.
2. Create the synthetic objectives: yield \(y_1(x)\) as a mixture of sinusoids plus Gaussian noise, cost \(y_2(x)\) as a log-normal function of \(x\), and a noisy safety constraint \(c(x)\) drawn from a Student-t distribution; standardize them to unit variance.
3. Initialize a multi-output GP surrogate with an RBF kernel across objectives, and a separate GP for the constraint; fit both via maximum likelihood with entropic regularization added to the ELBO following the insights from [arxiv:2404.09113](https://arxiv.org/pdf/2404.09113) to prevent overconfidence.
4. Implement EHVI acquisition with reparameterized expectations for the predictive multivariate normal and weight it by the CVaR-informed feasibility probability derived from the constraint GP’s predictive Student-t tails; optimize the acquisition using Adam over 100 restarts and 64 steps per restart, logging the acquisition value per iteration.
5. Evaluate the loop by sampling 30 iterations, plot the evolving Pareto front (yield vs. cost), and report CVaR (\(\alpha=0.95\)) on the sampled points; the artifact is a checkpointed BoTorch loop plus a visualization notebook showing the Pareto front growing with each iteration.

**Expected outcome:** a Colab notebook that reproduces the constrained Pareto front, logs the acquisition curve, and saves checkpoints of the surrogate; this notebook can be rerun on any T4-level GPU to explore different noise levels or constraint thresholds.

- **CS student:** Swap the optimizer to L-BFGS-B with 200 inner steps and use the free RTX 4070 in your lab machine; the extra precision should tighten the Pareto front within the same budget.
- **Applied engineer:** Quantize the GP kernel parameters to float16, export the acquisition optimizer as TorchScript, and serve the loop via a FastAPI endpoint producing acquisition suggestions with p95 latency ≤ 180 ms under batch size 4.
- **Applied researcher:** Replace EHVI with a CVaR-weighted Expected Improvement and compare the resulting Pareto fronts; the hypothesis is that CVaR-EI will dominate in the lower-left cost-yield region, with the falsification criterion being a ≥5% drop in hypervolume compared to standard EHVI.
- **Frontier researcher:** Extend the loop by integrating a graph-based surrogate that natively handles discrete molecular graphs (addressing the open question in §What’s still open) and measure whether the hypervolume gap closes without retraining the surrogate from scratch after each batch.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*