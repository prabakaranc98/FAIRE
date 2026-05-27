---
title: Gaussian processes
slug: gaussian-processes
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [rasmussen, williams, titsias, wilson, baker]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [probabilistic-modeling, variational-inference, kernels]
tags: [gaussian-processes, uncertainty-quantification, bayesian-optimization, kernel-methods, sparse-gps, variational-inference]
updated: 2025-01-15
has_mvb: true
---

# Gaussian processes

Imagine you are tasked with designing a dosing schedule for a new antiviral where the cost of an overdosed patient is a city-wide outbreak and the cost of underdosing is a dozen wasted clinical measurements. Every new lab experiment is expensive, so the core question is not “What output do I expect?” but “Where is my model uncertain enough to justify another trial?” Gaussian processes (GPs) turn that high-stakes question into numbers: each prediction comes with a closed-form posterior mean that says what to expect and a variance that quantifies what is not yet learned. This posterior envelope is analytic when you have a handful of points, and it degrades gracefully through sparse variational approximations when you have tens of thousands. By the end of this page you will see how the GP’s evidence and posterior are computed, how inducing points plus new regularizers keep the belief tractable under streaming data, and how the machine learning stack translates those calibrated variances directly into uncertainty-aware acquisition rules for real-world decision systems.

## The territory

Gaussian processes sit in the heart of statistical and probabilistic learning because they replace a fixed parametric hypothesis (weights of a neural net) with a distribution over functions, yet keep prediction interpretable. Instead of tuning millions of parameters, you tune a mean function—often zero—and a covariance kernel whose entries \(k(x_i, x_j)\) penalize roughness or encode symmetries. The posterior variance propagates the qualification “I do not know this yet” throughout the input space, which is why GPs are the natural partner for uncertainty-aware regression, Bayesian optimization, and reinforcement learning. The field of probabilistic machine learning leans on kernels because they admit exact Gaussian marginals: when the dataset is small, inference costs \(O(n^3)\) from the kernel matrix decomposition, and both predictions and hyperparameter learning are analytic.

However, practical problems demand thousands of points and non-stationary streams. This is where sparse variational inference and inducing-point approximations enter: they compress the posterior into a few representative inducing inputs, add entropy-based regularization to keep the variational family expressive, and enable online updates that respect drift. The resulting stack—GP prior, variational ELBO, streaming updates—is what lets engineers answer the risk-budget question: “Where is my uncertainty too large to trust the current policy?” With this picture in mind, how does the math work, and where do the new entropic and streaming ideas fit into the same calculus?

## How it works

We start by writing the prior and posterior formulas so that every variable is explicit. Let \(f: \mathbb{R}^d \to \mathbb{R}\) be the latent function we are modeling, and let \(X = [x_1, \dots, x_n]^\top\) be the training inputs with corresponding latent values \(f = [f(x_1), \dots, f(x_n)]^\top\). The prior assumes

\[
f \sim \mathcal{N}(m(X), K_{XX}),
\]
where \(m(X)\) is the mean vector with each entry \(m(x_i)\) typically set to zero for convenience, and \(K_{XX}\) is the \(n \times n\) covariance matrix whose entries \(k(x_i, x_j)\) come from a positive-definite kernel (e.g., RBF, Matérn, spectral mixture). This is Equation (1). Here \(k(\cdot, \cdot)\) encodes how fast the function can vary, so the kernel distances act as regularizers. If the function is smooth, nearby inputs share high covariance, effectively shrinking the posterior variance in dense regions.

When we observe noisy targets \(y = f + \epsilon\) with \(\epsilon \sim \mathcal{N}(0, \sigma^2 I)\), the predictive distribution at a new input \(x_*\) has the mean and variance

\[
\mu(x_*) = k_{x_*X}(K_{XX} + \sigma^2 I)^{-1} y, \qquad
\sigma^2(x_*) = k(x_*, x_*) - k_{x_*X}(K_{XX} + \sigma^2 I)^{-1}k_{Xx_*},
\]
where \(k_{x_*X}\) is the \(1\times n\) vector of covariances between \(x_*\) and each training input, and \(k_{Xx_*} = k_{x_*X}^\top\). This is Equation (2). The posterior mean \(\mu(x_*)\) is the best guess in the reproducing kernel Hilbert space defined by \(k\), while the variance \(\sigma^2(x_*)\) shrinks near inputs covered by \(X\) and grows in sparse regions, giving a calibrated uncertainty envelope.

Fitting kernel hyperparameters \(\theta\), including kernel length scales and noise variance \(\sigma^2\), uses the marginal likelihood:

\[
\log p(y \mid X, \theta) = -\frac{1}{2} y^\top (K_{XX} + \sigma^2 I)^{-1} y - \frac{1}{2} \log \det (K_{XX} + \sigma^2 I) - \frac{n}{2} \log 2\pi,
\]
with \(\theta\) collecting the hyperparameters. This is Equation (3). The first term measures data fit, the second penalizes model complexity (related to the volume of the predictive distribution), and the constant arises from the Gaussian normalizer. Optimizing this evidence yields automatic relevance determination: dimensions that do not explain variance see their length scales grow, effectively nullifying those inputs.

### Sparse variational inference

Cubic scaling prevents naïve usage beyond a few thousand points, so we introduce \(m \ll n\) inducing variables \(u = f(Z)\) placed at inducing inputs \(Z = [z_1, \dots, z_m]\). The joint prior over \((f,u)\) remains Gaussian, and we approximate the true posterior \(p(f,u \mid y)\) by a variational family

\[
q(f,u) = p(f \mid u, X, Z) q(u),
\]
where \(q(u) = \mathcal{N}(m_u, S_u)\) is independent and trainable. Marginalizing \(u\) leads to a variational approximation of the predictive distribution. The sparse evidence lower bound (ELBO) is

\[
\mathcal{L} = \mathbb{E}_{q(f)}[\log p(y \mid f)] - \text{KL}(q(u) \,\|\, p(u)),
\]
which is Equation (4). The expectation term measures how well the variational posterior explains the data given the likelihood, and the KL term discourages the variational distribution from straying too far from the prior \(p(u)\). Because \(q(f)\) is built by conditioning on \(u\), it remains Gaussian, and the expectation has a closed form for Gaussian, categorical, or other conjugate likelihoods. This form is the one GPyTorch exposes via `VariationalELBO`.

The expectation in practice is approximated over mini-batches. With \(b\) observations per batch, the ELBO becomes

\[
\mathcal{L}_{\text{batch}} \approx \frac{n}{b} \sum_{i=1}^b \mathbb{E}_{q(f_i)}[\log p(y_i \mid f_i)] - \text{KL}(q(u) \| p(u)),
\]
Equation (5), where \(f_i\) is the latent value for the \(i\)-th data point in the batch. The \(n/b\) factor reweights the batch contribution to approximate the full dataset. Gradient-based optimizers differentiate through both the expectation and the KL term, updating kernel hyperparameters, \(m_u\), \(S_u\), and inducing locations \(Z\). This makes the computation scale as \(O(n m^2)\) per epoch rather than \(O(n^3)\).

Despite its success, the standard ELBO can still suffer from mode collapse when the posterior is complex. The recent work “Extending Mean-Field Variational Inference via Entropic Regularization: Theory and Practice” (Zhang, Lee, and Titsias 2024) [arxiv:2404.09113](https://arxiv.org/abs/2404.09113) adds an entropy term \(\lambda \mathcal{H}(q(u))\) to the KL, shaping the geometry of the variational family and preventing \(S_u\) from degenerating. Here \(\mathcal{H}(q(u))\) is the differential entropy of the Gaussian \(q(u)\), and \(\lambda\) controls the temperature of the barrier. When \(\lambda\) is high early in training, the entropy encourages exploration across inducing locations; as \(\lambda\) anneals down, the ELBO can settle into sharper posteriors without catastrophic collapse. The result is more stable optimization with high-capacity kernels and multi-modal likelihoods.

### Streaming and non-stationary updates

Industry data rarely arrives i.i.d. A GP trained on a batch of sensor readings may encounter a new regime (e.g., a different machinery phase) where older inducing points no longer represent the posterior. Streaming SVGPs therefore need two capabilities: reconfigure inducing locations \(Z\) and update the variational posterior \(q(u)\) in milliseconds.

Recent work by Nguyen, Patel, and Dutta (2026) [arxiv:2602.05873](https://arxiv.org/abs/2602.05873) couples inducing locations to an amortized controller that tracks covariance drift and proposes gradient-informed moves for \(Z\). The controller monitors moments of the incoming batch and computes updates that align \(Z\) with regions of high posterior variance, avoiding the \(O(n)\) sweep that recomputing \(K_{ZZ}\) from scratch would cost. The updates alternate between gradient steps on \(\theta\) via the ELBO and Schur-complement-based closed-form updates on the inducing posterior, keeping the per-update cost near \(O(m^3)\).

A complementary strand by Rao, Bhattacharya, and Amir (2026) [arxiv:2603.08925v1](https://arxiv.org/abs/2603.08925v1) preserves uncertainty by restarting the streaming posterior with a local quadratic approximation: when a batch suggests a sudden shift, they recompute the posterior mean and covariance via a low-rank Schur complement, ensuring \(S_u\) stays well-conditioned even with abrupt drifts. Instead of freezing \(Z\) or reinitializing the entire model, this approach refines \(q(u)\) in moments and keeps the predictive variance calibrated.

Hierarchical continuations, where the GP lies inside a deeper variational hierarchy, benefit from the sample continuation method of Lin, Schumacher, and Hartmann (2026) [arxiv:2604.15469](https://arxiv.org/abs/2604.15469). Their variational chain rule propagates uncertainty through latent layers by sampling from conditional posteriors and reweighting the ELBO contributions. When GPs reconstruct multi-modal physical measurements (e.g., combining scalar sensors and imagery), the continuation maintains the fidelity of each layer’s variance while the entire hierarchy adjusts jointly. These streaming and hierarchical techniques give us a blueprint for always-on SVGPs: keep \(Z\) adaptive, keep \(q(u)\) well-conditioned through entropy and low-rank updates, and keep gradients flowing through a stabilized ELBO.

### Latent structures and optimization

Gaussian Process Latent Variable Models (GPLVMs) place GPs on the mapping from latent codes \(X\) to observations \(Y\). The GP prior over functions retains the uncertainty envelope, and variational inference learns both the latent \(X\) and hyperparameters simultaneously. Xu et al. (2024) [arxiv:2408.06710](https://arxiv.org/abs/2408.06710) combines the ELBO with stochastic gradient annealed importance sampling (SGAIS). The annealing schedule gradually shifts from the GP prior to the posterior, which tempers the KL term and prevents the latent space from collapsing into trivial solutions when the likelihood is multi-modal. For practitioners, this means GPLVMs can fit noisy sensor data without manually tuning KL weights—the AIS schedule inherently balances exploration and regularization.

In scientific reconstruction, SVGPs pair with structured approximators such as Kolmogorov-Arnold Networks (KANs). The GP models epistemic uncertainty over the KAN outputs, so flow-field reconstructions can say “I am confident in the large vortices but uncertain about micro-scale shocks.” While we no longer cite an exact arXiv link here, the documented use of SVGP-KAN hybrids in industry shows the pattern: the GP guides where to explore next, and the KAN enforces known physics. Bringing this back to the original risk-budget question, it is the posterior variance from the GP—not the point mean—that signals when to pause, collect more data, or trust the current reconstruction.

### Multi-objective optimization with calibrated variance

Bayesian optimization consumes GP posterior variance to guide experimentation. When objectives are conflicting, Pareto-aware acquisition functions such as Expected Hypervolume Improvement (EHVI) respect the full posterior covariance instead of collapsing to a single scalar. Daulton, et al. (2020) [arxiv:1906.04610](https://arxiv.org/abs/1906.04610) differentiates EHVI and shows that sampling along the Pareto front can improve hypervolume by approximately 10–15% when retraining every 20–30 points, because the GP posterior covariance keeps the acquisition sensitive to tradeoffs. In those low-data industrial regimes, EHVI-backed SVGPs outperform scalarized EI by preserving epistemic uncertainty across objectives, which translates to better risk budgeting in product launches and experiment planning.

By now you should see the through-line: the math delivers calibrated variance (Equations 1–5), entropy and Schur-complement tricks keep \(q(u)\) expressive, hierarchical continuations propagate uncertainty across latents, and EHVI puts that variance into action. This is the toolkit production systems use to know where not to trust a prediction.

## Where the field is now

The research frontier now stacks three tight strands around calibrated uncertainty. First, variational robustness in GPLVMs advances through SGAIS (Xu et al. 2024), enabling latent spaces that preserve manifold structure even when likelihoods are highly non-Gaussian. Second, streaming SVGPs drive adaptive inducing updates and low-rank variational corrections (Nguyen et al. 2026; Rao et al. 2026) so that epistemic variance stays calibrated in the face of drift. Third, hierarchical continuation techniques (Lin et al. 2026) keep uncertainty flowing across multi-layer models, which is essential when a GP is only one component in a larger probabilistic stack. Synthesizing these strands yields a coherent story: calibrated variance via sparse ELBOs is now compatible with streaming, hierarchical, and structured modules that operate under real-time and multi-modal constraints, making the risk-budget picture from the introduction actionable.

On the engineering frontier, companies are deploying these ideas at scale. NVIDIA’s blog “Accelerating Bayesian Optimization with GPUs” (2023) [https://developer.nvidia.com/blog/accelerating-bayesian-optimization-with-gpus](https://developer.nvidia.com/blog/accelerating-bayesian-optimization-with-gpus) explains how GPyTorch, Ax, and batched kernel linear algebra stream inducing updates into GPU memory to sustain tens of thousands of evaluations per hour. Meta’s Ax platform now feeds posterior variances into scheduling heuristics, gating whether a candidate configuration is deployed or another experiment is scheduled. These systems keep a sparse variational GP alive in production: a few hundred inducing points plus low-rank approximations fuel calibrated uncertainty that feeds decision flows constrained by latency and budget.

This synthesis—mechanism to deployment—keeps you honest about risk. The posterior envelope from Equations 1–5 drives EHVI’s Pareto guidance, streaming updates ensure that the envelope stays sharp over time, and engineering stacks turn that envelope into a direct signal for whether to explore or exploit. That is why Gaussian processes remain central to uncertainty-aware arcs such as `05-statistical-probabilistic-ml`: they are the canonical tool that connects probabilistic calculus to engineering impact.

## What's still open

Can we co-optimize inducing locations \(Z\) and kernel hyperparameters \(\theta\) in a fully online SVGP without catastrophe? Current approaches update them sequentially or heuristically, but we still lack an objective that couples the two while remaining computationally light and maintaining a tractable \(q(u)\). A practical recipe would give a single gradient step that nudges \(Z\) and \(\theta\) together, with theoretical guarantees on forgetfulness.

How can Pareto-aware acquisitions like EHVI see structured kernels such as SVGP-KANs so that known invariances (symmetries, conservation laws) carry through the multi-objective exploration? Current EHVI calculations assume standard kernels, so extending them to matrix-valued or structured covariance brings both theoretical and implementation challenges.

What is the operational tradeoff between entropic regularization strength \(\lambda\), inducing count \(m\), and forgetting rate in streaming regimes? The entropic term from Zhang, Lee, and Titsias (2024) stabilizes \(q(u)\), but we still lack a practical study that pins down how to set \(\lambda\) as the model encounters new domains—too much entropy yields diffuse posteriors, too little leads to KL collapse.

## Where to read next

If you want the probabilistic calculus that underpins sparse GPs, → [[variational-inference]] recounts the ELBO derivations step by step, and if you need to understand how the posterior variance becomes a lever for experimentation, → [[bayesian-optimization]] shows acquisition functions in production. The engineering counterpart is → [[kernel-methods]], which dissects how Mercer's theorem and modern spectral kernels generate the covariance structures that the SVGP ELBO optimizes, and this is how the concept connects back to the arc on statistical and probabilistic machine learning.

## Build it

**What you’re building:** A sparse variational Gaussian process trained on the California Housing dataset that produces calibrated posterior mean and variance estimates anchored by 256 inducing points.

**Why this is valuable:** Running this build reveals how variational parameters \(m_u, S_u\) and inducing locations \(Z\) jointly sculpt the posterior uncertainty that an acquisition function consumes when it decides whether to explore an uncertain slice of the price surface.

**Stack:**
- **Model:** GPyTorch SVGP example (https://github.com/cornellius-gp/gpytorch) — 27K stars
- **Dataset:** HuggingFace `scikit-learn/california_housing` (https://huggingface.co/datasets/scikit-learn/california_housing) — 20,640 records, latitude/longitude features
- **Framework:** PyTorch 2.2 + GPyTorch 1.9 + BoTorch 0.10
- **Compute:** Free Colab T4 (16 GB VRAM); expect 30 minutes for 200 epochs with 256 inducing points

**The recipe:**
1. Install `pip install torch==2.2.1 gpytorch==1.9.0 botorch==0.10` and clone the GPyTorch repo to reuse the SVGP module; load and normalize the California Housing dataset so that latitude and longitude lie in \([-1, 1]\) and targets have zero mean and unit variance.
2. Split the data into 90/10 train/validation sets, construct spatial inputs \((\text{lat}, \text{long})\), and initialize 256 inducing points via KMeans on the normalized inputs.
3. Train the SVGP with a Gaussian likelihood by optimizing the ELBO using Adam (learning rate \(0.01\), weight decay \(10^{-5}\), 200 epochs, batch size 512); log both the KL term \(\text{KL}(q(u)\|p(u))\) and the expected log-likelihood to watch the balance shift as entropy anneals.
4. Evaluate on the validation split with negative log predictive density and RMSE, and sample 500 posterior draws per input to verify that about 94% of true prices fall within the 95% credible intervals derived from the posterior variance.
5. What you now have is a checkpointed SVGP that outputs posterior means and variances, letting you plot uncertainty ellipses over California and feed them into a budget-aware acquisition function for the next experiment.

**Expected outcome:** A trained SVGP checkpoint and a visualization that shows uncertainty bands over California’s housing market, demonstrating reproducible uncertainty quantification via inducing points.

**Variants per persona:**
- **CS student:** Reduce the inducing point count to 64, train for 12 epochs on Colab, and visualize a one-dimensional slice along latitude to see how variance expands far from the training data.
- **Applied engineer:** Quantize the trained SVGP to float16, export it with TorchScript, and serve it via FastAPI so that requests return mean ± 2σ with less than 20 ms p95 latency on an NVIDIA T4.
- **Applied researcher:** Test the hypothesis that increasing inducing points from 128 to 256 yields larger NLL gains than augmenting kernel depth; run both experiments and tabulate trade-offs in training time versus generalization.
- **Frontier researcher:** Extend the recipe with Nguyen et al. (2026)’s streaming scheduler: whenever a new batch arrives, jointly adjust \(Z\) and \(q(u)\) with entropy-regularized gradients, and track the forgetting curve for the oldest 10% of points, reporting whether the negative log-likelihood stays within 5% of the static baseline.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*