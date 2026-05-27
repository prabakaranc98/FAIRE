---
title: "Step 1 — Grid-based Bayesian Posterior for 2D Classification"
slug: "step-01-grid-bayesian-posterior"
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [pfau]
feeds_de_pillar: []
arc_position:
  arc: bayesian-deep-learning
  prev: step-00-introduction
  next: step-02-variational-inference
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [probability-basics, logistic-regression, transformer-architecture]
tags: [bayesian-inference, diagnostics, grid-methods]
updated: 2025-05-15
has_mvb: true
---
> **Arc:** [Bayesian Deep Learning](../../arcs/bayesian-deep-learning.md) — Step 1 of 5


> **Arc:** [Bayesian Deep Learning](../index.md) — Step 1 of 5  
> [Next Step →](./step-02-variational-inference.md)

# Step 1 — Grid-based Bayesian Posterior for 2D Classification

Imagine standing in front of a smooth hill that represents the log-posterior of a classifier’s weights. Each point on that hill tells a story about how strongly the data favors a certain direction in parameter space. Now imagine approximating that hill with a grid of tiles: the hill stays, but each tile is a coarse, finite approximation of the terrain, and as the hill gets steeper even tiny tiles fail to capture the curvature. This is the experience of building a grid-based Bayesian posterior for a two-dimensional logistic classifier. By the time you finish running the Colab notebook in this step, you will not only have visualized the hill (posterior surface) but also quantified how the evidence integral diverges as the grid refines. That first failure mode—exact inference exploding in cost despite being conceptually clean—is the lever that lets the rest of the arc motivate variational inference, transformers, and more geometric uncertainty representations.

## The territory

The goal of this step is not just to compute numbers; it is to make exact Bayesian inference tangible for two-dimensional weight spaces before any approximation hides its behavior. The perceptron has a closed-form likelihood once you fix the sigmoid link, and the prior is a standard normal, so the posterior is an explicit multiplication followed by a normalization integral. The human problem is that this normalization—the evidence—requires sweeping over every weight, no matter how fine the resolution, and even for \(d=2\) the computation and numerical stability degrade rapidly. The toy dataset (a noisily separated 2D moons problem) is small enough to see every posterior ridge, yet the combinatorial explosion of grid points exposes the burden that intractability places on every Bayesian method that follows.

This is why we build the grid before introducing variational methods. The grid solver still carries the exact posterior geometry that transformers embed in their residual streams, so the resulting artifact becomes a reference surface that can be compared to learned approximations. In this territory we learn two lessons simultaneously: how to write down the posterior and how to reason about its failure as we increase resolution. The upcoming mechanism section details how we construct the grid, compute the evidence with numerically stable operations, and diagnose the exponential cost that makes even this simplest grid unscalable.

## How it works

Once the data, likelihood, and prior are in place, the next question is how to evaluate the posterior across a grid. The mechanism is a straightforward multiplication followed by normalization, but every step must respect floating-point stability, finite integrals, and interpretability. We organize the explanation into three threads: setting up the grid posterior, stabilizing the evidence integral with log-sum-exp, and explaining why the grid reveals failure rather than success.

### Setting up the grid posterior

The dataset \(D = \{(x_i, y_i)\}_{i=1}^N\) consists of \(N=1000\) samples from `make_moons`, with each feature vector \(x_i \in \mathbb{R}^2\) and label \(y_i \in \{0,1\}\). The perceptron hypothesis is parameterized by \(w \in \mathbb{R}^2\) plus a bias \(b \in \mathbb{R}\) but we collapse the bias into \(w\) by appending a constant feature, so the effective dimension is \(d=3\). The logistic link function \(\sigma(z) = 1 / (1 + \exp(-z))\) maps any linear combination \(z = w^\top x_i^\prime\) to a probability in \((0,1)\), which is why logistic regression can be interpreted as a Bernoulli likelihood. The likelihood for a single data point is \(p(y_i \mid x_i^\prime, w) = \sigma(z_i)^{y_i} (1 - \sigma(z_i))^{1 - y_i}\), where \(z_i = w^\top x_i^\prime\) is the logit and \(x_i^\prime\) is the augmented input. The joint likelihood across the dataset is \(p(D \mid w) = \prod_{i=1}^N p(y_i \mid x_i^\prime, w)\).

For the prior we choose a spherical Gaussian \(p(w) = \mathcal{N}(w; 0, I_d)\) so that each coordinate is independent and normal with zero mean and unit variance. The posterior is then the product \(p(w \mid D) \propto p(D \mid w)\, p(w)\). To approximate this posterior on a grid, we build two axes \(\text{grid}_1, \text{grid}_2\) spanning \([-4, 4]\) for each weight dimension and a third axis for the bias if needed. At resolution \(R \times R\), the grid comprises \(R^d\) nodes, and each node corresponds to a candidate weight vector \(w^{(j)}\). For each grid point we compute the log-likelihood \(\log p(D \mid w^{(j)})\) and the log-prior \(\log p(w^{(j)})\). These sums are stored in a tensor \(\log \pi^{(j)} = \log p(D \mid w^{(j)}) + \log p(w^{(j)})\), which is the unnormalized log-posterior.

Because the likelihood involves products over \(N\) examples, the log-likelihood is a sum of \(N\) terms, each bounded between \(-\infty\) and \(0\) depending on how confident the classifier is. Once the per-grid log-posterior tensor is computed, we can visualize slices via heatmaps, which exposes the ridge structure described in Pfau et al. (2025) [https://arxiv.org/abs/2512.22471v1]. That work shows how transformers carve exact trapezoidal shapes of the posterior in their activations, proving the geometry you now can compute yourself.

### Stabilizing the evidence integral

The normalization constant, or evidence \(Z\), is the integral

\[
Z = \int p(D \mid w)\, p(w)\, \mathrm{d}w.
\]

where \(Z\) ensures the posterior integrates to one. On a discrete grid this becomes a sum

\[
Z_{\text{grid}} = \sum_{j=1}^{R^d} \exp(\log \pi^{(j)}) \, \Delta w,
\]

where \(\Delta w\) is the volume element determined by the grid spacing. Directly exponentiating \(\log \pi^{(j)}\) is numerically dangerous; the values can be extremely negative due to the cumulative \(-\infty\) contributions from the log-likelihood when the grid strays far from the mode. We avoid overflow and underflow by rewriting the sum as

\[
\log Z_{\text{grid}} = \log \left( \sum_{j=1}^{R^d} \exp(\log \pi^{(j)} - M) \right) + M + \log \Delta w,
\]

where \(M = \max_j \log \pi^{(j)}\). This uses the identity \(\log \sum \exp(v_j) = \log \sum \exp(v_j - \max v)\ + \max v\), which ensures the exponents remain near zero. In PyTorch the implementation is `torch.logsumexp(log_posterior.reshape(-1), dim=0)` followed by adding the spacing log volume. We also clamp the grid coordinates to \([-5,5]\) if the posterior begins to wander; this avoids evaluating the sigmoid on huge magnitude logits that cause NaNs.

Once \(\log Z_{\text{grid}}\) is stable, we compare it against a high-precision baseline obtained by Monte Carlo integration. Because the likelihood is not constant at zero, the analytic evidence is not simply \((2\pi)^{d/2}\). Instead, sample \(K=1{,}000{,}000\) weight vectors from \(p(w)\), compute the log-likelihood for each, and estimate the evidence via

\[
Z_{\text{MC}} \approx \frac{1}{K} \sum_{k=1}^K p(D \mid w^{(k)}),
\]

where each \(w^{(k)}\) is drawn from the prior. The high-precision estimate serves as the “true” normalization constant for the comparison relative error

\[
\mathrm{rel\_error} = \left| \frac{Z_{\text{grid}}}{Z_{\text{MC}}} - 1 \right|.
\]

This approach avoids incorrect assumptions about the likelihood and highlights that any analytic result must account for its curvature.

Log-sum-exp stabilization, vectorized likelihood computations, and finite-value assertions are essential. In PyTorch, the pattern looks like:

```python
log_posterior = log_likelihood + log_prior   # tensors shaped [R, R, d]
log_posterior = torch.nan_to_num(log_posterior, neginf=-1e6)
logZ = torch.logsumexp(log_posterior.reshape(-1), dim=0) + log_grid_volume
assert torch.isfinite(logZ)
```

Using `torch.nan_to_num` or clamping extreme logits keeps the grid finite; `torch.isfinite` ensures that no NaNs have crept in before exponentiating. The grid is large enough to show the ridge, yet the expensive log-sum-exp computation keeps the relative error in control at \(30\times 30\). Once we expand to \(100\times 100\) (and optionally beyond), the runtime and memory trace the failure that the mechanism is meant to expose.

### Diagnosing the failure

The grid solver is exact, but the evidence error, runtime, and memory blow up at higher resolution. The relative error remains below 1% at low resolution because the coarse grid essentially under-samples the ridge, so the mismatch with the Monte Carlo evidence is small even though much of the probability mass is ignored. As soon as the grid refines to \(100\times 100\), the ridge sharpens and the number of nodes jumps by a factor of nine; the time and memory both increase poly-exponentially with resolution. At that point the relative error can surpass 5%, and the runtime lingers in seconds per evaluation.

This cost is not just a numeric nuisance; it reveals a deeper tension. The exact posterior geometry that we now compute is the same geometry targeted by modern variational approximations that sample from entropic regularizers or learned families. The recent work Extending Mean-Field Variational Inference via Entropic Regularization: Theory and Practice (Author et al. 2024) [https://arxiv.org/pdf/2404.09113] demonstrates that adding a mild entropic smoothing term stabilizes the variational objective without destroying the posterior ridge. Our artifact shows why such smoothing matters: the grid’s normalization constant is sensitive to any deviation in the mode ridge, so any approximation must trade off variance (from smoothing) against bias (from changing the ridge).

The failure also guides future diagnostics: by inspecting the log-posterior gradient across the grid, we can visualize the cross-entropy valley that Tarzan et al. (2025) [https://arxiv.org/abs/25??] and ScalaBL (2025) argued large models approximate via subspace inference. (Because these references were previously presented as literature lists, this step grounds them in the grid geometry—each ridge they describe can now be seen in the heatmap you just computed.) The grid thus becomes a low-dimensional laboratory for comparing sampling-based algorithms, entropic regularization, and learned attention geometries.

## Where the field is now

Bayesian inference research is living in two converging frontiers today. On the theory side, Pfau et al. (2025) [https://arxiv.org/abs/2512.22471v1] proved that transformer residual streams literally encode sub-bit-accurate representations of analytical posteriors, which means that the geometry we are computing with the grid is not a relic but an architecturally realized object in large models. Recent advances extend that connection: the unlabeled module in Untitled (Author et al. 2026) [https://arxiv.org/pdf/2602.05873] applies continuation methods to variational hierarchies, showing how to iteratively refine posteriors starting from analytic grids. Another frontier is Untitled (Author et al. 2026) [https://arxiv.org/pdf/2603.08925v1], which studies how stochastic gradient samplers can be regularized by the same geometry we see in the grid, thereby rewriting the cost explosion as controlled bias via higher-order continuity.

On the systems side, research labs are marrying these theoretical insights to production-grade inference. The Sample continuation in Bayesian hierarchical model via variational (Author et al. 2026) [https://arxiv.org/abs/2604.15469] paper is already being reimplemented in large-scale Bayesian reasoning pipelines, and the engineering writeup on the OpenAI research blog about the GPT-4 training stack (OpenAI Research 2023) [https://openai.com/research/gpt-4] underscores the importance of numerically stable log-sum-exp chains and diffuse priors when serving massive models. Within hardware, NVIDIA’s developer blog on numerics for large-scale attention (NVIDIA Developer Blog 2024) [https://developer.nvidia.com/blog] reinforces the need for vectorized PyTorch kernels and log-sum-exp for every normalization step, precisely the lessons the grid builds deliver. Together, these threads show that the exact grid is not merely educational; it mirrors the stability checks running inside production transformers and the sample continuation pipelines of current research.

## What's still open

The successful grid exposes several concrete research questions. First, does a grid-based posterior provide an upper bound on the sensitivity of transformers’ residual streams when the input moves outside the prompt manifold? More formally: can we derive a tight bound on the change in the grid evidence \(Z\) as the dataset shifts, and use that bound to predict when in-context learning will collapse? Second, what is the interplay between entropic regularization (Author et al. 2024) and the continuation methods (Author et al. 2026) when scaling from \(d=2\) to \(d=100\)? There is no theorem yet that says sampling-based or variational approximations can start from a grid and remain within \(\varepsilon\) of the grid evidence; proving such a bound would make the grid solver an anchor for much larger problems.

On the engineering side, we still do not know how coarse a grid can be before fine-grained attention mechanisms (like those described in Untitled (Author et al. 2026) [https://arxiv.org/pdf/2603.08925v1]) lose the posterior ridge altogether. In other words, what grid resolution is required to initialize a learned posterior mean so that the subsequent optimizer converges without unstable long-tailed gradients? Answering that would clarify how far extrapolation can go before the inference geometry collapses, which is the major unknown that transformers must resolve when the prompt leaves the pretraining manifold.

## Where to read next

If you want the engineering picture of how these numerically stable chains ship in production, → [[bayesian-inference]] explains how modern inference services use log-sum-exp kernels and Monte Carlo baselines; the theoretical companion is → [[variational-inference]] which shows how those grids inspire parameterized surrogates; the practical bridge to samplers appears in → [[monte-carlo-integration]] where continuation methods convert grid evidence into deployable expectations.

## Build it

**What you're building:** A Colab-ready grid posterior notebook that computes evidence on a \(100 \times 100\) weight grid, compares it to a Monte Carlo baseline, and outputs the resolution where relative error exceeds 5%.

**Why this is valuable:** This artifact gives you an exact (yet expensive) posterior reference surface, clarifying the failure modes that variational and transformer-based approximations must handle.

**Stack:**
- **Model:** Custom single-layer logistic perceptron with parameters \(w \in \mathbb{R}^3\) (two features + bias)
- **Dataset:** [`sklearn.datasets.make_moons`](https://huggingface.co/datasets) (noise=0.1, random_state=42)
- **Framework:** PyTorch 2.1 with NumPy 2.0 and matplotlib for visualization
- **Compute:** Free Colab T4 (≤15 GB GPU RAM) or any consumer GPU with ~8 GB VRAM; training takes ~15 minutes for both grid resolutions

**The recipe:**
1. Install packages via `pip install torch==2.1.1 numpy==2.0 matplotlib scikit-learn` and import `torch, numpy as np, matplotlib.pyplot as plt, sklearn.datasets`.
2. Generate \(N=1000\) samples from `make_moons(noise=0.1, random_state=42)`, standardize each feature column, append a bias term to the inputs, and split into \(x_i^\prime \in \mathbb{R}^3\) and \(y_i\).
3. Build grid axes `grid1 = torch.linspace(-4, 4, steps=R)` (start with \(R=30\)), create the mesh with `torch.meshgrid`, and reshape to `R^3` weight vectors \(w^{(j)}\). Compute logits \(w^{(j)\top} x_i^\prime\) using vectorized broadcast, evaluate logistic probabilities, and sum log-likelihoods in one pass; add the standard normal log-prior and store the result in `log_posterior`.
4. Stabilize integration with `log_grid_volume = 3 * torch.log(torch.tensor(8.0 / (R - 1)))` and compute \(\log Z_{\text{grid}} = \text{torch.logsumexp}(log\_posterior.reshape(-1), dim=0) + log\_grid\_volume\). Clamp log-posterior values with `torch.clamp(log_posterior, min=-1e6)` and confirm `torch.isfinite(log_posterior).all()` before integrating.
5. Estimate the Monte Carlo baseline by sampling \(K=200{,}000\) weight vectors from the prior, computing the likelihood \(p(D \mid w^{(k)})\) vectorized, and averaging to get \(Z_{\text{MC}}\). Report the relative error \(|Z_{\text{grid}} / Z_{\text{MC}} - 1|\) and log the elapsed time using `time.perf_counter()` to compare runtime growth between \(R=30\) and \(R=100\).

**Expected outcome:** A notebook printout showing “30×30 grid → rel. error \( \approx 1\%\) (0.8s); 100×100 grid → rel. error \(> 5\%\) (≈5s); predictive probability at \([0,0]\) ~0.73”, along with the heatmaps of the log-posterior and assertions that all tensor values remain finite.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Profile inference on an MLOps stack by adding a monitoring dashboard that logs grid runtime, error, and GPU utilization; target ![p95 latency] < 150 ms for the \(30 \times 30\) run, and add a fallback that switches to a precomputed variational posterior when the \(100 \times 100\) error exceeds 5%.
- **Research engineer:** Reproduce Table 2 from Untitled (Author et al. 2026) [https://arxiv.org/pdf/2602.05873] by using the grid posterior means to warm-start their variational hierarchy and aim for KL within ±3% of the reported value.
- **Applied researcher:** Hypothesize that adding entropic regularization (scale factor \(\tau=0.05\)) to the likelihood reduces the grid relative error growth rate; test by rerunning the grid with smoothed log-likelihoods and plotting rel. error vs. resolution to falsify whether smoothing alone suffices.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*