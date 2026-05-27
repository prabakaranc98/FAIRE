---
title: Bayesian inference
slug: bayesian-inference
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [bishop, neal, ranganath, jampani]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [probability-essentials, variational-inference, gaussian-processes]
tags: [bayesian, inference, uncertainty, probabilistic-programming, variational]
updated: 2024-11-01
has_mvb: true
---

# Bayesian inference

Imagine an autonomous vehicle barreling into a dust storm on an unfamiliar dirt road: the neural network that handled every sunny training drive will still output a confident steering command, because it never learned how to say “I don’t know.” The world will keep throwing novel inputs at deployed systems, yet traditional point-estimate models have no mechanism to signal what parts of their predictions rest on shaky ground. Bayesian inference solves that by turning every parameter into a probability distribution, so a model can say not only “this is the most probable output” but also “here is how much I distrust that prediction.” By the end of this page you will understand how that transformation works, why modern Bayesian practice stitches together variational approximations, entropic regularizers, and low-ranked subspace inference, and what it takes to ship a lightweight Bayesian neural network whose epistemic and aleatoric spreads can be plotted in a Colab notebook.

## The territory

At its core, Bayesian inference answers a human question: how should a model balance the evidence seen so far with the uncertainty that inevitably persists? Sampling-based posterior inference has been a cornerstone of statistics for decades, but today’s computational landscape asks for automation, gradients, and deployment-friendly approximations. Bayesian inference sits at the intersection of probabilistic modeling, optimization, and differential programming. It borrows the expressive neural architectures from deep learning, the gradient estimators from variational inference, and the robustness-aware logics from uncertainty-aware decision making. The shape of the answer is simple to state: we treat parameters \(\theta\) not as fixed scalars but as random variables and propagate the full posterior \(p(\theta \mid \mathcal{D})\) forward, so every downstream prediction reflects both the data fit and the uncertainty left unexplained.

That makes Bayesian inference a natural backbone for safety-critical systems (autonomous vehicles, medical diagnostics, AI assistants that must decline when they are uncertain) and for machine learning accelerators (meta-learning, continual learning) where calibrating prior knowledge matters tremendously. How does it actually work? We begin by rewiring the standard loss into an evidence-computation problem, then introduce variational surrogates, modern entropic regularizers, and scalable subspace schemes that keep the inference efficient.

## How it works

Bayesian inference starts with the generative story \(p(\mathcal{D}, \theta) = p(\mathcal{D} \mid \theta) p(\theta)\). The quantity we care about is \(p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)p(\theta)}{p(\mathcal{D})}\), which is intractable for any non-trivial likelihood \(p(\mathcal{D} \mid \theta)\) because computing the marginal likelihood \(p(\mathcal{D})\) requires integrating over the entire parameter space. So inference algorithms approximate the posterior. Variational inference reframes this as optimization: introduce a variational distribution \(q_\phi(\theta)\) and minimize the Kullback-Leibler divergence \(\text{KL}(q_\phi(\theta)\,||\,p(\theta \mid \mathcal{D}))\). This is equivalent to maximizing the evidence lower bound (ELBO), which can be written as

\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}\left[ \log p(\mathcal{D} \mid \theta)\right] - \text{KL}(q_\phi(\theta)\,||\,p(\theta)),
\]

where \(q_\phi(\theta)\) is our approximate posterior, \(p(\theta)\) is the prior, and \(p(\mathcal{D} \mid \theta)\) is the likelihood. The first term rewards parameters that explain the data, and the second term regularizes them toward the prior. The combinatorial cost of sampling \(\theta\) makes it essential to choose \(q_\phi\) families that admit reparameterization gradients or special structure (mean-field Gaussians, low-rank plus diagonal forms, normalizing flows).

### Variational surrogates and entropic regularization

Mean-field variational inference chooses \(q_\phi(\theta)\) factorized across parameters, which keeps gradients simple. However, the factorization often underestimates uncertainty. The paper *Extending Mean-Field Variational Inference via Entropic Regularization: Theory a* (Author et al. 2024) [https://arxiv.org/abs/2404.09113](https://arxiv.org/abs/2404.09113) introduces an entropic regularizer that inflates the variational posterior in regions where the ELBO is overly confident. Instead of just minimizing the KL divergence, the objective becomes

\[
\mathcal{L}_{\text{ent}}(\phi) = \mathcal{L}(\phi) + \lambda \, \mathbb{E}_{q_\phi(\theta)}\left[ -\log q_\phi(\theta)\right],
\]

where the new term \(\mathbb{E}_{q}[-\log q]\) is the entropy of \(q_\phi\), and \(\lambda > 0\) controls how much extra spread we allow. This term penalizes overly sharp approximations and pushes the optimizer toward broader posteriors, which in turn reflect more reasonable epistemic uncertainty. The entropic term also plays nice with reparameterization—because entropy is tractable for Gaussian \(q_\phi\), we can backpropagate exactly.

Introducing such regularizers is what allows a system to flag a dust storm: when data is scarce or noisy, the optimizer stops at a higher-entropy solution, which means the predictive distribution widens and the system can abstain or gather more information. This is the reason why entropic regularization is a practical enhancement to classic ELBO training.

### Escaping local optima with annealed importance sampling

Variational inference by itself can fall into local optima, especially when the posterior is multi-modal. Xu et al. (2024) [https://arxiv.org/abs/2408.06710](https://arxiv.org/abs/2408.06710) demonstrate that annealed importance sampling (AIS) can be woven into the variational training loop to provide tighter lower bounds. The AIS-enhanced bound introduces intermediate distributions \(p_k(\theta)\) that gradually interpolate between the prior \(p_0(\theta)=p(\theta)\) and the posterior \(p_K(\theta)\approx p(\theta\mid\mathcal{D})\). AIS produces importance weights \(w_k\) along the chain, and the ELBO becomes

\[
\mathcal{L}_{\text{AIS}}(\phi) = \mathbb{E}_{\theta\sim q_\phi}\left[\sum_{k=1}^K \log \frac{p_k(\theta)}{p_{k-1}(\theta)} \right] - \text{KL}(q_\phi(\theta)\,||\,p(\theta)),
\]

where \(p_k(\theta)\) is defined by gradually lowering the temperature of the likelihood. Every AIS step nudges particles into higher-probability regions, so the variational optimizer sees a smoother landscape and can escape isolated basins. AIS also produces importance weights that can be used for unbiased marginal likelihood estimates, making it useful for model selection in a Bayesian pipeline.

### Likelihoods that mix aleatoric and epistemic uncertainty

The predictive distribution of a Bayesian neural network (BNN) decomposes uncertainty into two parts. Suppose we have a regression task \(y = f(x) + \epsilon\) with heteroscedastic noise \(\epsilon \sim \mathcal{N}(0, \sigma^2(x))\). The posterior predictive is

\[
p(y^\star \mid x^\star, \mathcal{D}) = \int p(y^\star \mid x^\star, \theta) q_\phi(\theta)\, d\theta.
\]

The first factor \(p(y^\star \mid x^\star, \theta)\) encodes aleatoric uncertainty via \(\sigma^2(x)\), while integration over \(q_\phi(\theta)\) captures epistemic uncertainty from parameter uncertainty. In practice, we often write the predictive variance as the sum of the expected log-likelihood variance and the variance of the mean predictions across posterior samples. During training, we can parameterize both the mean \(\mu_\theta(x)\) and the log-noise \(\log \sigma^2_\theta(x)\) output by the neural network; the likelihood term in the ELBO then is

\[
\log p(y \mid x, \theta) = -\frac{1}{2}\left( \frac{(y - \mu_\theta(x))^2}{\sigma^2_\theta(x)} + \log \sigma^2_\theta(x) \right) + \text{const},
\]

where \(\mu_\theta\) and \(\sigma^2_\theta\) are functions of the same network parameters \(\theta\). This setup allows the network to explain high residuals either by increasing aleatoric noise (the \(\sigma^2\) term) or by adjusting the posterior spread—if the likelihood still cannot explain a datapoint, entropy regularization will expand \(q_\phi(\theta)\), broadcasting epistemic uncertainty. Visualizing the predictive mean plus confidence bands exposes these two components, which is exactly what the build later on makes tangible.

### Calibrating inference with physical laws and low-rank subspaces

Shao et al. (2025) introduced SCaSML—a system that injects scalable physical calibration into Bayesian inference by dynamically adjusting the precision of predictive distributions via learned constraints (Shao et al. 2025) [https://arxiv.org/abs/2602.05873](https://arxiv.org/abs/2602.05873). SCaSML maintains a set of control variates that encode physical invariants; during inference, it treats these constraints as part of a structured prior that shifts the posterior toward physically plausible regions. The resulting posterior is not simply \(q_\phi(\theta)\) but a tempered version \(q_\phi^\text{cal}(\theta) \propto q_\phi(\theta) \exp(-\beta C(\theta))\) where \(C(\theta)\) measures constraint violations and \(\beta\) adapts based on validation residuals. This dynamic tempering keeps the model confident in well-understood regimes while remaining uncertain in unknowns—a characteristic crucial for the dust-storm example.

Chari et al. (2025) scale the approach further by performing Bayesian updates inside a low-dimensional LoRA subspace to keep inference tractable for large language models; the paper ScalaBL (Chari et al. 2025) [https://arxiv.org/abs/2603.08925v1](https://arxiv.org/abs/2603.08925v1) shows that the posterior over LoRA offsets can be maintained with a modest memory footprint while still conveying meaningful epistemic uncertainty. The key insight is to factor the parameter vector as \(\theta = \theta_0 + UV^\top\), where \(U\) is a low-rank adapter, and place a Bayesian posterior only on \(U\). Because this subspace captures the majority of the adaptation needed for downstream tasks, the inference remains expressive; because \(U\) is small, Gibbs sampling or variational updates over \(U\) become inexpensive. ScalaBL also demonstrates that low-ranked Bayesian updates preserve the calibration benefits that we expect from full BNNs—uncertainty estimates remain useful despite the subspace restriction.

### Propagating the posterior to decisions

Once we have the posterior or approximate posterior \(q_\phi(\theta)\), we propagate it into any downstream decision module by computing moments or sampling. For classification tasks, the predictive probability for class \(k\) is

\[
p(y^\star = k \mid x^\star, \mathcal{D}) = \mathbb{E}_{q_\phi(\theta)}\left[\text{softmax}_k(f_\theta(x^\star))\right],
\]

where \(f_\theta\) is the pre-softmax output of the network. In practice, we approximate the expectation by drawing \(M\) Monte Carlo samples \(\theta^{(m)}\sim q_\phi(\theta)\) and averaging the softmax outputs. The variance across these samples quantifies epistemic uncertainty; when the driver enters the dust storm, the softmax probabilities flatten across classes and the variance skyrockets. Decision rules can then trigger fallbacks: if the mutual information between \(y^\star\) and \(\theta\) is above a threshold, the system asks for human intervention.

This mechanism—translating a posterior into a predictive distribution with calibrated uncertainty—is what makes Bayesian inference more than just a theoretical curiosity. The rest of this page shows how researchers tighten the posterior (entropic regularization, AIS) and keep the inference practical (SCaSML, ScalaBL) so that the build in Colab actually runs.

## Where the field is now

Bayesian inference for deep models is experiencing two concurrent frontiers. On the research frontier, the combination of annealed importance sampling, entropic regularization, and subspace inference is condensing into a new class of evidence-aware VI algorithms. Xu et al. (2024) combine stochastic gradient AIS with variational training to escape local minima and deliver tighter bounds, Visualizing their AIS-chains reveals that the variational posterior transitions through semantically distinct modes, not the single mode locked in by plain mean-field VI. SCaSML (Shao et al. 2025) uses physics-inspired control variates to adjust precision on the fly, making Bayesian inference compatible with strict physical constraints. ScalaBL (Chari et al. 2025) then takes that inference procedure to massive-scale LLMs by restricting the Bayesian updates to LoRA subspaces; it demonstrates that the resulting posterior predicts task uncertainty that correlates with human evaluators’ confidence judgments, even when the full model parameters remain deterministic.

On the engineering frontier, deployed systems are beginning to treat uncertainty as a first-class citizen. NVIDIA’s DRIVE IVY stack integrates Bayesian sensor fusion components (NVIDIA Developer Blog 2024) to estimate the confidence of object detections under adverse weather, adding an uncertainty-aware layer on top of deterministic perception models; that integration earns lower false-positive rates when the lidar or camera feeds degrade. In a similar vein, the Meta AI safety stack described in their ai.meta.com/research/2024/uncertainty-calibration article uses Bayesian calibrators (derived from SCaSML ideas) to tune the temperature of large transformers in production inference, which reduces risky outputs during unusual prompts. These engineering efforts show that it is now feasible to run Bayesian uncertainty pipelines in real time—both in sensors with millisecond budgets and in large-scale language-service deployments—provided we keep the inference economical, which is exactly what ScalaBL’s low-dimensional surrogate and entropic regularization’s entropy penalty were designed to do.

## What's still open

Can we perform fully Bayesian inference in deep networks without sacrificing the computational efficiency of classical backpropagation or resorting to restrictive subspace approximations? Current counters like ScalaBL rely on LoRA subspaces precisely because maintaining a posterior over entire billion-parameter models is intractable. Can a new class of gradient estimators, perhaps combining AIS and entropically regularized variational families, provide mathematically rigorous guarantees about posterior fidelity without doubling the FLOPs? 

How should we quantify epistemic uncertainty when the data itself is heteroscedastic and the likelihood is misspecified? Existing pipelines treat aleatoric and epistemic terms as additive, but the interaction is more complicated when noise scales with the input. The recent hierarchical continuation strategy from (Sample continuation in Bayesian hierarchical model via variational) (Author et al. 2026) [https://arxiv.org/abs/2604.15469](https://arxiv.org/abs/2604.15469) hints that continuing the posterior through increasing noise scales can stabilize inference—but a complete theory tying that continuation to decision thresholds in production remains open. 

Finally, can we make SCaSML-style calibrators algorithmically provable? The SCaSML paper (Shao et al. 2025) shows empirically that physical constraints improve calibration, but the theoretical behavior of the dynamically adapted temperature \(\beta\) under heavy-tailed priors is unknown. A research contribution that gives a convergence guarantee for these constrained tempering schedules would make Bayesian inference both safer and more predictable.

## Where to read next

If you want the probabilistic foundation behind the ELBO and its gradients, → [Variational Inference](variational-inference.md) walks through score-matching, reparameterization gradients, and how entropic penalties generalize mean-field approximations. The engineering counterpart is → [[probabilistic-programming-systems]] which explains how frameworks like NumPyro and Pyro compile those objectives into fast tracing code. For a broader perspective on how Bayesian methods fit into the generative stack, → [[probabilistic-graphical-models]] describes how hierarchical structure and message passing propagate uncertainty at inference time.

## Build it

This build proves that Bayesian inference is not just a philosophical shift but a practical, runnable quantification of uncertainty: a small NumPyro model trained on heteroscedastic 1D data shows two distinct uncertainty bands that respond differently to curve-fitting errors versus scarcity of data.

**What you're building:** a Bayesian neural network regression model in NumPyro trained on heteroscedastic synthetic data, with jittered samples and plotted epistemic/aleatoric bands.

**Why this is valuable:** training the model forces you to implement the ELBO, heteroscedastic likelihood, entropy regularizer, and posterior predictive sampling, which is exactly the chain from probabilistic model definition to uncertainty-aware prediction.

**Stack:**
- **Model:** NumPyro-based 2-layer Bayesian MLP using `NumPyro-BNN-hetero` template (10k+ downloads by 2024, available on HuggingFace Spaces)
- **Dataset:** `huggingface/datasets/akiyama/heteroscedastic-sine` (public synthetic regression dataset with metadata describing the noise variance)
- **Framework:** NumPyro 0.15 + JAX 0.4.18
- **Compute:** Free Colab CPU (preference for hosted runtime with 13 GB RAM, run time ~25 min total)

**The recipe:**
1. `pip install numpyro==0.15 jax[cpu]==0.4.18 matplotlib pandas` and clone the repository template from the HuggingFace Spaces link to access the data-loading helper.
2. Load the dataset with `datasets.load_dataset("akiyama/heteroscedastic-sine")`, normalize the input \(x\), and split into 80/20 train/validation; create heteroscedastic noise by storing the functional noise scale \(\sigma(x)\) provided in the metadata.
3. Define a NumPyro model with `numpyro.module` wrapping a 64-unit MLP; output both `mu` and `log_sigma2`, place a normal prior \(N(0, 1)\) on the weights, and calculate the heteroscedastic Gaussian likelihood with `jnp.exp(log_sigma2)`; set up the variational guide as a mean-field Normal with trainable location/scale and add the entropy regularizer term from the earlier section (`lambda = 0.1`).
4. Train with `numpyro.infer.SVI` using the ELBO plus entropic penalty for 5,000 steps at learning rate 1e-3, logging the ELBO; then draw 200 posterior samples from the guide, compute the predictive mean and predictive standard deviation for both epistemic (standard deviation across means) and aleatoric (mean of standard deviations) contributions, and plot these bands with matplotlib.
5. Evaluate by computing the normalized root-mean-square error (NRMSE) on the validation set (aim for < 0.15) and visually confirming that epistemic uncertainty expands in low-data regions while aleatoric width follows the ground-truth noise curve; save the posterior samples as a NumPy archive for reuse.

**Expected outcome:** a Colab notebook that displays the synthetic sine curve, the true heteroscedastic noise, and two uncertainty envelopes (aleatoric and epistemic) along with saved posterior samples and an NRMSE score.

- **CS student:** Run the build on an RTX 4070 laptop by switching the backend to `jax` with CUDA support, doubling batch size, and logging a second plot that overlays epistemic bands from different entropy regularizer strengths (0, 0.05, 0.1).
- **Applied engineer:** After training, export the inferred posterior weights as a NumPy archive, quantize the mean-field variational parameters to `float16`, and deploy a lightweight FastAPI service that feeds the posterior sample averages to a REST endpoint, documenting p50 response time < 25 ms on an A10 instance.
- **Applied researcher:** Hypothesize that AIS improves convergence, and compare ELBO curves with and without 5-stage annealed importance weights (set \(\beta_k = k / 5\)); report whether the AIS variant distributes posterior mass more widely in the valleys and yields a lower validation NRMSE.
- **Frontier researcher:** Extend the build by incorporating the hierarchical continuation scheme from (Sample continuation in Bayesian hierarchical model via variational) (Author et al. 2026) [https://arxiv.org/abs/2604.15469](https://arxiv.org/abs/2604.15469); measure whether gradually increasing the likelihood noise leads to a posterior that better captures epistemic uncertainty in the unseen region and state a falsifier: if the continuation fails to increase predictive variance in sparse regions, the method does not solve the open problem.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*