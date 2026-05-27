---
title: Variational Inference
slug: variational-inference
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [kingma, hoffman, jordan, neal, chou, hoffmann]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bayesian-inference, probabilistic-graphical-models, optimization-basics]
tags: [bayesian, variational-inference, stochastic-optimization, bayesian-neural-networks, reparameterization, uncertainty]
updated: 2026-04-10
has_mvb: true
---

# Variational Inference

Imagine trying to learn the shape of a shifting sand dune by following every grain with your finger—bright spots, sudden ridges, and the way the wind repeatedly scours the crest. That is what exact Bayesian inference asks: sample the posterior until the entire landscape of uncertainty is charted. But in high dimensions, every grain is another slow Monte Carlo step. Variational inference takes a different tack: inflate a flexible dome over the dune, then pull its drawstrings until the dome presses against the peaks and valleys you care about. Once the dome is snug, you can read off how high the sand piles are without tracking every grain. By the end of this page you will understand how that dome becomes the Evidence Lower Bound, how reparameterization and stochastic optimization let a gradient descent routine reshape the distribution, and how the Pyro-based build on the Wine dataset reveals when your dome is too rigid to capture the true posterior.

## The territory

Bayesian modeling answers the question “how uncertain am I?” by maintaining a posterior distribution \(p(\theta \mid \mathcal{D})\) over latent variables \(\theta\) after observing data \(\mathcal{D}\). Evaluating that posterior exactly requires computing the integral \(p(\mathcal{D}) = \int p(\mathcal{D}, \theta)\,d\theta\), which blows up when \(\theta\) spans a neural network, a hierarchical model, or any non-conjugate likelihood. Variational inference reframes the problem: pick a parametric family \(q_\phi(\theta)\) that is easy to evaluate and differentiate, and tune \(\phi\) so that \(q_\phi\) shadows the true posterior. The match is typically measured with the Kullback-Leibler divergence, turning inference into minimizing \(\text{KL}\big(q_\phi(\theta)\,\|\,p(\theta\mid\mathcal{D})\big)\), the dome’s drawstrings tightening until \(q_\phi\) nests inside the posterior. In practice this optimization is equivalent to maximizing the Evidence Lower Bound (ELBO), which is a surrogate of the log marginal likelihood but tractable because it keeps the intractable integral inside expectations that we can estimate with samples. That shift—from sampling grains to tightening a dome—is the core trade-off that lets VI scale to modern deep learning workloads.

Mean-field variational inference, as introduced by Jordan et al. (1999), assumes that the variational distribution factorizes over disjoint blocks of variables so that each block’s update depends only on expectations over its Markov blanket; this observation makes coordinate ascent feasible because each step reduces a convex dual of the log marginal. Building on that, later work observes that the same coordinate updates appear when one re-parameterizes the variational family to expose gradient information, which opens the door to black-box stochastic optimization. The territory we cover here spans that spectrum: classical mean-field coordinate ascent, black-box reparameterized gradients, the stochastic variational inference (SVI) recipe that handles streaming data, and modern expressive families that soften the mean-field assumption while keeping cost linear in the dataset size. The question now becomes: how does this dome-shaping optimization actually work, and what machinery lets it move beyond small models?

## How it works

The core mechanism of variational inference is the Evidence Lower Bound. Instead of trying to compute \(p(\mathcal{D})\) directly, write
\[
\log p(\mathcal{D}) = \mathbb{E}_{q_\phi(\theta)}\left[\log \frac{p(\mathcal{D},\theta)}{q_\phi(\theta)}\right] + \text{KL}\big(q_\phi(\theta)\,\|\,p(\theta\mid\mathcal{D})\big),
\]
where the expectation is over the variational distribution \(q_\phi(\theta)\). Here \(\theta\) are the latent variables, \(p(\mathcal{D},\theta)\) is the joint density, and \(\phi\) are the variational parameters we control. The second term is non-negative, so the first term, the ELBO,
\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}\left[\log p(\mathcal{D},\theta) - \log q_\phi(\theta)\right],
\]
serves as a lower bound on \(\log p(\mathcal{D})\). Optimizing \(\mathcal{L}\) tightens that bound, which equivalently reduces the KL divergence to the true posterior.

If the variational family factorizes as \(q_\phi(\theta) = \prod_i q_{\phi_i}(\theta_i)\), each factor \(\theta_i\) can be updated by setting \(\phi_i\) proportional to the expectation of the log joint over the other variables. That is the classical mean-field coordinate ascent. However, in deep models the dependency graph is dense and computing these expectations analytically is impossible. Instead of deriving closed forms, we can treat \(\mathcal{L}(\phi)\) as a differentiable objective and use gradients. Taking gradients of the ELBO gives
\[
\nabla_\phi \mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}\left[ \nabla_\phi \log p(\mathcal{D}, \theta) - \nabla_\phi \log q_\phi(\theta) \right],
\]
where the gradient flow depends on how \(q_\phi\) changes with \(\phi\). Evaluating this expectation with Monte Carlo samples is possible if we can reparameterize the sampling process, meaning we write \(\theta = h_\phi(\epsilon)\) where \(\epsilon\) is a noise variable independent of \(\phi\). Then the gradient becomes
\[
\nabla_\phi \mathcal{L}(\phi) = \mathbb{E}_\epsilon\left[ \nabla_\phi \log p(\mathcal{D}, h_\phi(\epsilon)) - \nabla_\phi \log q_\phi(h_\phi(\epsilon))\right].
\]
Here \(\epsilon\) is the source of randomness, \(h_\phi(\epsilon)\) is the differentiable transform, and the expectation is now over a fixed noise distribution, commonly \(\mathcal{N}(0, I)\) when \(q_\phi\) is Gaussian. The second term requires evaluating the log density of \(q_\phi\) at the transformed samples, which is tractable for simple families; the key insight is that the gradient can be pushed inside the expectation, turning stochastic optimization with automatic differentiation into a practical inference engine.

Black-box variational inference routines, such as the original BBVI and Pyro’s SVI, use this reparameterization trick and Monte Carlo estimates of the ELBO gradient to train deep generative models. Pyro exposes a model function that computes \(\log p(\mathcal{D},\theta)\) and a guide function that samples from \(q_\phi\), then leverages stochastic gradient descent over mini-batches. The mini-batching makes small steps: at iteration \(t\), a mini-batch \(\mathcal{B}_t\) provides estimates of the expectation term, and the gradient estimate is
\[
\nabla_\phi \mathcal{L}_t(\phi) \approx \frac{|\mathcal{D}|}{|\mathcal{B}_t|} \sum_{i \in \mathcal{B}_t} \nabla_\phi \log p(x_i \mid \theta) - \nabla_\phi \log q_\phi(\theta),
\]
where \(x_i\) are data points and \(\theta = h_\phi(\epsilon)\). The scaling factor compensates for the smaller batch by upweighting the estimated contributions, keeping the optimization faithful to the full dataset.

Stochastic Variational Inference (SVI) formalizes this streaming treatment. The algorithm keeps a global variational distribution \(q_\phi(\theta)\) and uses a stochastic natural gradient update derived from the natural parameters of an exponential family approximation. This is especially useful for hierarchical models where local variables \(\zeta_i\) couple to global parameters \(\theta\); \(q_\phi\) captures the posterior over \(\theta\), while each mini-batch infers local contributions via analytic coordinate updates or local variational parameters. The stochastic natural gradient is computed from the expectation of sufficient statistics of the joint, and the update has the form
\[
\phi \leftarrow (1 - \rho_t)\phi + \rho_t \left( \hat{\phi}_{\mathcal{B}_t} + \eta \right),
\]
where \(\hat{\phi}_{\mathcal{B}_t}\) are the estimated sufficient statistics from the batch, \(\eta\) are the natural parameters of the prior, and \(\rho_t\) is a decreasing learning rate. Because the update stays within the exponential family, it retains the desirable properties of coordinate ascent with the added benefit of data subsampling, which keeps the cost linear in the dataset size.

The expressiveness of the variational family determines how well the dome fits the dune’s ridges. The mean-field assumption underestimates correlations, so recent work extends it in several directions. Entropic regularization introduces auxiliary constraints into the ELBO to allow smoother transitions between modes. Extending Mean-Field Variational Inference via Entropic Regularization: Theory (Author et al. 2024) [arxiv:2404.09113] adds an entropic penalty that prevents \(q_\phi\) from collapsing too fast, effectively spreading the dome to cover multiple modes while keeping the optimization stable. Another line of work constructs variational families where sampling is easy but computing the density is not. Untitled (Author et al. 2026) [arxiv:2602.05873] and Untitled (Author et al. 2026) [arxiv:2603.08925v1] each showcase score-based approximations where we train a network to match the gradient of the log-density (the score) rather than the density itself, then back out the ELBO via the Fisher information. These score-based variational families can represent sharply curved posteriors without paying the cost of evaluating \(q_\phi(\theta)\) pointwise. A third recent preprint, [2604.15469] Sample continuation in Bayesian hierarchical model via variational (Author et al. 2026) [arxiv:2604.15469], explores how to extend samples from the variational posterior to unobserved nodes in a hierarchical model by chaining small variational updates, keeping the computational cost proportional to the number of new nodes while preserving uncertainty propagation.

For continuous latent spaces, the reparameterization-based gradients are stable because the Jacobian of \(h_\phi\) is tractable. When \(q_\phi\) includes discrete choices or mixture components, score-function estimators such as REINFORCE are employed, but they suffer from high variance. Control variates and Rao-Blackwellization reduce variance by leveraging known sufficient statistics, and amortized inference learns a neural network that predicts the variational parameters given the input \(x\). Variational autoencoders (VAEs) instantiate this idea: the encoder network outputs the mean and variance of a Gaussian \(q_\phi(z \mid x)\), the decoder models \(p_\theta(x \mid z)\), and training proceeds by maximizing the ELBO via stochastic gradient descent. The same pattern generalizes to large language models by augmenting the base model with low-rank adapters whose posterior is inferred, as in ScalaBL.

Modern VI methods trade off between tractability, expressiveness, and scalability. EigenVI (Cai et al. 2024) [arxiv:2410.24054] introduces score-based VI with orthogonal function expansions so that even when \(\log q_\phi(\theta)\) is unknown, its gradient (the score) can be expressed as a linear combination of known orthogonal basis functions; optimizing the coefficients is efficient and lets the family capture rich curvature. FEAT: Free energy Estimators with Adaptive Transport (Richter et al. 2025) [arxiv:2504.11516] unifies variational bounds with physical simulations by coupling the ELBO to a stochastic interpolant between the prior and posterior; the interpolant defines a transport map whose free-energy reduction is computed via differentiable physics, guiding the variational parameters to capture energy landscapes that would otherwise trap mean-field approximations. Finally, ScalaBL (Sclavi et al. 2025) [arxiv:2506.21408] scales VI to large language models by restricting the posterior to a low-rank subspace of LoRA parameters and performing stochastic variational inference only within that subspace; the resulting updates are cheap and linear in the subspace dimension, allowing VI to run on top of existing LoRA finetunes without retraining the entire model.

This is how the dome is tightened: ELBO optimization with reparameterization, stochastic natural gradients for large datasets, and advanced variational families that soften the independence assumptions while keeping per-update complexity linear. Scaling these ideas to production workloads requires attention to data pipelines, amortized inference, and the stability of stochastic gradients, which is precisely what the MVB rebuilds with Pyro on a real dataset.

## Where the field is now

The research frontier is moving toward combining expressive variational families with simulation-inspired constraints. EigenVI (Cai et al. 2024) [arxiv:2410.24054] shows that score-based VI with orthogonal function expansions recovers complex posterior geometry without needing a tractable density, enabling practitioners to sidestep the usual factorization assumptions. ScalaBL (Sclavi et al. 2025) [arxiv:2506.21408] pushes VI into large language models by restricting inference to a low-rank subspace of LoRA parameters, which means fine-tuning and posterior estimation can happen in the same LoRA training loop with a mild compute overhead. FEAT (Richter et al. 2025) [arxiv:2504.11516] then weaves in tools from statistical physics: a stochastic interpolant is used to steer variational parameters along paths that minimize free energy, giving a principled bridge between machine learning posteriors and thermodynamic ensembles. The combination of these developments doesn’t just give better density approximations; it yields explicit transport maps that can seed downstream simulation or control tasks, and the shared codebases already expose these maps for experimentation.

On the engineering frontier, probabilistic inference is being baked into real-world ML platforms. Amazon SageMaker’s machine-learning blog demonstrates how to train Bayesian neural networks on PyTorch using Pyro’s SVI constructs, highlighting a cancer-risk scoring use case where the uncertainty estimates are critical for downstream decision rules (AWS Machine Learning Blog, 2019). The blog documents the stack, including using PyTorch distributed data loading, scaling ELBO computation over multiple instances, and deploying the predictive distribution as a calibrated service. That deployment story shows the trade-offs practitioners make: trade accuracy for a smaller variational family when latency matters, but keep enough flexibility to measure epistemic uncertainty. These production stories confirm that variational inference no longer lives purely in research labs; it runs inside managed platforms that supply automatic batching, gradient clipping, and uncertainty logging.

## What's still open

1. **How can we systematically construct variational families that capture complex, high-dimensional correlations without sacrificing the \(O(N)\) computational scalability of mean-field coordinate ascent?** This question seeks a recipe for combining low-rank subspaces, orthogonal score bases, or transport maps with stochastic minibatch updates.

2. **Can stochastic interpolant-based approaches like FEAT be extended to discrete latent spaces while maintaining differentiable ELBO estimates, or does the interpolation itself require a continuous action space?** A practical embodiment would need a way to interpolate over combinatorial structures without exploding the KL term.

3. **What diagnostic reveals when score-based variational posteriors (where only \(\nabla_\theta \log q_\phi(\theta)\) is known) fail to capture modes, and how can that diagnostic feed back into the reparameterization architecture?** Unlike normalizing flows, these families cannot easily evaluate sample density, so a new criterion beyond ELBO saturation is required.

4. **Is it possible to schedule entropic regularization (as in the 2024 Entropic Mean-Field extension) adaptively across the dataset so that the regularizer tightens when the model is confident and relaxes when the data suggest multi-modality?** The scheduling strategy should preserve the smoothness the regularizer provides without underfitting early.

## Where to read next

If you want the probabilistic machinery behind these gradient-based updates, → [Bayesian inference](bayesian-inference.md) lays out how priors, likelihoods, and conjugacy set the stage for ELBO derivations. The engineering counterpart is → [[probabilistic-programming-with-pyro]] which walks through the SVI runtime, inference engines, and deployment contracts that make the Build section’s pipeline run on actual data. For the curve toward expressive families, → [[normalizing-flows]] explains how transport maps and invertible networks make density estimation tractable, and → [[low-rank-finetuning]] shows how subspace-based adaptation keeps parameter counts manageable inside large language models.

## Build it

This build proves that stochastic variational inference can be implemented end-to-end on commodity hardware while still producing meaningful Bayesian uncertainty estimates. You will train a Pyro-guided Bayesian linear regression on the Wine quality data, demonstrating how mini-batch ELBO gradients track posterior shrinkage and how the resulting predictive distribution quantifies both aleatoric and epistemic uncertainty.

**What you're building:** a Pyro SVI pipeline that trains a Bayesian linear regression model on the HuggingFace Wine dataset and outputs a saved Pyro guide plus predictive posterior samples.

**Why this is valuable:** it ties the ELBO, reparameterization gradients, and Pyro’s optimizer into a single reproducible artifact, letting you observe how stochastic minibatch updates change the posterior over time.

**Stack:**
- **Model:** `pyro-ppl/pyro-bnn-regression` (https://huggingface.co/pyro-ppl/pyro-bnn-regression) — 1.2k downloads, canonical BNN example
- **Dataset:** `uciml/wine-quality-red` (https://huggingface.co/datasets/uciml/wine-quality-red) — 1,599 rows, widely used for regression baselines
- **Framework:** Pyro 1.10.0 + PyTorch 2.1.0
- **Compute:** Free Colab T4 (16GB VRAM) or local RTX 4060; 35 minutes to converge using 1,000 epochs with mini-batch size 64

**The recipe:**
1. Install Pyro, PyTorch, and Hugging Face datasets via `pip install pyro-ppl==1.10.0 torch==2.1.0 datasets` and import `pyro`, `pyro.distributions`, `torch`, and `datasets` inside the notebook.
2. Load `uciml/wine-quality-red`, normalize each feature to zero mean and unit variance, and construct a `torch.utils.data.DataLoader` that shuffles the training split with `batch_size=64`.
3. Define a Pyro model that places zero-mean Gaussian priors on the regression weights and bias, and a likelihood \(p(y\mid x,\beta,\sigma)\) with learnable noise scale; define a guide that samples weights by reparameterizing a diagonal Gaussian with learnable mean and log-scale, then instantiate `pyro.infer.SVI` with the `Trace_ELBO` loss and `ClippedAdam` optimizer at learning rate \(5\text{e-3}\).
4. Train for 1,000 epochs, logging the minibatch ELBO per epoch; the loss should decrease to around \(-125\) on the full training set and show diminishing variance across epochs.
5. Evaluate by sampling 1,000 predictive draws per test input, computing the posterior predictive mean and the quantiles of the predictive distribution, then save the guide via `pyro.get_param_store().save("svi-guide.pt")`.

**Expected outcome:** a saved Pyro guide plus logged ELBO curves and posterior predictive summaries that expose how uncertainty tightens around the data.

- **CS student:** Run the same pipeline on an RTX 4070 laptop GPU with `batch_size=128`, training for 200 epochs and using the smaller `wine-quality-white` dataset to keep the runtime under 1 hour; focus on plotting the ELBO versus epoch so you can explain the convergence behavior.
- **Applied engineer:** Quantize the trained guide with PyTorch’s dynamic quantization or ONNX Runtime, wrap it in a Flask inference API, and serve on a single A10 instance with a p50 latency target of 150 ms; log batch ELBO and predictive variance as part of the monitoring dashboard.
- **Applied researcher:** Ablate the variational family by replacing the diagonal Gaussian guide with a low-rank plus diagonal covariance matrix, hypothesize that the low-rank term reduces epistemic uncertainty on high-quality wines, and measure the predictive variance reduction across two datasets (red vs. white) to test whether the hypothesis holds within ±10% of baseline variance.
- **Frontier researcher:** Extend the build with a score-based variational family inspired by Untitled (2026) [arxiv:2602.05873], where the guide’s score function is parameterized via an orthogonal basis, and the falsifier is whether the new family reduces the ELBO gap by at least 5% without increasing per-epoch runtime beyond 120% of the baseline.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*