---
title: Variational Inference
slug: variational-inference
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [kingma, hoffman, jordan, neal]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bayesian-inference, probabilistic-graphical-models, optimization-basics]
tags: [bayesian, variational-inference, stochastic-optimization, bayesian-neural-networks, reparameterization, uncertainty]
updated: 2026-02-12
has_mvb: true
---

# Variational Inference

Imagine you are dropped in a vast underground cavern that no map can capture: every twist both confirms and contradicts your current assumption about where the cavern’s walls might be. Monte Carlo methods feel like scrambling through that cave with a headlamp—useful, but liable to wander in circles when the passages multiply. Variational inference takes a different strategy: instead of walking the cave, you inflate a flexible balloon inside it and adjust the balloon’s shape until the rubber presses everywhere the cave does. Once the balloon matches, you can read off your belief about the space without another exhaustive search. By the end of this page you will see precisely how that metaphor becomes the Evidence Lower Bound and gradient-based optimization, how stochastic variational inference scales the balloon-fitting procedure to streaming data, and how a hands-on PyTorch build reveals the failure modes that tell you whether the balloon is too rigid to reflect the posterior you care about.

## The territory

Estimating a full posterior \(p(\theta\mid \mathcal{D})\) over latent variables \(\theta\) given data \(\mathcal{D}\) is the core statistical demand any uncertainty-aware system faces. Computing that posterior exactly requires evaluating the marginal likelihood \(p(\mathcal{D})=\int p(\mathcal{D},\theta)\,d\theta\), which boils down to high-dimensional integration when the latent space is rich. Variational inference replaces that integral with an optimization: choose a parametric family \(q_\phi(\theta)\) whose members are tractable to evaluate and differentiate, and then adjust the parameters \(\phi\) so \(q_\phi(\theta)\) matches \(p(\theta\mid \mathcal{D})\) as closely as possible. That match is typically measured with the Kullback-Leibler divergence, so VI turns the inference problem into minimizing \(\text{KL}(q_\phi(\theta)\,\|\,p(\theta\mid \mathcal{D}))\), the balloon-pushing against the cave walls.

The classical mean-field interpretation, introduced by Jordan et al. (1999) — “An Introduction to Variational Methods for Graphical Models” — describes how convex duality bounds the marginal likelihood: if the variational family factorizes over blocks of variables, coordinate ascent becomes possible because each block’s optimal update depends only on expectations over its Markov blanket. That insight lays the theoretical bedrock, and successive developments bring in stochastic optimization and differentiable generative models. The next section explains how that objective actually expands into the Evidence Lower Bound and how each term corresponds to a component of this balloon-fitting metaphor.

## How it works

### The Evidence Lower Bound

The starting point is the marginal log-likelihood, which we rewrite by introducing \(q_\phi(\theta)\) and subtracting the KL divergence:

\[
\log p(\mathcal{D}) = \mathcal{L}(\phi) + \text{KL}(q_\phi(\theta)\,\|\,p(\theta\mid\mathcal{D}))
\]
where \(q_\phi(\theta)\) is the variational density we control, \(\mathcal{L}(\phi)\) is the Evidence Lower Bound (ELBO), and the KL divergence is always non-negative, making \(\mathcal{L}(\phi)\) a lower bound on \(\log p(\mathcal{D})\).

This rearrangement reveals two things: maximizing \(\mathcal{L}(\phi)\) is equivalent to minimizing the KL divergence, and the ELBO itself decomposes into two practical terms:

\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D}, \theta)] - \mathbb{E}_{q_\phi(\theta)}[\log q_\phi(\theta)]
\]
where the expectation of \(\log p(\mathcal{D}, \theta)\) rewards placing mass where the joint likelihood is high, and the negative entropy \(-\mathbb{E}_{q_\phi}[\log q_\phi]\) acts as a regularizer that keeps the balloon’s shape from collapsing. Those two terms are why the optimization is tractable: we only ever need expectations under the chosen variational family \(q_\phi\), which is why we “inflate a balloon with a known surface tension”.

### Factorization and coordinate ascent

Drawing from Jordan et al. (1999), a common template for constructing \(q_\phi\) is to factorize it across blocks: \(q_\phi(\theta)=\prod_{i} q_{i}(\theta_i)\). Each component is then optimized holding the others fixed, and the optimal update for block \(i\) is proportional to the exponentiated expected log joint:

\[
q_i(\theta_i) \propto \exp\!\left(\mathbb{E}_{q_{-i}}[\log p(\mathcal{D}, \theta)]\right)
\]
where \(q_{-i}\) denotes the current variational distributions over all variables except \(\theta_i\). The expectation is taken with respect to those other variational factors, and since each block only depends on its Markov blanket in a graphical model, this leads to closed-form coordinate ascent updates whenever the joint belongs to an exponential family. The advantage: you can interpret each coordinate ascent step as inflating a local part of the balloon using an analytic formula, which explains why early VI work could run efficiently even before gradient-based optimization was ubiquitous.

### Scaling via stochastic variational inference

Fixed-point coordinate ascent breaks when the dataset \(\mathcal{D}\) is massive, because the expectation \(\mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D}, \theta)]\) decomposes over data points yet computing it across all points every iteration is prohibitive. Hoffman et al. (2013) provide the operational bridge: partition the latent parameters into global and local variables, then apply stochastic optimization with natural gradients to the global variational parameters.

More concretely, assume the joint is \(p(\theta, z, \mathcal{D})\) where \(z\) are local latent variables for each datum. If \(q_\phi(\theta)\) is the global variational factor and \(q_{\lambda_n}(z_n)\) are local factors, the global ELBO gradient can be estimated with a minibatch \(\mathcal{B}\) of datapoints:

\[
\nabla_\phi \mathcal{L}(\phi) \approx \nabla_\phi \left[ \mathbb{E}_{q_\phi(\theta)}[\log p(\theta)] - \log q_\phi(\theta) + \frac{N}{|\mathcal{B}|} \sum_{n\in\mathcal{B}} \mathbb{E}_{q_\phi(\theta)q_{\lambda_n}(z_n)}[\log p(x_n, z_n\mid\theta)] \right]
\]
where \(N\) is the dataset size, \(x_n\) the \(n\)-th observation, and the minibatch approximates the full sum. Hoffman's key recognition is that when both the prior and variational family are exponential-family distributions, the natural gradient of this ELBO can be computed by updating the sufficient statistics using the minibatch, yielding stable convergence even with large stepsizes.

The practical upshot is that you only need to inspect a small minibatch per update, and the local variational parameters for those datapoints can be computed analytically or with short inner-loop optimizations before stepping the global parameter. That is how the balloon analogy extends to streaming data: each minibatch reveals only a patch of the cave, but the natural-gradient step pulls the global balloon to match that patch without requiring a full sweep over the cavern.

### Reparameterization and amortized inference

The coordinate / natural-gradient view handles conjugate models well, but modern deep generative models break conjugacy by design. Kingma & Welling (2013) unleash the second part of our balloon metaphor: backpropagating through stochastic nodes is possible if the sampling operation is written via a deterministic transformation of noise. For instance, if \(q_\phi(z\mid x)=\mathcal{N}(\mu_\phi(x), \sigma^2_\phi(x))\), we can sample

\[
z = \mu_\phi(x) + \sigma_\phi(x)\cdot \epsilon
\]
where \(\epsilon\sim \mathcal{N}(0, I)\), and the gradient \(\nabla_\phi \mathcal{L}\) flows through \(\mu_\phi(x)\) and \(\sigma_\phi(x)\) directly because the randomness now enters only through \(\epsilon\), which is independent of \(\phi\). This reparameterization trick is what allows us to inflate a highly parameterized balloon (a neural network encoder) while still computing gradients with respect to \(\phi\).

From there we build amortized inference: instead of optimizing separate \(\lambda_n\) per datapoint, we train an inference network \(q_\phi(z\mid x)\) that maps each observation \(x\) to the parameters of a variational distribution, and the shared parameters \(\phi\) are updated with gradients computed on minibatches. The ELBO becomes

\[
\mathcal{L}(\phi) \approx \frac{1}{|\mathcal{B}|} \sum_{x\in\mathcal{B}} \left[ \mathbb{E}_{q_\phi(z\mid x)}[\log p_\theta(x\mid z)] - \text{KL}(q_\phi(z\mid x)\,\|\,p(z)) \right]
\]
where \(p_\theta(x\mid z)\) is the generative decoder parameterized separately, and the expectation is evaluated with a single reparameterized sample \(z\). This is exactly the VAE training routine, which Kingma & Welling (2013) cast as optimizing the ELBO for generative modeling; the balloon is now a neural map whose surface tension is maintained by the KL term.

### Black-box gradients and stochastic VI in deep models

When even the KL term lacks closed form, we fall back to Monte Carlo estimators with variance reduction. The general recipe, described in the untitled 2016 note (https://www.arxiv.org/pdf/1601.00670v4), is to use control variates and adaptive importance sampling to keep the gradient estimates stable when the variational family is expressive (e.g., normalizing flows or neural splines). When such black-box estimators are combined with the stochastic optimization techniques from Hoffman et al. (2013), the result is a versatile SVI engine that can be plugged into deep learning frameworks: the balloon’s shape is now parameterized by autoregressive flows or attention layers, but the optimization still only requires minibatches and differentiable samples.

The SVI loop thus alternates among: (1) evaluating the minibatch ELBO via reparameterized samples; (2) backpropagating gradients through both the inference network and the generative model; and (3) updating the global parameters with optimizers such as Adam or its natural-gradient cousins. Diagnostics come from monitoring the ELBO and the KL term: if the KL term collapses to zero while reconstruction errors remain high, the balloon is not stretching into the cave—either the variational family is too rigid or the encoder has collapsed to a degenerate solution.

## Where the field is now

The research frontier keeps expanding from the original VAE story. The untitled 2016 note (https://www.arxiv.org/pdf/1601.00670v4) introduced techniques like importance-weighted objectives and hierarchical encoders that push the balloon into more intricate caves; these ideas inspired the Importance Weighted Autoencoder family and flow-based variational families used in recent work. More recently, approaches such as deep Gaussian processes and structured variational families explore ways to encode dependencies among latent blocks without sacrificing the tractable gradients that make SVI viable, making inference in latent-variable models like hierarchical time series amenable to gradient-based optimization.

On the production frontier, companies are deploying variational inference across applications with streaming data. Google Research’s Vertex AI Platform has published at [research.google](https://research.google/pubs/pub47903/) how probabilistic embeddings, trained with VI, maintain uncertainty when serving recommendations to billions of users, letting the inference service update global parameters incrementally from logged interactions. Similarly, Databricks’s 2024 machine-learning blog describes how SVI fits naturally into Delta Live Tables for near-real-time anomaly detection, because the natural-gradient stochastic optimizer can absorb new rows of data without retraining from scratch. These stories reveal that the balloon-fitting metaphor is not just a toy: it is the operational core beneath confidence-aware ranking, anomaly monitoring, and downstream deciders that need fast, differentiable posterior approximations.

## What's still open

Can we design variational families whose expressivity matches that of full posterior covariances while still allowing closed-form, blockwise coordinate updates like mean-field inference? Which family of flows or implicit distributions delivers the best trade-off between flexibility and the stability of the natural-gradient updates that Hoffman et al. (2013) championed?

What are the minimal conditions under which amortized inference generalizes across domains? VAEs demonstrate transfer when the encoder is reused, but a systematic theory that predicts when an inference network trained on distribution \(A\) can be reused on distribution \(B\) without catastrophic collapse is still missing.

Is there an efficient procedure to choose between KL directions? The standard ELBO uses \(\text{KL}(q\|p)\), which leads to mode-seeking behavior; can we construct an objective that interpolates toward \(\text{KL}(p\|q)\) while keeping stochastic gradients stable enough for SGD-scale minibatching?

## Where to read next

If you want the deterministic approximation perspective, → [[mean-field-variational-inference]] spells out how coordinate ascent works on tree-structured graphs; the engineering counterpart is → [[stochastic-variational-inference]] where minibatches and natural gradients keep large datasets manageable, and for the deep generative modeling angle, → [Variational Autoencoders](../../02-generative-modeling/concepts/variational-autoencoders.md) explains how reparameterization pairs with modern neural architects.

## Build it

We can now prove the insides of the ELBO by building a full stochastic variational inference loop for Bayesian linear regression, from synthetic data generation through ELBO monitoring on a Colab-scale GPU.

**What you're building:** a PyTorch SVI engine that fits a Bayesian linear regression posterior on `huggingface/datasets/uci/diabetes`, tracks ELBO vs. SGD steps, and outputs a posterior predictive interval chart.

**Why this is valuable:** this build forces you to derive the ELBO term-by-term, implement the natural-gradient-like update on the global weight mean and precision, and watch when the variational covariance underestimates real uncertainty—precisely the failure that exposes a too-rigid balloon.

**Stack:**
- **Model:** custom Bayesian linear regression defined from scratch (mean vector and precision matrix optimized via PyTorch tensors)
- **Dataset:** [https://huggingface.co/datasets/uci/diabetes](https://huggingface.co/datasets/uci/diabetes) — 442 samples, 10 features
- **Framework:** PyTorch 2.1 + Pyro 2.0 for its SVI scaffolding
- **Compute:** single RTX 3070 or Colab T4 (16 GB VRAM) — training finishes in ~20 minutes per run

**The recipe:**
1. pip install `torch==2.1.0`, `pyro-ppl==2.0.0`, `huggingface_hub`, and `matplotlib`, then load the dataset via `datasets.load_dataset("uci/diabetes")` and standardize features.
2. Construct a model where the prior over weights \(w\) is \(\mathcal{N}(0, \sigma_w^2 I)\), the likelihood is \(\mathcal{N}(x^\top w, \sigma_y^2)\), and the variational distribution is a Gaussian with learnable mean \(\mu_\phi\) and log-precision \(\log \lambda_\phi\); pack the parameters as Pyro `param` nodes.
3. Define the ELBO manually by computing \(\mathbb{E}_{q_\phi(w)}[\log p(\mathcal{D}\mid w)] - \text{KL}(q_\phi(w)\,\|\,p(w))\) for each minibatch, using the analytic form of the KL for Gaussians and the minibatch average of the log likelihood, then take gradients with Adam (learning rate \(1e-3\), batch size 64) and step the Pyro optimizer.
4. Every epoch, sample from the variational Gaussian to produce posterior predictive means on a validation split, compute the mean squared error and the ELBO, and plot both curves; expect the ELBO to climb while the MSE stabilizes around the closed-form ridge regression baseline.
5. You now have an artifact: a checkpoint containing \(\mu_\phi\) and \(\lambda_\phi\), a plot of ELBO + MSE vs. steps, and a predictive interval visualization showing whether the variational covariance captures the residual variance.

**Expected outcome:** a reproducible SVI loop that outputs ELBO diagnostics, a posterior predictive interval figure, and a saved checkpoint for the Bayesian linear regression variational parameters.

- **CS student:** Run the same code on free Colab (T4) but reduce minibatch size to 32 and log the ELBO vs. steps in a Colab cell so you can explain why the Bayesian posterior is tighter than the point estimate on the validation set.
- **Applied engineer:** Extend the artifact by exporting the variational checkpoint to ONNX, quantizing the weight mean to FP16, and serving it as a TensorRT engine with p50 latency < 4 ms on an A10 using the same ELBO monitor for drift detection.
- **Applied researcher:** Treat \(\sigma_y^2\) as a tunable parameter and record how the ELBO and predictive intervals change as you sweep \(\sigma_y^2\); the hypothesis is that a fixed noise variance fails to capture heteroscedasticity, so evidence is the widening gap between ELBO and validation log-likelihood when \(\sigma_y^2\) is misspecified.
- **Frontier researcher:** Incorporate a small normalizing flow (e.g., two-planar flows) into the variational family and measure whether the increased expressivity actually reduces the KL term without destabilizing the minibatch gradients—if the KL fails to shrink, the hypothesis that flow expressivity improves mean-field-like updates is falsified.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*