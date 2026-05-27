---
title: Variational Inference
slug: variational-inference
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [jordan, hoffman, kingma, welling, blei, pasley]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bayesian-inference, probabilistic-graphical-models, optimization-basics]
tags: [bayesian, variational-inference, reparameterization, stochastic-optimization, probabilistic-programming]
updated: 2026-04-10
has_mvb: true
---

# Variational Inference

Imagine trying to learn the average shape of a shifting sand dune by inspecting every grain: you walk around it, you scoop samples from the crest and the troughs, and you try to sum their heights to a single number. Exact Bayesian inference faces that same grain-counting nightmare when the latent space is high-dimensional; the normalizing constant of the posterior is the sum of an astronomical number of probabilities. Variational inference takes a different tack. Instead of sampling every grain, it takes a pliable dome—usually a multivariate Gaussian or a mixture you can evaluate quickly—and bends and stretches its parameters so that the dome hugs the knobby shape of the sand pile as closely as possible. By the end of this page you will understand how that dome is described by the Evidence Lower Bound, how the reparameterization trick and stochastic gradients let modern autodiff toolchains carve it against data, and why a hands-on Black Box Variational Inference build on the Wine dataset proves the idea works even with a handful of lines of PyTorch.

## The territory

The Bayesian answer to “how uncertain am I?” is a posterior \(p(\theta \mid \mathcal{D})\) over latent variables \(\theta\) after seeing data \(\mathcal{D}\). The textbook route rewrites this as
\[
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D}, \theta)}{p(\mathcal{D})},
\]
where \(p(\mathcal{D})=\int p(\mathcal{D},\theta)\,d\theta\) is the marginal likelihood that normalizes the posterior.
\(\mathcal{D}\) is the observed data. \(\theta\) is the vector of latent variables. The intractability lurks in \(p(\mathcal{D})\): for neural networks, hierarchical models, or non-conjugate likelihoods, that integral has no closed form, so the posterior cannot be written down or sampled exactly. The classical alternative—Markov Chain Monte Carlo—draws samples until the empirical distribution matches the posterior, but it does so at the cost of slow convergence and poor parallelism in high dimensions.

Variational inference reframes the problem. Borrowing the mean-field intuition from Jordan et al. (1999) [http://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf], it chooses a tractable family \(q_\phi(\theta)\) parameterized by \(\phi\) and then solves an optimization problem to make \(q_\phi\) “as close as possible” to the true posterior. \(\phi\) are variational parameters. The goal is to minimize the Kullback-Leibler divergence \(\mathrm{KL}(q_\phi(\theta) \,\|\, p(\theta \mid \mathcal{D}))\), which is equal to maximizing the Evidence Lower Bound (ELBO) on \(\log p(\mathcal{D})\). This turns inference into an optimization problem where each gradient step tightens the dome so it sits snugly against the posterior, trading off some approximation error for tractability. Because the dome can be evaluated and differentiated quickly, variational inference scales to millions of datapoints once we pair it with stochastic optimization and autodiff—this is the shift from counting every grain to sculpting the dome.

The territory also spans probabilistic programming (Pyro, Gen, PyMC) and the rich set of amortized inference techniques used inside VAEs. How does it actually work? The remainder of the explanation walks through the ELBO, how the reparameterization trick creates usable gradients for continuous latents, and how modern stochastic variational inference lets these gradients train on minibatches.

## How it works

### The ELBO and the mean-field dome

The Evidence Lower Bound is the workhorse objective. Starting from the KL divergence, we rewrite
\[
\text{KL}(q_\phi(\theta)\,\|\,p(\theta\mid\mathcal{D})) = \mathbb{E}_{q_\phi}\left[\log q_\phi(\theta) - \log p(\mathcal{D},\theta) \right],
\]
where \(q_\phi\) is the variational distribution, and \(p(\mathcal{D},\theta)\) is the joint of the data and latents.
This divergence is non-negative, so rearranging gives
\[
\log p(\mathcal{D}) = \underbrace{\mathbb{E}_{q_\phi}[\log p(\mathcal{D},\theta) - \log q_\phi(\theta)]}_{\text{ELBO}(\phi)} + \text{KL}(q_\phi\,\|\,p(\theta\mid\mathcal{D})).
\]
The ELBO is the part we can compute, because the intractable \(\log p(\mathcal{D})\) has been replaced with an expectation under \(q_\phi\), which is drawn from a family we control. Every gradient step that increases the ELBO simultaneously raises the log-marginal bound and decreases the KL divergence.

The dome metaphor becomes concrete when we choose a mean-field family:
\[
q_\phi(\theta) = \prod_{i=1}^n q_{\phi_i}(\theta_i),
\]
where each coordinate \(\theta_i\) has its own variational factor \(q_{\phi_i}\). \(n\) is the number of latent dimensions.
Mean-field inference, as explained by Jordan et al. (1999), decouples dependencies by cycling through coordinate updates for each \(\theta_i\), similar to the way coordinate descent squeezes a high-dimensional dome along one axis at a time. The ELBO decomposes accordingly, and in the case of exponential-family models the update for \(\phi_i\) is often available in closed form. Even when the model is non-conjugate, the ELBO retains its form, and we can now plug in generic optimizers.

However, mean-field also reveals the fundamental tension: the dome cannot capture correlated posterior mass once the variational family factorizes. Later sections show how richer families and normalizing flows mitigate this, but every additional degree of freedom increases compute. The job of a good variational posterior is to be expressive enough to wrap around the posterior’s major modes while retaining the ability to evaluate and differentiate the density quickly.

### Reparameterization, gradients, and black-box VI

Black Box Variational Inference (BBVI) generalizes the dome by allowing any \(q_\phi(\theta)\) we can sample from; the gradient is estimated via Monte Carlo. However, naive gradients over the ELBO suffer from high variance because the sample \(\theta \sim q_\phi\) depends on \(\phi\). The reparameterization trick, introduced by Kingma and Welling and formally described in Auto-Encoding Variational Bayes (2014) [https://www.arxiv.org/pdf/1601.00670v4], sidesteps the issue by expressing \(\theta\) as a deterministic transformation of a noise variable:
\[
\theta = g_\phi(\epsilon),\qquad \epsilon \sim p(\epsilon),
\]
where \(g_\phi\) is differentiable, and \(p(\epsilon)\) is a fixed noise distribution, usually \(\mathcal{N}(0,I)\).
\(\phi\) again are variational parameters. \(\epsilon\) is an auxiliary noise source. The expectation in the ELBO now becomes
\[
\mathcal{L}(\phi) = \mathbb{E}_{\epsilon \sim p(\epsilon)}\left[\log p(\mathcal{D}, g_\phi(\epsilon)) - \log q_\phi(g_\phi(\epsilon)) \right].
\]
The gradient \(\nabla_\phi \mathcal{L}(\phi)\) can be pushed through \(g_\phi\) with automatic differentiation, just like any deterministic neural network. The only stochasticity comes from \(\epsilon\), whose distribution does not depend on \(\phi\), so the gradient estimator has lower variance. In practice, \(g_\phi(\epsilon)\) is implemented as \(\mu_\phi + \sigma_\phi \odot \epsilon\) for a Gaussian factor, where \(\mu_\phi\) and \(\sigma_\phi\) are learned mean and standard-deviation parameters, and \(\odot\) is element-wise multiplication.

Kingma and Welling’s trick unlocks two critical advances. First, it works whenever the latent is continuous and can be written as a differentiable transform of fixed noise, making it simple to integrate with PyTorch, JAX, or TensorFlow. Second, it turns the ELBO maximization into a form that SGD can handle, which lets us use the same tooling as supervised deep learning.

Black-box VI algorithms—take the gradient estimator, average over a few \(\epsilon\) samples, and update \(\phi\)—are now routine. They form the backbone of variational autoencoders, where an encoder network outputs \(\mu\) and \(\sigma\) and the decoder reconstructs the data. The only stochastic computation is the reparameterization, so training time per batch is comparable to training a deterministic encoder-decoder pair.

### Stochastic Variational Inference and scaling to big data

Black-box gradients give us per-sample updates, but real data arrives in millions of points. Stochastic Variational Inference (SVI) lifts the ELBO to minibatches. As Hoffman, Blei, Wang, and Paisley (2013) [https://www.cs.columbia.edu/~blei/papers/HoffmanBleiWangPaisley2013a.pdf; http://www.cs.columbia.edu/~blei/papers/HoffmanBleiWangPaisley2013.pdf] show, we can define a global variational parameter \(\phi\) for shared latent structure and local variational parameters \(\lambda_d\) for each datum \(d\). The joint ELBO over the dataset \(\mathcal{D}=\{x_d\}_{d=1}^D\) is
\[
\mathcal{L}(\phi, \{\lambda_d\}) = \sum_{d=1}^D \mathbb{E}_{q_{\phi,\lambda_d}}[\log p(x_d, z_d) - \log q_{\phi,\lambda_d}(z_d)],
\]
where \(z_d\) are the latent variables attached to datapoint \(x_d\), and \(\lambda_d\) controls their local approximate posterior. \(D\) is the total number of datapoints.
The expectation is taken over both global and local latent variational factors. Hoffman et al. derive natural gradient updates for \(\phi\) using the structure of conditional conjugacy; in the non-conjugate case, we simply plug the noisy gradients from reparameterization.

SVI replaces the full sum with a stochastic estimate using a minibatch \(\mathcal{B}\subset\mathcal{D}\):
\[
\mathcal{L}_{\mathcal{B}}(\phi) \approx \frac{D}{|\mathcal{B}|} \sum_{d\in\mathcal{B}} \mathbb{E}_{q_{\phi,\lambda_d}}[\log p(x_d, z_d) - \log q_{\phi,\lambda_d}(z_d)].
\]
\(|\mathcal{B}|\) is the minibatch size.
This estimator is unbiased and can be differentiated via backpropagation. We then update \(\phi\) with an optimizer like Adam, treating the scaled minibatch objective as if it were the entire dataset.

SVI’s key insight is that the gradient with respect to global variational parameters aggregates contributions from each minibatch, so we can scale to massive data volumes while keeping the optimization cheap per step. The only price we pay is variance from subsampling, which is amortized by large but manageable batch sizes and adaptive learning rates.

### Failure modes and diagnostics

Every variational approximation makes a choice. Mean-field factors assume independence, so they underestimate uncertainty when dimensions are correlated. Reparameterization only applies cleanly to continuous latents, which makes discrete latent models require methods like the Concrete relaxation or score-function estimators instead. The optimization can get stuck in local optima, especially when the ELBO landscape has multiple modes. Common diagnostics include comparing predictive log-likelihoods on held-out data, using simulation-based calibration, and monitoring the KL divergence terms for collapse (e.g., a too-large \(\sigma_\phi\) in Gaussian approximations indicates underfitting). Another red flag is the mismatch between the inductive biases of the variational family and the true posterior; the next section explores how richer families and flows are employed to relax those assumptions.

## Where the field is now

The research frontier is sharpening around expressive variational families. Rezende and Mohamed (2015) [https://arxiv.org/abs/1505.05770] introduced normalizing flows, successively composing simple invertible transformations so a base Gaussian can morph into a complex posterior shape while retaining a tractable Jacobian. They report log-likelihood improvements of several nats on MNIST and CIFAR-10 compared to diagonal mean-field; those benchmarks remain reference points for comparing new flows. Kucukelbir et al. (2017) [https://arxiv.org/abs/1603.00788], in their Automatic Differentiation Variational Inference (ADVI), combine reparameterization with automatic differentiation to automatically generate variational families from model specifications and to report ELBO convergence curves within seconds on datasets such as Boston housing, showing that VI can be as black-box as modern neural networks. Together these papers advance the frontier by asking: how rich can the variational family be before the optimization loses stability and scalability?

On the engineering side, Uber AI Labs’ Pyro platform exemplifies how variational inference runs in production. Pyro builds on PyTorch and exposes Stochastic Variational Inference primitives, tensors for parameterized guide networks, and hardware-aware compilation. According to the Pyro announcement on the Uber Engineering blog [https://eng.uber.com/pyro/], the framework supports hundreds of hierarchical production models for demand and pricing, processing millions of trips and tens of thousands of inference calls per second with GPU acceleration; the blog highlights multi-city supply forecasting as a concrete use case. This shows that VI is not a research curiosity but part of an end-to-end engineering stack that must juggle latency, hardware, and streaming data.

## What's still open

1. Can we design variational families that capture highly correlated, multi-modal posteriors without exploding the \(O(1)\) cost of mean-field updates? In particular, is there a composable recipe that mixes flows with structured covariances while still allowing per-data-point minibatch updates?
2. How can discrete and mixed continuous-discrete latent spaces be reparameterized without resorting to high-variance score-function estimators? A working solution would allow VI to compete with autoregressive sampling in structured prediction tasks like probabilistic programming and combinatorial optimization.
3. What are reliable diagnostics that distinguish variational under-fitting from model misspecification at train time, especially when ELBO gradients appear to converge but downstream predictive distributions deviate from held-out data? A single metric that flags such failure would make VI safe for regulatory applications.
4. Can amortized inference networks be trained to generalize across datasets, supporting rapid few-shot posterior estimation without re-optimizing the entire ELBO? Solving this would bridge few-shot learning with probabilistic reasoning in a principled way.

## Where to read next

For the probabilistic foundation of reparameterization and score estimation, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) explains how denoising and Tweedie’s formula recover gradients of log-densities without normalizing constants. The engineering counterpart is → [[probabilistic-programming-platforms]] which surveys how Pyro and TensorFlow Probability keep VI running at the low latencies production requires. If you want to see these ideas applied to robust uncertainty quantification in neural nets, → [Bayesian neural networks](bayesian-neural-networks.md) walks through Bayesian linear layers, dropout approximations, and their evaluation metrics.

## Build it

This build proves that variational inference is not academic: a single PyTorch script using reparameterized Gaussian guides and minibatch ELBO gradients recovers the true posterior over a Bayesian linear regression on real tabular data.

**What you're building:** a raw PyTorch Black Box Variational Inference implementation that fits mean and variance parameters via the reparameterization trick to approximate the posterior over weights and bias for Bayesian linear regression.

**Why this is valuable:** the build forces you to work through the ELBO, derive its gradient as a reparameterized expectation, and compare those gradients with the true ground-truth posterior of a conjugate model, so you see where the approximation errors come from.

**Stack:**
- **Model:** [hf-internal-testing/tiny-random-mlp](https://huggingface.co/hf-internal-testing/tiny-random-mlp) — minimal multi-layer perceptron used here to instantiate the deterministic part of the regression head (downloads: 3.6k).
- **Dataset:** [UCI/wine](https://huggingface.co/datasets/uci/wine) — classic tabular dataset with \(n=178\) samples and \(13\) features, recommended in probabilistic classrooms.
- **Framework:** PyTorch 2.2 + NumPy + Matplotlib + `torchdistributions`.
- **Compute:** Colab T4 GPU or local CPU (4 vCPUs); the training loop with 1k gradient steps finishes in under 10 seconds.

**The recipe:**
1. Install `pip install torch==2.2 torchdistributions matplotlib datasets`. Load the Wine dataset with `datasets.load_dataset("uci/wine")` and standardize each feature column to zero mean and unit variance.
2. Define a Bayesian linear regression by wrapping the HuggingFace tiny MLP: replace the deterministic final layer with variational parameters \(\mu\) and \(\rho\) (where \(\sigma=\log(1+\exp(\rho))\)). Sample weights via \(\epsilon \sim \mathcal{N}(0,I)\) and set \(w=\mu + \sigma \odot \epsilon\). \(I\) is identity, and \(\odot\) is elementwise multiplication.
3. Implement the ELBO: \(\mathcal{L}=\mathbb{E}_{\epsilon}[\log p(y\mid X, w) + \log p(w) - \log q(w)]\) where \(p(y\mid X, w)\) is Gaussian likelihood with fixed noise \(\sigma_y=0.1\), and \(p(w)=\mathcal{N}(0,I)\) is the prior. Use the same \(\epsilon\) sample for both forward and KL terms to keep the estimator low variance. Optimize \(\mu\) and \(\rho\) with Adam, learning rate \(1\mathrm{e}{-3}\), minibatch size 32,