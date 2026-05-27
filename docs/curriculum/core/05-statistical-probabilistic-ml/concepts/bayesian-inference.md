---
title: Bayesian inference
slug: bayesian-inference
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [bayes, martin, hoffman, khan, rue]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [probability-essentials, variational-inference, gaussian-processes]
tags: [bayesian, inference, uncertainty, probabilistic-programming, variational]
updated: 2024-11-15
has_mvb: true
---

# Bayesian inference

Imagine a self-driving car creeping through a patch of fog. Ahead is a shape that could be a harmless plastic bag on the curb or a concrete barrier directly in its lane. A frequentist system would dutifully turn that silhouette into a “most likely” label and commit to action, which is how one illusion of certainty turns into a real accident. The car we want instead keeps a distribution over “bag” and “barrier,” widening it when the noise grows and tightening it as the lidar returns cluster. Bayesian inference is the language for that distributional belief state: every sensor update nudges the posterior, every prior belief tethers it, and every lane-change request is weighed against the confidence in those probabilities. By the end of this page you will understand how Bayes turns estimation into continuous belief updates, how modern systems keep those updates tractable through stochastic variational inference, kernel embeddings, and learning-rule unifications, and how to ship a lightweight Bayesian neural network in Pyro on free Colab hardware that visibly widens its epistemic uncertainty whenever it leaves the training manifold.

## The territory

The field’s core question has remained almost unchanged since “An Essay towards solving a Problem in the Doctrine of Chances” (Bayes 1763) showed that a coin’s bias should not be a single scalar but a distribution that the next coin flip updates analytically. That moment made two commitments: parameters are random, and uncertainty must be propagated forward, not just reported as a single number. Every probabilistic modeling effort since depends on circulating those commitments through likelihoods, priors, and computation. Martin et al. (2020) [arxiv:2004.06425](https://arxiv.org/abs/2004.06425) traces how this circulation has been chained to the evolution of algorithms and hardware, from Laplace’s analytical conjugacy to modern MCMC and variational programs that rely on GPUs and autodiff. The practical territory now sits between three families: expressive probabilistic modeling for likelihood design, differentiable optimization for gradient-based posterior refinement, and dependable systems engineering to keep the engine running on streaming telemetry. The shape of the answer therefore is not just Bayes’ rule, it is a dynamic engine in which priors, likelihoods, and budgeted computation cycle continually through observations. The mechanism is best understood by starting from that rule itself and tracing how real-world constraints force approximations.

## How it works

Bayesian inference rewrites estimation as the iterative update of a distribution. The canonical formula is

\[
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta) p(\theta)}{\int p(\mathcal{D} \mid \theta') p(\theta') d\theta'}.
\]

where \(\theta\) is the latent quantity we care about, \(\mathcal{D}\) is the observed data, \(p(\theta)\) is the prior belief before seeing \(\mathcal{D}\), \(p(\mathcal{D} \mid \theta)\) is the likelihood packet, and the denominator is the marginal likelihood \(p(\mathcal{D})\) that normalizes the posterior. This is the mathematical statement of the self-driving car’s behavior: every new depth map multiplies the prior by the new likelihood and renormalizes, so uncertainty compresses if the likelihood is sharp and inflates when data is ambiguous.

Carrying that denominator forward is what makes exact inference expensive, so the field builds tractable engines that approximate the same multiplication. When the likelihood is conjugate to the prior (e.g., Gaussian-Gaussian), the integral collapses into closed form, but most modern models—nonlinear neural networks, heavy-tailed noise models—are far from conjugacy. That is where approximation gears spin: rather than computing \(p(\mathcal{D})\), we replace it with an optimization problem that finds a tractable surrogate \(q(\theta)\) close to the true posterior.

### Variational inference and stochastic gradients

In variational inference, we posit a family \(\mathcal{Q}\) of distributions and minimize the Kullback-Leibler divergence from \(q(\theta)\) to the true posterior. Practically we maximize the evidence lower bound (ELBO)

\[
\mathcal{L}(q) = \mathbb{E}_{q(\theta)}\left[\log p(\mathcal{D}, \theta) - \log q(\theta)\right]
\]

where \(q(\theta)\) is our variational approximation, \(p(\mathcal{D}, \theta)\) is the joint density, and the expectation is taken over the variational distribution. The KL term ensures the approximation does not stray from the prior or the likelihood. Hoffman et al. (2013) [arxiv:1206.5533](https://arxiv.org/abs/1206.5533) introduced stochastic variational inference (SVI) that makes this optimization scalable: instead of summing over the entire dataset, we draw a minibatch \(\mathcal{B}\), compute a stochastic gradient of \(\mathcal{L}(q)\) with respect to the variational parameters \(\lambda\), and take a small optimizer step. The key insight is that the global ELBO decomposes into local contributions from each datapoint, so the minibatch gradient is an unbiased estimator of the full gradient. SVI’s machinery of reparameterization gradients, natural gradient steps, and amortized inference lets Bayesian inference run on datasets with millions of examples while keeping the posterior updated after each minibatch.

### The Bayesian learning rule as optimizer + probabilistic inference

The step from SVI to production-grade training involves more than minibatches: it is about picking update rules that respect the geometry of the posterior. Khan & Rue (2021) [arxiv:2103.04883](https://arxiv.org/abs/2103.04883) argue that standard optimization algorithms like Adam or RMSProp are instances of a broader “Bayesian learning rule,” in which each parameter update performs a localized Bayes’ rule: gradient steps are natural gradients that move along the manifold defined by the current posterior approximation, and second-order adaptation arises from carrying posterior covariance information. Casting Adam as a weighted combination of prior precision and observed gradient covariance reveals that even deterministic optimizers are implicitly managing uncertainty. In practice, this perspective yields hybrid algorithms where the prior is regularized through Bayesian updates, while the optimizer’s adaptive learning rates are grounded in posterior scale. The result is convergence behavior that mirrors variational inference while remaining compatible with existing deep learning frameworks.

### Kernel Bayes’ rule for nonparametric posteriors

When likelihoods are not easily specified but we can sample from generative simulators or streams, we can still apply Bayes via embeddings. Kernel Bayes’ Rule (Fukumizu et al. 2013) [https://jmlr.csail.mit.edu/papers/volume14/fukumizu13a/fukumizu13a.pdf] lifts priors and likelihoods into reproducing kernel Hilbert spaces and performs the multiplication there. The core idea is that the conditional embedding \(C_{Y|X}\) satisfies

\[
\mu_{Y \mid x} = C_{Y X} C_{XX}^{-1} \phi(x)
\]

where \(C_{Y X}\) and \(C_{XX}\) are cross-covariance operators in the RKHS, \(\phi(x)\) is the feature map of \(x\), and \(\mu_{Y \mid x}\) is the embedded conditional distribution. Estimating these operators from samples and multiplying them is computationally cheaper than performing likelihood evaluations, yet it captures nonparametric structure because the feature map can be any universal kernel. Kernelized Bayes is therefore a way of updating beliefs when the modeler does not trust parametric likelihoods, which keeps the “foggy” prediction alive even when only complex simulators or high-dimensional summaries are available. This is especially relevant for systems like the brain, which avoid explicit likelihood functions: Doya (2007) [arxiv:0402205](https://bayes.wustl.edu/etj/articles/how.does.the.brain.orig.pdf) describes neural circuitry as performing approximate Bayesian updates through recurrent loops and gain modulation, akin to kernelized inference in biological feature spaces.

### Composing priors, likelihoods, and heteroscedastic noise

A Bayesian neural network (BNN) is assembled by placing priors over the weights of an MLP and computing a posterior given the dataset. In practice, we specify \(p(\theta)\) as independent Gaussians for each weight tensor, \(p(y \mid x, \theta)\) via a heteroscedastic likelihood (e.g., \(y \sim \mathcal{N}(f_\theta(x), \sigma^2(x))\)), and approximate the posterior \(q(\theta)\) with a mean-field Gaussian whose parameters are learned through SVI. The heteroscedastic likelihood is crucial for revealing regions of the input space with sparse data: the network predicts both mean and variance, so the epistemic uncertainty from the posterior blends with aleatoric noise to widen credible intervals away from the training manifold. When training on synthetic heteroscedastic data (e.g., \(y = \sin(x) + \epsilon(x)\) with \(\epsilon(x) \sim \mathcal{N}(0, 0.05 + 0.2|x|)\)), the posterior’s variance tracks the increasing noise, which reinforces the idea that Bayesian inference is not just about point prediction but about learning to “know what you do not know.”

### Productionizing the posterior

Deploying Bayesian inference requires pipelines that update posterior beliefs as data streams in. Production systems run either streaming SVI (updating \(\lambda\) with each incoming batch) or Bayesian filtering (Kalman or particle filters) for time series. A practical pattern is to warm-start the BNN on historical data, freeze the prior covariance, and then keep measuring the divergence between the incoming batch predictions and the historical posterior. If divergence exceeds a threshold, the system increases exploration (wider priors, lower learning rates) until confidence stabilizes. This continual recalibration is how modern telemetry systems avoid cascading outages: they downgrade the influence of anomalous batches while retaining the learned posterior from “normal” operation.

## Where the field is now

Recent research still grapples with scaling inference to the billions of parameters in modern neural networks. The Bayesian Learning Rule (Khan & Rue 2021) has triggered experiments that reinterpret Adam and other adaptive optimizers as approximate natural-gradient steps with explicit covariance tracking, opening a research frontier in which optimizers come with statistical certificates on how much uncertainty they propagate. On the other hand, stochastic variational inference continues to drive the practical side: there are now libraries that wrap PyTorch models and automatically handle reparameterization, Monte Carlo sampling, and distributed gradients for millions of datapoints. The research frontier lies in closing the gap between these adaptive optimizers and full posterior inference; if the optimizer’s intrinsic uncertainty estimates could be promoted to true posterior covariances, then the sampling burden might vanish.

At the engineering frontier, Google’s AutoBNN (Google Research blog, 2024) stitches together compositional Bayesian neural networks that forecast time series with uncertainty-aware attention, all deployed in production time-series forecasts. The system allocates compute to Bayesian layers only where uncertainty spikes and quantizes the rest with deterministic backbones, delivering the 30% better anomaly detection that the blog reports without exceeding existing latency targets. That deployment shows how modern inference must be framed: not as a single SVI run, but as a flow of Bayes updates that rescale compute based on the posterior’s entropy.

Alongside these two frontiers, Kernel Bayes’ Rule still influences simulators and likelihood-free inference; researchers are now combining kernel embeddings with neural conditional density estimators to update posteriors when both the simulator and the latent space are high-dimensional. The core tension remains: our posterior approximations must live inside practical optimizers, yet they must stay expressive enough to highlight when “fog” makes the next decision uncertain.

## What's still open

Can we preserve the covariance structure between billions of parameters without collapsing to mean-field approximations? Current Bayesian neural networks and variational programs routinely assume independence between weights to keep computations feasible, but that assumption erases important correlations that modulate uncertainty propagation. A concrete research question is: when training a Transformer-scale model, can we carry a low-rank approximation of the weight covariance through SVI updates that still fits inside 16 GB of VRAM and that meaningfully changes downstream uncertainty metrics?

How should streaming inference reconcile stale priors with sudden regime shifts? Systems like the self-driving car need to keep beliefs updated when the world changes faster than the posterior can adapt, yet resetting the prior throws away hard-won knowledge. Is there a Bayesian forgetting rule—perhaps inspired by Doya’s brain models—that automatically reweights old posterior mass against recent evidence without manual interventions?

Finally, what’s the minimal posterior representation that lets us deploy uncertainty-aware models at production latencies below 50 ms p95? The engineering solutions today either quantize weights or push posterior updates off the critical path, but neither delivers full Bayesian fidelity. Identifying a representational bottleneck that captures the “essence” of the posterior while staying within these latency constraints would unlock the next wave of robust inference in deployed systems.

## Where to read next

If you want the optimization perspective, → [Variational Inference](variational-inference.md) explains how the ELBO and reparameterization gradients turn probabilistic questions into differentiable programs; the engineering counterpart is → [[probabilistic-programming]] where frameworks wrap everything into deployable services; for tighter uncertainty calibration, → [Gaussian processes](gaussian-processes.md) provides the nonparametric counterpart whose posteriors inspire kernelized Bayes.

## Build it

Training a Bayesian neural network on synthetic heteroscedastic data proves that the posterior is a living, widening belief state rather than a single weight vector.

**What you're building:** a Pyro + PyTorch Bayesian MLP that learns both a mean and heteroscedastic variance on handcrafted \$x \mapsto \sin(x)\$ data, showing wide posterior bands away from training points.

**Why this is valuable:** the recipe forces you to implement priors, variational posteriors, and SVI updates so you can directly observe how epistemic uncertainty grows where the likelihood provides no information.

**Stack:**
- **Model:** `hf-internal-testing/tiny-random-mlp` — (92 downloads, minimal downloadable checkpoint) wrapped into a Bayesian torch module with priors on weights.
- **Dataset:** `huggingface/california_housing` — use features plus a synthetic heterosced noise schedule derived from the dataset to scale noise with feature magnitude.
- **Framework:** Pyro 1.9 + PyTorch 2.2 for the probabilistic programming layer, Matplotlib 3.9 for plots.
- **Compute:** Free Colab T4 (16 GB VRAM) or any single RTX 4090 (24 GB); training runs in ~35 minutes.

**The recipe:**
1. `pip install pyro-ppl==1.9 torch==2.2 matplotlib pandas scikit-learn`
2. Load `california_housing`, standardize the feature matrix \(X\), and create targets \(y = \sin(3x_0) + \epsilon\) where \(\epsilon \sim \mathcal{N}(0, 0.1 + 0.4 |x_0|)\) to embed heteroscedasticity.
3. Define a Pyro module that wraps the Tiny Random MLP, replacing each weight tensor with a Normal prior and implementing a variational posterior that samples per forward pass; optimize the ELBO using SVI with learning rate 0.005 and 800 epochs, tracking the loss on a held-out slice.
4. Evaluate by sampling 100 posterior models and computing the 95% credible bands over a dense grid; the expected number is that the bands tighten inside the training region and expand to cover ±0.5 outside.
5. You now have a trained Bayesian MLP checkpoint plus plot artifacts that visualize epistemic uncertainty shifting with the data density.

**Expected outcome:** a saved Pyro checkpoint containing the posterior parameters plus a batch of plots showing Bayesian credible intervals that widen away from the training manifold.

- **CS student:** On free Colab or an RTX 4070 laptop, reduce the hidden width to 32 units and halve the epochs so the build still runs in 20 minutes while keeping the same visual posterior playbook.
- **Applied engineer:** Quantize the learned variational parameters (8-bit) and serve the model with vLLM (or Triton) at p95 ≤ 45 ms by caching posterior samples as a small ensemble.
- **Applied researcher:** Hypothesize that replacing the heteroscedastic likelihood with a Student-\(t\) noise model yields wider tails; implement the alternative likelihood, retrain, and falsify by comparing RMSE and average uncertainty width to the original.
- **Frontier researcher:** Probe the open question from above by attaching a low-rank plus diagonal covariance parameterization to the posterior, training on the same data, and measuring whether the richer covariance changes predictive entropy outside the training region.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*