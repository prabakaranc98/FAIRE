---
title: Expectation-Maximization
track: 05-statistical-probabilistic-ml
tags: [latent-variables, maximum-likelihood, optimization, inference]
depth: foundational
prereqs: [maximum-likelihood-estimation, gaussian-mixture-models]
updated: 2025-05-14
has_mvb: true
---

# Expectation-Maximization

> **TL;DR:** The Expectation-Maximization (EM) algorithm is an iterative optimization framework for finding maximum likelihood estimates of parameters in models with latent variables, where direct optimization is analytically intractable.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is

Imagine you are trying to characterize the distribution of heights in a population, but your data is incomplete: some individuals are hidden behind a curtain, and you only observe a subset of the total population. If you knew which individuals belonged to which group, estimating the mean and variance would be trivial. Because the group membership is unknown—a latent variable, or a hidden factor that influences the observed data but is not directly measured—you cannot directly calculate the likelihood of the observed data.

The Expectation-Maximization algorithm solves this by alternating between two steps. In the Expectation (E) step, you estimate the distribution of the hidden variables given your current best guess of the parameters. In the Maximization (M) step, you update your parameters to maximize the likelihood of the observed data, assuming the estimates from the E-step are correct. 

This iterative process is guaranteed to increase the likelihood of the observed data at every step. The algorithm effectively navigates the likelihood surface by creating a sequence of lower-bound approximations that are easier to optimize than the original, complex objective.

## Why it matters at the frontier

EM is the backbone of many foundational statistical models, including Gaussian Mixture Models (GMMs) and Hidden Markov Models (HMMs). It enables the training of models that can discover hidden clusters or temporal patterns in data without requiring explicit labels for every observation.

The algorithm serves as a bridge to modern variational inference. While EM provides a point estimate for parameters, variational methods extend this logic to approximate the full posterior distribution. Understanding EM is essential for grasping why modern probabilistic models rely on lower-bound optimization, as the E-step is essentially the maximization of the Evidence Lower Bound (ELBO).

## Core concepts

- **Latent variables** — Unobserved variables that influence the distribution of the observed data.
- **E-step** — The calculation of the posterior distribution of latent variables given the current parameter estimates.
- **M-step** — The update of model parameters to maximize the expected log-likelihood found in the E-step.
- **Incomplete data** — A scenario where the likelihood function involves marginalizing over unobserved variables.
- **Convergence** — The property where the algorithm reaches a stationary point of the likelihood surface.
- **ELBO** — The Evidence Lower Bound, a lower bound on the log-likelihood used as a surrogate objective for optimization.

## Mathematical foundations

The goal is to maximize the log-likelihood of the observed data \(X\) given parameters \(\theta\):

\[ \log p(X|\theta) = \log \sum_Z p(X, Z|\theta) \]

where \(X\) is the observed data, \(Z\) is the latent variable, and \(\theta\) represents the model parameters. This equation requires marginalizing over all possible states of the hidden variables to obtain the likelihood of the observed data.

We introduce a distribution \(q(Z)\) over the latent variables to define the ELBO:

\[ \mathcal{L}(q, \theta) = \sum_Z q(Z) \log \frac{p(X, Z|\theta)}{q(Z)} \]

where \(q(Z)\) is an arbitrary distribution over the latent variables, and \(\mathcal{L}\) is the Evidence Lower Bound. This equation provides a lower bound on the log-likelihood, which we maximize iteratively.

The E-step sets \(q(Z) = p(Z|X, \theta^{(t)})\), and the M-step sets \(\theta^{(t+1)} = \arg \max_\theta \sum_Z q(Z) \log p(X, Z|\theta)\). This sequence ensures that the log-likelihood is monotonically non-decreasing.

## Key algorithms / techniques

- **GMM-EM** — The standard application of EM to Gaussian Mixture Models, where the E-step computes responsibilities and the M-step updates cluster means and covariances.
- **Baum-Welch** — A specialized version of EM for Hidden Markov Models used to estimate transition and emission probabilities by iteratively updating the forward-backward variables.
- **Amortized EM** — A modern technique that replaces the iterative E-step with a neural network inference model, commonly used in Variational Autoencoders.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| "Maximum Likelihood from Incomplete Data via the EM Algorithm" | 1977 | Dempster et al. | The foundational proof of the algorithm's convergence properties. |
| "Extending Mean-Field Variational Inference via Entropic Regularization" | 2024 | Wu et al. | Connects EM to modern variational inference frameworks. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Dempster et al. (1977) | 1977 | Formalized the EM algorithm for general incomplete data problems. |
| Neal and Hinton (1998) | 1998 | Provided the "view as coordinate ascent" interpretation of EM. |

## Current SotA

Modern variants of EM are integrated into deep probabilistic frameworks. EigenVI achieves state-of-the-art performance in score-based variational inference by using orthogonal function expansions to approximate gradients (Cai et al., 2024, https://arxiv.org/abs/2410.24054). Furthermore, recent work by Kim (2026, https://arxiv.org/abs/2604.15469) demonstrates that score-based variational posterior inference can scale to Bayesian deep neural networks with millions of parameters, significantly outperforming traditional mean-field approximations on standard benchmarks like CIFAR-10.

## What's happening now

Research is currently focused on scaling EM-like procedures to high-dimensional deep neural networks (Kim, 2026, https://arxiv.org/abs/2604.15469). These methods aim to bypass the need for explicit E-step marginalization, which is often intractable in high-dimensional spaces.

Engineering efforts are shifting toward hardware-accelerated M-steps. Liu et al. (2026, https://arxiv.org/abs/2604.15469) explored sample continuation techniques in hierarchical models, allowing for faster convergence on GPU clusters by parallelizing the parameter update phase.

Open problems remain regarding the non-convexity of the likelihood surface. Xu and Campbell (2021, https://arxiv.org/pdf/2104.05886v1) highlighted that the computational asymptotics of variational inference are highly sensitive to the initialization of latent distributions, a problem that remains largely unsolved for complex architectures.

## Open questions

> **Researcher:** How can we derive convergence guarantees for EM-based procedures in non-convex deep learning landscapes where the ELBO is only locally optimized?

> **Engineer:** What are the optimal hardware primitives for accelerating the M-step in large-scale Bayesian hierarchical models when the latent space dimension exceeds \(10^6\)?

> **Open:** Can we develop a universal initialization strategy for latent distributions that guarantees global convergence for EM-based variational inference in deep neural networks?

## In production

- **Google** — Ads Click-Through Rate Prediction — Uses EM-based clustering to handle latent user intent at scale (research.google).
- **Netflix** — Recommendation Systems — Employs latent factor models trained via EM-like iterations to personalize content (Netflix TechBlog, 2023).

## Minimum Valuable Build

**Build:** Train a Gaussian Mixture Model (GMM) on the MNIST dataset to cluster digits.
**Compute:** Runs on RTX 3080 (10GB VRAM) or free Colab T4.

1. Load the MNIST dataset using `torchvision.datasets.MNIST`.
2. Flatten the images and use `sklearn.mixture.GaussianMixture` to fit 10 components.
3. Set `n_init=5` and `max_iter=100` to ensure stable convergence.
4. Extract the cluster centers using `gmm.means_` and reshape them to 28x28.
5. Plot the cluster centers to visualize the "prototypical" digits learned by the model.

**Expected outcome:** A plot of 10 images representing the learned cluster centers.
**Metric:** Log-likelihood improvement of >10% over random initialization.
**Researcher Variant:** Implement a custom EM loop in JAX to track the ELBO at each iteration, comparing the convergence rate against the standard `sklearn` implementation.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations

- [scikit-learn GMM](https://scikit-learn.org/stable/modules/generated/sklearn.mixture.GaussianMixture.html) — The standard production-grade implementation of EM for Gaussian mixtures.
- [Pyro](https://pyro.ai/) — A deep probabilistic programming language that implements modern variational EM variants.

## This concept appears in

- [[../../arcs/probabilistic-ml/step-01-em.md]] — This page serves as the entry point for understanding iterative parameter estimation in latent variable models.

## What comes next

Understanding EM provides the necessary intuition for how we optimize models when we cannot observe the full state of the world. This serves as the foundation for more complex inference techniques like Variational Inference and MCMC.

- [[variational-inference]] — Extends the EM framework to approximate complex posterior distributions.
- [[hidden-markov-models]] — Uses the Baum-Welch algorithm, a specific application of EM, to learn temporal dependencies.
- [[bayesian-neural-networks]] — Applies EM-like principles to estimate the distribution of weights in deep networks.

## Connected topics

- [[bayesian-inference]] — EM is a primary algorithm used in Bayesian inference for point estimation.
- [[bayesian-nn]] — EM provides the optimization framework for training Bayesian neural networks.
- [[bootstrapping-methods]] — EM can be viewed as a form of bootstrapping for parameter estimation.
- [[bias-variance-tradeoff]] — EM is sensitive to the bias-variance tradeoff when fitting models with latent variables.
- [[disentanglement]] — EM is used in latent variable models that aim to learn disentangled representations.

## Further reading

- [Lilian Weng's survey on EM](https://lilianweng.github.io/posts/2018-08-12-em/) — An intuitive walkthrough of the algorithm's derivation and its relation to coordinate ascent.
- [Dempster et al. (1977)](https://doi.org/10.1111/j.2517-6161.1977.tb01600.x) — The original paper defining the EM framework, published in the Journal of the Royal Statistical Society.
- [Bishop, Pattern Recognition and Machine Learning](https://www.microsoft.com/en-us/research/people/cmbishop/prml-book/) — Chapter 9 provides the definitive textbook treatment of EM and its applications.