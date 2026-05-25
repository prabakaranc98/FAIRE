---
title: Statistical & Probabilistic ML
tags: [bayesian, probabilistic, graphical-models, inference, statistics]
---

# Track 05 · Statistical & Probabilistic ML

> The probabilistic foundations of machine learning: Bayesian inference, graphical models, approximate inference, and uncertainty quantification.

Statistics is the backbone of learning from data. This track covers the principled probabilistic framework — from Bayes' theorem to variational inference to modern neural probabilistic models — that underlies much of what deep learning does implicitly.

---

## Topics

### Bayesian Methods
- [Bayesian Inference](bayesian-inference.md) — prior, likelihood, posterior, conjugate distributions
- [Gaussian Processes](gaussian-processes.md) — kernels, posterior regression, sparse GPs
- [Bayesian Neural Networks](bayesian-nn.md) — weight uncertainty, Laplace approximation, MC dropout

### Graphical Models
- [Directed Graphical Models](directed-graphical-models.md) — Bayesian networks, d-separation, factorization
- [Undirected Graphical Models](undirected-graphical-models.md) — Markov random fields, partition function, energy
- [Hidden Markov Models](hidden-markov-models.md) — forward-backward, Viterbi, EM for HMMs

### Inference
- [Variational Inference](variational-inference.md) — ELBO, mean-field, reparameterization, ADVI
- [MCMC Sampling](mcmc.md) — Metropolis-Hastings, Gibbs, Hamiltonian Monte Carlo
- [Expectation Maximization](em.md) — the EM algorithm, mixture models, latent variables

### Uncertainty
- [Uncertainty Quantification](uncertainty-quantification.md) — epistemic vs. aleatoric, calibration, conformal prediction
- [Distribution Shift](distribution-shift.md) — covariate shift, concept drift, domain generalization

---

## Connections to frontier research

- **Probabilistic generative models** — VAEs and diffusion as instantiations of latent variable models
- **Bayesian deep learning** — scalable posterior inference for large models
- **Conformal prediction** — distribution-free uncertainty quantification at scale
- **Causal graphical models** — the bridge to causal AI (Track 08)

---

## Recommended entry points

Start with [Bayesian Inference](bayesian-inference.md) and [Variational Inference](variational-inference.md) — these underpin almost everything else in the track and connect directly to generative modeling.
