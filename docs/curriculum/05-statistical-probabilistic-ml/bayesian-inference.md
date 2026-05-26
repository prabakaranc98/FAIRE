---
title: Bayesian Inference
track: 05-statistical-probabilistic-ml
tags: [bayesian, prior, posterior, likelihood, inference]
depth: foundations
prereqs: []
updated: 2026-05-25
---

# Bayesian Inference
> **TL;DR:** The framework for updating beliefs in light of evidence — prior belief × likelihood of data → posterior belief — and the foundation of principled uncertainty quantification in ML.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **Prior p(θ)** — belief about parameters before seeing data
- **Likelihood p(D|θ)** — probability of data given parameters
- **Posterior p(θ|D)** — updated belief; proportional to prior × likelihood
- **Bayes' theorem** — p(θ|D) ∝ p(D|θ)p(θ)
- **Conjugate prior** — prior and posterior have the same functional form
- **MAP estimate** — mode of the posterior; reduces to regularized MLE

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Probabilistic Machine Learning: An Introduction](https://probml.github.io/pml-book/book1.html) | 2022 | Murphy | Authoritative reference (MIT Press open-access) |

## Connected topics
- [Variational Inference](./variational-inference.md) — scalable approximate Bayesian inference
- [Gaussian Processes](./gaussian-processes.md) — Bayesian nonparametrics via kernel functions
- [Variational Autoencoders](../02-generative-modeling/variational-autoencoders.md) — VAEs are Bayesian latent variable models
