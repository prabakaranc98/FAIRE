---
title: Variational Inference
track: 05-statistical-probabilistic-ml
tags: [variational-inference, elbo, mean-field, kl-divergence, approximate-inference]
depth: foundations
prereqs: [bayesian-inference]
updated: 2026-05-25
---

# Variational Inference
> **TL;DR:** Approximate Bayesian inference by optimization — cast intractable posterior inference as minimizing KL divergence between a tractable variational family and the true posterior.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **ELBO** — Evidence Lower BOund; the objective maximized in VI
- **KL divergence** — KL(q‖p); measures how far q is from the true posterior
- **Mean-field VI** — q factorizes across all dimensions; coordinate ascent updates
- **Reparameterization** — enables backprop through stochastic sampling
- **ADVI** — Automatic Differentiation VI; general-purpose VI using reparameterization

## Mathematical foundations
ELBO = E_q[log p(x,z)] - E_q[log q(z)] = log p(x) - KL(q(z)‖p(z|x))

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | 2013 | Kingma & Welling | VAE = VI + reparameterization trick |
| [Automatic Differentiation Variational Inference](https://arxiv.org/abs/1603.00788) | 2016 | Kucukelbir et al. | ADVI — general VI via autodiff |

## Connected topics
- [Bayesian Inference](./bayesian-inference.md) — VI approximates intractable posteriors
- [Variational Autoencoders](../02-generative-modeling/variational-autoencoders.md) — VAEs apply VI to deep generative models
- [Diffusion Models](../02-generative-modeling/diffusion-models.md) — DDPM training loss is a form of variational bound
