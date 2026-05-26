---
title: Variational Autoencoders
track: 02-generative-modeling
tags: [vae, elbo, latent-variable, reparameterization, generative-models]
depth: foundations
prereqs: [bayesian-inference, variational-inference]
updated: 2026-05-25
---

# Variational Autoencoders
> **TL;DR:** Latent variable models that learn to encode data into a structured distribution and decode samples back — the foundational framework for learned latent spaces in modern generative AI.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
A VAE jointly trains an encoder q_φ(z|x) that maps data to a latent distribution and a decoder p_θ(x|z) that generates data from latent samples. Training maximizes the Evidence Lower BOund (ELBO): reconstruction quality plus a KL divergence term that regularizes the latent space toward a prior.

## Why it matters at the frontier
VAEs provide the latent space backbone for latent diffusion models (Stable Diffusion, FLUX). The ELBO objective is the template for a large family of variational methods. Understanding VAEs is prerequisite to understanding most modern generative systems.

## Core concepts
- **Latent variable model** — data x is explained by unobserved latent z
- **ELBO** — Evidence Lower BOund; the tractable training objective
- **Reparameterization trick** — sample z = μ + σε (ε ~ N(0,I)) to enable backpropagation
- **KL divergence** — regularizer pushing q(z|x) toward prior p(z)
- **Reconstruction loss** — log p(x|z); encourages decoder to reconstruct input
- **Posterior collapse** — failure mode where decoder ignores z; common in powerful decoders

## Mathematical foundations
ELBO objective:
\[
\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - \text{KL}(q_\phi(z|x) \| p(z))
\]

Reparameterization:
\[
z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
\]

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | 2013 | Kingma & Welling | Original VAE paper — start here |
| [An Introduction to Variational Autoencoders](https://arxiv.org/abs/1906.02691) | 2019 | Kingma & Welling | Authoritative review by the original authors |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | 2013 | VAE + ELBO + reparameterization trick |
| [β-VAE: Learning Basic Visual Concepts](https://openreview.net/forum?id=Sy2fchgRb) | 2017 | Higgins et al. | Disentanglement via KL weighting |

## Current SotA
> *Updated: 2026-05-25*
VAEs are now primarily used as compression backbones rather than standalone generative models. Latent diffusion models (Stable Diffusion 3, FLUX) use VQ-VAE or continuous VAE to encode images into 4-16× compressed latents before applying diffusion. The VAE itself is often pretrained and frozen.

## Connected topics
- [Diffusion Models](./diffusion-models.md) — latent diffusion uses VAE as encoder
- [Variational Inference](../05-statistical-probabilistic-ml/variational-inference.md) — ELBO is the central tool
- Normalizing Flows — alternative to VAE's approximate posterior

## Further reading
- [Tutorial on Variational Autoencoders](https://arxiv.org/abs/1606.05908) — Doersch 2016; clear pedagogical treatment
