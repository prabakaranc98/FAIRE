---
title: Generative Modeling
tags: [generative-models, diffusion, flow-matching, vae, gan, score-matching]
---

# Track 02 · Generative Modeling

> Learning the structure of data distributions: from variational autoencoders to score-based diffusion to flow matching.

Generative modeling is one of the most active frontiers in deep learning. The question — how do you learn to sample from a complex, high-dimensional distribution — has produced a cascade of architectures and training objectives over the past decade.

---

## Topics

### Foundational Models
- [Variational Autoencoders](variational-autoencoders.md) — ELBO, reparameterization, latent spaces
- [Generative Adversarial Networks](generative-adversarial-networks.md) — minimax training, mode collapse, Wasserstein distance

### Score-Based & Diffusion Models
- [Diffusion Models](diffusion-models.md) — DDPM, DDIM, score matching, noise schedules
- [Score Matching & SDEs](score-matching.md) — Stein score, Langevin dynamics, stochastic differential equations
- [Consistency Models](consistency-models.md) — distillation, consistency training, single-step generation

### Flow-Based Models
- [Normalizing Flows](normalizing-flows.md) — change of variables, RealNVP, Glow
- [Flow Matching](flow-matching.md) — continuous normalizing flows, optimal transport, Rectified Flow

### Energy-Based Models
- [Energy-Based Models](energy-based-models.md) — contrastive divergence, MCMC sampling, EBM training

---

## Connections to frontier research

- **Latent diffusion** — operating diffusion in compressed latent space (Stable Diffusion, FLUX)
- **Video generation** — temporal consistency and causal attention in generative video
- **Protein structure** — diffusion and flow matching applied to molecular geometry (AlphaFold 3, RFDiffusion)
- **Scientific simulation** — generative models as surrogates for physical simulators

---

## Recommended entry points

Start with [Variational Autoencoders](variational-autoencoders.md) for the latent variable framework, then [Diffusion Models](diffusion-models.md) for the dominant modern paradigm. [Flow Matching](flow-matching.md) is the current frontier.
