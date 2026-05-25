---
title: "Arc: The Generative Stack — VAEs to Flow Matching"
arc: generative-stack
super_domain: B-Modeling
tracks: [02-generative-modeling, 03-representation-learning, 05-statistical-probabilistic-ml]
estimated_depth: "6-8 weeks, ~25 papers"
prereqs: [variational-inference, backpropagation, optimization]
---

# Arc: The Generative Stack
> **What this arc builds:** A deep understanding of how generative models work — from the foundational latent variable framework through diffusion to flow matching — and why the current dominant paradigm looks the way it does.

## Why this arc exists

Generative modeling asks one question: how do you sample from a complex distribution you've never seen directly? The answer has been refined three times in a decade: VAEs gave us continuous latent spaces and the ELBO; GANs gave us sharp samples via adversarial training; score-based models and diffusion gave us principled iterative refinement; flow matching gave us straight paths and faster inference.

Each generation solved a specific problem its predecessor had. Following this arc in order means you understand *why* each method was invented and what it's actually doing — not just how to call the API.

## Prerequisites

Be comfortable with: the ELBO and KL divergence (variational inference), backpropagation, basic probability (Gaussian distributions, change of variables). Some familiarity with PyTorch is helpful for the applied nodes.

## The sequence

**Foundational Framework**

1. **Latent variable models & EM** (foundational) — the general framework: p(x) = ∫ p(x|z)p(z)dz; intractable; need approximation.
2. **Variational Inference & ELBO** (theoretical) — maximize Evidence Lower Bound; KL divergence as the gap. [→](../../curriculum/05-statistical-probabilistic-ml/variational-inference.md)
3. **VAEs** (applied) — encoder + decoder + reparameterization trick; the first practical deep generative model. [→](../../curriculum/02-generative-modeling/variational-autoencoders.md)
4. **β-VAE & disentanglement** (theoretical) — KL weighting; what it means to have a structured latent space.

**Adversarial & Energy-Based**

5. **GANs** (applied) — minimax game; discriminator/generator; why they work and why they fail.
6. **Wasserstein GAN** (theoretical) — Earth Mover distance; Lipschitz constraint; more stable training.
7. **Energy-Based Models** (theoretical) — define unnormalized probability; MCMC for sampling; contrastive divergence.

**Score-Based & Diffusion**

8. **Score matching** (theoretical) — estimate score ∇_x log p(x) directly; Hyvärinen 2005.
9. **Denoising score matching** (applied) — match score of noisy data; connects to denoising autoencoders.
10. **DDPM** (applied) — forward noising + learned reverse denoising; the canonical diffusion model. [→](../../curriculum/02-generative-modeling/diffusion-models.md)
11. **Score-based SDEs** (theoretical) — unified SDE framework; DDPM and NCSN as special cases. [→](https://arxiv.org/abs/2011.13456)
12. **DDIM** (applied) — deterministic sampling; 50 steps instead of 1000. [→](https://arxiv.org/abs/2010.02502)
13. **Classifier-free guidance** (applied) — trade diversity for quality; the standard inference trick.
14. **Latent diffusion** (applied) — diffuse in compressed VAE latent space; Stable Diffusion's key insight.
15. **Consistency models** (frontier) — distill diffusion to 1-step; consistency training vs. distillation.

**Flow Matching — Current Frontier**

16. **Normalizing flows** (theoretical) — exact likelihood via change of variables; limited expressiveness but principled.
17. **Continuous normalizing flows (CNFs)** (theoretical) — define flow via ODE; exact likelihood; slow.
18. **Flow Matching** (frontier) — simulation-free CNF training; conditional flows between noise and data. [→](../../curriculum/02-generative-modeling/flow-matching.md)
19. **Rectified Flow** (frontier) — straight-line paths; reflow distillation; FLUX and SD3 foundation. [→](https://arxiv.org/abs/2209.03003)
20. **OT-CFM** (frontier) — optimal transport paths; better coupling quality; fewer steps to convergence.

## Key figures

- **Diederik Kingma** (Google) — VAEs, normalizing flows, diffusion
- **Ian Goodfellow** — GANs
- **Yang Song** (Stanford → OpenAI) — score matching, SDEs, consistency models
- **Yaron Lipman** (Meta AI) — flow matching, stochastic interpolants
- **Xinquan Liu** — Rectified Flow (FLUX, SD3)

## Essential reading sequence

1. [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — Kingma & Welling 2013 — VAE
2. [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — Goodfellow et al. 2014 — GANs
3. [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — Ho et al. 2020 — DDPM
4. [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — Song et al. 2020 — unified SDE view
5. [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — Song et al. 2020 — DDIM
6. [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — Lipman et al. 2022 — conditional flow matching
7. [Flow Straight and Fast: Rectified Flow](https://arxiv.org/abs/2209.03003) — Liu et al. 2022 — straight-line flows

## Current frontier anchors
> As of 2026-05-25

- **FLUX.1** — flow matching + diffusion transformer; current best image generation
- **AlphaFold 3** — diffusion-based joint structure prediction for proteins + nucleic acids + ligands
- **Voicebox / Voituras** — flow matching for audio generation
- **Flow Matching Guide and Code** ([arXiv:2412.06264](https://arxiv.org/abs/2412.06264)) — comprehensive 2024 tutorial by original authors

## What you'll know when done

1. Derive the ELBO and explain what the KL and reconstruction terms do
2. Explain the forward and reverse processes of DDPM from scratch, with the math
3. Implement a minimal diffusion model and sample from it
4. Explain why flow matching is faster to train and sample than classical diffusion
5. Distinguish the Rectified Flow (straight paths), OT-CFM (optimal transport paths), and stochastic interpolant approaches

## Branch points to other arcs

- **→ Language Models arc**: Discrete diffusion/flow matching for token generation
- **→ Scientific AI arc**: AlphaFold 3 uses diffusion; molecular generation; protein design
- **→ Robotics arc**: Diffusion Policy and π0 (flow matching) for robot action generation

## Where to go next

[Scientific AI arc →](../scientific-ai/index.md) — How generative models power AlphaFold and molecular design

[Language Models arc →](../language-models/index.md) — Discrete diffusion and generation for language
