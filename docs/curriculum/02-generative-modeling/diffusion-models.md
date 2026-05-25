---
title: Diffusion Models
track: 02-generative-modeling
tags: [diffusion, ddpm, score-matching, denoising, generative-models]
depth: all
prereqs: [variational-autoencoders, score-matching]
updated: 2026-05-25
has_mvb: true
---

# Diffusion Models
> **TL;DR:** Generative models that learn to reverse a gradual noising process — currently the dominant paradigm for high-quality image, audio, video, and molecular structure generation.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Fine-tune or run a diffusion model |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Understand why diffusion became dominant |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Derive the ELBO and the ε-prediction objective |
| Researcher / frontier | [Current SotA](#current-sota) → [What's happening now](#whats-happening-now) | Know where diffusion ends and flow matching begins |

---

## What it is

Diffusion models learn to generate data by reversing a destruction process. In the **forward process**, Gaussian noise is added to data across T steps until the signal is completely destroyed. A neural network is then trained to **reverse** this process: given a noisy sample at step t, predict the clean signal (or equivalently, the noise that was added). At inference, you start from pure noise and apply the learned denoiser iteratively to generate new samples.

What makes this work is a key mathematical insight: the forward process has a closed-form marginal. You don't need to run T steps to compute q(x_t | x_0) — you can jump directly. This makes training efficient: sample a random t, add the appropriate noise in one step, predict, compute loss.

The dominant paradigm that emerged from this — DDPM (Ho et al. 2020) — predicts the noise ε rather than the clean image directly, which leads to better training stability and sample quality.

## Why it matters at the frontier

Diffusion models power Stable Diffusion, DALL-E 3, Sora (video), AlphaFold 3 (protein structure), and Stable Audio. The framework is remarkably general: the same forward-reverse process applies to images, audio waveforms, video tokens, molecular coordinates, and robot trajectories. Wherever you need to generate high-quality samples from a complex distribution, diffusion is either the current solution or the departure point.

The reason diffusion beat GANs is training stability: no adversarial game, no mode collapse. The reason flow matching is now challenging diffusion is inference speed: diffusion needs 50-1000 steps; flow matching can do it in 8-20 with comparable quality.

## Core concepts

- **Forward process** — q(x_t | x_{t-1}): adds Gaussian noise at each step; parameterized by a noise schedule β_1, ..., β_T
- **Closed-form marginal** — q(x_t | x_0) = N(√ᾱ_t x_0, (1−ᾱ_t)I) where ᾱ_t = ∏βs; enables direct sampling without T steps
- **Reverse process** — p_θ(x_{t-1} | x_t): learned denoising; parameterized as UNet (pixel space) or transformer (latent space)
- **ε-prediction** — predict the noise added at step t; most common parameterization; equivalent to score matching
- **Noise schedule** — linear (Ho 2020), cosine (Nichol & Dhariwal 2021), or learned; controls information destruction rate
- **DDIM** — Denoising Diffusion Implicit Models; deterministic sampler; reduces sampling steps from 1000 → 50
- **Classifier-free guidance** — jointly train conditional and unconditional model; at inference, interpolate scores to trade diversity for quality; standard in all production systems
- **Latent diffusion** — run diffusion in a compressed VAE latent space rather than pixel space; 4-8× compute reduction (Stable Diffusion's core idea)

## Mathematical foundations

Forward process (closed-form marginal, the key identity):
$$q(x_t | x_0) = \mathcal{N}\!\left(x_t;\, \sqrt{\bar{\alpha}_t}\, x_0,\; (1 - \bar{\alpha}_t)I\right)$$

where $\bar{\alpha}_t = \prod_{s=1}^{t}(1 - \beta_s)$ and $\beta_t$ is the noise schedule.

Training objective (simplified ε-prediction loss):
$$\mathcal{L} = \mathbb{E}_{t \sim \mathcal{U}[1,T],\, x_0,\, \epsilon \sim \mathcal{N}(0,I)}\!\left[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\right]$$

Reverse process step (from ε-prediction to x_{t-1}):
$$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\!\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar{\alpha}_t}}\epsilon_\theta(x_t, t)\right) + \sigma_t z, \quad z \sim \mathcal{N}(0,I)$$

## Key algorithms / techniques

- **DDPM** (Ho et al. 2020) — the canonical formulation; 1000-step linear schedule; ε-prediction
- **DDIM** (Song et al. 2020) — deterministic ODE sampler over the same trained model; 10-50× fewer steps
- **Classifier-free guidance (CFG)** (Ho & Salimans 2021) — condition on class label or text; scale guidance strength w at inference
- **Latent Diffusion (LDM)** (Rombach et al. 2022) — diffuse in VAE latent space; the architecture of Stable Diffusion
- **Improved DDPM** (Nichol & Dhariwal 2021) — cosine schedule, learned variance, log-likelihood improvements
- **DiT** (Peebles & Xie 2023) — Diffusion Transformer; replaces UNet with a transformer; backbone for Sora and FLUX

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 | Ho et al. | Canonical DDPM — the foundation; read this first |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | 2020 | Song et al. | Unified SDE framework; shows DDPM and NCSN as special cases |
| [Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) | 2020 | Song et al. | Deterministic ODE sampling; essential for fast inference |
| [High-Resolution Image Synthesis with Latent Diffusion](https://arxiv.org/abs/2112.10752) | 2022 | Rombach et al. | Latent diffusion — Stable Diffusion's architecture |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| [Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) | 2015 | Sohl-Dickstein et al. — original diffusion formulation |
| [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) | 2019 | Song & Ermon — score matching at scale |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 | Ho et al. — made diffusion practical; dominant paradigm |
| [Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) | 2023 | Peebles & Xie — replaced UNet with transformer backbone |

## Current SotA
> *Updated: 2026-05-25*

**Image:** FLUX.1 (Black Forest Labs, 2024) uses flow matching + diffusion transformer on a Rectified Flow backbone — current best open-weight image generation. DALL-E 3 and Imagen 3 remain frontier closed systems.

**Video:** Sora (OpenAI, 2024) uses a DiT backbone with spatiotemporal attention. CogVideoX (Zhipu AI) is the open-weight frontier.

**Structure:** AlphaFold 3 (DeepMind, 2024) applies diffusion over atomic coordinates for all biomolecule types.

## What's happening now
> *Research · Engineering · Systems*

**Research:** Flow matching (Lipman et al. 2022, Liu et al. 2022) is now the primary challenger — it achieves comparable quality with 4-8× fewer inference steps by using straighter paths. Discrete diffusion (D3PM, MDLM) extends the paradigm to token-space for language generation.

**Engineering & Systems:** Production systems use latent diffusion + CFG + DDIM/DPM-Solver samplers. Consistency models (Song et al. 2023) distill diffusion to 1-2 step generation, enabling real-time use.

**Open problems:** How to choose the noise schedule optimally? Can diffusion compete with autoregressive models for language? How to handle variable-length outputs (audio, video) without fixed-resolution assumptions?

## In production
> *How top labs and companies have deployed this at scale*

- **Stability AI (Stable Diffusion 1.5 → 3 → FLUX):** Latent diffusion + CFG on a UNet backbone, then migrated to DiT. SD1.5 ran on consumer GPUs; FLUX.1 uses Rectified Flow on a 12B DiT. [stabilityai.com/research](https://stability.ai/research)
- **OpenAI (DALL-E 2 → 3):** DALL-E 2 uses CLIP image embeddings + a diffusion prior; DALL-E 3 integrates with ChatGPT for recaptioning and prompt improvement. [arxiv.org/abs/2204.06125](https://arxiv.org/abs/2204.06125)
- **Google DeepMind (Imagen, AlphaFold 3):** Imagen uses cascaded diffusion (64px → 256px → 1024px). AlphaFold 3 applies diffusion over atomic coordinates for all biomolecule types. [arxiv.org/abs/2205.11487](https://arxiv.org/abs/2205.11487)
- **Meta AI (Make-A-Video, Emu):** Video diffusion built on image diffusion by adding temporal attention layers — no video-text training data needed for the first model.
- **Runway ML:** Text-to-video production systems (Gen-2, Gen-3) power professional creative workflows; core is a video diffusion transformer.

## Minimum Valuable Build

A practical recipe: something real you can build with what's on this page.

**What you're building:** A class-conditional image generator on CIFAR-10, controllable by label — generate any of 10 classes on demand.

**Why this is valuable:** You'll understand every training loop element (noise schedule, ε-prediction, CFG), have a model you can compare against baselines, and have the foundation to fine-tune a larger model with your own data.

**Stack:**
- **Model:** [google/ddpm-cifar10-32](https://huggingface.co/google/ddpm-cifar10-32) — pretrained DDPM on CIFAR-10 (or train from scratch via `diffusers`)
- **Dataset:** [cifar10](https://huggingface.co/datasets/uoft-cs/cifar10) on HuggingFace Datasets
- **Framework:** [HuggingFace Diffusers](https://huggingface.co/docs/diffusers) + PyTorch

**The recipe:**

1. **Understand the pipeline:** Load `DDPMPipeline.from_pretrained("google/ddpm-cifar10-32")`, generate 16 samples, visualize — see what a trained diffusion model produces.
2. **Train your own from scratch:** Use `diffusers` `UNet2DModel` + `DDPMScheduler`. Train with ε-prediction loss on CIFAR-10. ~50 epochs gets recognizable images.
3. **Add class conditioning:** Modify the UNet to accept a class embedding. Add CFG: randomly drop labels during training; at inference, interpolate conditional and unconditional predictions with guidance scale w.
4. **Evaluate quality:** Compute FID score vs. the pretrained baseline. See how guidance scale w trades diversity for quality.

**Expected outcome:** A working conditional image generator you trained yourself, with quantitative FID evaluation and a grid of class-conditional samples.

**Stretch goals:**
- Swap CIFAR-10 for your own image dataset (any folder of images)
- Implement DDIM sampling on top of your trained model — compare 1000-step vs. 50-step quality
- Fine-tune [stabilityai/stable-diffusion-2-1](https://huggingface.co/stabilityai/stable-diffusion-2-1) on a custom domain using LoRA via Diffusers' `train_dreambooth_lora.py`

## Code & implementations

- [huggingface/diffusers](https://huggingface.co/docs/diffusers) — production diffusion library; DDPM, DDIM, latent diffusion, DiT
- [openai/consistency_models](https://github.com/openai/consistency_models) — official consistency model code

## Connected topics

- [[flow-matching]] — the emerging alternative; simpler training, fewer inference steps
- [[score-matching]] — theoretical foundation (Stein score, Langevin dynamics)
- [[variational-autoencoders]] — latent diffusion uses a VAE encoder
- [[consistency-models]] — single-step distillation of a trained diffusion model
- [[diffusion-transformer]] — DiT backbone replacing UNet in modern systems

## Further reading

- [What are Diffusion Models?](https://arxiv.org/abs/2208.11970) — Luo 2022; accessible survey
- [Improved Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2102.09571) — Nichol & Dhariwal 2021; cosine schedule + learned variance
- [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — Karras et al. 2022; systematic comparison of design choices
