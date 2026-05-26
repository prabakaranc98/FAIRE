---
title: Energy-Based Models
track: 02-generative-modeling
tags: [generative-modeling, energy-based-models, inference, optimization, sampling]
depth: foundational
prereqs: [variational-autoencoders, score-matching]
arc_refs: []
updated: 2025-05-14
has_mvb: true
---

# Energy-Based Models

> **TL;DR:** Energy-based models (EBMs) learn a scalar energy landscape where data points reside at low-energy minima, enabling generative modeling through optimization-based inference rather than iterative path-following.

---

## Who this page is for

| Persona | What you get | Jump to |
|---|---|---|
| Curious learner | Intuition on landscape-based generation | [§What it is](#what-it-is) |
| CS student / tinkerer | Langevin dynamics build on MNIST | [§MVB — CS student](#mvb-cs-student) |
| Applied engineer | Production framing + quantization tips | [§In production](#in-production), [§MVB — Applied engineer](#mvb-applied-engineer) |
| Applied researcher | Hypothesis-driven ablation build | [§What's happening now](#whats-happening-now), [§MVB — Applied researcher](#mvb-applied-researcher) |
| Theory student | Energy landscape derivations | [§Mathematical foundations](#mathematical-foundations) |
| Frontier researcher | Open problems in equilibrium matching | [§Open questions](#open-questions), [§MVB — Frontier researcher](#mvb-frontier-researcher) |
| PM / decision-maker | Why EBMs matter for static priors | [§Why it matters](#why-it-matters), [§Current SotA](#current-sota) |

---

## What it is

Modern generative models often rely on iterative processes, such as diffusion, which require hundreds of sequential steps to transform noise into data. This path-following approach is computationally expensive because it forces the model to simulate a trajectory through probability space. Energy-based models (EBMs) bypass this by learning a static, universal scalar field where the data distribution is defined by the energy landscape itself.

In this framework, the model assigns low energy values to regions of high data density and high energy to regions of low density. Instead of following a pre-defined path, generation becomes a single optimization problem: navigating this landscape to find the low-energy valleys. This shift from "simulating a path" to "navigating a landscape" allows EBMs to act as flexible, universal priors that can be applied to sampling, denoising, and solving inverse problems without needing to retrain for specific tasks.

The consequence of this design is that EBMs unify generative modeling with energy minimization. By learning the underlying potential function of the data, the model effectively captures the constraints of the data manifold. This provides a robust alternative to likelihood-based models, as EBMs do not require the explicit calculation of the partition function, which is often intractable in high-dimensional spaces.

## Why it matters

EBMs are central to the frontier of generative modeling because they provide a bridge between traditional probabilistic modeling and modern optimization-based inference. By learning a static energy landscape, researchers can perform inference using Langevin dynamics, which allows for the integration of external constraints directly into the sampling process. This is why labs are increasingly looking at EBMs to solve complex inverse problems where the goal is to generate data that satisfies specific physical or logical conditions.

The current tension in the field lies in the trade-off between training stability and sample quality. While diffusion models have dominated due to their stable training objectives, EBMs offer a more elegant theoretical framework for equilibrium-based generation. Recent breakthroughs in equilibrium matching suggest that we can achieve state-of-the-art performance by learning implicit energy landscapes, potentially eliminating the need for the long, iterative chains that currently bottleneck production-scale generative systems.

## Core concepts

- **Energy Function** — A scalar mapping \(E_\theta(x)\) that assigns lower values to data configurations that are more probable.
- **Partition Function** — The normalization constant \(Z(\theta)\) that ensures the energy-based distribution integrates to one.
- **Langevin Dynamics** — An MCMC sampling technique that uses the gradient of the energy function to navigate toward low-energy regions.
- **Contrastive Divergence** — A training objective that minimizes energy for real data while maximizing it for model-generated samples.
- **Boltzmann Distribution** — The probabilistic interpretation of energy, where \(p(x) \propto \exp(-E(x))\).

## Mathematical foundations

\[
p(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}
\]
where \(p(x)\) is the probability density of data \(x\), \(E_\theta(x)\) is the energy function parameterized by \(\theta\), and \(Z(\theta) = \int \exp(-E_\theta(x)) dx\) is the partition function. This equation defines the Boltzmann distribution, linking the scalar energy landscape to a valid probability distribution.

\[
x_{t+1} = x_t - \eta \nabla_x E_\theta(x_t) + \sqrt{2\eta} \epsilon_t
\]
where \(x_t\) is the sample at step \(t\), \(\eta\) is the step size, \(\nabla_x E_\theta\) is the gradient of the energy function, and \(\epsilon_t \sim \mathcal{N}(0, I)\) is Gaussian noise. This is Langevin Dynamics, the core mechanism for sampling from an EBM by navigating the energy landscape.

\[
\mathcal{L}(\theta) = \mathbb{E}_{x \sim p_{data}} [E_\theta(x)] + \mathbb{E}_{x \sim p_{model}} [E_\theta(x)]
\]
where \(p_{data}\) is the empirical data distribution and \(p_{model}\) is the model distribution. This is the contrastive divergence objective, used to push down energy on data and pull up energy on generated samples.

## Key algorithms / techniques

- **Langevin MCMC** (2006) — Uses gradient-based updates to sample from the energy landscape; essential for generating high-quality images from EBMs.
- **Energy Matching** (2025) — A modern approach that unifies flow matching with EBMs by learning a static potential that guides samples into equilibrium.
- **Equilibrium Matching** (2025) — Replaces time-conditional dynamics with an implicit energy landscape to achieve state-of-the-art generative performance.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| A Tutorial on Energy-Based Learning | 2006 | LeCun et al. | Foundational framework for energy-based architectures. |
| Energy Matching | 2025 | Arbel et al. | Bridges flow matching and EBMs for static potential learning. |
| Equilibrium Matching | 2025 | Arbel et al. | Demonstrates SOTA performance via implicit energy landscapes. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Implicit Generation and Modeling with EBMs | 2019 | Du & Mordatch | Scaled MCMC training for high-dimensional image generation. |
| Generalized Energy Based Models | 2020 | Arbel et al. | Kernelized Stein discrepancy for EBM training. |

## Current SotA

Equilibrium Matching achieves an FID of 1.90 on ImageNet 256x256 (2025), outperforming traditional diffusion-based baselines by replacing time-conditional dynamics with an implicit energy landscape.

## What's happening now

Research is currently focused on replacing the computationally expensive MCMC sampling during training with implicit equilibrium landscapes. Arbel et al. (2025) (https://arxiv.org/abs/2510.02300) showed that learning these landscapes directly allows for optimization-based sampling that is significantly faster than traditional Langevin dynamics.

From an engineering perspective, the challenge is implementing these energy landscapes in production environments. Because EBMs rely on gradient-based sampling, they require efficient automatic differentiation and high-precision kernels. Adobe’s implementation of generative video systems (https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/) highlights the necessity of FP8 quantization and TensorRT optimization for these types of gradient-heavy workloads.

The open problem remains the stability of the training objective. While contrastive divergence is the standard, it often suffers from mode collapse in high-dimensional spaces. Researchers are currently exploring kernelized Stein discrepancy as a more stable alternative to ensure the energy landscape covers the entire data manifold (Arbel et al., 2020, https://arxiv.org/abs/2003.05033).

## In production

- **Adobe** — Firefly Video Generation — Deployed using NVIDIA TensorRT with FP8 quantization on Hopper GPUs — [Source](https://developer.nvidia.com/blog/optimizing-transformer-based-diffusion-models-for-video-generation-with-nvidia-tensorrt/)
- **Amazon Ads** — Generative lifestyle product image generation — Fully managed pipeline on Amazon SageMaker — [Source](https://aws.amazon.com/blogs/machine-learning/learn-how-amazon-ads-created-a-generative-ai-powered-image-generation-capability-using-amazon-sagemaker/)

## Minimum Valuable Builds — by persona

### 1. For the curious learner (30 min · free tier)
**Build:** Visualize the energy landscape of a 2D Gaussian distribution.
**Artifact:** A Colab notebook plotting the energy surface and Langevin trajectories.
**Success:** Trajectories converge to the high-density regions of the Gaussian.
**Stack:** `matplotlib`, `numpy`, `pytorch`.

### 2. For the CS student / tinkerer (1 day · RTX 4070 / M-series)
**Build:** Train an EBM on MNIST using Langevin Dynamics.
**Artifact:** A checkpoint and a grid of generated digits.
**Success:** FID ≤ 20 on MNIST 32×32.
**Stack:** `sanganaka/Word_Segmentation_in_Sanskrit_Using_Energy_Based_Models` (architecture reference), `mnist` dataset.

### 3. For the applied / production engineer (1 week · A10 / L4 / cloud)
**Build:** Deploy an EBM-based sampler as a REST endpoint.
**Artifact:** A vLLM-style endpoint serving samples at p50 < 2s.
**Success:** Throughput of 10 samples/sec on A10.
**Stack:** `pytorch`, `fastapi`, `nvidia-tensorrt`.

### 4. For the applied researcher (3 days · A100)
**Build:** Ablate the effect of Langevin step size \(\eta\) on mode coverage.
**Artifact:** A comparison plot of FID vs. step size.
**Success:** Evidence that smaller steps improve mode coverage but increase compute time.
**Stack:** `pytorch`, `mnist`, `a100-80gb`.

### 5. For the theory student (1 day · CPU)
**Build:** Derive the gradient of the energy function for a simple RBM.
**Artifact:** A plot comparing theoretical energy gradients to numerical approximations.
**Success:** Residual error below 1e-4.
**Stack:** `numpy`, `scipy`.

### 6. For the frontier researcher (1 week+ · A100 cluster)
**Build:** Probe the stability of equilibrium matching in high-dimensional latent spaces.
**Artifact:** Evidence of training divergence or convergence in 1024-dim space.
**Success:** Falsification criterion: if training loss diverges within 10k steps, the objective is unstable.
**Stack:** `pytorch`, `a100-cluster`.

## Open questions

!!! researcher "For researchers"
    Can we design a universal, stable training objective for EBMs that eliminates the need for MCMC-based sampling during training while maintaining the ability to perform exact likelihood estimation in high-dimensional latent spaces?

!!! engineer "For engineers"
    How can we optimize the Langevin sampling loop using custom CUDA kernels to achieve sub-100ms latency for 256x256 image generation?

!!! open "Think about this"
    If the energy landscape is static, does the "path" taken by Langevin dynamics contain information about the data manifold that we are currently discarding?

## This concept appears in
- Arc step pages for this concept are being generated.

## Connected topics

- [Generative Adversarial Networks](./generative-adversarial-networks.md) — Both are frameworks for generative modeling that learn complex data distributions.
- [Bayesian Inference](../05-statistical-probabilistic-ml/bayesian-inference.md) — Energy-based models often represent unnormalized posterior distributions used in Bayesian inference.
- [Contrastive Learning](../03-representation-learning/contrastive-learning.md) — Contrastive learning is frequently used to train energy-based models by shaping energy landscapes.
- [Entropy](../15-ml-theory-foundations/entropy.md) — Energy-based models are fundamentally linked to maximum entropy principles in statistical mechanics.
- [Expectation-Maximization](../05-statistical-probabilistic-ml/em.md) — EM algorithms are often employed to estimate parameters in latent variable energy-based models.
- [Equivariant Networks](../12-physics-scientific-ai/equivariant-networks.md) — Energy-based models often incorporate symmetries to define physically meaningful energy functions.


## Further reading

- [A Tutorial on Energy-Based Learning](https://cs.nyu.edu/~sumit/publications/assets/ebmtutorial.pdf) — The foundational text for understanding energy-based architectures.
- [Implicit Generation and Modeling with EBMs](https://arxiv.org/abs/1903.08689) — A key paper for scaling MCMC-based training to high-dimensional images.
- [Generalized Energy Based Models](https://arxiv.org/abs/2003.05033) — Provides the kernelized framework for stable EBM training.