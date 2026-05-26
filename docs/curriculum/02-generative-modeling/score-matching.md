---
title: Score Matching
track: 02-generative-modeling
tags: [score-matching, generative-models, diffusion, energy-based-models, density-estimation]
depth: foundations
prereqs: [probability-theory, stochastic-calculus, energy-based-models]
updated: 2025-01-15
has_mvb: true
---

# Score Matching
> **TL;DR:** Score matching learns the gradient of a log-probability density instead of the density itself, sidestepping the intractable normalizing constant and providing the mathematical foundation for modern diffusion models.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms-techniques) → [MVB](#minimum-valuable-build) | Implement denoising score matching on a toy density |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Understand why diffusion models work |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Derive the integration-by-parts trick |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers-test-of-time) | Find open questions on error propagation |

---

## What it is

Suppose someone hands you an unnormalized model of natural images — a neural network \(f_\theta(x)\) that assigns a scalar "energy" to every \(1024 \times 1024\) RGB image. To turn this into a proper probability distribution you would need to compute \(Z_\theta = \int \exp(-f_\theta(x))\, dx\), the partition function, integrated over the entire \(\mathbb{R}^{3{,}145{,}728}\) pixel space. Even if you could evaluate \(f_\theta\) at a trillion points per second on every GPU ever manufactured, this integral would not finish before the heat death of the universe. Maximum likelihood training is therefore not just slow on high-dimensional unnormalized models — it is computationally impossible.

Score matching is the trick that escapes this trap. Instead of learning the density \(p(x)\), you learn its **score function**: the gradient of the log-density with respect to the input, \(s(x) = \nabla_x \log p(x)\). The crucial observation is that this gradient annihilates the partition function entirely, because \(\log p_\theta(x) = -f_\theta(x) - \log Z_\theta\) and \(Z_\theta\) does not depend on \(x\), so \(\nabla_x \log Z_\theta = 0\). A global integration problem has been converted into a local differentiation problem — and local differentiation is exactly what neural networks and autograd are built for.

The remaining difficulty is that we do not know the true score \(\nabla_x \log p_{\text{data}}(x)\) either; we only have samples. Hyvärinen's 2005 paper resolved this with an integration-by-parts manoeuvre that turns the intractable score-matching loss into an expectation over the trace of the model's Jacobian — quantities you can estimate from data alone. Vincent's 2011 paper then showed that if you perturb the data with Gaussian noise, the score-matching objective collapses to a clean denoising regression, which is the form used in every modern diffusion model.

## Why it matters at the frontier

Score matching is not a competitor to diffusion models — it is their mathematical engine. Every score-based generative model (SGM), every continuous-time SDE-based diffusion model, and every flow-matching variant ultimately learns a score field or a closely related vector field. When Song & Ermon (2019) introduced Noise Conditional Score Networks, they fixed score matching's notorious failure on data lying near low-dimensional manifolds by adding noise at multiple scales, and that fix is precisely what made DDPM, Stable Diffusion, and the entire generative imaging stack possible. Understanding score matching makes the noise schedule in DDPM precise rather than heuristic, and it makes the choice of the reverse SDE solver a question of numerical analysis rather than empirical tuning.

The open theoretical problems remain active. The most consequential question — and the one labs like Stability AI, NVIDIA, and Google DeepMind quietly run into when scaling — is how score estimation errors in low-density regions propagate through the reverse-time SDE to affect final sample quality. Existing bounds (e.g. Chen et al., 2023; Lee et al., 2023) require global Lipschitz constants on the score that are vacuously large for real image distributions, so they predict failure where models in fact succeed. Tightening these bounds without unrealistic regularity assumptions is the central theoretical question gating principled improvements to sampler design, guidance scales, and consistency-model training.

## Core concepts

- **Score function** — the gradient of the log-density with respect to the input, \(s(x) = \nabla_x \log p(x)\), which points in the direction of steepest density increase and is invariant to the normalizing constant.
- **Partition function** — the normalizing integral \(Z_\theta = \int \exp(-f_\theta(x))\, dx\) that makes an unnormalized energy model into a probability distribution; intractable in high dimensions and the central obstacle that score matching circumvents.
- **Explicit score matching (ESM)** — the naive objective \(\tfrac{1}{2}\,\mathbb{E}_{p_{\text{data}}}\!\left[\lVert s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\rVert^2\right]\), which is unusable because the data score is unknown.
- **Implicit score matching (ISM)** — Hyvärinen's reformulation that, via integration by parts, replaces the unknown data score with the trace of the model Jacobian plus the squared model score norm.
- **Denoising score matching (DSM)** — Vincent's variant: perturb data with Gaussian noise \(\tilde{x} = x + \sigma\epsilon\) and regress the model score onto \(-\epsilon/\sigma\), the (tractable) score of the perturbation kernel.
- **Sliced score matching** — Song et al.'s trick to avoid the \(O(d)\) cost of computing the full Jacobian trace by projecting onto random unit vectors, restoring linear-time training.
- **Noise-conditional score network (NCSN)** — a score model conditioned on noise level \(\sigma_t\), trained jointly across a geometric ladder of noise scales to cover both low- and high-density regions of the data manifold.
- **Langevin dynamics** — the sampling procedure \(x_{k+1} = x_k + \tfrac{\eta}{2} s_\theta(x_k) + \sqrt{\eta}\,z_k\) that uses a learned score field to produce samples from the implied density.

---

## Mathematical foundations

**The unnormalized model and the partition function problem.** For an energy-based model parametrized by a neural network,

\[
p_\theta(x) = \frac{\exp(-f_\theta(x))}{Z_\theta}, \qquad Z_\theta = \int \exp(-f_\theta(x))\, dx
\]

where \(x \in \mathbb{R}^d\) is the data point, \(f_\theta : \mathbb{R}^d \to \mathbb{R}\) is the learned energy function with parameters \(\theta\), and \(Z_\theta\) is the partition function — a scalar that depends on \(\theta\) but not on \(x\), and whose evaluation requires integrating over all of \(\mathbb{R}^d\).

**The score annihilates \(Z\).** Taking the gradient of the log-density with respect to the input,

\[
s_\theta(x) := \nabla_x \log p_\theta(x) = -\nabla_x f_\theta(x) - \nabla_x \log Z_\theta = -\nabla_x f_\theta(x)
\]

where \(s_\theta(x) \in \mathbb{R}^d\) is the model score (a vector field over input space), and the term \(\nabla_x \log Z_\theta = 0\) vanishes because \(Z_\theta\) has no dependence on \(x\). This is the core trick: a quantity we cannot compute has been removed from the problem entirely by differentiation.

**Explicit score matching (ESM).** The natural — but useless — objective is the squared distance between model and data scores,

\[
J_{\text{ESM}}(\theta) = \tfrac{1}{2}\, \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\lVert s_\theta(x) - \nabla_x \log p_{\text{data}}(x)\rVert_2^2\right]
\]

where \(p_{\text{data}}\) is the unknown true data distribution and \(\nabla_x \log p_{\text{data}}(x)\) is its score — which we cannot evaluate from samples.

**Hyvärinen's integration-by-parts identity (ISM).** Expanding the square and applying integration by parts under the regularity conditions \(p_{\text{data}}(x)\, s_\theta(x) \to 0\) as \(\lVert x\rVert \to \infty\), the cross term simplifies and the objective becomes, up to a constant independent of \(\theta\),

\[
J_{\text{ISM}}(\theta) = \mathbb{E}_{x \sim p_{\text{data}}}\!\left[\tfrac{1}{2}\lVert s_\theta(x)\rVert_2^2 + \operatorname{tr}\!\left(\nabla_x s_\theta(x)\right)\right]
\]

where \(\nabla_x s_\theta(x) \in \mathbb{R}^{d \times d}\) is the Jacobian of the score (equivalently the Hessian of \(\log p_\theta\)), and \(\operatorname{tr}(\cdot)\) is its trace — the sum of second partial derivatives \(\sum_{i=1}^d \partial s_\theta^{(i)}/\partial x^{(i)}\). The unknown \(\nabla_x \log p_{\text{data}}\) no longer appears; everything is computable from \(\theta\) and samples.

**The Jacobian trace bottleneck.** Computing \(\operatorname{tr}(\nabla_x s_\theta(x))\) exactly requires \(d\) backward passes per sample — prohibitive when \(d \sim 10^6\) for images. Sliced score matching (Song et al., 2019) replaces this with the Hutchinson estimator,

\[
\operatorname{tr}(\nabla_x s_\theta(x)) \approx \mathbb{E}_{v \sim \mathcal{N}(0,I)}\!\left[v^\top \nabla_x s_\theta(x)\, v\right]
\]

where \(v \in \mathbb{R}^d\) is a random projection direction; the inner Jacobian-vector product \(\nabla_x s_\theta(x)\, v\) is one backward pass, reducing per-sample cost to \(O(1)\) in \(d\).

**Vincent's denoising equivalence (DSM).** Define the perturbed density \(q_\sigma(\tilde{x}\mid x) = \mathcal{N}(\tilde{x};\, x,\, \sigma^2 I)\) and the marginal \(q_\sigma(\tilde{x}) = \int q_\sigma(\tilde{x}\mid x)\, p_{\text{data}}(x)\, dx\). Vincent (2011) showed that matching the score of the *perturbed* density is equivalent, up to a \(\theta\)-independent constant, to a tractable denoising regression:

\[
J_{\text{DSM}}(\theta) = \tfrac{1}{2}\, \mathbb{E}_{x \sim p_{\text{data}},\, \tilde{x} \sim q_\sigma(\cdot \mid x)}\!\left[\left\lVert s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x} \mid x)\right\rVert_2^2\right]
\]

where the conditional score \(\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}\mid x) = -(\tilde{x} - x)/\sigma^2 = -\epsilon/\sigma\) for \(\tilde{x} = x + \sigma\epsilon\), \(\epsilon \sim \mathcal{N}(0,I)\). The model is regressing onto the (scaled) noise direction — which is exactly the \(\epsilon\)-prediction loss in DDPM.

**Langevin sampling.** Once \(s_\theta \approx \nabla_x \log p_{\text{data}}\), samples are drawn via discretized Langevin dynamics,

\[
x_{k+1} = x_k + \tfrac{\eta}{2}\, s_\theta(x_k) + \sqrt{\eta}\, z_k, \qquad z_k \sim \mathcal{N}(0, I)
\]

where \(\eta > 0\) is the step size and \(z_k\) is fresh Gaussian noise; as \(\eta \to 0\) and \(k \to \infty\) the chain converges to the stationary distribution \(p_\theta\).

---

## Key algorithms / techniques

- **Implicit Score Matching (ISM)** — Hyvärinen's original estimator using the trace of the score Jacobian; mathematically clean but \(O(d)\) per sample, so only practical for \(d \lesssim 100\).
- **Denoising Score Matching (DSM)** — Vincent's noise-perturbation variant that replaces the Jacobian trace with a simple denoising MSE; the workhorse of every modern diffusion model.
- **Sliced Score Matching (SSM)** — Hutchinson-trace approximation that restores \(O(1)\) training cost while preserving the ISM form, useful when noise perturbation is undesirable (e.g. discrete data).
- **Noise Conditional Score Networks (NCSN)** — train a single network \(s_\theta(x, \sigma)\) on a geometric ladder of noise scales \(\sigma_1 > \cdots > \sigma_L\), fixing the low-density-region failure of single-scale score matching.
- **Annealed Langevin Dynamics** — sample by running Langevin steps at decreasing \(\sigma\), using the previous scale's output as the next scale's initialization; the discrete predecessor to continuous-time SDE sampling.
- **Score SDE** — Song et al. (2021) reframed score matching as learning the drift of a reverse-time SDE, unifying NCSN and DDPM under one continuous-time framework and enabling probability-flow ODE sampling.

---

## Essential reading

| Paper | Authors, Year | Why it matters |
|---|---|---|
| [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/old.pdf) | Hyvärinen, 2005 | The original integration-by-parts derivation that makes the entire field possible. |
| [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) | Vincent, 2011 | Proves DSM equivalence; collapses score matching into a tractable regression and unlocks high-dimensional scaling. |
| [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) | Song & Ermon, 2019 | Introduces NCSN and annealed Langevin sampling — the direct conceptual precursor to DDPM and Stable Diffusion. |
| [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) | Song et al., 2021 | Unifies discrete and continuous formulations and introduces probability-flow ODE sampling. |

---

## Seminal papers & test-of-time

| Paper | Authors, Year | Lasting contribution |
|---|---|---|
| [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/old.pdf) | Hyvärinen, 2005 | Defines the field; the ISM identity is still cited verbatim in every modern derivation. |
| [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) | Vincent, 2011 | DSM is the loss function inside every production diffusion model today. |
| [Sliced Score Matching: A Scalable Approach to Density and Score Estimation](https://arxiv.org/abs/1905.07088) | Song et al., 2019 | Made ISM scalable; still the method of choice for energy-based models on discrete or non-smooth data. |
| [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) | Song & Ermon, 2019 | NCSN diagnosed and solved the manifold-hypothesis failure mode of single-scale score matching. |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | Song et al., 2021 | The continuous-time SDE framework that unified DDPM, NCSN, and probability-flow ODEs. |

---

## Current SotA

Score matching itself is a training objective rather than a benchmarkable system, but its direct descendants set the state of the art on every major generative-imaging benchmark. NVIDIA's EDM2 (Karras et al., 2024) reaches **FID 1.81 on ImageNet-512** using a refined DSM objective with a principled noise schedule derived from the score-SDE framework, while Stable Diffusion 3 (Esser et al., 2024) trains a rectified-flow variant of the score objective at the 8B-parameter scale and dominates human-preference benchmarks for text-to-image generation. On molecular conformer generation, score-based models such as Torsional Diffusion (Jing et al., 2022) and follow-ups now produce conformer ensembles within chemical accuracy of DFT calculations, with score matching providing the underlying gradient field over \(SE(3)\)-equivariant manifolds.

---

## What's happening now

**Research.** The frontier of pure score-matching theory has shifted toward error propagation. Chen et al. (2023, "Sampling is as easy as learning the score") gave the first polynomial-time convergence guarantees for score-based samplers under bounded second-moment assumptions, but their bounds require a global Lipschitz constant on the score that is vacuously large for real image distributions. Subsequent work — Benton et al. (2024), Conforti et al. (2024) — has been chipping away at these assumptions, replacing global Lipschitzness with local or moment-based conditions, but the gap between theoretical bounds and empirical sample quality remains the central open question. A parallel research thread is consistency models (Song et al., 2023) and flow matching (Lipman et al., 2023), both of which generalize the score-matching objective to learn either the integrated trajectory directly or an arbitrary interpolant vector field, dramatically reducing sampling steps.

**Engineering & Systems.** The DSM loss is now the default training objective in every major open-source diffusion library — `diffusers` from Hugging Face, NVIDIA's NeMo, and Meta's PyTorch ecosystem all expose it as the primary entry point. Practical engineering effort has moved to noise-schedule design (EDM's \(\sigma\)-parametrization, SD3's logit-normal sampling), preconditioning (so the network always sees inputs of unit variance regardless of noise level), and mixed-precision Jacobian-vector products for sliced-score variants used in scientific applications. On the sampling side, DPM-Solver++ (Lu et al., 2023) and the recent UniPC framework reduce score-SDE sampling to 10–20 function evaluations while preserving the DSM-trained score network unchanged.

**Open problems.** The deepest unresolved question is the one stated in this page's framing: how do score estimation errors in low-density regions of the data manifold propagate through the reverse-time SDE to affect final sample quality, *without* assuming global Lipschitz continuity of the score network? Related open problems include: (i) whether sliced score matching's variance can be reduced enough to make ISM-style training competitive with DSM on images, sidestepping the noise-perturbation bias; (ii) how to do score matching on discrete or mixed discrete-continuous data without resorting to dequantization; and (iii) whether there exists a score-matching objective for which the optimal \(s_\theta\) is provably the score of a tractable, sampleable density rather than only an approximation.

---

## In production

- **Stability AI — Stable Diffusion 3 / 3.5** — trained with a rectified-flow score-matching objective at the 8B-parameter scale; powers the public Stable Diffusion API and serves billions of generations across DreamStudio and downstream integrations. [Stability AI research post](https://stability.ai/news/stable-diffusion-3-research-paper).
- **NVIDIA — EDM2 and the Picasso platform** — the EDM family of score-based models (Karras et al., 2022, 2024) is the reference implementation used inside NVIDIA's enterprise generative-imaging stack and underlies the Picasso foundry service; EDM2-XXL trains on 512×512 ImageNet to FID 1.81 with the DSM objective. [NVIDIA research](https://research.nvidia.com/labs/toronto-ai/EDM2/).
- **Google DeepMind — Imagen 3** — uses a score-based diffusion backbone trained with denoising score matching across a learned noise schedule; deployed in Gemini and the Vertex AI image-generation endpoints handling production traffic across Google Cloud. [Imagen 3 technical report](https://arxiv.org/abs/2408.07009).
- **OpenAI — DALL·E 3 and Sora** — both are score-based diffusion systems (transformer-parametrized score networks in Sora's case) trained with DSM-style objectives; Sora generates minute-long 1080p video clips and is integrated into ChatGPT for paying subscribers. [Sora technical report](https://openai.com/research/video-generation-models-as-world-simulators).

## Minimum Valuable Build

**Goal.** Implement Denoising Score Matching (DSM) from scratch and learn the score field of a 2D Gaussian-mixture "double-well" target. By the end you will have a small MLP whose gradient field, when plotted as a quiver over \([-4,4]^2\), points toward the two mixture modes — and you will have done it by minimizing a single, one-line loss that never touches a partition function.

**Compute.** Free Colab T4, or any CPU. End-to-end runtime: ~90 seconds.

**Stack.** PyTorch ≥ 2.0, NumPy, Matplotlib. No external model weights, no dataset download.

**The target distribution.** A 2-component isotropic Gaussian mixture:
\[
p_{\text{data}}(x) = \tfrac{1}{2}\mathcal{N}(x;\mu_1,\sigma^2 I) + \tfrac{1}{2}\mathcal{N}(x;\mu_2,\sigma^2 I)
\]
with \(\mu_1 = (-2, 0)\), \(\mu_2 = (2, 0)\), \(\sigma = 0.5\).

**The DSM objective.** For a noise level \(\sigma_n\), draw \(x \sim p_{\text{data}}\) and \(\tilde{x} = x + \sigma_n \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\). Train \(s_\theta(\tilde{x})\) to minimize
\[
\mathcal{L}(\theta) = \mathbb{E}_{x, \epsilon}\left[\left\| s_\theta(\tilde{x}) + \frac{\epsilon}{\sigma_n} \right\|^2\right]
\]
where \(s_\theta : \mathbb{R}^2 \to \mathbb{R}^2\) is our MLP, and the target \(-\epsilon/\sigma_n\) is the exact closed-form score of the Gaussian-perturbed conditional \(p(\tilde{x} \mid x)\). No partition function appears anywhere.

**Recipe.**

1. **Build the sampler.** Write `sample_data(n)` that draws \(n\) points by flipping a fair coin between the two modes and adding \(\mathcal{N}(0, 0.25 I)\) noise.
2. **Build the network.** A 4-layer MLP: `Linear(2,128) → SiLU → Linear(128,128) → SiLU → Linear(128,128) → SiLU → Linear(128,2)`. No normalization layers.
3. **Set the noise level.** Fix \(\sigma_n = 0.1\) for a first pass. (Stretch goal below: condition on \(\sigma_n\).)
4. **Training loop.** For 5,000 steps: draw a batch of 512 clean points \(x\), sample \(\epsilon\), form \(\tilde{x} = x + \sigma_n \epsilon\), compute the loss above, step with Adam at lr \(= 10^{-3}\).
5. **Visualize the learned score.** Build a \(30 \times 30\) grid on \([-4, 4]^2\), evaluate \(s_\theta\) at every grid point, and plot the resulting vector field with `matplotlib.pyplot.quiver`. Overlay 1,000 samples from `sample_data` as a scatter.
6. **Sanity check against analytic score.** For the same grid, compute the closed-form score \(\nabla_x \log p_{\text{data}}(x)\) analytically (a sum of two Gaussian gradients weighted by posterior responsibility) and plot it side-by-side. The two fields should be visually indistinguishable in the mode regions.
7. **Sample from the learned score via Langevin dynamics.** Starting from \(x_0 \sim \mathcal{N}(0, I)\), iterate \(x_{k+1} = x_k + \tfrac{\eta}{2} s_\theta(x_k) + \sqrt{\eta}\, z_k\) with \(z_k \sim \mathcal{N}(0,I)\), \(\eta = 0.01\), for 1,000 steps. Plot the resulting samples — they should concentrate on the two modes.

**Expected outcome.** Two side-by-side quiver plots that are nearly identical, plus a scatter of Langevin samples that visibly bimodal. Loss should drop from ~100 to under 1.0 within 2,000 steps. Total artifact: one `.ipynb`, one PNG of the score-field comparison, one PNG of the Langevin samples.

**Stretch goals.**

- **Multi-noise (NCSN-style) training.** Sample \(\sigma_n\) from a geometric schedule \(\{1.0, 0.6, 0.36, \ldots, 0.01\}\), condition the network on \(\sigma_n\) via a sinusoidal embedding concatenated to \(\tilde{x}\), and weight the loss by \(\sigma_n^2\). This reproduces Song & Ermon (2019) at toy scale and fixes mixing between the two modes in Langevin sampling.
- **Annealed Langevin dynamics.** With the multi-noise model, run Langevin at decreasing \(\sigma_n\) from large to small. Compare sample quality (mode coverage) to fixed-\(\sigma_n\) Langevin.
- **Sliced score matching.** Replace DSM with sliced score matching (Song et al., 2019) using a single random projection — no noise injection needed. Verify the learned field still matches.
- **Probability-flow ODE.** Replace stochastic Langevin with the deterministic probability-flow ODE \(\dot{x} = -\tfrac{1}{2}\sigma_n^2 s_\theta(x)\) using `torchdiffeq`. Observe that you get the same marginal distribution but deterministic trajectories.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations

| Repository | What it provides |
|---|---|
| [yang-song/score_sde_pytorch](https://github.com/yang-song/score_sde_pytorch) | Reference implementation of Song et al. (2021), "Score-Based Generative Modeling through SDEs." Includes VE, VP, and sub-VP SDEs with both Langevin and probability-flow ODE samplers. |
| [ermongroup/ncsnv2](https://github.com/ermongroup/ncsnv2) | Official NCSNv2 (Song & Ermon, 2020) — noise-conditional score networks at CIFAR-10 / CelebA scale. |
| [NVlabs/edm](https://github.com/NVlabs/edm) and [NVlabs/edm2](https://github.com/NVlabs/edm2) | NVIDIA's EDM and EDM2 — the cleanest modern score-matching training recipe; the EDM paper isolates the design choices (preconditioning, loss weighting, noise schedule) that matter. |
| [openai/consistency_models](https://github.com/openai/consistency_models) | Consistency models — distill a score-based teacher into a one-step student; directly built on the DSM objective. |
| [huggingface/diffusers](https://github.com/huggingface/diffusers) | Production-grade implementations of score-based and DDPM-style schedulers (`ScoreSdeVeScheduler`, `EulerDiscreteScheduler`, etc.) usable with any pretrained checkpoint. |

## What comes next

Score matching is the probabilistic substrate; everything downstream is either a discretization of it, a parametrization of it, or an alternative to it.

- Denoising Diffusion Probabilistic Models — DDPM is the discrete-time training procedure that re-derives DSM under a fixed forward Markov chain; the \(\epsilon\)-prediction objective is exactly DSM with a specific noise-level weighting.
- Score Based Sde — the continuous-time generalization where the discrete noise schedule becomes a stochastic differential equation, and sampling becomes either reverse-time SDE integration or probability-flow ODE solving.
- Langevin Dynamics — the MCMC sampler that turns a learned score field into actual samples; the bridge from "we know \(\nabla_x \log p\)" to "we can draw \(x \sim p\)."
- [Flow Matching](./flow-matching.md) — an alternative regression target (velocity fields of probability paths) that subsumes score matching as a special case and currently dominates state-of-the-art image and video generation.
- Consistency Models — distillation of a score-based teacher into a one- or few-step student, trading the elegant SDE structure for inference speed.

## Connected topics

- Energy Based Models — the family of unnormalized density models that motivated score matching in the first place; Hyvärinen's 2005 paper was written explicitly to fit EBMs without computing \(Z(\theta)\).
- Denoising Autoencoders — Vincent (2011) proved DAEs trained with Gaussian noise are implicitly performing score matching on the perturbed density, retroactively explaining why DAEs learned useful representations.
- Stein Discrepancy — a kernel-based generalization that uses the score function to test goodness-of-fit; shares the integration-by-parts trick that makes score matching tractable.
- Normalizing Flows — the alternative route to tractable likelihoods; flows compute \(\log p(x)\) exactly via change-of-variables, where score-based models compute only \(\nabla_x \log p(x)\) but scale better.
- Fisher Divergence — the divergence score matching actually minimizes; understanding why Fisher divergence is well-behaved where KL is not is the theoretical core of why score matching works.

## Further reading

- Hyvärinen, A. (2005). [Estimation of Non-Normalized Statistical Models by Score Matching](https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/old.pdf) — the original paper; read sections 2–3 for the integration-by-parts derivation that makes the entire field possible.
- Vincent, P. (2011). [A Connection Between Score Matching and Denoising Autoencoders](https://www.iro.umontreal.ca/~vincentp/Publications/smdae_techreport.pdf) — the four-page proof that DSM is exact for Gaussian-perturbed densities; the practical foundation of every modern diffusion model.
- Song, Y. & Ermon, S. (2019). [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600) — introduces noise-conditional score networks and explains the manifold-hypothesis failure mode of single-noise score matching.
- Song, Y., Sohl-Dickstein, J., Kingma, D. P., Kumar, A., Ermon, S., & Poole, B. (2021). [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) — the unifying SDE framework that ties score matching, DDPM, and probability-flow ODEs into one mathematical picture.
- Karras, T., Aittala, M., Aila, T., & Laine, S. (2022). [Elucidating the Design Space of Diffusion-Based Generative Models](https://arxiv.org/abs/2206.00364) — the EDM paper; an empirical and theoretical clean-up that identifies which design choices in score-based training actually matter.
- Lilian Weng, [Generative Modeling by Estimating Gradients of the Data Distribution](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) (lil'log, 2021) — for an intuitive walkthrough of how DSM connects to the broader diffusion story; useful as a second pass after the primary papers.
- Chen, S., Chewi, S., Li, J., Li, Y., Salim, A., & Zhang, A. R. (2023). [Sampling is as easy as learning the score](https://arxiv.org/abs/2209.11215) — the current best convergence bounds for score-based samplers and the natural entry point for the open problem on error propagation in low-density regions.