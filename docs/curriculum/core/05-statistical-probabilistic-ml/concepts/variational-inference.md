---
title: Variational Inference
slug: variational-inference
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [kingma, hoffman, jordan, blei]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bayesian-inference, probabilistic-graphical-models, optimization-basics]
tags: [bayesian, variational-inference, stochastic-optimization, uncertainty, reparameterization]
updated: 2026-02-12
has_mvb: true
---

# Variational Inference

Imagine you were dropped into a sprawling, crooked cave and tasked with measuring its volume. One way is to wander through every nook and cranny with a lantern (a Monte Carlo expedition), but that takes forever and yields noisy maps. Variational inference instead lets you inflate a mathematically tame balloon inside the cavern and watch how it squishes against the walls. By tweaking the balloon’s parameters until it fills the space as closely as possible, you replace the endless integration over posterior mass with a deterministic optimization problem solved by gradients. By the end of this page you will see exactly how that balloon is defined, why its fit is measured by the Evidence Lower Bound, where stochastic gradients allow it to scale, and how the build at the end makes the optimizer’s resistance visible on a simple Bayesian linear regression problem in PyTorch.

## The territory

Modern production systems often demand uncertainty estimates. A recommendation system needs to know when it is guessing on cold-start users before it applies an exploration policy; a medical model needs to report how confident it is before a clinician acts on its output. Bayesian inference formalizes this by computing the full posterior \(p(\theta \mid \mathcal{D})\) over latent model parameters \(\theta\) given data \(\mathcal{D}\), but the integral required to normalize \(p(\theta \mid \mathcal{D})\) grows combinatorially with model complexity. Variational inference reframes the problem as optimization: we introduce a parametrized family of densities \(q_\phi(\theta)\) that are easy to evaluate and differentiate, and then adjust \(\phi\) so that \(q_\phi(\theta)\) is “tight” against the true posterior. Conceptually, this is the balloon being squeezed against the cave—each gradient step pushes it to wrap more tightly around the high-probability regions instead of wandering through every passage.

This framing sits at the intersection of probabilistic graphical models, where the original integrals live, and gradient-based optimization, which gives us a practical lever. As Jordan et al. (1999) [An Introduction to Variational Methods for Graphical Models](https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf) first explained, variational methods take graphical model structure and turn inference into a structured optimization problem over tractable families. Blei et al. (2017) [Variational Inference: A Review for Statisticians](https://www.arxiv.org/pdf/1601.00670v4) later refreshed the statistics community by showing that Coordinate Ascent Variational Inference (CAVI) essentially performs block-wise optimization of the Evidence Lower Bound, while Hoffman et al. (2013) [Stochastic Variational Inference](https://arxiv.org/abs/1206.7051v3) extended the idea to streaming data via stochastic gradients. This page occupies the junction of those insights: we will see how the ELBO encodes the balloon’s fit, how gradients climb it, and how stochastic approximations keep the build feasible for large datasets. The mechanism is best understood by starting with the ELBO itself.

## How it works

The goal of variational inference is to approximate the posterior \(p(\theta \mid \mathcal{D})\) when direct computation is impossible, so we introduce a tractable candidate \(q_\phi(\theta)\) parameterized by \(\phi\). Instead of minimizing the KL divergence \(\text{KL}(q_\phi(\theta)\,\|\,p(\theta \mid \mathcal{D}))\) directly, which requires the intractable posterior, we maximize the Evidence Lower Bound (ELBO):
\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D}, \theta) - \log q_\phi(\theta)]
\]
where \(p(\mathcal{D}, \theta)\) is the joint likelihood and \(q_\phi(\theta)\) is the surrogate density. The ELBO equals the log marginal likelihood \(\log p(\mathcal{D})\) minus the KL divergence to the posterior, so maximizing \(\mathcal{L}(\phi)\) simultaneously approximates the posterior and tightens the lower bound.

This expression splits into two interpretable terms: the expected joint log-likelihood \(\mathbb{E}_{q_\phi}[\log p(\mathcal{D}, \theta)]\) rewards explaining the data, while the entropy term \(\mathbb{E}_{q_\phi}[-\log q_\phi(\theta)]\) prevents \(q_\phi\) from collapsing to a point mass. When \(q_\phi\) is a mean-field factorization \(q_\phi(\theta) = \prod_j q_{\phi_j}(\theta_j)\), each coordinate update can be derived analytically, leading to CAVI. Specifically, for a factor \(q_{\phi_j}(\theta_j)\), the optimal update is proportional to the exponentiated expectation of the complete conditional:
\[
\log q_{\phi_j}^\star(\theta_j) = \mathbb{E}_{q_{\phi_{-j}}}[\log p(\theta_j \mid \theta_{-j}, \mathcal{D})] + \text{const}
\]
where \(\phi_{-j}\) denotes all parameters except \(\phi_j\). This coordinate ascent is the original balloon-inflation: each update adjusts one shape parameter while assuming the rest fixed, iteratively molding \(q_\phi\) closer to the posterior.

However, exact computation of these expectations still requires summing over the entire dataset and the other latent variables, which becomes intractable in deep models. Hoffman et al. (2013) introduced Stochastic Variational Inference (SVI) to resolve this. SVI replaces full expectations with stochastic estimators computed on minibatches, using natural gradients when available to keep the updates numerically stable. The key realization is that the variational parameters \(\phi\) can be treated like weights in a neural network: the ELBO is an expectation, and its gradient can be estimated by sampling a minibatch \(\mathcal{D}_b\) and computing
\[
\nabla_\phi \mathcal{L}(\phi) \approx \frac{N}{|\mathcal{D}_b|} \mathbb{E}_{q_\phi(\theta)}[\nabla_\phi \log p(\mathcal{D}_b, \theta)] - \nabla_\phi \mathbb{E}_{q_\phi(\theta)}[\log q_\phi(\theta)]
\]
where \(N\) is the total dataset size. The gradient is unbiased, and with an appropriate learning-rate schedule the optimizer tracks the true ELBO gradient over time.

To backpropagate through stochastic latent variables, Kingma & Welling (2013) [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) proposed the reparameterization trick. For continuous latent variables \(\theta \sim q_\phi(\theta)\), we rewrite the sampling as \(\theta = g_\phi(\epsilon)\) where \(\epsilon \sim p(\epsilon)\) is a fixed noise variable and \(g_\phi\) is a deterministic function. The ELBO gradient becomes
\[
\nabla_\phi \mathcal{L}(\phi) = \mathbb{E}_{p(\epsilon)}[\nabla_\phi \log p(\mathcal{D}, g_\phi(\epsilon)) - \nabla_\phi \log q_\phi(g_\phi(\epsilon))]
\]
so we can apply automatic differentiation like in standard neural network training. This view reveals why variational inference can be implemented in PyTorch or JAX without symbolic derivations: the gradient flows through the reparameterized sample.

Selecting the variational family \(q_\phi\) determines the balloon’s flexibility. Mean-field families are fast but often miss posterior correlations, leading to underestimation of uncertainty. More expressive families—normalizing flows, rank-structured covariance, implicit distributions—allow \(q_\phi\) to bend around curved posterior manifolds, at the cost of higher gradient variance and more expensive evaluation. Practically, we balance this trade-off by starting with diagonal Gaussians for speed and later introducing learned transformations if the balloon refuses to cover all high-probability regions.

A worked example on Bayesian linear regression illustrates these pieces. Suppose \(y = X\theta + \epsilon\) with \(\epsilon \sim \mathcal{N}(0, \sigma^2 I)\) and a Gaussian prior \(\theta \sim \mathcal{N}(0, \tau^2 I)\). The ELBO becomes
\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}\left[-\frac{1}{2\sigma^2}\|y - X\theta\|^2 - \frac{1}{2\tau^2}\|\theta\|^2\right] + \mathbb{H}[q_\phi(\theta)]
\]
where \(\mathbb{H}[q_\phi(\theta)]\) is the entropy of the variational distribution. If \(q_\phi\) is Gaussian with mean \(\mu\) and variance \(\Sigma\), gradients flow through \(\mu\) and \(\Sigma\) straightforwardly once we reparameterize \(\theta = \mu + \Sigma^{1/2}\epsilon\). The optimizer inflates the balloon by pulling \(\mu\) toward the maximum a posteriori (MAP) direction while \(\Sigma\) expands to cover directions with high uncertainty, revealing which features are poorly identified. Implementing this with PyTorch and minibatches is the focus of the build at the end. Before that, we need to understand how modern systems estimate these gradients efficiently and incorporate the structure of graphical models when they exist.

## Where the field is now

The last few years have seen both theoretical advances in constructing richer variational families and engineering efforts that push VI into production scale. Recent research such as Wu et al. (2024) [arxiv:2404.09113](https://arxiv.org/abs/2404.09113) studies hierarchical normalizing flows that maintain tractable densities while encoding complex posterior dependencies, showing that bridging expressive families with structured variational inference recovers credit assignment in hierarchical Bayesian models. Kim et al. (2026) [arxiv:2602.05873](https://arxiv.org/abs/2602.05873) pushes this further by coupling graph-based inductive biases with implicit variational distributions, yielding tighter ELBOs on image segmentation tasks. On the production side, Stitch Fix’s recommendation system employs a scalable SVI implementation (Hoffman et al. 2013) that processes streaming user interactions by maintaining a variational posterior over item attractiveness and updating it with natural gradients on batches of new feedback. The engineering frontier is the combination of amortized inference and streaming updates: Meta’s Pyro team describes how automated guide generation plus SVI keeps the inference balloon adaptive while models continuously train on live event logs.

A small comparative study illustrates where these advances land. Traditional CAVI and mean-field VI remain baseline choices when the graphical structure is explicit and datasets fit in memory, but SVI becomes essential when hundreds of gigabytes stream in, and expressive guides (flows, implicit models) are now practical thanks to richer gradient estimators such as pathwise derivatives and doubly reparameterized gradients. This parallel development—stochastic gradients enabling production scale, richer families enabling better approximations—defines the current frontier. The next section explains where the balloon still resists inflation.

## What's still open

Can we automatically construct variational families that match the posterior’s conditional independence structure without resorting to hand-designed flow architectures? The current practice either assumes mean-field factorization or manually engineers coupling layers, so a general construction that reads the graphical structure and suggests tractable transforms remains elusive.

How can gradient variance be controlled when the variational guide is expressive? Adding flows or implicit distributions dramatically increases gradient noise, which in turn forces smaller learning rates and longer training. A systematic variance-reduction strategy tied to the guide’s architecture would make expressive VI practical for larger datasets.

Is there a principled way to trade off variational expressivity and computational budget in a data-dependent manner? Presently, practitioners tune this manually: simpler families for large datasets, richer ones when accuracy demands it. A method that evaluates dataset complexity and allocates “balloon degrees of freedom” accordingly would remove guesswork.

Can amortized inference generalize across datasets in streaming settings without degrading uncertainty calibration? The combination of amortization (training an inference network once) and SVI updates (fine-tuning on new minibatches) is promising, but the conditions under which it maintains posterior fidelity over time are not well understood.

## Where to read next

If you want the probabilistic foundation that explains why minimizing the ELBO is equivalent to minimizing a moment-matching divergence, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) shows the likelihood-free training perspective that unifies VI with diffusion models; the engineering counterpart is → [[flash-attention]] which explains how modern GPUs compute the same gradients used in VI efficiently. For those curious about the next inference paradigm beyond balloons and ELBOs, → [Flow matching](../../02-generative-modeling/concepts/flow-matching.md) generalizes the noising process to continuous paths and raises similar questions about variational families.

## Build it

Completing this build makes the ELBO and reparameterization trick tangible: you will see how stochastic gradients inflate the variational balloon and how its failure to stretch exposes calibration issues.

**What you're building:** A PyTorch implementation of Bayesian linear regression trained with stochastic variational inference on synthetic 1D data, producing credible intervals for the learned weights.

**Why this is valuable:** The build nudges the reader to implement the ELBO, sample through the reparameterization trick, and optimize with minibatch gradients, making it obvious when the variational guide collapses or underestimates uncertainty.

**Stack:**
- **Model:** Custom lightweight PyTorch linear model plus variational Gaussian guide — no public HF ID because the artifact is self-contained code.
- **Dataset:** `huggingface/datasets` synthetic regression generator (e.g., `synthetic_regression` with configurable noise) for reproducibility.
- **Framework:** PyTorch 2.1 with `torch.distributions` and `torch.optim.AdamW`.
- **Compute:** Free Colab T4 (approx. 16 GB VRAM) — training completes in ~15 minutes for 10k datapoints and 20 epochs.

**The recipe:**
1. `pip install torch datasets matplotlib` then import `torch`, `torch.distributions`, and the HF dataset loader; seed the RNG for reproducibility.
2. Load the synthetic dataset with 1 feature and noise `σ=0.1`, normalize the feature, and split into train/validation loaders with batch size 128.
3. Define the variational guide \(q_\phi(\theta)\) as a diagonal Gaussian with learnable mean `μ` and log-variance `ρ`; reparameterize samples with `ε ~ Normal(0,1)` and compute \(\theta = μ + \exp(ρ/2) * ε\); optimize the ELBO estimated on each minibatch using AdamW with LR=1e-3.
4. Evaluate by sampling 100 posterior weight draws to compute predictive intervals on the validation set; report the average width and whether the true weights fall within the 95% credible interval.
5. The artifact is a trained PyTorch module that outputs both point predictions and credibility intervals and logs the ELBO curve.

**Expected outcome:** A reproducible Colab notebook that trains SVI on a toy regression, visualizes credible intervals, and exports the checkpoint to share as a small demonstration of Bayesian uncertainty.

- **CS student:** Keep everything on Colab but reduce dataset size to 2k points and batch size 64; compare ELBOs for different learning rates to see how gradient noise affects the balloon.
- **Applied engineer:** Quantize the trained guide to float16, export it via TorchScript, and serve it behind a simple Flask API that responds within 120 ms on an A10 GPU while returning posterior mean ± SD.
- **Applied researcher:** Hypothesize that increasing guide expressivity (adding a second affine flow) should narrow the ELBO gap; train both the diagonal and flow guides for the same budget and report the ELBO and calibration error.
- **Frontier researcher:** Probe the open question of amortized inference drift by adding streaming minibatches with shifting noise levels, then test whether continuing SVI updates keeps calibration within the original 95% interval; the falsifier criterion is whether the credibility intervals widen beyond 20% of their initial width.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*