---
title: Variational Inference
slug: variational-inference
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [kingma, hoffman, jordan, neal]
feeds_de_pillar: []
mvb_personas: [applied-ai-ml-engineer, research-engineer, applied-researcher]
prereqs: [bayesian-inference, probabilistic-graphical-models, optimization-basics]
tags: [bayesian, variational-inference, stochastic-optimization, bayesian-neural-networks, reparameterization, uncertainty]
updated: 2026-02-12
has_mvb: true
---

# Variational Inference

Every time an autonomous system or a clinician issues a probability statement—“this patient has a 2% risk of complication,” “that vehicle will overtake with 95% confidence”—they are implicitly working with a posterior distribution that could easily be intractable to compute exactly. Direct Monte Carlo integration over hundreds of latent variables would demand so much compute that the system would time out before it completed the prediction. Variational inference answers this practical bottleneck: rather than sampling through the entire posterior, it posits a flexible but tractable surrogate distribution and optimizes its parameters like tuning the shape of a domed cover until it conforms to the unknown landscape beneath. This page walks through how that optimization—the Evidence Lower Bound, stochastic updates, and the reparameterization trick—turns approximate inference into a scalable software primitive, how it performs in practice on a small benchmark, and how the contemporary arc of work still pushes that dome to fit ever more complex posterior geometries.

## The territory

The central question is classical Bayesian inference: after observing data \(\mathcal{D}\), how can one compute the posterior \(p(\theta\mid\mathcal{D}) = \frac{p(\mathcal{D},\theta)}{\int p(\mathcal{D},\theta)\,d\theta}\) when the marginal likelihood \(p(\mathcal{D})\) is intractable to integrate over \(\theta\)? Variational inference (VI) reframes this problem by introducing a parameterized family \(q_\phi(\theta)\) and seeking the member whose density is closest to the true posterior. Jordan et al. (1999) [https://people.eecs.berkeley.edu/~jordan/papers/variational-intro.pdf] showed that maximizing the Evidence Lower Bound (ELBO) is equivalent to minimizing the Kullback-Leibler divergence \(\text{KL}(q_\phi(\theta)\parallel p(\theta\mid\mathcal{D}))\), turning a difficult integral into a deterministic optimizable objective. That optimization perspective underlies mean-field coordinate ascent, amortized inference, and the stochastic gradients that make VI practical on modern datasets.

## How it works

The mechanism starts with the identity
\[
\mathcal{L}(\phi) = \mathbb{E}_{\theta \sim q_\phi(\theta)}[\log p(\mathcal{D},\theta)] - \mathbb{E}_{\theta \sim q_\phi(\theta)}[\log q_\phi(\theta)],
\]

where \(\mathcal{L}(\phi)\) is the ELBO for variational parameters \(\phi\), \(q_\phi(\theta)\) is the tractable surrogate distribution over latent vector \(\theta\), and \(p(\mathcal{D},\theta)\) is the joint model density. Because
\[
\log p(\mathcal{D}) = \mathcal{L}(\phi) + \text{KL}(q_\phi(\theta)\parallel p(\theta\mid\mathcal{D})),
\]

the ELBO lower-bounds the log marginal likelihood and shrinking the KL divergence is equivalent to raising the lower bound. The dome metaphor reflects this: adjusting \(\phi\) inflates the surrogate \(q_\phi\) until it presses against the unknown posterior, and the optimization is entirely deterministic once the expectations can be evaluated.

### Mean-field and coordinate ascent

Jordan et al. (1999) introduced a mean-field factorization \(q_\phi(\theta) = \prod_i q_i(\theta_i)\), yielding coordinate updates that are expectations of the joint log probability under all other factors:
\[
\log q_i^\star(\theta_i) \propto \mathbb{E}_{q_{-i}(\theta_{-i})}[\log p(\mathcal{D},\theta)],
\]

where \(q_{-i}\) denotes the product of the other factors. Each update therefore requires only the expected sufficient statistics of the joint density, which are available in conjugate models. Mean-field VI is deterministic because every coordinate update solves a fixed-point equation: compute the expectation, normalize, and iterate until the ELBO saturates. It is this fixed-point view that best explains why VI is an optimization problem rather than a sampling process.

The deterministic coordinate updates and the later gradient-based updates share the same ELBO objective, but they differ in how they handle the expectations. Mean-field VI handles expectations in closed form by exploiting conjugacy, while stochastic VI approximates expectations with unbiased samples and alternates between local and global parameter updates. The result is a unified ELBO-driven picture: any variational algorithm is permissible if it can evaluate or approximate the gradients of \(\mathcal{L}(\phi)\). Blei et al. (2017) [https://arxiv.org/pdf/1601.00670v4] provides a comprehensive overview of how structured approximations, implicit flows, and amortized networks fit within this shared objective, showing that deterministic coordinate ascent, implicit variational distributions, and amortized inference are special cases of the same optimization framework.

### The reparameterization trick and generic gradients

Mean-field updates collapse when the model lacks conjugacy or when the variational family is not tractable. Kingma & Welling (2013) [https://arxiv.org/abs/1312.6114] remapped the expectation over the random latent \(\theta\) into an expectation over noise \(\epsilon\) that has a fixed distribution, enabling backpropagation through stochastic nodes. For example, a Gaussian variational factor \(q_\phi(\theta) = \mathcal{N}(\mu_\phi,\sigma_\phi^2)\) can be written as
\[
\theta = \mu_\phi + \sigma_\phi \odot \epsilon,\qquad \epsilon \sim \mathcal{N}(0, I),
\]

so the ELBO gradient becomes
\[
\nabla_\phi \mathcal{L}(\phi) = \mathbb{E}_{\epsilon \sim \mathcal{N}(0, I)}\left[\nabla_\phi \log p(\mathcal{D}, \theta(\phi, \epsilon)) - \nabla_\phi \log q_\phi(\theta(\phi, \epsilon))\right],
\]

where \(\theta(\phi, \epsilon)\) is the deterministic reparameterized latent. The gradient estimates now flow through both the model likelihood and the variational entropy, and automatic differentiation can handle complex networks that output the variational parameters. The reparameterization trick is thus the workhorse that extends VI into deep generative models and neural amortized inference.

Rather than crafting custom updates for \(\log q_\phi(\theta)\), Ranganath et al. (2014) rewrote the entropy gradient in a generic form using the log-derivative trick:
\[
\nabla_\phi \mathbb{E}_{q_\phi}[\log q_\phi(\theta)] = \mathbb{E}_{q_\phi}[(\log q_\phi(\theta)) \nabla_\phi \log q_\phi(\theta)],
\]

allowing any differentiable implementation of \(q_\phi(\theta)\) to participate. The combination of reparameterization gradients for the likelihood and score-function gradients for the entropy is the core loop in Black Box Variational Inference (BBVI): sample \(\epsilon\), compute \(\theta\), evaluate \(\log p(\mathcal{D},\theta)\) and \(\log q_\phi(\theta)\), and backpropagate. The variance of these gradients provokes the next major engineering challenge—developing diagnostics and control techniques that keep the dome from vibrating as it converges.

### Stochastic Variational Inference

To scale VI to massive datasets, Hoffman et al. (2013) introduced Stochastic Variational Inference (SVI) [https://www.cs.columbia.edu/~blei/papers/HoffmanBleiWangPaisley2013a.pdf][https://www.cs.columbia.edu/~blei/papers/HoffmanBleiWangPaisley2013.pdf][https://arxiv.org/abs/1206.7051v3], which replaces full-dataset updates with minibatch-based stochastic gradients while incorporating natural gradients based on the exponential family geometry. The global variational parameters \(\lambda\) govern the entire dataset, while each datum \(x_i\) has local variables \(z_i\) with variational parameters \(\phi_i\). The ELBO decomposes into a sum over data points, so the global gradient is approximated by scaling the minibatch contribution:
\[
\nabla_\lambda \mathcal{L}(\lambda) \approx \frac{N}{|B|} \sum_{i \in B} \left[\nabla_\lambda \mathbb{E}_{q_{\phi_i}}[\log p(x_i, z_i)] - \nabla_\lambda \mathbb{E}_{q_{\phi_i}}[\log q_{\phi_i}(z_i)]\right],
\]

where \(N\) is the dataset size, \(B\) is the minibatch, and \(|B|\) its cardinality. The natural gradient rescales the ordinary gradient by the inverse Fisher information of the variational family, producing updates that respect the information geometry and avoid runaway steps. The net effect is a data-efficient dome-fitting strategy: SVI processes each batch once and steadily adjusts the global parameters while local parameters are optimized on the fly, making VI viable for streaming and billion-token corpora.

### Practical example: Bayesian logistic regression

BBVI’s mechanics become tangible when applied to Bayesian logistic regression on the UCI Breast Cancer dataset. The probabilistic model places a standard normal prior \(p(w) = \mathcal{N}(0, I)\) on the weights \(w \in \mathbb{R}^d\), and the likelihood is logistic: \(p(y_i\mid x_i, w) = \text{Bernoulli}(\sigma(w^\top x_i))\), where \(\sigma\) is the sigmoid. The variational posterior \(q_\phi(w) = \mathcal{N}(\mu, \text{diag}(\exp(\log\sigma)))\) is reparameterized as \(w = \mu + \exp(0.5 \log \sigma) \odot \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\). The ELBO becomes
\[
\mathcal{L}(\phi) = \mathbb{E}_{w \sim q_\phi}[ \sum_i \log p(y_i\mid x_i, w) + \log p(w) - \log q_\phi(w)],
\]

and each minibatch of size 64 yields an unbiased Monte Carlo estimate of the likelihood term. Adam with learning rate \(10^{-3}\) and weight decay \(10^{-4}\) produces a smooth rise in ELBO when the variational parameters are well-initialized, and typical predictive accuracy lies between 93% and 96% depending on the random seed and preprocessing. The outcome is a posterior over weights with calibrated uncertainty, which can be inspected to understand which features drive the classifier’s confidence.

### Failure modes and diagnostics

The methodology falters when the variational family is too rigid, when gradient estimates carry high variance, or when stochastic updates oscillate. A fully factorized Gaussian cannot capture multi-modal or strongly correlated posteriors, leading to KL terms that saturate quickly while the predictive likelihood still lags. Monitoring the KL term relative to the likelihood reveals these failures: if the KL fails to increase but the log-likelihood stagnates, the surrogate is underfitting the true posterior. The gradient variance problem that arises in BBVI stems from the log-derivative estimator in the entropy term—this is precisely why Ranganath et al. (2014) recommends control variates that subtract baselines from the log-likelihood-to-weight scores. The same section also noted that SVI introduces noisy updates, so the annealing schedule \(\rho_t = (\tau + t)^{-\kappa}\) with \(\kappa \in (0.5,1]\) from Hoffman et al. (2013a) is critical; natural gradients further stabilize these noisy steps by adapting to the curvature of the variational family. These diagnostics thus close the loop with the earlier sections: the mean-field failure emerges from the deterministic expectations, the gradient variance issue originates from the BBVI log-derivative, and the noisy stochastic updates are the reason for careful step-size control in SVI.

## Where the field is now

Modern work on VI sits at the intersection of reparameterization-friendly families, score-function gradients, and computational kernels that make ELBO evaluation cheap. Subedar et al. (2025) [https://arxiv.org/abs/2506.21408] exemplifies this synthesis: ScalaBL constrains LoRA adapters to a low-rank subspace and performs SVI within that subspace, mixing reparameterization gradients for the adapter subspace with Score-function terms when the posterior over the remaining dimensions lacks a simple reparameterization. Their pipeline demonstrates that VI can run in a megabyte-scale inference cabinet while still delivering calibrated uncertainty, showing how the natural-gradient SVI steps described above become practical when the variational family is constrained to a subspace but still retains expressiveness.

At the same time, system-level advances such as Meta’s FlashAttention 2 (2024) [https://ai.meta.com/research/flashattention2] supply fast kernels for the reparameterization-based components of the ELBO. When vision-language or reasoning transformers add variational regularizers to their loss, the same FlashAttention 2 kernels that accelerate attention also reduce the time spent computing the likelihood and its gradients, making it feasible to treat reparameterization-based VI as part of the training loop rather than a post-hoc calibration pass. These engineering advances tie directly back to the mathematical trade-offs discussed earlier: reparameterized gradients benefit from fast matrix operations, while score-function terms can be kept in check by control variates and natural-gradient preconditioning.

### Where this concept appears

Variational inference anchors the approximate-inference arc by connecting conjugacy-guided Bayesian inference (see [[bayesian-inference]]) with neural amortized approaches such as [[bayesian-neural-networks]] and [[variational-autoencoders]]. The approach also reappears in broader arcs focused on uncertainty quantification and Bayesian deep learning, where the ELBO objective is re-used to shrink posterior approximations while keeping inference tractable. The principle is that any arc that adds uncertainty-aware objectives to neural models will sooner or later instantiate the ELBO, so this page is a central node linking probabilistic foundations to applied deep-learning builds.

## What's still open

Can automated search over variational families discover structures that approximate the true posterior geometry without human engineering? Current flows and blockwise approximations still rely on intuition about posterior shape. A general-purpose procedure that chooses the right structural constraint for a given likelihood and data distribution remains elusive.

What finite-sample guarantees can be proven when combining SVI with amortized inference networks? The ELBO is controlled in expectation, but gradients now depend on approximations to both the likelihood and the amortized encoder, leaving theoretical guarantees for convergence and bias open.

How can variational inference retain memory of previous modes when learning from streaming data? The standard ELBO chases the new data, especially when natural gradients accentuate recent observations. A formal analysis of trust-region-constrained SVI or memory-augmented variational families could connect continual learning with Bayesian robustness.

## Where to read next

The probabilistic foundations live in [[bayesian-inference]], which details conjugacy and marginalization stories that gave rise to the ELBO, while the deep-learning implementation leap is documented in [[bayesian-neural-networks]] where amortized VI and reparameterization are the workhorses of uncertainty in large models; the engineering counterpart is [stochastic-gradient-optimization](stochastic-gradient-optimization.md), which describes the optimizer choices that keep ELBO penalties stable over billions of datapoints.

## Build it

This build demonstrates that deterministic coordinate ascent is not required: a BBVI loop with pretrained features can train a calibrated Bayesian classifier, exposing how reparameterization gradients, entropy estimates, and minibatched likelihoods interact.

**What you're building:** A Black Box Variational Inference pipeline over a pretrained transformer feature extractor that fits a Bayesian classification head on GLUE/MRPC, producing calibrated predictive intervals.

**Why this is valuable:** It turns the ELBO into a deployable optimization with concrete uncertainty outputs, highlighting how parameter uncertainty influences downstream calibration.

**Stack:**
- **Model:** `bert-base-uncased` feature extractor plus a small Bayesian logistic head
- **Dataset:** `glue/mrpc` from Hugging Face (https://huggingface.co/datasets/glue/viewer/mrpc)
- **Framework:** PyTorch 2.1 + functorch 2.1 for per-sample gradients
- **Compute:** RTX 4060 (8 GB) or free Colab T4, ~40 minutes for 200 epochs of the logistic head over frozen BERT embeddings

**The recipe:**
1. Install `torch`, `functorch`, `transformers`, `datasets`, and `scikit-learn`, then load `glue/mrpc` splits and tokenize inputs with `AutoTokenizer.from_pretrained("bert-base-uncased")`; cache the tokenized inputs to disk to avoid repeated tokenization.
2. Freeze the `BertModel` trunk, extract the `[CLS]` embeddings, and initialize variational parameters \(\mu\) and \(\log \sigma\) for the logistic head (matching the embedding size). Implement the reparameterization \(w = \mu + \exp(0.5\log\sigma)\odot\epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\) sampled per minibatch of 32.
3. Compute the ELBO as the minibatched negative binary cross-entropy plus the KL divergence between \(q(w)\) and the standard normal prior, using 10 MC samples per batch. Update the variational parameters with AdamW (lr \(1e{-3}\), weight decay \(1e{-4}\)) and the ELBO gradient estimated by functorch-supported vectorized Jacobian.
4. Evaluate by sampling 100 posterior weight draws to compute predictive accuracy and log-likelihood on the validation split; typical accuracy lies between 84% and 88% and the negative log-likelihood should settle in the range 0.35–0.50 depending on random seeds and preprocessing.
5. The artifact is a saved checkpoint of \(\mu\) and \(\log \sigma\) plus plots of the ELBO trace, calibration curves, and predictive intervals from the sampled weights, which can feed into downstream uncertainty-aware pipelines.

**Expected outcome:** A PyTorch checkpoint for the Bayesian classification head, calibration metrics, and plots showing posterior marginals and ELBO development.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Quantize the variational head weights using PyTorch dynamic quantization, serve via TorchServe on an L4, and implement temperature-scaled predictive intervals by taking 10⁴ posterior samples, converting logits to probabilities via \(\sigma(w^\top x_i / T)\) for \(T \in \{0.8,1.0,1.2\}\), and publishing the 5th/95th percentile ranges to monitor calibration drift under real traffic.
- **Research engineer:** Reproduce Table 2 from Subedar et al. (2025) by constraining the logistic head to a rank-2 LoRA adapter over `bert-base-uncased`, match the reported ScalaBL negative ELBO within ±0.03 by tuning the natural gradient step size and minibatch scale, and log the ELBO components for inspection.
- **Applied researcher:** Hypothesize that inflating the KL weight sharpens posterior uncertainty but lowers accuracy; test by scaling the KL term by coefficients {0.5, 1.0, 2.0} and plotting held-out accuracy versus predictive variance to falsify the hypothesis “more KL weight does not increase calibration.”

What can you build next: extend the pipeline with streaming SVI updates on continuously arriving text, or swap the logistic head for a simple flow-based variational family to probe whether richer approximations improve calibration without sacrificing throughput.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/frontier-ml/FAIRE) is the only signal we collect.*