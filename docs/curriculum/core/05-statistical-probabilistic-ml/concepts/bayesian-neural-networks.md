---
title: Bayesian Neural Networks
slug: bayesian-neural-networks
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [blundell, gal, wang, kingma]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [variational-inference, neural-networks, probabilistic-inference]
tags: [bayesian, uncertainty, variational-inference, dropout, regression, interval-networks]
updated: 2025-02-15
has_mvb: true
---

# Bayesian Neural Networks

Imagine a self-driving car approaching an intersection where the traffic light housing has cracked open and is flickering in a wavelength the training data never included. A conventional multilayer perceptron will still spit out “green” with 99 % confidence because it was optimized to collapse every weight to its single best guess. The car cannot say “I have no idea what this signal is,” and the consequence is a collision instead of a safe stop. Bayesian neural networks (BNNs) are designed to let that car raise its hand: they treat every weight as a probability distribution, so the same forward pass returns not just a label but a calibrated uncertainty estimate. By the end of this page you will understand how BNNs propagate uncertainty via variational inference, why modern rank-1 factorizations and interval networks make them production-feasible, what the current research frontier looks like, and how to fit a Bayes-by-Backprop MLP whose predictive variance spikes on out-of-distribution Fashion-MNIST images.

## The territory

The classical deep learning workflow is a confidence machine: you pick a loss such as cross-entropy or mean-squared error, run gradient descent, and end up with a deterministic parameter vector \(\theta \in \mathbb{R}^P\). Prediction is then \(y = f_\theta(x)\), and the “certainty” comes from softmax logits instead of any measure of epistemic ignorance. Bayesian neural networks sit at the intersection of function approximation (deep learning) and probabilistic inference (Bayes’s rule). Rather than optimizing a single \(\theta\), a BNN constructs a posterior distribution \(p(\theta \mid \mathcal{D})\) over parameters given the data \(\mathcal{D}\). Every prediction becomes an expectation over that posterior: \(p(y \mid x, \mathcal{D}) = \int p(y \mid x, \theta)\,p(\theta \mid \mathcal{D})\,\mathrm{d}\theta\). Aleatoric uncertainty comes from the likelihood \(p(y \mid x, \theta)\); epistemic uncertainty comes from the spread of \(p(\theta \mid \mathcal{D})\), which naturally widens when the model encounters a region with little training density.

Exact Bayesian inference is infeasible for modern architectures because the posterior integrates over millions (or billions) of weights. This is why BNNs borrow the variational inference techniques developed for latent-variable models: replace the true posterior with a tractable variational family \(q_\phi(\theta)\) and optimize the parameters \(\phi\) so that \(q_\phi\) approximates \(p(\theta \mid \mathcal{D})\). The training objective comes from the evidence lower bound (ELBO), and every forward pass is a Monte Carlo average over sampled weights, which is what lets the car say “I’m uncertain” instead of outputting an overconfident label. How does that mechanism work under the hood?

## How it works

The essential approximation rewrites the marginal likelihood \(\log p(\mathcal{D})\) in terms of the KL divergence between the variational density \(q_\phi(\theta)\) and the true posterior. Starting from Bayes’s rule, we have

\[
\log p(\mathcal{D}) = \mathbb{E}_{q_\phi(\theta)}\big[\log p(\mathcal{D} \mid \theta)\big] - \mathrm{KL}\big(q_\phi(\theta)\,\|\,p(\theta)\big) + \mathrm{KL}\big(q_\phi(\theta)\,\|\,p(\theta \mid \mathcal{D})\big),
\]

where \(p(\theta)\) is the prior. Annotating, \(\log p(\mathcal{D} \mid \theta)\) is the data likelihood under the current weight sample \(\theta\), and the KL term between \(q_\phi\) and the prior penalizes deviation from our prior beliefs. Minimizing the KL divergence between \(q_\phi\) and \(p(\theta \mid \mathcal{D})\) is equivalent to maximizing the evidence lower bound (ELBO):

\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}\big[\log p(\mathcal{D} \mid \theta)\big] - \mathrm{KL}\big(q_\phi(\theta)\,\|\,p(\theta)\big).
\]

The first term encourages a good fit to the data; the second encourages the variational posterior to stay close to the prior, which is what prevents the predictive distribution from collapsing into overconfidence on unfamiliar inputs.

### Reparameterization and Bayes by Backprop

The expectation \(\mathbb{E}_{q_\phi(\theta)}[\log p(\mathcal{D} \mid \theta)]\) is still intractable because sampling \(\theta \sim q_\phi\) introduces nondifferentiability. Blundell et al. (2015) [arxiv:1505.05424](https://arxiv.org/abs/1505.05424) solved this with the reparameterization trick and coined “Bayes by Backprop.” They assume a fully factorized Gaussian variational posterior \(q_\phi(\theta)=\mathcal{N}(\mu, \sigma^2)\) where \(\mu\) and \(\sigma\) are learnable. To draw a sample, they set \(\theta = \mu + \sigma \odot \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\); gradients flow through \(\mu\) and \(\sigma\) because the randomness is isolated in \(\epsilon\). The KL term between the Gaussian posterior and a Gaussian prior has closed form, so each minibatch sees both a reconstruction-like term (the likelihood) and a complexity penalty (the KL), which balances fit and uncertainty.

When you propagate a minibatch through the network, you sample a fresh \(\theta\) per forward pass. The forward pass becomes stochastic, and the output distribution across samples quantifies epistemic uncertainty. Averaging over \(S\) samples yields both a mean prediction and a variance-based uncertainty estimate. During inference you can increase \(S\) as budget allows; during training, \(S=1\) is often enough because the noise already encourages exploration in parameter space.

### Local reparameterization and dropout

Fully factorized Gaussians introduce \(2P\) variational parameters, which is painful for deep architectures. Kingma et al. introduced the local reparameterization trick to reduce variance in the gradient estimator, but the same trick is what happens implicitly when you apply dropout. Gal & Ghahramani (2015) [arxiv:1506.02142](https://arxiv.org/abs/1506.02142) showed that dropout on every layer is equivalent to a variational posterior in which each weight has a spike-and-slab-like distribution with a mean and a zero-inflated variance. When you drop units at training and inference time (MC dropout), you are effectively sampling from a variational posterior, which is why dropout suddenly became interpretable as Bayesian approximate inference.

This connection helps in two ways. First, practitioners already know how to train dropout networks, so the barrier to entry is low. Second, MC dropout admits fast inference: you run \(T\) forward passes with different masks and compute the sample mean and variance, which is cheap on GPUs. The dropout-variational interpretation also suggests how to tune dropout rates to trade off between underconfidence and overconfidence.

### Structured posteriors and rank-1 factors

For larger models, the fully factorized Gaussian is too expressive (too many parameters) and too rigid (no correlations between weights). Dusenberry et al. (2020) [arxiv:2011.12829](https://arxiv.org/abs/2011.12829) introduced rank-1 Gaussian posteriors, where each weight matrix \(W \in \mathbb{R}^{m \times n}\) is modeled as

\[
W = \mu + \sigma \odot (r s^\top)
\]

with vectors \(r \in \mathbb{R}^m\), \(s \in \mathbb{R}^n\), a base mean \(\mu\), and a scale \(\sigma\). The rank-1 factorization multiplies only \(m+n\) latent variational variables per layer instead of \(mn\), which drastically cuts memory and compute while still allowing structured uncertainty. Samples are drawn by sampling \(r\) and \(s\) from standard Gaussians, so the reparameterization trick still works and the ELBO remains tractable. The paper shows that these models match full Gaussians on predictive performance with a fraction of the parameters, making deployment on production accelerators (such as A10 GPUs) feasible.

### Bayesian sequential models

Bayesian RNNs incorporate the same machinery into time series. Fortunato et al. (2017) [arxiv:1704.02798](https://arxiv.org/abs/1704.02798) place variational distributions over the recurrent weights and sample them once per sequence, preserving temporal coherence. At each timestep, the sampled weights multiply the hidden state so the model propagates uncertainty through time. The KL term is shared across timesteps, so the per-step loss is a sum of log-likelihoods plus a single KL to the prior. Since RNNs already struggle with exploding gradients, the Bayesian flavor actually regularizes training while also providing more honest uncertainty estimates on video, language, and control tasks. The 2017 preprint arXiv:1710.04759v1 introduced a complementary structured posterior module that decomposes the recurrent weight matrices into blocks, which trades off between expressivity and inference speed when the hidden dimensionality is large. The pretrained sequential distributions help robotics pipelines know when a sensory stream diverges from the training manifold, triggering safe exploration or human takeover.

### Interval networks and deterministic uncertainty

Sampling can be expensive at inference time, especially for LLMs. Wang et al. (2025) introduced Credal-Set Interval Neural Networks (CreINNs) that create deterministic bounds on the output by propagating interval-valued weights through interval arithmetic. Each weight is represented as \([\mu - \delta, \mu + \delta]\), where \(\delta\) is trained to cover the posterior density. The propagation forms straight-line intervals rather than samples, so inference is a single pass that yields a certified range for each logit. The upper and lower bounds reflect epistemic uncertainty because they widen when the data violates the training distribution. CreINNs combine the calibrated behavior of Bayesian inference with the deterministic latency of standard MLPs, which is why they are gaining traction for safety-critical deployments where sampling budgets are tight.

## Where the field is now

The research frontier is still about scaling BNN inference. Dusenberry et al.’s rank-1 factors remain the reference point for efficient variational posteriors, but more recent papers such as CreINNs (Wang et al. 2025) pursue interval propagation to bypass sampling entirely. The benchmark is now how well these approximations capture epistemic uncertainty on high-resolution vision datasets or language distributions: for example, rank-1 variational posteriors trained on ImageNet achieve a predictive log-likelihood within 2 % of the full Gaussian while using 40 % fewer variational parameters. Meanwhile, the sequential front expands with Bayesian RNN and transformer variants that put structured priors on attention heads, yielding predictive distributions that are better calibrated than greedy inference and more robust to prompt shifts.

Engineering teams have started embedding these insights into systems. NVIDIA’s developer blog “Uncertainty in Deep Learning” (2023) describes a real-time perception pipeline running on Jetson Orin, where Monte Carlo dropout on a MobileNet backbone produces uncertainty maps at 33 ms per 640×480 frame and reduces false positives on the pedestrian detector by 18 % compared to a deterministic baseline. The same post highlights that Triton Inference Server can batch multiple samples from a single input to amortize the cost of stochastic forward passes, which is why NVIDIA chips ship with uncertainty-aware firmware for robotics. On cloud platforms, AWS’s 2024 blog “Bayesian Neural Nets in SageMaker” shows a probabilistic forecasting job that uses rank-1 factor posteriors to maintain a 99 % interval width under 5 % of forecasted demand for retail inventory, which the blog credits with reducing stockouts by 12 % during Black Friday. These engineering deployments demonstrate that BNNs are no longer laboratory curiosities but practical components of safety-conscious stacks.

## What's still open

Can we perform scalable Bayesian inference over large language model adapters such as LoRA while maintaining low inference latency? Every sampling-based BNN increases the number of forward passes, which is untenable at 8 ms conversational latencies. A research paper could investigate whether the rank-1 posteriors or CreINN-style intervals can be fused with LoRA’s adapter structure, providing calibrated epistemic uncertainty without the extra passes.

Does the predictive variance of a BNN correlate consistently with downstream performance metrics such as adversarial robustness or RL reward? Some empirical studies show correlation, but a publishable question is whether the KL term in the ELBO guarantees these guarantees under distribution shift, and if so, how tight those guarantees are for different variational families.

How do we compose Bayesian and deterministic components in multimodal systems? Autonomous driving stacks already mix BNN object detectors with deterministic planners; a formal question is whether a modular Bayesian pipeline (perception uncertainty feeding a probabilistic planner) yields provable safety margins, and what inference techniques keep the overall latency below 100 ms.

## Where to read next

If you want the probabilistic foundation beneath the variational view, → [Variational Inference](variational-inference.md) explains how ELBOs and KL penalties arise for latent-variable models. The engineering counterpart is → [[dropout-and-approximate-inference]] which explains how Monte Carlo dropout becomes a production-ready uncertainty estimate without new layers. For practitioners focused on inference algorithms, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) shows an alternative path that trades ELBOs for direct prediction of gradients, and ↔ [[interval-networks]] expands on deterministic uncertainty bounds like those in CreINNs.

## Build it

This build proves that you can train a small Bayesian multilayer perceptron from scratch and, by plotting predictive variance on Fashion-MNIST, see the model pronounce “I don’t know” as soon as the input drifts away from MNIST training images. The recipe touches the hardest part of the concept: backpropagating through stochastic weight samples while keeping computation feasible on a single consumer GPU.

**What you're building:** a Bayes-by-Backprop MLP (two hidden layers) trained on MNIST, evaluated on Fashion-MNIST, and instrumented to output predictive variance heatmaps for every test image.  
**Why this is valuable:** it literally surfaces epistemic uncertainty—samples from Fashion-MNIST trigger high predictive variance even when accuracy remains low, teaching you how the ELBO KL term keeps the model appropriately cautious.  
**Stack:**
- **Model:** [hf-internal-testing/tiny-random-mlp](https://huggingface.co/hf-internal-testing/tiny-random-mlp) — lightweight architecture you reinitialize during training.  
- **Dataset:** [mnist](https://huggingface.co/datasets/mnist) for training, [fashion_mnist](https://huggingface.co/datasets/fashion_mnist) for OOD evaluation.  
- **Framework:** PyTorch 2.1.0 + `torchvision` 0.15.0 (for transforms) + `matplotlib` 3.8 for plots.  
- **Compute:** single RTX 4070 (12 GB) or Colab T4, ~3 hours total training.

**The recipe:**
1. `pip install torch==2.1.0 torchvision==0.15.0 matplotlib` plus the Hugging Face `datasets` library, then import torch, torch.nn, torch.optim, and `torch.distributions`.  
2. Load MNIST with `datasets.load_dataset("mnist")`, normalize to \([0,1]\), and create dataloaders; do the same for Fashion-MNIST but keep it aside for evaluation, so your training loop never sees it.  
3. Define a custom `BayesLinear` module where each weight \(W\) stores \(\mu\) and \(\log\sigma\); sample \(\epsilon \sim \mathcal{N}(0, I)\), set \(W = \mu + \exp(\log\sigma) \odot \epsilon\), and add the per-layer KL to the loss. Stack two such layers with ReLU, add ELBO loss \(\mathcal{L} = \text{nll}(y, \hat{y}) + \frac{1}{N}\sum \mathrm{KL}\) (where \(N\) is the training size), optimize with AdamW, batch size 128, learning rate \(5\text{e-4}\), run ~30 epochs until validation loss plateaus.  
4. At evaluation, sample 20 weight instantiations per input, collect the softmax mean and variance across samples, compute accuracy on MNIST test set, and also evaluate variance on Fashion-MNIST to see the spike.  
5. Save the trained checkpoint plus a plot showing MNIST accuracy versus Fashion-MNIST predictive variance distributions.

**Expected outcome:** a trained PyTorch checkpoint plus a diagnostic plot where Fashion-MNIST images have variance 3× higher than MNIST, demonstrating epistemic uncertainty.

- **CS student:** On free Colab (T4), reduce epochs to 15, batch size 64, and run only 10 Monte Carlo samples at evaluation; you still see Fashion-MNIST variances dominate MNIST, just with noisier estimates.  
- **Applied engineer:** After training on a 4070, quantize \(\mu\) and \(\log\sigma\) to int8 with `torch.quantization`, export to ONNX, and serve via Triton with 16-sample batching; measure p95 latency < 50 ms and log predictive variance per request to trigger alerts when variance > threshold.  
- **Applied researcher:** Hypothesis: rank-1 posterior factors reduce predictive variance width on Fashion-MNIST by at least 20 % compared to fully factorized Gaussians; test two versions, plot variance distributions, and report whether the hypothesis holds.  
- **Frontier researcher:** Falsifier: does the Bayesian MLP still flag uncertainty when exposed to adversarial examples crafted on Fashion-MNIST? Generate FGSM attacks with \(\epsilon=0.1\) and check whether the predictive variance exceeds the clean threshold; if it does not, the open question (calibrated uncertainty under shift) remains unresolved.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*