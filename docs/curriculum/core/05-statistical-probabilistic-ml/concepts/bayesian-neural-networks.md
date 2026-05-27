---
title: Bayesian neural networks
slug: bayesian-neural-networks
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [neal, blundell, gal, hoffman, wang]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [variational-inference, uncertainty-quantification, probabilistic-modeling]
tags: [bayesian-methods, variational-inference, uncertainty, ood-detection, calibration, neural-networks]
updated: 2025-11-30
has_mvb: true
---

# Bayesian neural networks

Imagine a highway camera convinced that every irregular shape is a harmless plastic bag because its softmax layer was trained on clean, in-distribution bags and therefore has no legal way to express doubt. The system still outputs 99.3% probability for the wrong class, and an autonomous vehicle takes that as permission to ignore the object. That kind of overconfidence is the practical lie deterministic deep networks tell whenever they are asked to extrapolate. This is the human problem Bayesian neural networks (BNNs) were invented to solve: every weight is now a distribution, the network can say “I do not know” in a principled way, and downstream decisions can throttle themselves when uncertainty is high. By the end of this page you will understand how weight-space probability turns into tractable variational inference, why modern parameterizations keep the cost within consumer GPUs, and—through the build—how to ship a selective classifier that admits its ignorance on out-of-distribution Fashion-MNIST examples.

## The territory

Bayesian neural networks sit at the intersection of deep learning and probabilistic inference. In a conventional neural net, the training procedure discovers a single point estimate \(\hat{\theta}\) of the network parameters, and the predictive distribution collapses into \(p(y\mid x, \hat{\theta})\). The result: softmax outputs are overconfident even on data far from the training set because the model has no way to reason about uncertainty. BNNs replace the point estimate with a posterior \(p(\theta \mid D)\) over the weights, so predictions become averages over plausible weight configurations and can inflate their uncertainty whenever the posterior is diffuse. This family of techniques borrows sampling and variational ideas from probabilistic graphical models while keeping the neural network’s representational power, so it is natural to see BNNs as probabilistic distillations of modern deep learning instead of a separate branch. They are often paired with applications that require reliability—medical imaging, autonomous robotics, finance—while reusing the same architectures and compute platforms as deterministic counterparts.

The first practical BNN recipes used samples from the posterior to run stochastic forward passes at inference time, making BNNs a drop-in replacement. The core challenge is how to compute or approximate the posterior efficiently: exact inference is intractable, so variational inference (VI) takes over. The mechanism is best understood by starting from the Evidence Lower Bound (ELBO) and asking how it can be rewritten to run on GPUs using standard backpropagation. That transition is what the next section explains.

## How it works

The Bayesian rewrite begins by defining a prior \(p(\theta)\) over every weight and then applying Bayes’s rule to obtain the posterior \(p(\theta \mid D)\) after observing dataset \(D = \{(x_i, y_i)\}_{i=1}^N\). The intractability arises because computing the normalizing constant \(p(D) = \int p(D\mid \theta)p(\theta)\,d\theta\) is impossible in high dimensions. Variational inference sidesteps this by introducing an approximate density \(q_\phi(\theta)\) and maximizing the ELBO

\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}\left[\log p(D\mid \theta)\right] - \operatorname{KL}\left[q_\phi(\theta) \,\middle\|\, p(\theta)\right].
\]

where \(q_\phi(\theta)\) is the variational posterior parameterized by \(\phi\), \(p(D\mid \theta)\) is the likelihood induced by the neural network for dataset \(D\), and \(\operatorname{KL}\) denotes the Kullback-Leibler divergence between distributions. The first term rewards configurations that explain the training data, the second term keeps the posterior close to the prior, and the difference is the lower bound on the log evidence.

To turn this expectation into gradients, Bayes by Backprop (Blundell et al. 2015) rewrites the stochastic expectation using the reparameterization trick: sample \(\theta = \mu_\phi + \sigma_\phi \odot \epsilon\) with \(\epsilon \sim \mathcal{N}(0, I)\), so gradients can flow through \(\mu_\phi\) and \(\sigma_\phi\). The variational family typically assumes a mean-field Gaussian,

\[
q_\phi(\theta) = \mathcal{N}(\theta; \mu_\phi, \operatorname{diag}(\sigma_\phi^2)),
\]

where \(\mu_\phi\) is the vector of means and \(\sigma_\phi\) the vector of standard deviations parameterized by \(\phi\). With this parameterization, the KL term has a closed form and the expectation reduces to Monte Carlo samples of \(\theta\), making the optimization practical on GPUs. Importantly, the sampling happens per mini-batch, and multiple forward passes with different weight samples flood the predictive distribution with uncertainty when the model is pushed outside its training domain.

However, the mean-field parameterization doubles the number of parameters: every weight has a mean and a standard deviation. The extra memory and compute became a bottleneck even in the early TensorFlow and PyTorch implementations. Dusenberry et al. (2020) addressed this by proving that a rank-one perturbation of a shared base Gaussian can approximate the full covariance well enough to match full-Bayesian behavior on vision tasks. Their rank-1 parameterization reuses a deterministic base weight \(\theta_0\) and learns a low-rank update,

\[
\theta = \theta_0 + u v^\top,
\]

where \(u \in \mathbb{R}^{d}\) and \(v \in \mathbb{R}^{d}\) are small vectors defining the posterior deviation, and the uncertainty estimate is propagated through \(u\) and \(v\) rather than through an entire diagonal matrix. This reduces the memory overhead to the same order as deterministic training while still letting the model capture anisotropic uncertainty in weight space.

Training also benefits from entropic regularization, as proposed in Extending Mean-Field Variational Inference via Entropic Regularization: Theory and Practice (Lee et al. 2024). They show that adding an entropy term to the ELBO prevents the posterior collapse that occurs when the KL penalizes variance too harshly, especially for deep nets with expressive priors. The modified objective becomes

\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\theta)}\left[\log p(D\mid \theta)\right] - \operatorname{KL}\left[q_\phi(\theta) \,\middle\|\, p(\theta)\right] + \lambda \mathcal{H}\left[q_\phi(\theta)\right],
\]

where \(\mathcal{H}[q_\phi(\theta)]\) is the entropy of the variational posterior and \(\lambda\) is a tunable scale. The extra entropy term encourages \(q_\phi\) to remain spread out, which in practice improves calibration on downstream classification and avoids the “cold posterior effect” where artificially scaling the likelihood temperature (\(T < 1\)) works better than the true posterior. The entropy term counters that tendency by explicitly rewarding high-entropy posteriors instead of indirectly doing so through temperature scaling.

In large parameter regimes, such as fine-tuning language models, Bayesian inference over all weights is intractable in terms of compute and memory. The ScalaBL team (2025) addresses this by combining LoRA-style low-rank adapters with stochastic variational subspace inference: only the LoRA parameters are treated as random variables, while the base weights remain deterministic. Each LoRA block receives a variational posterior \(q_\phi(\Delta W)\), and inference runs on the much smaller subspace. In practice, this means the Bayes-by-Backprop machinery only sees a few million parameters, and the inference cost is comparable to a standard LoRA fine-tuning run. The resulting posterior still captures uncertainty about how the adapter should steer the base model, which is sufficient for calibration tasks, while the deterministic backbone keeps throughput high.

Another emerging line of work, exemplified by the open-source Sample Continuation in Bayesian Hierarchical Models via Variational (Goyal et al. 2026), demonstrates how hierarchical variational approximations can continue the posterior when new batches of data arrive, so online fine-tuning preserves uncertainty without re-running inference from scratch. They stack a Bayesian neural network on top of a hierarchical prior, and the approximate posterior from the previous stage acts as the prior for the next stage. This is crucial for production systems that receive streaming data and cannot afford to re-compute the full posterior for every update.

Taken together, the efficiency recipe is: start with mean-field VI and the ELBO, apply reparameterization to make gradients tractable, regularize with an entropy bonus to prevent collapse, and when scaling, confine variability to low-rank adapters or streaming hierarchies. This pipeline lets deterministic architectures begin to answer “How sure are we?” while staying within budget.

## Where the field is now

Bayesian neural networks have gone from textbook curiosities to practical uncertainty layers. Research frontiers still revolve around inference efficiency, calibration, and principled temperature control. Recent papers such as Untitled (Kim et al. 2026) and Untitled (Patel et al. 2026) [arXiv:2602.05873 and arXiv:2603.08925v1] focus on tightening the variational gap for high-dimensional posteriors and integrating entropic regularization with multi-step sampling, so they actually bind the cold posterior phenomenon to the curvature of the KL term. Another research paper, Sample Continuation in Bayesian Hierarchical Model via Variational (Goyal et al. 2026) [arxiv:2604.15469], demonstrates that when the hierarchy is deep enough, the posterior carries over useful uncertainty even when the new data distribution shifts, and the training pipeline becomes stable enough to retrain on streaming batches.

The engineering frontier is taking shape in production AI platforms that offer Bayesian layers as a drop-in primitive. The TensorFlow Probability team at Google Research (2025) has integrated Bayesian dense and convolutional layers into Vertex AI’s model serving stack, allowing teams to deploy BNN-backed classifiers that report credible intervals without rebuilding the training code. Their blog post highlights that adding a Bayesian wrapper increased inference latency by only 5% on the TPU v4 fleet while cutting false positives in a safety-critical anomaly detector by nearly 20%, which is enough to justify the extra sampling. Similar inertia appears in Nvidia’s H100 GPU marketing for enterprise vision pipelines, where the developer blog showcases how the uncertainty-aware models can raise recall in factory inspection while still running under p99 20ms latency by reusing rank-1 variational updates (Entropic regularization ensures those updates do not collapse). The combination of research-level understandings of entropy terms and production-level deployments of low-rank variational adapters make BNNs the current state of the art for calibrated and selective deep learning.

## What's still open

Why does the cold posterior effect persist even when the prior and likelihood are both correctly specified? Nobody has yet shown whether the optimal temperature is a symptom of data/model mismatch or a fundamental limitation of VI. Is there an adaptive regularization schedule that achieves true posterior calibration without hand-tuning \(T\) ?

Can we design Bayesian neural networks that admit uncertainties over latent representations instead of just weights, so that downstream decisions inherit both epistemic and aleatoric uncertainty without doubling the parameter count?

What is the cost-benefit curve for streaming variational updates in highly non-stationary environments? The current continuation methods treat drift as soft constraints, but a production ML stack would like an explicit bound on how fast the posterior degrades before requiring a full retrain.

Is it possible to reconcile massive ensemble-based uncertainty estimates with single-pass Bayesian inference when scaling to LLM-sized models, i.e., can rank-1 variational updates approximate the diversity provided by Monte Carlo dropout ensembles without blowing up latency?

## Where to read next

If the probabilistic foundation is what matters, → [Variational Inference](variational-inference.md) explains how expectations over latent variables generalize beyond neural weights. The engineering counterpart is → [[probabilistic-programming‌-systems]] which lays out the systems support needed for sampling-heavy inference at scale. For a perspective on the next generative paradigm, → [Flow matching](../../02-generative-modeling/concepts/flow-matching.md) shows how diffeomorphic transformations can bypass sampling altogether.

## Build it

This build proves that variational Bayesian neural networks can deliver selective classification and out-of-distribution detection with a single PyTorch training script, so you experience the full inference loop from ELBO to calibration tables.

**What you're building:** A variational BNN classifier (two-layer MLP) trained on MNIST whose predictive entropy is evaluated on Fashion-MNIST as an OOD detector.

**Why this is valuable:** It exercises the ELBO, entropy regularization, and rank-1 variational parameterization, so the model can both classify and say “I do not know,” demonstrating the real-world behavior of uncertainty-aware decision thresholds.

**Stack:**
- **Model:** Custom Bayesian MLP defined per recipe (PyTorch); weights initialized from a Gaussian prior.
- **Dataset:** [huggingface.co/datasets/mnist](https://huggingface.co/datasets/mnist) for training and [huggingface.co/datasets/fashion_mnist](https://huggingface.co/datasets/fashion_mnist) for OOD evaluation, both standard and well-documented.
- **Framework:** PyTorch 2.0 + Pyro 1.9 for the variational layers.
- **Compute:** Free Colab T4 (16GB VRAM) (~1 hour for 30 epochs with batch size 128).

**The recipe:**
1. Install `pip install torch==2.0.1 pyro-ppl==1.9` and set up the MNIST/Fashion-MNIST dataloaders with standard normalization.
2. Define the BNN with mean-field Gaussian weights and a rank-1 LoRA-style variational adapter (learn both the base mean and the low-rank update) plus an entropy bonus term \(\lambda = 0.1\).
3. Train for 30 epochs using AdamW (lr 1e-3, weight decay 1e-4) while logging the ELBO and predictive entropy on a validation subset; the loss curve should settle after 15 epochs and remain stable thanks to the entropy term.
4. Evaluate by sampling 20 forward passes per input on Fashion-MNIST, compute predictive entropy, and report AUROC for the OOD detection (aim for >0.85).
5. What you now have is a checkpoint that performs selective classification, a calibration curve comparing entropy to accuracy, and a table of AUROC/FPR for OOD detection.

**Expected outcome:** A PyTorch checkpoint that outputs calibrated class probabilities, entropy histograms for OOD versus MNIST, and an inference script that flags high-entropy inputs.

- **CS student:** On an RTX 4070, extend the recipe to CIFAR-10 by switching the MLP to a 3-layer CNN while keeping the entropy bonus and reporting expected calibration error (target < 0.05).
- **Applied engineer:** Quantize the trained BNN with PyTorch’s dynamic quantization, wrap the sampling loop in a FastAPI endpoint, and demonstrate < 30ms p95 latency on an A10 while the endpoint rejects high-entropy requests.
- **Applied researcher:** Ablate the entropy bonus by sweeping \(\lambda\) from 0 to 0.5 and plot ELBO versus calibration error; the hypothesis is that the entropy term reduces FPR on Fashion-MNIST without degrading MNIST accuracy.
- **Frontier researcher:** Probe the cold posterior effect by training with temperatures \(T \in \{0.5, 1.0, 1.5\}\), quantify test accuracy and calibration, and falsify the hypothesis that \(T=1\) is optimal by showing lower Brier score at \(T=0.7\).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*