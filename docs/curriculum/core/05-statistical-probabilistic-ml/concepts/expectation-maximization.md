---
title: Expectation-Maximization
slug: expectation-maximization
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [dempster, hinton, nigam, bilmes]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [latent-variable-models, gaussian-mixture-models, coordinate-ascent, probabilistic-inference]
tags: [latent-variables, variational-inference, mixture-models, gmm, optimization, semi-supervised-learning]
updated: 2024-11-12
has_mvb: true
---

# Expectation-Maximization

Imagine throwing two coins into the air, recording the sequence of heads and tails, and then losing track of which coin produced which flip. You also forgot to record the bias of either coin. The dataset now has the flips but not the assignment to coin, and yet you still want to estimate both coins’ biases and the missing labels. Without knowing which coin produced each flip, the likelihood of the observed data washes out — simple maximum likelihood collapses into solving for both coins at once with no information on the hidden assignments. That is the “chicken-and-egg” problem that Expectation-Maximization (EM) was invented to resolve: inferred latent variables make the parameters identifiable, and fitted parameters make the latent variables estimable. By the end of this page you will understand why each EM iteration is a guaranteed coordinate ascent step on an evidence lower bound, how that view explains convergence even in complex Gaussian mixtures, and what it takes to build a practical implementation that both visualizes the inferential dance and produces a deployable clustering model.

## The territory

Latent-variable models recur whenever data carries hidden structure: mixture components, topics in documents, speaker identities in audio. The classical answer is maximum likelihood, but the marginal likelihood requires integrating out the latent variables — an intractable integral unless the model happens to be conjugate. Expectation-Maximization lives in the middle ground. It is not the gradient-based, adversarial, or variational inference framework per se; rather, it is the alternating coordinate-ascent algorithm that directly targets the marginal likelihood by descending the complete-data likelihood that one would have written if the missing assignments were known. Dempster, Laird, and Rubin (1977) [arxiv: https://web.mit.edu/6.435/www/Dempster77.pdf] formalized this alternating scheme and proved that each iteration monotonically increases the marginal likelihood, even though we never compute the intractable integral. (A parallel version of the same paper appears at https://www.ece.iastate.edu/~namrata/EE527_Spring08/Dempster77.pdf.) That foundation makes EM the entry point for numerous fields: early speech recognition employed EM to train Gaussian mixture models (GMMs) before neural networks were added; modern semi-supervised NLP uses EM to grow supervised signal with unlabeled text; and variational inference views EM as coordinate ascent on a free energy, which explains incremental or sparse variants that arise in deep learning. This is why EM is still one of the few optimization primitives that a production ML engineer can explain to both their data scientist and their systems team. The mechanism is best understood by starting from the complete-data log likelihood and defining a surrogate objective that alternates between imputing latent variables and maximizing parameters.

## How it works

The core mathematical insight is to pretend the data were fully observable, derive the complete-data log likelihood, and then rewrite the marginal likelihood so that expectations over the missing data appear transparently. Let the observed data be \(\mathcal{X} = \{x^{(n)}\}_{n=1}^N\), the latent assignments be \(Z = \{z^{(n)}\}_{n=1}^N\), and the parameters be \(\theta\). The marginal log likelihood is

\[
\log p(\mathcal{X} \mid \theta) = \log \sum_Z p(\mathcal{X}, Z \mid \theta),
\]

where \(\mathcal{X}\) is the observed dataset, \(Z\) is the missing label configuration, and \(\theta\) parameterizes the joint model.

Direct maximization is hard because the sum over \(Z\) ties all latent assignments together. EM avoids summing by introducing a variational distribution \(q(Z)\) and now writing

\[
\log p(\mathcal{X} \mid \theta) = \mathbb{E}_{q}[\log p(\mathcal{X}, Z \mid \theta)] - \mathbb{E}_{q}[\log q(Z)] + \mathrm{KL}(q(Z) \,\|\, p(Z \mid \mathcal{X}, \theta)).
\]

In this decomposition, the first term is the complete-data log likelihood averaged under \(q\), the second is the entropy of \(q\), and the third is the Kullback-Leibler divergence between \(q\) and the true posterior \(p(Z \mid \mathcal{X}, \theta)\). This identity shows that maximizing the marginal likelihood is equivalent to maximizing the evidence lower bound (ELBO)

\[
\mathcal{L}(q, \theta) = \mathbb{E}_{q}[\log p(\mathcal{X}, Z \mid \theta)] + \mathcal{H}(q),
\]

where \(\mathcal{H}(q)\) is the entropy of \(q\). (Neal and Hinton (1998) [http://www.cs.toronto.edu/~hinton/absps/emk.pdf] reinterpreted EM precisely as coordinate ascent on this free-energy, justifying the convergence guarantees even when you update \(q\) or \(\theta\) incrementally or sparsely.) Value of EM comes from the fact that one can alternate between optimizing \(q\) with \(\theta\) fixed and optimizing \(\theta\) with \(q\) fixed, and each step increases \(\mathcal{L}(q, \theta)\), thereby ascending the marginal likelihood while never computing the posterior normalization constant.

### The E-step: expectation of the complete-data log likelihood

During the E-step we let \(q(Z) = p(Z \mid \mathcal{X}, \theta^{(t)})\) be the exact posterior under the current parameters \(\theta^{(t)}\). The \(Q\)-function is defined as

\[
Q(\theta \mid \theta^{(t)}) = \mathbb{E}_{Z \sim p(Z \mid \mathcal{X}, \theta^{(t)})}[\log p(\mathcal{X}, Z \mid \theta)],
\]

where \(Q(\cdot \mid \theta^{(t)})\) is the expected complete-data log likelihood and the expectation is taken over the true posterior with parameters \(\theta^{(t)}\). In this step, we “fill in” the missing assignments with their posterior probabilities under the current guess. Because the posterior depends on \(\theta^{(t)}\) only, this expectation is tractable whenever \(\log p(\mathcal{X}, Z \mid \theta)\) has a conjugate form with respect to the latent variables.

For a GMM with \(K\) Gaussian components, let \(x^{(n)} \in \mathbb{R}^D\), the component mean \(\boldsymbol{\mu}_k \in \mathbb{R}^D\), covariance \(\mathbf{\Sigma}_k \in \mathbb{R}^{D \times D}\), and mixing proportions \(\pi_k\). The complete-data likelihood is

\[
p(x^{(n)}, z^{(n)} \mid \theta) = \prod_{k=1}^K [\pi_k \mathcal{N}(x^{(n)}; \boldsymbol{\mu}_k, \mathbf{\Sigma}_k)]^{\mathbb{I}[z^{(n)} = k]},
\]

where \(\mathbb{I}[\cdot]\) is the indicator function, and \(\theta = \{\pi_k, \boldsymbol{\mu}_k, \mathbf{\Sigma}_k\}_{k=1}^K\). The posterior responsibility of component \(k\) for data point \(x^{(n)}\) is

\[
r_k^{(n)} = p(z^{(n)} = k \mid x^{(n)}, \theta^{(t)}) = \frac{\pi_k^{(t)} \mathcal{N}(x^{(n)}; \boldsymbol{\mu}_k^{(t)}, \mathbf{\Sigma}_k^{(t)})}{\sum_{j=1}^K \pi_j^{(t)} \mathcal{N}(x^{(n)}; \boldsymbol{\mu}_j^{(t)}, \mathbf{\Sigma}_j^{(t)})}.
\]

This is the only step where the latent variables appear explicitly: EM computes soft assignments (responsibilities) rather than hard labels.

### The M-step: maximizing parameters with soft labels

With responsibilities \(r_k^{(n)}\) fixed, the M-step maximizes \(Q(\theta \mid \theta^{(t)})\) with respect to \(\theta\). Because \(Q\) factorizes over components and the complete-data log likelihood is quadratic in the Gaussian parameters, the updates have closed-forms:

\[
\pi_k^{(t+1)} = \frac{1}{N} \sum_{n=1}^N r_k^{(n)}, \qquad \boldsymbol{\mu}_k^{(t+1)} = \frac{\sum_{n=1}^N r_k^{(n)} x^{(n)}}{\sum_{n=1}^N r_k^{(n)}}, \qquad \mathbf{\Sigma}_k^{(t+1)} = \frac{\sum_{n=1}^N r_k^{(n)}(x^{(n)} - \boldsymbol{\mu}_k^{(t+1)})(x^{(n)} - \boldsymbol{\mu}_k^{(t+1)})^\top}{\sum_{n=1}^N r_k^{(n)}}.
\]

Each sum is weighted by the responsibilities, turning the E-step posterior into soft sufficient statistics. This looks identical to the maximum likelihood estimates for fully labeled data, except that each data point is fractionally assigned across components instead of deterministically assigned.

EM therefore becomes the repeated execution of these two steps: compute soft labels from the current parameters, then recompute parameters from the soft labels. The monotonic increase in \(\mathcal{L}(q, \theta)\) is immediate: holding \(\theta^{(t)}\) fixed, the E-step chooses \(q\) to minimize the KL divergence term, and holding \(q\) fixed, the M-step maximizes the expected complete-data likelihood. This coordinate-ascent view explains why even non-convex families like GMMs or HMMs have the familiar plateau-then-convergence behavior: each iteration can only improve the surrogate objective or leave it unchanged, so the algorithm settles in a stationary point, which was the main convergence claim of Dempster et al. (1977) [https://web.mit.edu/6.435/www/Dempster77.pdf and https://www.ece.iastate.edu/~namrata/EE527_Spring08/Dempster77.pdf].

### From EM to variational inference and extensions

Neal and Hinton’s (1998) view opened the door to a wider class of variational EM algorithms. When \(q(Z)\) is restricted to a tractable family (e.g., mean-field), the E-step becomes variational inference, and the M-step is the usual maximum likelihood update — hence the naming “variational EM.” Because the variational parameters now approximate the posterior instead of computing it exactly, one can derive incremental updates where only one latent variable is updated per step or sparse updates that touch only a subset of the components, as long as they still increase the ELBO by at least a small amount. This insight explains how EM morphs into the modern stochastic variational inference algorithms used in deep generative models: the same alternation principle holds even if the posterior estimate uses gradient-based updates rather than closed-form updates.

Extensions such as Generalized EM (GEM) or Expectation-Conditional Maximization (ECM) relax the requirement that either step be globally optimal; it suffices that the step increases the bound. This flexibility is crucial in production, where the M-step might involve backprop through a neural network rather than closed forms, and the E-step might involve sampling rather than exact enumeration.

### Failure modes and practical heuristics

EM can converge to poor local maxima because the ELBO is non-convex. Initialization is therefore critical: random starts, \(k\)-means seeding, or spectral methods are common tricks to avoid mixing components that should stay separated. Regularization of the covariance matrices (adding a small multiple of the identity) stabilizes the M-step when some responsibilities collapse to zero. Finally, monitoring the ELBO rather than the raw log likelihood helps detect when the algorithm gets stuck in a plateau: because the ELBO lower bounds the marginal likelihood, a sudden drop signals a numerical artefact or underflow in the E-step.

The GMM example also shows why EM is an accessible building block for semi-supervised tasks. Nigam et al. (2000) [https://www.cs.cmu.edu/~dgovinda/pdf/emcat-mlj99.pdf] used EM to blend a small labeled corpus with a far larger unlabeled corpus by treating document topics as latent variables. The labeled documents fix the responsibilities for their known label, while the unlabeled documents receive soft labels that are refined with each EM iteration. This made EM a practical way to bootstrap text classifiers long before large language models existed.

## Where the field is now

The research frontier sits at the junction of EM’s convergence guarantees and modern high-dimensional statistics. Lu and Zhou (2016) [https://arxiv.org/abs/1608.00982] analyze a specific EM trajectory for two-component spherical Gaussian mixtures and show that with careful initialization, the algorithm converges linearly to the global optimum; later work builds on this by characterizing the basin of attraction for more than two components and for mixtures with non-isotropic covariances. A 2023 follow-up (Xu et al. 2023) further quantifies the sample complexity needed for the E-step to approximate the true posterior well enough for the M-step to stay in the contraction region, thereby transforming EM from an empirical heuristic into a procedure with sharp statistical bounds. These papers show that EM is still an active research topic rather than a solved textbook exercise.

The engineering frontier illustrates that EM remains embedded inside deployed systems. Kaldi’s open-source toolkit, which underpins speech systems at Amazon, Microsoft, and Google, still runs EM-style iterations to estimate GMM–HMM acoustic models before the neural acoustic model takes over; see the “Kaldi for Dummies” tutorial for explicit expectation and maximization loops that process tens of thousands of hours of audio. Similarly, edge-oriented speaker diarization systems at IBM Watson use EM to estimate per-speaker Gaussians when bootstrapping labeled data in each call center, enabling sub-second latency deployments of diarization despite limited ground truth. These production stacks demonstrate that even as deep learning dominates, EM continues to provide a stable, interpretable initialization and an explicit lower bound that keeps retraining loops grounded.

## What's still open

- Can we characterize a general initialization strategy for EM that provably lands inside the attraction basin of the global optimum for high-dimensional, highly over-specified mixture models without relying on random restarts?

- Does a variant of EM exist whose E-step is amortized by a learned inference network but whose ELBO guarantees still imply monotonic improvement in the marginal likelihood?

- In production pipelines with billions of examples, how do we multiplex EM updates across shards without breaking the coordinate-ascent guarantee—i.e., can we interleave asynchronous E-steps that pool partial responsibilities and still ensure non-decreasing ELBO?

- When using EM to fuse labeled and unlabeled data, can we design stopping criteria that detect when adding unlabeled data hurts rather than helps, and does that detection admit a statistical test with controlled error rates?

## Where to read next

If you want the probabilistic foundation, → [Variational Inference](variational-inference.md) explains how EM’s ELBO generalizes to the structured mean-field approximations used in VAEs; the engineering counterpart is → [[kaldi-asr]] showing how the same expectation-maximization loops run in Kaldi’s acoustic model training; to understand how this alternating structure compounds into more expressive models, → [Mixture of experts](../../01-ai/concepts/mixture-of-experts.md) lays out how EM-like gating updates coordinate cubes of experts.

## Build it

Implementing a Gaussian Mixture Model from scratch and tracking its EM iterations makes the algorithm concrete and gives intuition for why the bound improves even when the latent labels are unobserved.

**What you're building:** a NumPy + SciPy implementation of EM for a GMM trained on the Old Faithful geyser dataset that renders responsibilities, centroid shifts, and covariance ellipses across iterations in a Colab notebook.

**Why this is valuable:** each iteration fills the missing latent labels with probabilistic responsibilities (E-step) and then recomputes parameters (M-step), so watching the centroids move shows precisely how the algorithm climbs the ELBO; this build also prepares you to scope production clusters or semi-supervised classifiers.

**Stack:**
- **Model:** custom NumPy/SciPy GMM, no pretrained weights (you implement expectations and maximizations yourself).
- **Dataset:** [huggingface.co/datasets/old_faithful](https://huggingface.co/datasets/old_faithful) — 272 data points of geyser eruptions with two continuous features.
- **Framework:** Python 3.11 with NumPy 2.1, SciPy 1.11, Matplotlib 3.8.
- **Compute:** Free Colab T4 (16 GB RAM, ~12 GB GPU memory); each EM run finishes within seconds, even with animation.

**The recipe:**
1. pip install `numpy scipy matplotlib pandas` (Colab already has these; reinstall with `pip install --upgrade numpy scipy matplotlib pandas` to match versions).
2. Load the Old Faithful dataset from HuggingFace, standardize the two features, and shuffle; maintain a DataFrame with eruption durations so you can color-code points.
3. Initialize \(K=3\) Gaussian components with random centroids from the data and identity covariances scaled by the dataset variance; implement the E-step to compute responsibilities via the multivariate Gaussian PDF and normalize across components, then implement the M-step updates for \(\pi_k\), \(\boldsymbol{\mu}_k\), and \(\mathbf{\Sigma}_k\).
4. Run 20 EM iterations, logging the ELBO (use the expected complete-data log likelihood plus entropy) and plotting the centroids and covariance ellipses after each iteration; expect the ELBO curve to rise quickly at first and plateau around iteration 10 with the final responsibilities concentrating.
5. Save the fitted means, covariances, mixing weights, and the ELBO trace so you can compare runs that started from different seeds; this is your artifact that demonstrates EM’s convergence behavior.

**Expected outcome:** a Colab notebook that visualizes the responsibility contours and ELBO curve for a GMM EM fit on Old Faithful, plus saved parameters for downstream clustering or labeling.

- **CS student:** Run the same notebook on an RTX 4070 laptop, increasing \(K\) to 5 and plotting the responsibility heatmap on a 2D grid to see how the additional components carve up the space.

- **Applied engineer:** Extend the notebook by serializing the learned parameters and deploying them as a Flask API that returns posterior probabilities for new data, then benchmark latency on an A10 instance (target p50 < 15 ms with vectorized NumPy inference).

- **Applied researcher:** Hypothesize that shrinking the covariance regularization improves generalization; run an experiment that varies the covariance floor, tracks the ELBO, and measures log likelihood on a held-out 30% of the geyser dataset to confirm where overfitting begins.

- **Frontier researcher:** Probe the open question above by swapping the random initialization for spectral clustering seeds, then measure whether the spectral seed increases the probability of reaching the globally highest ELBO across 50 random restarts; the falsifier is observing that spectral seeding still plateaus in the same local maximum more than 75% of the time.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*