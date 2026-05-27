---
title: Expectation-Maximization Algorithm
slug: em-algorithm
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [dempster,laird,rubin,neal,hinton,bilmes]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [latent-variable-models, variational-inference, gaussian-mixture-models, probability-foundations]
tags: [em-algorithm, latent-variable-models, gaussian-mixture-models, variational-inference, maximum-likelihood, unsupervised-learning]
updated: 2024-10-02
has_mvb: true
---

# Expectation-Maximization Algorithm

Imagine trying to split a crowd of festival-goers into male and female height distributions when you only have their heights. You cannot compute the average height for each group without knowing who belongs where, and you cannot assign people to groups without knowing the averages. That chicken-and-egg dilemma repeats itself any time a model posits hidden variables—cluster assignments, hidden states, speaker identities—but the training data only records partial observations. The Expectation-Maximization algorithm solves that paradox by repeatedly pretending it knew the missing pieces well enough to compute a nice objective, tightening that pretend objective until the true evidence it bounds stops improving. By the end of this page you will understand how EM not only rewrites an intractable marginal likelihood as a monotonic lower bound but also how that bound can be updated with simple coordinate steps, how modern applications reuse those same updates, and how to implement a minimal Gaussian mixture EM loop yourself.

## The territory

The statistical territory EM inhabits is the gap between models that offer clean probabilistic interpretations—mixture models, hidden Markov models, latent Dirichlet allocation—and datasets where one or more of the random variables is never observed. When the latent variable is discrete (cluster identity, hidden state index), the marginal likelihood of the observed data becomes a sum over exponentially many possibilities, so naive maximum likelihood is impossible to compute directly. EM answers “what parameter would have made the data most probable if we knew the hidden state?” by alternating between estimating the hidden states given the current parameters (the E-step) and maximizing the parameters given those estimated states (the M-step). This alternating structure is not heuristic; it constructs a tight variational lower bound on the log-likelihood and guarantees that each iteration increases or at least does not decrease the true likelihood. The territory is therefore shared with variational inference, coordinate ascent, and stochastic optimization—EM is the original coordinate-ascent-on-a-bound algorithm for latent-variable models. How does this lower bound get built, and what exactly is being optimized at each step?

## How it works

The starting point is the incomplete-data log-likelihood. Let \(X = \{x^{(1)}, \dots, x^{(N)}\}\) be the observed samples and \(Z = \{z^{(1)}, \dots, z^{(N)}\}\) the corresponding latent variables (e.g., cluster indicators), and let \(\theta\) parametrise the joint distribution \(p(x, z \mid \theta)\). The marginal likelihood is

\[
\log p(X \mid \theta) = \sum_{n=1}^{N} \log \sum_{z^{(n)}} p(x^{(n)}, z^{(n)} \mid \theta)
\]

where \(N\) is the dataset size, \(x^{(n)}\) is the \(n\)-th observed sample, and the inner sum marginalises over the latent variable. That inner sum makes the log-likelihood non-linear and hard to differentiate. The key idea of EM is to introduce an auxiliary distribution \(q^{(n)}(z)\) over the latent variable for each data point and rewrite the log-likelihood as a sum of a lower bound plus a KL divergence:

\[
\log p(X \mid \theta) = \sum_{n=1}^{N} \left[\mathcal{L}(q^{(n)}, \theta) + \mathrm{KL}\left(q^{(n)}(z) \, \| \, p(z \mid x^{(n)}, \theta)\right)\right]
\]

where \(\mathcal{L}(q^{(n)}, \theta) = \sum_{z^{(n)}} q^{(n)}(z^{(n)}) \log \frac{p(x^{(n)}, z^{(n)} \mid \theta)}{q^{(n)}(z^{(n)})}\) is the variational lower bound, and \(\mathrm{KL}(q\,\|\,p)\) is the Kullback-Leibler divergence between the auxiliary distribution and the true posterior. Because the KL divergence is non-negative, \(\mathcal{L}(q^{(n)}, \theta)\) is always a lower bound on \(\log p(x^{(n)} \mid \theta)\); it becomes tight when \(q^{(n)}(z)\) matches the posterior \(p(z \mid x^{(n)}, \theta)\). Each EM iteration alternates between making this bound tight (E-step) and improving the parameters to raise the bound (M-step).

### The E-step: tightening the bound

In the E-step, the current parameter estimate \(\theta^{(t)}\) is used to compute posterior responsibilities

\[
q^{(n)}(z) \leftarrow p(z \mid x^{(n)}, \theta^{(t)})
\]

where \(p(z \mid x^{(n)}, \theta)\) denotes the posterior probability of latent configuration \(z\) given the data point \(x^{(n)}\) and current parameters. Computing this posterior is tractable in many models: in Gaussian mixture models it reduces to Bayes’ rule—each component’s likelihood times its mixing proportion normalized over components—and in hidden Markov models it is the forward–backward probability. The E-step therefore closes the KL gap by setting the auxiliary distribution equal to the true posterior, which makes \(\mathcal{L}(q^{(n)}, \theta)\) equal to the log-likelihood when the parameters do not change. Because each data point’s contribution is independent, the responsibilities can be computed in parallel across the dataset without changing the algorithmic structure.

### The M-step: maximizing the bound

After the E-step, the auxiliary distributions are fixed, so \(\mathcal{L}(q, \theta)\) becomes a surrogate objective that is linear in \(\log p(x, z \mid \theta)\). The M-step maximises this surrogate with respect to \(\theta\), namely

\[
\theta^{(t+1)} \leftarrow \arg\max_{\theta} \sum_{n=1}^{N} \sum_{z^{(n)}} q^{(n)}(z^{(n)}) \log p(x^{(n)}, z^{(n)} \mid \theta)
\]

where \(q^{(n)}(z^{(n)})\) are the responsibilities computed in the E-step. This sum is the expected complete-data log-likelihood under the current posterior; in many models the expectation creates sufficient statistics that have closed-form updates. In a GMM with diagonal covariances, for example, the new mixture weight is the average responsibility, the mean is the responsibility-weighted average of the data assigned to that component, and the covariance is the responsibilities-weighted empirical covariance. Because this expected log-likelihood is a lower bound on the true log-likelihood, and because the M-step maximises it exactly (or increases it in the case of constrained or numerical optimisation), Dempster, Laird, and Rubin (1977) proved that \(\log p(X \mid \theta^{(t+1)}) \geq \log p(X \mid \theta^{(t)})\), guaranteeing monotonic ascent with each iteration. That ascent cannot jump to the global optimum in the presence of multiple modes, but the algorithm cannot decrease the likelihood, which makes EM attractive for problems where a local maximum is sufficient.

### Variational and coordinate-ascent view

Neal and Hinton (1998) recast EM as coordinate ascent on the variational free energy, \(\mathcal{F}(q, \theta)\), which is equivalent to \(\mathcal{L}(q, \theta)\) up to a constant. Their view clarifies why EM admits sparse, incremental, and stochastic variants: each E or M step optimises a different block of variables while holding the other fixed, just like coordinate ascent on a convex (or quasi-convex) surrogate. Hinton’s ALGORITHM (1998) emphasises that each coordinate update reduces the variational free energy, documenting how an M-step that only partially maximises the expected log-likelihood still produces valid updates as long as it increases the surrogate. That relaxation is what lets practitioners tidy EM for modern neural latent-variable models, where an exact M-step is impossible. Variational autoencoders replace the M-step with gradient descent, treating the mean-field parameters as differentiable surrogates, but the original descent-ascent structure remains EM in spirit.

### Practical EM considerations

Even with convergence guarantees, practical EM requires care. Initialization strongly influences the mode to which EM converges, so practitioners use heuristics such as K-means clustering or spectral decomposition to seed \(\theta^{(0)}\). Dempster et al. (1977) noted this sensitivity as part of the algorithm’s definition: while each iteration increases likelihood, the landscape can have multiple basins; the algorithm only finds the basin into which the initialization falls. Solutions include running EM multiple times with different seeds and picking the fit with the highest final likelihood, or injecting regularisation into the M-step to penalize unlikely parameter values. The E-step has a natural interpretation as soft labeling, so in semi-supervised settings where some labels are observed and others are missing, the observed ones fix their responsibilities and the E-step only fills the rest. For example, Nigam et al.’s Text Classification from Labeled and Unlabeled Data (1999) trains a naive Bayes classifier by treating unlabeled documents as having fractional labels and updating those fractions via EM. That application demonstrates EM’s ability to blend supervisory signals without altering the core algorithm.

Another practical challenge is computational efficiency on large datasets. Batch EM requires a pass through all data for each iteration, but since the E-step only produces sufficient statistics, these can be accumulated incrementally. Neal and Hinton’s coordinate-ascent view legitimizes stochastic and online EM variants because each mini-batch can be treated as an approximate M-step that increases the surrogate expected log-likelihood. Modern frameworks implement EM-like loops by stacking neural networks for the E-step (amortized inference) and gradient-based optimizers for the M-step, but the essential expectation-maximization skeleton remains.

## Where the field is now

After the foundational unification of EM by Dempster et al. (1977) [https://web.mit.edu/6.435/www/Dempster77.pdf](https://web.mit.edu/6.435/www/Dempster77.pdf) [https://www.ece.iastate.edu/~namrata/EE527_Spring08/Dempster77.pdf](https://www.ece.iastate.edu/~namrata/EE527_Spring08/Dempster77.pdf), two research threads dominate the current landscape: turning EM into a coordinate ascent tool for variational inference, and scaling stochastic EM to industrial-size datasets. The variational perspective was sharpened by Neal and Hinton (1998) [http://www.cs.toronto.edu/~radford/ftp/em.pdf](http://www.cs.toronto.edu/~radford/ftp/em.pdf), which explains why EM supports incremental and sparse updates; those innovations are still the research frontier when combining EM with deep latent-variable models such as VAEs or implicit generative models. Another influential lens is provided by Hinton’s ALGORITHM paper (1998) [http://www.cs.toronto.edu/~hinton/absps/emk.pdf](http://www.cs.toronto.edu/~hinton/absps/emk.pdf), which derives variants where one coordinate step (say the M-step) only partially optimizes, leaving the algorithm flexible enough for gradient-based deep architectures. These papers form the backbone of current work seeking to replace analytical M-step updates with differentiable networks that can still guarantee monotonic likelihood increases.

On the applied side, Nigam et al. (1999) [https://www.cs.cmu.edu/~dgovinda/pdf/emcat-mlj99.pdf](https://www.cs.cmu.edu/~dgovinda/pdf/emcat-mlj99.pdf) demonstrated EM’s value for semi-supervised text classification by gradually incorporating unlabeled documents as fractional counts; that same idea still powers production data augmentation and weak supervision pipelines where estimating reliable soft labels is essential. Beyond text, Google’s Switch Transformers (Fedus et al. 2021) [https://arxiv.org/abs/2101.03961](https://arxiv.org/abs/2101.03961) represent the current engineering frontier: their router behaves like an expectation step that assigns each token to a small subset of experts, and the experts are trained (via backpropagation) to maximize a weighted likelihood whose weights come from the router. The system’s throughput improvements—up to 10× savings in FLOPs for trillion-parameter models—are possible because the router’s soft assignments act like responsibilities in EM, making the gating decision differentiable yet sparse. That architecture is deployed within Google-scale language-model training to keep inference and training costs manageable while retaining the probabilistic semantics that EM theory supplies. Meanwhile, research frontiers seek to extend this paradigm to streaming, federated, and unsupervised mixture-of-experts problems where the latent assignment itself must be inferred under privacy or resource constraints.

## What's still open

- Can we guarantee a global-optimum certificate for EM in high-dimensional, non-convex latent-variable models without relying on repeated random restarts or ensembles? If not, what additional structure (such as over-parameterization or sparsity) needs to be enforced to make EM globally convergent?
- How can we combine amortized inference with EM so that the E-step can be executed in constant time per data point but still provides a bound tight enough to control likelihood improvement, especially for discrete latent variables where reparameterization is impossible?
- In large mixture-of-experts systems, can we design routers whose soft assignment probabilities admit closed-form expected log-likelihood updates, enabling an EM-like convergence proof for the sparse, differentiable gating layers in models such as Switch Transformers?

## Where to read next

If you want to understand the probabilistic core that EM alternates over, → [[variational-inference]] explains the coordinate-ascent view in full detail; for the specific models EM is most often applied to, → [[gaussian-mixture-models]] shows how the responsibilities and parameter updates look when the latent variable indexes Gaussians; the engineering counterpart is → [[mixture-of-experts]] where EM-style assignment dictates production-scale router behaviour.

## Build it

Implementing EM from scratch on a two-component Gaussian mixture makes the alternating E- and M-steps visible and lets you control every detail of the responsibilities and updates.

**What you're building:** A NumPy-only EM loop that clusters the Old Faithful geyser dataset into two Gaussians while animating the decision boundary and parameter updates.

**Why this is valuable:** The build forces you to compute the posterior responsibilities, form the expected complete-data log-likelihood, and maximize it analytically, so you see how each step constructs the variational lower bound and why the log-likelihood never decreases.

**Stack:**
- **Model:** none (you implement the Gaussian mixture yourself; no pretrained checkpoint)
- **Dataset:** [uciml/old-faithful](https://huggingface.co/datasets/uciml/old-faithful) — 272 observations of eruption waiting times
- **Framework:** NumPy 2.0 + Matplotlib 3.9 in a Colab notebook
- **Compute:** Free Colab T4 / CPU equivalent, training per iteration < 2s

**The recipe:**
1. `pip install numpy matplotlib huggingface-datasets` and load `uciml/old-faithful`; normalise the two columns to zero mean and unit variance so the Gaussians stay well-conditioned.
2. Initialise two mean vectors randomly inside the data range, set covariances to identity, and set mixing weights to 0.5; compute responsibilities with Bayes’ rule using the current parameters.
3. Run 20 EM iterations, each time recalculating responsibilities (E-step) and updating means, covariances, and mixing weights by maximizing the expected complete-data log-likelihood (M-step); plot the likelihood curve to confirm monotonic growth.
4. Evaluate by overlaying the soft decision boundary (per-pixel label probability) and the final Gaussian ellipses; the expected metric is that the log-likelihood curve is monotonically increasing and that the two component means align with the tallest and shortest eruption durations.
5. What you now have is a complete EM implementation that you can use as a baseline for warm-starting more complex models or for studying how bad initialization leads to inferior local optima.

**Expected outcome:** A Colab notebook showing the Old Faithful data, iterative updates of the Gaussian parameters, and a monotonic log-likelihood curve.

- **CS student:** Run the same notebook on an RTX 4060 but add a 30-iteration animation of the contours for each iteration; compare the log-likelihood curves from three different random initialisations to understand sensitivity.
- **Applied engineer:** Quantize the NumPy loops into Numba-compiled functions or port them to a minimal PyTorch script, then deploy a microservice that scores new eruption durations in <5 ms and logs the likelihood so you can monitor convergence drift.
- **Applied researcher:** Replace the Gaussian kernels with Student-t components and test the hypothesis that heavier tails prevent singular covariance collapse by comparing final likelihoods and cluster assignments on perturbed data.
- **Frontier researcher:** Probe the open question about global convergence by tracking the basin of attraction for each initialization; specifically, record how often EM converges to the same solution under small perturbations to the initial means and whether adding noise to the E-step responsibilities helps escape poor basins.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*