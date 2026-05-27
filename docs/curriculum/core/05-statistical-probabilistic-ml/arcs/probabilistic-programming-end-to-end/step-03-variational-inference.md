---
title: "Variational Inference for Bayesian Neural Networks"
slug: "variational-inference-bnn"
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [jordan, blei]
feeds_de_pillar: []
arc_position:
  arc: probabilistic-programming-end-to-end
  prev: step-02-bayesian-inference
  next: step-04-mcmc
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [bayesian-inference, probabilistic-programming, pyro-basics]
tags: [variational-inference, bayesian-neural-networks, pyro]
updated: 2024-11-20
has_mvb: true
---
> **Arc:** [Probabilistic Programming End To End](../../arcs/probabilistic-programming-end-to-end.md) — Step 3 of 5


How do you know when an auto-piloted vehicle is truly unsure about what it sees, rather than just having a poorly tuned point estimate? The standard answer is “look at the posterior over model weights,” but computing that posterior means integrating a function whose shape you can’t even plot when the network has millions of parameters. You could rely on a sampler, but Monte Carlo over such a high-dimensional space takes days per update and still only gives you noisy traces. What you really need is a scalable procedure that trades off exactness for tractable uncertainty, flags guide collapse when it happens, and plugs straight into the optimizer you already use. By the end of this page you will see how variational inference reframes posterior computation as optimization, how that gradient objective is implemented inside Pyro so it behaves like familiar training loops, why the Evidence Lower Bound (ELBO) unifies the estimation and regularization forces, and how to ship a minimally viable Bayesian neural net that reports both ELBO and a calibration check for the guide.

# The territory

Variational inference sits at the junction of two design constraints: the desire for Bayesian uncertainty, and the requirement that inference run alongside deep models inside probabilistic programming. A MAP estimate selects the single most probable weight configuration, ignoring how broad the posterior really is, while a Monte Carlo sampler estimates the whole posterior at the cost of thousands of expensive likelihood evaluations. Jordan et al. (1999)’s foundational survey showed that the Bayes evidence \(\log p(x)\) can be bounded from below by any tractable distribution \(q(z;\lambda)\), which means we do not need the exact posterior—as long as we can find a distribution whose Evidence Lower Bound (ELBO) is high. This mean-field perspective, which decomposes the original graph into tractable pieces, is what lets us restore posterior-like behavior when a model has millions of coupled parameters. Stochastic Variational Inference (Hoffman et al. 2013a) built on that foundation by showing how these bounds become optimizable via minibatch gradients, letting the variational parameters \(\lambda\) sweep through data without recomputing the full marginal likelihood each time. This page therefore plays the inference step in the [[probabilistic-programming-end-to-end]] arc: read the previous Bayesian inference node for the definition of posteriors and consult the next MCMC step for verification, but here we replace expensive sampling with ELBO-driven optimization that keeps training loops as familiar as any MAP procedure.

The rest of the chapter explains how the ELBO encodes accuracy and regularization, how reparameterization turns sampling into differentiable computation, and how Pyro wires the guide and model together so that optimization is simply an SVI loop. Along the way we connect the ELBO’s decomposition to the diagnostics you will log and the Dirichlet-normal benchmark you can reproduce.

## How it works

### Translating inference into optimization

When \(p(z\mid x)\) is intractable, the new question becomes: what parameterized distribution \(q(z; \lambda)\) can we choose so that the joint probability \(p(x, z)\) stays high while \(q(z; \lambda)\) stays close to the prior? The ELBO formalizes this intuition:

\[
\mathcal{L}(\lambda) = \mathbb{E}_{q(z;\lambda)}\left[\log p(x, z) - \log q(z;\lambda)\right].
\]

Here \(x\) is the observed dataset, \(z\) are the latent weights in the Bayesian neural network, \(p(x,z)\) is the model-defined joint density, \(q(z;\lambda)\) is the variational guide with parameters \(\lambda\), and \(\mathcal{L}(\lambda)\) is the lower bound on the marginal likelihood \(\log p(x)\); the expectation is taken with respect to the guide. This objective rewards guides that propose latent samples making the data likely while penalizing overly diffuse guides through the entropy term.

Rewriting the same equation reveals the connection to KL divergence:

\[
\log p(x) - \mathcal{L}(\lambda) = \mathrm{KL}(q(z;\lambda) \,\|\, p(z\mid x)).
\]

The left-hand side contains the intractable \(\log p(x)\) but the right-hand side is KL divergence between the guide and the true posterior. Since the KL is non-negative, \(\mathcal{L}(\lambda)\) really is a lower bound, and maximizing \(\mathcal{L}\) minimizes the divergence—without ever evaluating \(p(z\mid x)\) directly. This reframing, “optimize the guide instead of integrate the posterior,” is why the ELBO makes variational inference practical even when MAP and Monte Carlo cannot.

Because regression models factorize across datapoints, the ELBO also decomposes:

\[
\mathcal{L}(\lambda) = \sum_{i=1}^N \mathbb{E}_{q(z;\lambda)}\left[\log p(x_i \mid z)\right] - \mathrm{KL}(q(z;\lambda) \,\|\, p(z)),
\]

where \(N\) is the total number of observations, \(p(x_i \mid z)\) is the likelihood for the \(i\)th datum, and \(p(z)\) is the prior. The first term encourages fitting the likelihood, while the second term prevents the guide from straying too far from the prior, behaving like a regularizer. This decomposition is the mathematical anchor we return to when interpreting ELBO diagnostics and when calibrating KL-weight schedules later.

These equations collectively turn the inference problem into standard optimization: we can compute gradients of \(\mathcal{L}(\lambda)\) with respect to the guide parameters and run Adam.

### Stochastic gradients for ELBO optimization

The guide \(q(z;\lambda)\), often chosen as mean-field Gaussian over each weight, allows us to sample quickly but still requires careful gradient estimation because the expectation in \(\mathcal{L}(\lambda)\) depends on the guide parameters. Kingma & Welling (2014) introduced the reparameterization trick to address this: express each sample \(z\) as a deterministic function of \(\lambda\) and noise \(\epsilon\), so the dependency is explicit and differentiable,

\[
z = \mu_\lambda + \sigma_\lambda \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).
\]

This equation means that the guide outputs \(\mu_\lambda\) and \(\sigma_\lambda\), \(\epsilon\) is independent standard Normal noise, and the samples \(z\) are differentiable transformations of \(\lambda\). The expectation in the ELBO now becomes \(\mathbb{E}_\epsilon[\log p(x,z(\epsilon)) - \log q(z(\epsilon);\lambda)]\), allowing autodifferentiation to propagate through \(\mu_\lambda\) and \(\sigma_\lambda\). Kingma & Welling (2014) describe this trick in detail [https://www.arxiv.org/pdf/1601.00670v4], and it is the mechanism that makes SVI a “black-box” gradient estimator.

With large datasets we cannot evaluate the full \(\mathcal{L}(\lambda)\) every epoch, but Hoffmann et al. (2013) showed that we can use minibatches because the ELBO sums over datapoints. Their stochastic variational inference (SVI) objective is

\[
\mathcal{L}_\text{mini}(\lambda) = \frac{N}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \mathbb{E}_{q(z;\lambda)}[\log p(x_i, z) - \log q(z;\lambda)],
\]

where \(|\mathcal{B}|\) is the minibatch size, and the scaling factor \(N/|\mathcal{B}|\) corrects for using only part of the dataset. Because each minibatch shares the same guide \(q(z;\lambda)\), the gradients of \(\mathcal{L}_\text{mini}\) remain unbiased and tractable for Adam, and natural-gradient preconditioning can accelerate convergence. Hoffman et al. (2013) also show how to combine these gradients with adaptive learning rates for hierarchical models [https://www.cs.columbia.edu/~blei/papers/HoffmanBleiWangPaisley2013a.pdf], and the analytical natural-gradient updates arise from the same optimization perspective (also detailed in [http://www.cs.columbia.edu/~blei/papers/HoffmanBleiWangPaisley2013.pdf]). This stochastic framing is why the guide parameters can be updated batch by batch without ever computing the full posterior.

### Encoding the prior, likelihood, and guide in Pyro

Inside Pyro the model is implemented with `pyro.sample` statements for each layer’s weights and biases, all drawn from standard Normal priors. The likelihood is a Normal distribution around the network output with fixed scale \(\sigma_\text{obs}\); the guide mirrors the model structure with `pyro.param` nodes for each \(\mu\) and unconstrained `\theta \equiv \log \sigma\). The KL divergence between each Gaussian posterior and its prior can be computed analytically:

\[
\mathrm{KL}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0,1)) = \frac{1}{2} \left(\mu^2 + \sigma^2 - 1 - \log \sigma^2 \right),
\]

where \(\mu\) and \(\sigma\) are the guide parameters for a single weight. Summing this expression across all parameters yields the total KL penalty that appears in the ELBO decomposition above. This analytical KL gives the diagnostics—if the KL stays near zero, the guide is collapsing toward the prior; if it explodes, the guide is drifting away without improving likelihood.

Pyro’s `Trace_ELBO` loss already estimates the expectation over \(q\) via the reparameterized guide, so the training loop looks familiar: iterate over batches, call `svi.step(batch_x, batch_y)`, and log the ELBO and the analytical KL computed from `pyro.param`. Because the ELBO gradient flows through the guide’s \(\mu\) and \(\sigma\) via the reparameterization equation, the optimization runs with the same toolkit as MAP training, but the optimizer now updates the guide parameters instead of bare weights.

### Connecting to MAP and why the shift matters

MAP maximizes \(\log p(x,z)\) with respect to a single point \(z^*\). Variational inference maximizes \(\mathcal{L}(\lambda)\) with respect to the distribution parameters \(\lambda\). Rewriting the ELBO as

\[
\mathcal{L}(\lambda) = \mathbb{E}_{q(z;\lambda)}[\log p(x\mid z)] - \mathrm{KL}(q(z;\lambda) \,\|\, p(z)),
\]

the first term is the expected log likelihood (which still drives accuracy) and the second term is the KL to the prior (which prevents overconfidence). In the S-curve regression example, the expected likelihood wants to explain the curved data, while the KL term forces the guide to keep variance when the data is ambiguous. This balance gives a diagnostic handle: track the ELBO and KL, and if the former improves while the latter collapses, the guide has overfit; if neither moves, the optimizer may need a schedule change.

## Where the field is now

The ELBO optimization framework keeps evolving, but recent work has been explicit about shrinking the KL gap we control. Rezende & Mohamed (2015) introduced normalizing flows to transform a simple guide into one capable of modeling heavy tails, showing that the variational KL goes down without requiring expensive sampling because the flow layers appear inside the KL term of the ELBO [https://arxiv.org/abs/1505.05770]. Kucukelbir et al. (2017) automated that process with ADVI, generating mean-field guides from the model specification and differentiating the ELBO with automatic differentiation [https://arxiv.org/abs/1603.00788]; their implementation of ELBO automatic differentiation is now the foundation of Pyro’s and NumPyro’s ADVI tooling. Tran et al. (2015) explored structured variational families that capture correlations across layers, demonstrating empirically that ELBOs computed with richer guides improve held-out likelihood on deep generative models [https://arxiv.org/abs/1505.03925]. Each of these contributions can be read as extending the ELBO decomposition: flows increase the expressiveness of \(q(z;\lambda)\), ADVI automates differentiation of both terms, and structured families reshape the KL term to include inter-layer dependencies.

On the engineering front, Pyro 2.0’s rollout centralized guide modularity, GPU compilation, and tighter `torch.compile` integration so that ELBO training stays competitive on datasets with millions of records [https://pyro.ai/blog/pyro-2-0/]. NumPyro’s JAX backend continues to serve as a lightweight engine for SVI on accelerators with minimal boilerplate, and its composable primitives let researchers prototype ELBO objectives that mix Gaussian and Student’s-t priors. Production teams such as Uber AI (Fraccaro et al. 2016) deploy variational models for time series forecasting, logging ELBO windows as part of their monitoring stack to ensure the approximation is not drifting and to calibrate predictive intervals [https://arxiv.org/abs/1607.03074]. These efforts keep the ELBO optimization story honest: its stochastic gradient form stays tractable, but observability around KL and likelihood is now standard in deployable systems.

Where this concept appears after this page: the inference slot in the [[probabilistic-programming-end-to-end]] arc (from [[step-02-bayesian-inference]] to [[step-04-mcmc]]) and in the [[bayesian-neural-networks]] hub, where the optimizer-level perspective on uncertainty is essential before touching sampling-based verification.

## What's still open

Stochastic ELBO optimization relies on the balance between expected likelihood and KL; that is the backbone of every open question. One precise question is whether the trade-off weight \(\beta\) on the KL term (so that the objective becomes \(\mathbb{E}_q[\log p(x\mid z)] - \beta\,\mathrm{KL}(q\|p))\)) can be learned online so that the guide begins mean-field and only increases expressiveness when validation ELBO variance shrinks. Another frontier is designing minibatch schedules that monitor the variance of \(\mathcal{L}_\text{mini}(\lambda)\) and grow batch size only when the stochastic gradient noise triggers a statistically significant drift in the ELBO estimate, instead of fixing the batch size in advance. Finally, mixing variational inference with deep ensembles invites the question of how to synchronize multiple ELBO traces: can we coordinate the KL terms across ensemble members so that they jointly cover multimodal posterior components without doubling compute, while still treating the ELBO decomposition from earlier as the guiding objective?

## Where to read next

If you want to see how this inference node fits into deployable applications, → [[probabilistic-programming]] recounts how Pyro and NumPyro push ELBO-based models beyond training notebooks; if you want the theoretical opposite to these stochastic optimizers, → [[mean-field-variational-inference]] walks through the convex duality that underpins Jordan et al. (1999); if you want to validate a guide instead of optimize it, → [[mcmc-in-probabilistic-programming]] shows how to compare posterior samples against variational diagnostics; this page also anchors the inference slot of the [[probabilistic-programming-end-to-end]] arc so you can see where you came from and where to go next.

## Build it

**What you’re building:** A Pyro-based Bayesian neural network trained on the Hugging Face [uci/housing](https://huggingface.co/datasets/uci/housing) regression dataset that reports ELBO, analytic KL, and predictive log-likelihood while providing diagnostics that detect guide collapse.

**Why this is valuable:** The checkpoint and diagnostics prove whether ELBO optimization actually delivers meaningful uncertainty for a realistic regression problem instead of just converging to MAP; the harness also provides the monitoring hooks you need before deploying to a production stack.

**Stack:**
- **Model:** Custom three-layer MLP (input → 128 → 64 → 1) with Normal priors on each weight/bias, so the architecture is explicit and understandable without referencing a confusing placeholder model ID.
- **Dataset:** Hugging Face dataset [uci/housing](https://huggingface.co/datasets/uci/housing) holding 506 rows of real-valued features and a standard regression target.
- **Framework:** PyTorch 2.1 + Pyro 2.0 with `torch.compile` enabled for compilation-friendly ELBO loops.
- **Compute:** One RTX 4090 or A100 (24 GB) runs the 40-epoch recipe in under two hours; fallback to Colab T4 (16 GB) in ~4 hours with gradient accumulation.

**The recipe:**
1. Install `pip install pyro-ppl==2.0.0 torch torchvision torchaudio datasets matplotlib` and load the dataset with `datasets.load_dataset("uci/housing")`, fitting a `StandardScaler` on the training split and applying the same transformation to validation and test splits to avoid leakage.
2. Define the Pyro model with `pyro.sample` nodes for each layer weight and bias drawn from `torch.distributions.Normal(0., 1.)`, and a Normal likelihood `torch.distributions.Normal(f(x), 0.5)` so \(\sigma_\text{obs}=0.5\) keeps the ELBO finite.
3. Build a mean-field guide registering `mu = pyro.param(...); log_sigma = pyro.param(..., constraint=constraints.real)` for each parameter, compute `sigma = torch.nn.functional.softplus(log_sigma).clamp(min=1e-3)`, and sample via `pyro.sample(name, dist.Normal(mu, sigma))`.
4. Instantiate `SVI(model, guide, pyro.optim.Adam({"lr": 1e-3}), loss=Trace_ELBO())`; evaluate the initial ELBO with `svi.evaluate_loss(train_x, train_y)` to verify the graph.
5. For each batch compute `elbo = svi.step(batch_x, batch_y)` and `kl = sum(torch.distributions.kl_divergence(torch.distributions.Normal(pyro.param(f"{name}_mu"), torch.nn.functional.softplus(pyro.param(f"{name}_log_sigma")).clamp(min=1e-3)), torch.distributions.Normal(0., 1.)).sum() for name in param_names)` to monitor the KL contribution explicitly, then log both scalars per epoch.
6. After training draw 200 predictive samples via `pyro.infer.Predictive(model, guide=guide, num_samples=200)` and compute the per-example log-likelihood on the test set using the Gaussian log-probability: \(\log p(y_i\mid f_i) = -\frac{1}{2} \log(2\pi\sigma_\text{obs}^2) - \frac{(y_i - f_i)^2}{2\sigma_\text{obs}^2}\); report the mean across test examples to compare with the MAP baseline.
7. Fit a MAP model with the same architecture trained purely with MSE for 40 epochs; convert the aggregate test MSE \( \mathrm{MSE} = \frac{1}{n} \sum_i (y_i - f_i)^2\) to log-likelihood via \(\log p(y \mid f) = -\frac{n}{2}\log(2\pi\sigma_\text{obs}^2) - \frac{1}{2\sigma_\text{obs}^2} \sum_i (y_i - f_i)^2\), reporting the per-example mean log-likelihood when dividing by \(n\).
8. Compare the variational predictive mean log-likelihood to the MAP baseline: success means the variational value is no more than 0.25 nats lower than the MAP mean log-likelihood, ensuring a consistent absolute difference rather than mixing percent and units.
9. Save the guide via `pyro.get_param_store().save("bnn_variational.pt")` and the point-model state dict via `torch.save(model.state_dict(), "bnn_model.pt")` for later reuse.
10. Plot the predictive mean \( \pm \) one standard deviation against the true targets and persist the KL trace per epoch to show that the KL remains positive and reacts to curvature rather than collapsing to zero.

**Expected outcome:** A checkpoint whose ELBO falls between \(-1150\) and \(-1080\), a KL trajectory that decreases but does not vanish, a test log-likelihood within 0.25 nats of the MAP baseline, and saved predictive uncertainty plots plus checkpoints.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Deploy this Pyro model with TorchServe, exposing ELBO and KL as Prometheus metrics and setting an alert when the 15-minute ELBO average drops more than 0.2 nats below baseline, while keeping median latency under 500 ms on an RTX 4090.
- **Research engineer:** Reproduce Table 2 from Hoffman et al. (2013a)’s “Stochastic Variational Inference” by implementing a minibatch natural-gradient SVI for a Dirichlet-multinomial model in Pyro and hitting held-out perplexity within ±3 points on the same dataset, with code instrumented to log both ELBO and KL per minibatch.
- **Applied researcher:** Test the hypothesis that KL annealing improves predictive calibration by training two runs (with and without an exponential KL weight schedule) on the housing dataset, plotting calibration error curves, and accepting the hypothesis only if annealing reduces calibration error by at least 10%.

**What can you build next:** Extend this harness by swapping the mean-field guide for a low-rank plus diagonal covariance (see Rezende & Mohamed 2015) and observe whether the ELBO/KL trade-off from this recipe tightens without tuning the optimizer.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*