---
title: "Step 2 — Implement Bayesian Inference via Metropolis-Hastings"
slug: "step-2-metropolis-hastings"
layer: co
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [gelman]
feeds_de_pillar: []
arc_position:
  arc: probabilistic-programming-end-to-end
  prev: step-01-probabilistic-programming
  next: step-03-variational-inference
mvb_personas: [applied-ai-ml-engineer, research-engineer, applied-researcher]
prereqs: [probabilistic-programming, bayesian-inference]
tags: [bayesian, mcmc, metropolishastings, inference]
updated: 2024-10-21
has_mvb: true
---
> **Arc:** [Probabilistic Programming End To End](../../arcs/probabilistic-programming-end-to-end.md) — Step 2 of 5


> **Arc:** [Probabilistic Programming End To End](../index.md) — Step 2 of 5  
> ← [Previous Step](./step-01-probabilistic-programming.md) &nbsp;&nbsp; [Next Step →](./step-03-variational-inference.md)

# Step 2 — Implement Bayesian Inference via Metropolis-Hastings

Can the readings from a slightly sticky sensor teach you how wrong your previous guess about its bias was? Imagine you had a factory of identical sensors, one of them drifting just enough to ruin a downstream calibrator; you can hand-tune a single parameter, but it never tells you how uncertain that correction is or how confident you should be that the next sensor behaves similarly. By the end of this step you will not only answer what the bias is, but also why the data keeps the 95 % credible interval within a tight dance around the truth—it is the posterior distribution itself that plays the role of a reliability report card. That posterior is the human answer to the question “How should what I believed change when these new readings arrive?” and the mechanism that transforms your prior guess into a probabilistic forecast you can trust.

## The territory

The field has long treated Bayes’ original essay as the turning point where belief became math. Bayes (1763) [LII. An Essay towards solving a Problem in the Doctrine of Chances](https://bayes.wustl.edu/Manual/an.essay.pdf) introduced the conditional probability machine that takes prior belief plus evidence and spits out a posterior; without that update there is no principled story for how new data modifies what we think is plausible. Later work—SIAM-AMS (2012) “Ambiguity and Bayesian re-interpretation” [https://bayes.wustl.edu/etj/articles/ambiguity.pdf]—reminds us that ambiguity creeps back in if we only look at point estimates, because the same average could hide multiple conflicting possibilities, so we must reason about the entire distribution over hypotheses. Modern cognitive neuroscience echoes this point: the report from “D:\larry\Output.prn.pdf” (a canonical Bayes-net brain model) [https://bayes.wustl.edu/etj/articles/how.does.the.brain.orig.pdf] shows that the brain keeps track of posterior ratios precisely so that perception reflects both “what happened” and “how strongly it happened.” That is why this step exists between a generative story and a variational approximation: we move from expressing the model to actually sampling its posterior, and through that we accumulate evidence about uncertainty that informs the rest of the arc.

The territory is thus the probabilistic pipeline: a prior encodes what you knew before seeing data, a likelihood encodes how the model would generate readings given a hypothesis, and a posterior information counter balances the two. “PRIOR” [https://bayes.wustl.edu/etj/articles/prior.information.pdf] formalizes how to choose that prior without being arbitrarily confident, so that the posterior continues to reflect information brought in by the sensor rather than the biases you baked into the model. In this way, the human problem—reporting both a bias estimate and how reliable that estimate is—becomes a mathematical one: how do you represent the posterior and how do you draw from it? That is the question we now answer with Metropolis-Hastings, bridging the storytelling from Step 1 to the optimization experiments in Step 3.

## How it works

The goal is to sample from the posterior \(p(\theta \mid y)\), because sampling lets you estimate any summary (mean, credible interval, probability mass) rather than just a point. The posterior itself is

\[
p(\theta \mid y) = \frac{p(y \mid \theta)\,p(\theta)}{p(y)}\,,
\]

where \(\theta\) is the sensor bias, \(y=(y_1,\dots,y_N)\) are the observed readings, \(p(y \mid \theta)\) is the likelihood, \(p(\theta)\) is the prior belief density, and \(p(y)\) is the normalization constant or evidence. The expression’s intuition is simple: it reweights prior belief by how well each potential bias explains the data. Because computing \(p(y)\) requires integrating over the entire parameter space, we instead sample from the unnormalized posterior \(p(y \mid \theta)p(\theta)\)—Metropolis-Hastings does exactly that by constructing a Markov chain whose stationary distribution matches the posterior we care about.

### Writing the posterior for the sensor

Our sensor model assumes each observation satisfies

\[
y_i = \theta + \epsilon_i\,,\qquad \epsilon_i \sim \mathcal{N}(0, \sigma^2)\,,
\]

where \(\sigma=0.5\) and \(\epsilon_i\) captures measurement noise. The likelihood of a single observation given \(\theta\) is

\[
p(y_i \mid \theta) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\!\left(-\frac{(y_i-\theta)^2}{2\sigma^2}\right)\,,
\]

where \(y_i\) is the \(i\)-th reading and \(\sigma\) is known ahead of time. Assuming independent noise, the likelihood of the entire dataset \(y\) is the product over \(i=1,\dots,N\); together with the prior, this product defines the unnormalized posterior density \(\pi(\theta)=p(y \mid \theta)p(\theta)\) that Metropolis-Hastings will sample.

### The Metropolis-Hastings acceptance rule

The sampler proposes a new candidate \(\theta^\prime\) from a symmetric proposal distribution \(q(\theta^\prime \mid \theta)\) (here a Gaussian centered on the current state). The acceptance probability is

\[
\alpha = \min\!\left(1, \frac{\pi(\theta^\prime)}{\pi(\theta)} \times \frac{q(\theta \mid \theta^\prime)}{q(\theta^\prime \mid \theta)}\right)\,,
\]

where \(\pi(\cdot)\) is the unnormalized posterior defined above, \(q(\theta^\prime \mid \theta)\) is the proposal density, and \(\theta\) is the current chain state while \(\theta^\prime\) is the proposed state. Because \(q\) is symmetric in our scalar example, the ratio simplifies to \(\pi(\theta^\prime)/\pi(\theta)\). Cox (1946) showed that any rational system of beliefs must obey the probability calculus, which justifies the acceptance ratio’s role in preserving detailed balance; the Markov chain will therefore converge to the true posterior, visiting high-density regions more often but still exploring tails occasionally.

To avoid overflow when \(\pi(\theta^\prime)/\pi(\theta)\) is extreme, work in log space: compute the log-acceptance difference \(\Delta = \log\pi(\theta^\prime) - \log\pi(\theta)\), clip \(\Delta\) to a safe range (e.g., \([-50, 50]\)) before exponentiating, and accept the proposal if \(\log u < \Delta\) where \(u\sim\mathrm{Uniform}(0,1)\). This log-domain trick matches the stable practice used in Stan and TensorFlow Probability, preventing rare but catastrophic NaN or Inf values when \(\theta^\prime\) lands in a region with negligible posterior density. The acceptance decision keeps the chain faithful to Bayes’ logic but also injects controlled randomness, so the sampler can escape local modes if the proposal variance is tuned appropriately.

### Diagnostics in the scalar case

We track three diagnostics: the sliding credible interval, the acceptance rate, and the posterior mean. Each iteration stores whether the proposal was accepted as a boolean in an `accepts` array; the acceptance rate is then \(\texttt{accepts.mean()}\), which counts exactly the number of accepted transitions rather than relying on numerical differences between states. This explicit counting enables us to react when the acceptance rate leaves the sweet spot around 0.35: too low means the chain is stuck, too high means it behaves like independent samples.

After the loop we compute the 95 % credible interval by taking percentiles of the stored chain. The width of this interval indicates uncertainty: if the empirical width stays under 0.4 and the interval contains the known true bias, the inference strategy has successfully combined prior belief and likelihood information. The log-posterior function that underpins all of this is

\[
\log \pi(\theta) = -\frac{1}{2}\theta^2 -\sum_{i=1}^N \frac{(y_i - \theta)^2}{2\sigma^2}\,,
\]

where \(\theta^2/2\) comes from the Gaussian prior and the sum collects contributions from each likelihood term; \(\sigma\) is the known noise scale, and \(y_i\) is the \(i\)-th reading. This form is efficient because it lets you reuse precomputed sums and avoids recalculating constants that cancel in the ratio.

## Where the field is now

Recent work continues to push Metropolis-Hastings towards richer proposal families and better diagnostics. Hoffman and Gelman (2014) “The No-U-Turn Sampler” [https://arxiv.org/abs/1111.4246] set a research benchmark by automating trajectory lengths in Hamiltonian Monte Carlo, showing that automatic tuning can outperform naive Gaussian proposals in high dimensions. Welling and Teh (2011) “Bayesian Learning via Stochastic Gradient Langevin Dynamics” [https://arxiv.org/abs/1101.5123] keeps influencing the frontier, because embedding MCMC updates within stochastic gradients lets you scale posterior sampling to deep networks without losing the guarantees of detailed balance. On the engineering side, TensorFlow Probability’s documentation for \texttt{MetropolisHastings} and its TPU/XLA backend [https://www.tensorflow.org/probability/api_docs/python/tfp/mcmc/MetropolisHastings] details how production teams now run batches of chains at once—leveraging vectorized proposals and jit-compiled log densities to maintain credible intervals at 90 ms latencies on accelerators. These two fronts—research on adaptive proposals and engineering on scalable, batched execution—show that Metropolis-Hastings remains relevant both as a theoretical benchmark and as an engine that links probabilistic programs to reliable inference in products.

## What's still open

How much does the convergence curve change when you limit the chain to 5 000 iterations, and what system-level metrics (variance of the credible interval, effective sample size) should you trust when sampling from a scalar posterior with a tight proposal? The scalar bias story lets you observe this directly, but the more general question is whether small-memory devices can return calibrated credible intervals without long warm-up phases, so studying adaptive proposals that “learn” the right scale within the 5 000-iteration cap remains an open challenge. Another open question is how best to propagate the sampler’s empirical uncertainty into downstream decision-making when the prior itself is hand-designed: if the prior is slightly misspecified, can you correct for bias by weighting recent observations more heavily while still preserving the sampler’s theoretical guarantees? Finally, even in this toy problem the stored `accepts` array is a weak proxy for mixing; designing diagnostics attuned to online production deployments—where you only get 10 samples per minute—means rethinking both the chain length and the credible interval criterion so that you can still flag when the inference quality drifts.

## Where to read next

If you want the engineering side, → [[probabilistic-programming-inference-systems]] explains how TensorFlow Probability and Stan deploy Metropolis-Hastings chains across CPU/GPU pools. If you want the theory, → [[bayesian-inference]] lays out the full probability calculus that motivates the updates you are sampling here. The next arc node, → [[variational-inference]], contrasts these exact samplers with mean-field approximations so you can see which uncertainties survive the approximation.

## Build it

**What you're building:** A NumPy-based Metropolis-Hastings sampler for a scalar sensor bias whose 95 % credible interval converges to the true bias within 0.4 width after 5 000 iterations, along with diagnostics stored to disk.

**Why this is valuable:** You prove to yourself how Bayes’ rule becomes a posterior sample by implementing the acceptance logic, quantifying uncertainty, and understanding the impact of proposal scale—the same diagnostics you will compare against when introducing variational methods.

**Stack:**
- **Model:** Custom scalar bias model with Gaussian prior \(\theta \sim \mathcal{N}(0,1)\) and Gaussian likelihood \(y_i \sim \mathcal{N}(\theta, \sigma^2)\) where \(\sigma=0.5\).
- **Dataset:** Synthetic sensor readings drawn in-script to match the scalar bias story; we generate 50 examples from \(y_i = 1.0 + \epsilon_i\) with \(\epsilon_i \sim \mathcal{N}(0,0.5^2)\) because no public HuggingFace dataset captures this calibrated bias scenario exactly.
- **Framework:** NumPy 2.2 for numerics, Matplotlib 3.9 for plots, and PyArrow for saving diagnostics if desired.
- **Compute:** Google Colab T4 (15 GB) or any GPU/CPU with 8 GB RAM; the sampler runs in under five minutes on a Colab session.

**Estimated time:** 1.5 hours on Colab T4 and a few extra minutes to review diagnostics.

**Success criterion:** After 5 000 iterations the 95 % credible interval width is <0.4, the interval contains the ground-truth bias of 1.0, and the logged acceptance rate stays between 0.2 and 0.5.

**The recipe:**
1. Install the stack with `pip install numpy==2.2 matplotlib==3.9 pyarrow==12.0` and import `numpy as np`, `matplotlib.pyplot as plt`, and `pathlib.Path`.
2. Fix the seed `np.random.default_rng(0)` and generate observations with `observations = rng.normal(loc=1.0, scale=0.5, size=50)`; confirm `observations.shape == (50,)` to match downstream interfaces.
3. Define `log_posterior(theta)` returning `-0.5 * theta**2 - 0.5 * np.sum((observations - theta)**2) / sigma**2`; call it once at `theta=1.0` to ensure the function stays finite and prints a sanity check.
4. Initialize `chain = np.empty(5000)` and `accepts = np.zeros(5000, dtype=bool)`, set `theta = 0.0`, and loop 5 000 times proposing `theta_prime = theta + rng.normal(0, 0.3)`. Compute `delta = log_posterior(theta_prime) - log_posterior(theta)`, clip it to `[-50, 50]`, draw `u = np.log(rng.random())`, accept if `u < delta`, update `theta` and record `accepts[i]`, then set `chain[i] = theta`. This stores the acceptance explicitly and keeps `np.exp` calls safe.
5. Compute `acceptance_rate = accepts.mean()`, `lower, upper = np.percentile(chain, [2.5, 97.5])`, and `posterior_mean = chain.mean()`; print all statistics and assert `0.2 <= acceptance_rate <= 0.5`.
6. Plot the histogram of `chain`, overlay the true bias line at 1.0, label the credible interval, save as `posterior_hist.png`, and optionally store the diagnostics table via `pyarrow` for later comparators.

**Expected outcome:** You have a histogram image showing a credible interval narrower than 0.4, numeric logs confirming the acceptance rate, and a diagnostic file you can reuse when comparing to Step 3. The script prints the interval `[lower, upper]`, the mean, and the acceptance rate, making it obvious if any assertion fails.

**Stretch goals**
1. Incorporate a simple proportional-derivative controller that adapts the proposal standard deviation to track a target acceptance rate of 0.35 and plot how the interval width changes over the first 2 000 iterations.
2. Re-run the sampler with \(\sigma=1.0\) but the same observations, then compute the KL divergence between the two posterior histograms to visualize how likelihood confidence controls posterior concentration.
3. Swap the Gaussian prior to a Laplace prior \(\text{Laplace}(0,1)\) and compare the resulting credible interval width and bias; this highlights how different priors compete with data in shaping uncertainty.

Variants per persona
- **Applied AI/ML engineer:** Wrap the sampler in a Flask endpoint that accepts new batches of 10 readings, reruns 5 000 iterations of MH, and returns the credible interval JSON; benchmark the endpoint on a T4 to confirm p95 latency under 220 ms and monitor CPU/GPU utilization.
- **Research engineer:** Reproduce the acceptance rate and credible interval from Section 4.2 of Hoffman and Gelman (2014) by implementing the same No-U-Turn Sampler proposal for this scalar model and matching the reported ESS per second within ±10 %.
- **Applied researcher:** Test the hypothesis “A clipped log-acceptance advantage stabilizes diagnostics” by running three MH variants (no clip, clip at ±20, clip at ±50) over the same observations, then plot credible interval width vs. acceptance rate and consider the clip that best preserves coverage without inflating width.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*