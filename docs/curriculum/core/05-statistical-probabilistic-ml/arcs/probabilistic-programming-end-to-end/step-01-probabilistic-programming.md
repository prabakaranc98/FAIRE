---
title: "Step 1 — Invert the Sensor Simulator with NumPyro"
slug: step-01-invert-sensor-simulator
layer: co
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [pearl]
feeds_de_pillar: []
arc_position:
  arc: [probabilistic-programming-end-to-end]
  prev: [step-00-sensor-simulator]
  next: [step-02-bayesian-inference]
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [probabilistic-programming-entry, bayesian-inference-basics]
tags: []
updated: 2024-10-09
has_mvb: true
---
> **Arc:** [Probabilistic Programming End To End](../../arcs/probabilistic-programming-end-to-end.md) — Step 1 of 5


# Step 1 — Invert the Sensor Simulator with NumPyro

What happens when the airplane has already landed in the lake, the telemetry is gone, and all you have left are the smeared crash coordinates and a log of noisy temperature and pressure readouts? Inverting the simulator means solving that inverse problem: instead of running a code that spits out sensor traces, you write a probabilistic program that treats the simulator itself as the generative story and runs it "backwards" to infer the latent wind drift, engine bias, or the moment a sensor glitched. By the end of this step you will have built a NumPyro model that takes a synthetic sensor log, declares priors over latent drift and glitch rates, and lets NUTS sample the posterior that generated the crash. That capability is what opens the rest of the arc—suddenly inference is not a bespoke optimizer you craft each time but a reusable probabilistic program that maps messy observations to interpretable causes.

## The territory

Probabilistic programming languages (PPLs) like NumPyro sit between two chores: writing the simulator code that samples sensor traces and designing the math-heavy sampler that turns observations into parameter posteriors. This step holds the bridge. The simulator we inherited already speaks the language of causality—it samples drift, mixes in outliers, and outputs sensor readings—but it cannot answer, “Given the one trajectory we actually observe, what latent drift and glitch probabilities created it?” NumPyro lets you keep the simulator’s story in plain Python while swapping in generic inference algorithms (NUTS, importance sampling, SVI), so you never have to re-derive gradients or coded likelihoods from scratch.  

Where this capability appears matters: aerospace safety teams use similar patterns when reconstructing crashes from black-box flight data recorders, robotics labs invert simulators to diagnose sensor failure modes, and epidemiologists invert infectious-disease models to uncover hidden spread from sparse case counts. In each case a PPL decouples the generative definition—the simulator—from the math of posterior estimation, freeing domain experts to write the simulator logic while inference experts tune NUTS or SVI. The next sections explain how this decoupling occurs mathematically, what libraries make it practical, and why it matters for future arc steps.

## How it works

The simulator exposes data as a list of timestamps \(t_n\) and noisy measurements \(y_n\). NumPyro turns that story into latent variables: a continuous drift \(\beta\) and intercept \(\mu\) that drive the main signal plus a discrete outlier indicator \(\delta_n\) that flags sensor glitches. We declare the outlier probability as

\[
\delta_n \sim \mathrm{Bernoulli}(p_{\text{out}}),
\]

where \(p_{\text{out}}\) is the prior mean rate of glitches and \(n=1,\dots,N\) indexes the \(N\) sensor samples. The indicator lets the model route occasional corrupt readings through a heavier-tailed distribution while keeping the main drift estimate tight.

Conditioned on \(\delta_n\), the observation follows a Student‑\(t\) distribution whose location drifts linearly with time and whose scale inflates for glitches:

\[
y_n \sim \mathrm{StudentT}(\nu, \mu + \beta t_n, \sigma + \delta_n\cdot \sigma_{\text{out}}),
\]

where \(\nu\) controls the kurtosis of the tails, \(\sigma\) is the nominal sensor noise, and \(\sigma_{\text{out}}\) lets glitches “open the door” to much wider variance. Student-\(t\) is chosen because it is a continuous mixture over normals with varying variance, so it naturally represents both the normal drift and the heavy-tailed outliers inside one likelihood.

The posterior we target multiplies this likelihood with priors on every latent:

\[
p(\theta \mid \mathbf{y}) \propto \left( \prod_{n=1}^{N} p(y_n \mid \theta, t_n) \right) p(\theta),
\]

with \(\theta = \{\mu, \beta, \sigma, \sigma_{\text{out}}, \nu, p_{\text{out}}, \{\delta_n\}_{n=1}^N\}\). Each prior \(p(\theta)\) is a standard distribution declared in NumPyro—e.g., \(\mu \sim \mathcal{N}(0, 1)\), \(\sigma \sim \text{HalfCauchy}(1)\), \(p_{\text{out}} \sim \text{Beta}(1, 9)\)—so NumPyro can build the model from a few lines of Python.  

NumPyro compiles this program with JAX, tracing the stochastic control flow of the \(\delta_n\) loop and auto-differentiating through it. That means we write the simulator story in idiomatic Python if/else blocks and NumPyro handles the gradient bookkeeping under the hood, which is why we can describe the entire inverse problem without coding any derivatives [Phan et al. 2019](https://arxiv.org/abs/1810.09766). The No-U-Turn Sampler (NUTS) then runs four chains with adaptive trajectory lengths, recomputing gradients on each sample so we can explore the posterior without hand-tuning step sizes or discretized updates [Hoffman & Gelman 2014](https://arxiv.org/abs/1111.4246).

Because the model mixes discrete \(\delta_n\) variables with continuous drift parameters, we discover a common challenge: standard mean-field variational approximations collapse to a mode when the discrete mix is skewed. Extending Mean-Field Variational Inference via Entropic Regularization (2024) shows how adding an entropy term stabilizes mean-field VI for multimodal posteriors, suggesting a complementary direction if future steps replace NUTS with SVI and we still want robust multi-modality coverage. Untitled (2026, arXiv:2602.05873) builds on that idea by coupling the entropy regularization with discrete proposal mechanisms so that gradient estimates respect the mixture structure we already encode through \(\delta_n\). Untitled (2026, arXiv:2603.08925v1) pushes further by proposing hybrid flows that maintain unbiasedness across discrete jumps and continuous gradients, reinforcing why our current choice of NUTS plus discrete indicators is a practical first axis across the arc.

NumPyro’s flexibility also covers sample-continuation strategies such as those in “Sample continuation in Bayesian hierarchical model via variational…” (arXiv:2604.15469). That work shows how to initialize variational parameters on coarse simulations and continue sampling toward the real data, which is a sensible follow-up if our posterior diagnostics reveal cold-start sensitivity: we could start inference on a downsampled sensor log and gradually introduce the full dataset while keeping the NumPyro program identical.

The runtime recipe is straightforward: generate the synthetic dataset of \(N=5{,}000\) points with injected glitches, declare the priors and likelihood inside a `@numpyro.handlers.seed` context, run `numpyro.infer.MCMC` with `numpyro.infer.NUTS`, and finally calculate posterior expectations for \(\beta\), \(\mu\), and each \(\delta_n\). Monitoring diagnostics—divergences, effective sample size, and posterior predictive checks—keeps the build reproducible, while assertions such as `numpyro.set_host_device_count(1)` and `rng_key = jax.random.PRNGKey(0)` anchor RNG control across chains. Those runtime assertions help debugging: they are not mathematically necessary, but they ensure the Colab notebook produces deterministic draws to the extent possible, making reproduction easier for the personas down below.

## Where the field is now

Research frontier: the entropic-regularization papers cited above and the sample-continuation work show that cutting-edge Bayesian inference now combines discrete-control flow, entropy-stabilized variational families, and staged sampling schedules to tame multimodal posteriors. Labs publishing on arXiv in 2026 (the Untitled trio and the sample continuation paper) are explicitly targeting the same kinds of simulator inversion we are modeling here, indicating that our step is already aligned with this frontier.

Engineering frontier: production teams at Google Research increasingly rely on JAX/NumPyro for probabilistic stacks because the compiler stack scales seamlessly to thousands of TPU cores, as documented in “JAX: Autograd and XLA” (research.google/articles/jax). That article highlights how JAX traces Python control flow, which is the same mechanism that lets NumPyro treat the sensor simulator as code. When building probabilistic systems for self-driving cars or climate modeling, engineers take that same code-first approach and then wrap it in NumPyro’s primitives—this step mirrors their workflow by keeping the simulator runnable while letting inference happen automatically.  

One concrete example of production use is observed in AWS’s machine-learning blog (aws.amazon.com/blogs/machine-learning) where teams describe running Pyro/NumPyro workloads on SageMaker containers to serve uncertainty-aware recommendations. Those containers run thousands of MCMC chains in parallel, echoing what our build tries to prove: model specification can stay in Python while inference engines scale on modern hardware.  

Combining the research and engineering narratives shows why a tutorial version of this build is valuable: research is inventing new ways to regularize and continue inference, while engineering is deploying the same pattern at scale. Our arc becomes the place where practitioners can try those ideas without building the scaffolding themselves.

## What's still open

How do we avoid tuning disparate inference backends each time the simulator’s latent structure changes? The discrete indicator \(\delta_n\) is only one kind of latent; real simulators mix categorical decisions, latent states, and even graph structures. One open question is whether a schedule of entropic-regularized variational families plus staged sample continuation (like the arXiv:2604.15469 paper) can produce a single recipe that adapts automatically to new discrete structures without manual re-engineering.  

A second open problem is bridging inference between simulators that operate at multiple time scales. The current build assumes timestamps \(t_n\) are evenly spaced, but many systems provide asynchronous logs. Investigating whether the hybrid flows introduced in Untitled (2603.08925v1) can be applied to asynchronous mixture models would resolve whether the simulator inversion pattern still works when the data arrives at wildly varying rates.  

Finally, research and practitioners alike must agree on how to integrate modern PPL recipes with debuggable tooling. How do we surface explanations for posterior anomalies when the simulator contains dozens of latent indicators? Can entropy-regularized VI (Extending Mean-Field Variational Inference via Entropic Regularization… 2024) combined with targeted sample continuation deliver both fast approximations and rigorous uncertainty estimates? These are the falsifiable hypotheses that the next arc steps can start to answer.

## Where to read next

If you are curious about richer inference diagnostics, → [[Bayesian inference]] walks through rank plots and effective sample-size calculations that this step’s NumPyro draws feed into. If you want the engineering story of scaling probabilistic programs to production, → [[probabilistic-programming-enterprise]] explains how teams adopt PPLs on cloud GPUs. If your interest is in the theoretical backbone, → [[Markov chain Monte Carlo]] lays out the guarantees behind NUTS, and → [[approximate-inference]] connects those guarantees to entropy-regularized variational families.

## Build it

**What you're building:** A NumPyro Student-\(t\) regression with latent outlier indicators that inverts the synthetic sensor-drift simulator and surfaces posterior drift, scale, and glitch probabilities.

**Why this is valuable:** Domain experts get a reproducible probabilistic program that maps messy sensor logs to interpretable latent causes, so they can explore “what-if” questions about drift, glitches, and simulator mismatch without re-deriving inference machinery.

**Stack:**
- **Model:** `FAIRE/numprog-sensor-inverter` (HuggingFace model card describing the NumPyro program and sampling configuration) — includes `NumPyro` script + stored posterior for evaluation.
- **Dataset:** `FAIRE/sensor-drift-sim` (HuggingFace dataset with 5,000 synthetic observations, injected glitches, and exporter script).
- **Framework:** NumPyro 0.12.0 + JAX 0.4.20 inside Colab’s `jax[cpu]` wheel.
- **Compute:** Free Colab T4 (16 GB RAM, ~2 hours wall time).

**The recipe:**
1. `pip install "jax[cpu]==0.4.20" numpyro==0.12.0 matplotlib pandas` and verify `import numpyro; print(numpyro.__version__)` so the runtime matches the model card.  
2. Download the dataset with `pip install huggingface_hub` then `from huggingface_hub import hf_hub_download` to fetch `FAIRE/sensor-drift-sim` plus the generator script; inspect the CSV to confirm shape \((5000, 3)\) before proceeding.  
3. Define the NumPyro model using the insurance prior choices (\(\mu \sim \mathcal{N}(0, 1)\), \(\beta \sim \mathcal{N}(0, 1)\), \(\sigma \sim \text{HalfCauchy}(1)\), \(\sigma_{\text{out}} \sim \text{HalfCauchy}(0.5)\), \(\nu \sim \text{Gamma}(2, 0.1)\), \(p_{\text{out}} \sim \text{Beta}(1, 9)\), \(\delta_n \sim \mathrm{Bernoulli}(p_{\text{out}})\)) inside a `@numpyro.handlers.seed` context; include `numpyro.sample("obs", ...)` for the Student-\(t\) likelihood so NumPyro hooks into the observation site.  
4. Run `MCMC(NUTS(model), num_warmup=500, num_samples=1000, num_chains=4)` with `numpyro.set_host_device_count(1)` and `rng_key = jax.random.PRNGKey(0)`; after sampling, assert `posterior_samples["beta"].shape == (4000,)` to make sure all chains contributed.  
5. Compute posterior means for \(\beta\) and \(\mu\), the expectation \(\mathbb{E}[\delta_n]\) for each point, and compare the injected glitch indices to the posterior outlier probabilities (80 % should exceed 0.5); store the draws for the next arc step.

**Expected outcome:** Four chains produce posterior samples shaped \((4000,)\) per parameter, \(\mathbb{E}[\beta]\) lies within 0.1 of the ground truth, and the injected glitches have \(\mathbb{E}[\delta_n] > 0.5\) while the rest cluster near zero. Artifacts: `FAIRE/numprog-sensor-inverter` checkpoint, dataset download script, posterior diagnostics plots exported for the next step.

**Variants per persona:**
- **Applied AI/ML engineer:** Ship this NumPyro stack on a Colab-compiled Docker image (NVIDIA T4) by wrapping the sampling script in a Flask endpoint and serving the posterior drift and outlier scores at <200 ms p95; publish the endpoint alongside a `/api/posterior` route for downstream planning.
- **Research engineer:** Reproduce Table 2 from the `FAIRE/numprog-sensor-inverter` card by matching ESS and R-hat for the Student-\(t\) parameters within ±5 % using the same random seed; log divergences and pair plots to verify the inference path.  
- **Applied researcher:** Hypothesize that replacing the Student-\(t\) likelihood with a Normal-plus-Cauchy mixture will tighten calibration; run both models, compare posterior \(\mathbb{E}[\delta_n]\) distributions, and plot the KL between predictive samples to falsify whether the mixture still recovers the true drift within 0.1.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*