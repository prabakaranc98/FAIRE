---
title: Probabilistic programming  
slug: probabilistic-programming  
layer: core  
subject: 05-statistical-probabilistic-ml  
page_type: concept  
state: drafted  
authors_anchored: [bishop, tran, mansinghka, wood, becker]  
feeds_de_pillar: []  
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]  
prereqs: [bayesian-inference, probabilistic-modeling, markov-chain-monte-carlo, variational-inference]  
tags: [probabilistic-programming, bayesian-inference, numpyro, pyro, hmc, variational-inference]  
updated: 2025-05-20  
has_mvb: true  
---

# Probabilistic programming

Imagine writing a physics simulator for a delivery drone that captures wind, lift, drag, and actuator delays. Now imagine you do not run that simulator forward to forecast where the drone will land, but run it *backward* so that the exact gusts that pushed it off course fall out of the code as posterior samples. That “reverse execution” is what probabilistic programming offers: you sketch the generative story in familiar Python or Julia, sprinkle in stochastic nodes where uncertainty belongs, and hand the program to the runtime. The runtime then derives posterior distributions, gradient estimators, and samplers without you ever touching the calculus for the latent variables. By the time you finish this page you will know how PPLs transform modeling effort into a software-engineering workflow, how they marry Bayesian statistics to modern compiler techniques, what the current bottlenecks are, and how to build a working Bayesian regression with NumPyro that proves the compiler really does the heavy lifting.

## The territory

Conventional machine learning often assumes a rigid architecture: “fit a neural network, tune regularization, push data through.” Probabilistic programming flips the emphasis back to “what generative process could have produced that data?” That shift is what Bishop (2013) [Model-Based Machine Learning](https://www.cs.columbia.edu/~blei/fogm/2020F/readings/Bishop2013.pdf) called when he argued for designing bespoke models that capture domain structure instead of forcing every use case through a handful of fixed likelihood functions. In a PPL you describe that bespoke model as ordinary code—loops for hierarchies, conditionals for switching regimes, recursion for repeated reasoning—and the language runtime derives the posterior inference without manual derivations.

This approach sits at the junction of Bayesian statistics, programming languages, and high-performance systems. Bayesian statistics gives you the language of priors, likelihoods, and posterior distributions. Programming-language theory gives you the control-flow primitives, continuations, and trace representations that make it possible to inspect and transform user code. High-performance engineering gives you compilation to XLA/LLVM, vectorized sampling kernels, and autodiff-backed gradient estimators. Together they let a modeler swap out likelihoods or add latent variables without touching the sampler. That is why the fundamental promise of PPLs is “write once, infer everywhere”—the model specification is decoupled from inference execution. How does that actually work?

## How it works

The starting point is a generative story: a joint density \(p(x, z)\) that models observed data \(x\) and latent variables \(z\). The modeler writes this story as executable code that first samples from priors and then produces observations under a likelihood, e.g. the hierarchical linear model that samples global slope \(\beta_{\text{global}}\), per-group offsets \(\beta_{g}\), and finally noisy observations \(x_i \sim \mathcal{N}(z_i^\top w, \sigma^2)\). Because it is written as code, the program can use conditionals, loops, recursion, and higher-order functions, which is essential for expressing real-world structure.

The posterior we care about is

\[
p(z \mid x) = \frac{p(x, z)}{p(x)} = \frac{p(x, z)}{\int p(x, z)\,dz}.
\]

where \(z\) is the latent state, \(x\) is the observed data, and \(p(x)\) is the evidence (the integral of the joint over \(z\)). This ratio is intractable except in toy cases, which is why inference engines like Hamiltonian Monte Carlo (HMC) or variational families take over. The key becomes expressing the joint \(p(x,z)\) and then letting the runtime differentiate or sample through it.

PPL runtimes achieve that by tracing code execution. Every time the user calls a probabilistic primitive such as `sample("beta", Normal(0, 1))`, the runtime records the statement in a trace that logs the distribution, the site name, and the sampled value. That trace is the interface between the user code and the inference engine. Wood et al. (2015) [A New Approach to Probabilistic Programming Inference](https://arxiv.org/abs/1507.00996) formalized this idea: a universal execution protocol separates model execution from inference execution so that external engines can plug into any user-defined program. When inference needs to evaluate the joint density or compute gradients, it walks the trace, accumulates log probabilities, and backpropagates using autodiff on the stored samples.

One common inference strategy is Hamiltonian Monte Carlo, which requires gradients of the log joint. The runtime rewrites the trace into a log-density function \(\mathcal{L}(z) = \log p(x, z)\). To run HMC you supply this function and let autodiff compute \(\nabla_z \mathcal{L}(z)\). The runtime also handles proposals: it uses the sampler’s kinetic energy parameterization to propose new latent states, evaluates \(\mathcal{L}\) on the traced code, and accepts or rejects according to the Metropolis–Hastings rule. Because the trace already contains the sampled values, every evaluation is exactly the generative story you wrote. When you change a prior or likelihood, the next trace reflects that change automatically, and the gradient generator downstream sees the new math without you calculating anything by hand.

On the variational side, the runtime defines an Evidence Lower Bound (ELBO) and optimizes it. The ELBO is written as

\[
\mathcal{L}_{\text{ELBO}}(\phi) = \mathbb{E}_{q_\phi(z)}[\log p(x,z) - \log q_\phi(z)],
\]

where \(q_\phi(z)\) is the variational distribution parameterized by \(\phi\), \(p(x, z)\) is the joint from your program, and the expectation averages over the variational samples.
This equation states that optimizing \(\phi\) asymptotically makes \(q_\phi\) close to the true posterior while reusing the same trace information from the model. Runtimes such as Edward (Tran et al. 2017) [Deep Probabilistic Programming](https://arxiv.org/abs/1701.03757) built this pipeline on top of TensorFlow, letting the variational guide share neural-network parameters with the generative model to handle high-dimensional neural weight posteriors.

What makes the PPL story compelling is that the runtime does not need to know anything about the particular latent structure. The trace can be transformed by program transformations such as reparameterization (changing how randomness is introduced), control-flow unrolling, or symbolic differentiation. These transformations are the domain of probabilistic program compilers. For example, NumPyro compiles Pyro-style models to XLA kernels by analyzing the trace and generating batched log-density and gradient functions that run efficiently on GPUs or TPUs. Pyro and NumPyro both rely on the `trace` and `plate` abstractions to describe dependencies, so adding a nested plate means a new batch dimension, which the compiler handles for you.

Functional probabilistic programming, described in the 2019 paper “Functional probabilistic programming for scalable Bayesian modeling” (1908.02062), extends this flexible trace idea to programs where control flow depends on sampled values and where recursion defines the structure. In such programs, the execution graph changes every run. The paper shows how to capture continuations so that inference engines can resume execution from a sample site and how to generate unbiased gradient estimates for the resulting random-control-flow programs. The sampler then works even when your code uses recursion to represent, say, a generative grammar that keeps expanding until a stop condition depending on a random string length is met. The inference engine treats such probabilistic branching like any trace entry: it records distributions, replays them, and provides the same gradients regardless of whether the stack grew deep or not.

While description of a trace and inference protocol might sound abstract, real PPLs back them with user-facing APIs. In NumPyro, you wrap the generative code in a `model` function and invoke `mcmc = numpyro.infer.MCMC(...)`, passing in `numpyro.handlers` such as `trace` and `seed`. When you add an outlier component—say an additional Bernoulli that flips whether an observation is corrupted—the runtime sees the new sample site, the new log-probability terms, and the new gradients without requiring new math. That is the pragmatic payoff: “change one line of code, inference updates automatically.” It is how you can go from a simple linear regression to a robust mixture model with heavy tails using the same compiler pipeline.

Probabilistic databases like BayesDB (Mansinghka et al. 2015) [BayesDB: A probabilistic programming system for querying the probable implications of data](https://arxiv.org/abs/1512.05006) take the same programming paradigm and wrap it in a declarative query layer. They allow analysts to write SQL-like queries that under the hood instantiate probabilistic programs, run inference engines behind the scenes, and return probabilistic answers (e.g., “how likely is it that this field is missing because of a systematic bias?”). This makes the PPL machinery accessible to non-experts while still preserving the decoupling between modeling and inference.

The modern twist, illustrated by Becker et al. (2024) [Probabilistic Programming with Programmable Variational Inference](https://arxiv.org/abs/2406.15742), is to treat inference itself as code. Instead of a monolithic VI engine with fixed ELBO, their framework exposes program transformations that let you swap in custom objectives, clipping strategies, and gradient estimators while keeping a single trace interface. This modularity is what allows PPLs to grow beyond black-box samplers: you can craft the variational loss that suits your application, plug in Rao-Blackwellization, or mix HMC with amortized inference, all without rewriting the generative encoding.

When you piece these ideas together—the traced generative story, the automation of gradient and proposal generation, the ability to mutate control flow, and the programmable inference—you see why probabilistic programming languages decouple specification from inference execution. The structure is “model code + inference compiler + runtime engine,” and every change to the model code is transparently reflected in the traces the compiler inspects. That transparency is what makes PPLs a practical bridge between domain experts who think in models and systems engineers who deploy inference at scale.

## Where the field is now

The current research frontier is programmable inference itself. Becker et al. (2024) [Probabilistic Programming with Programmable Variational Inference](https://arxiv.org/abs/2406.15742) demonstrates a framework where variational inference is not a fixed ELBO but a compositional program transformation. Their compiler allows developers to write custom gradient estimators, clip terms, and even mix Monte Carlo and analytic components inside the same objective. That flexibility is what lets researchers tailor inference to complex posteriors such as hierarchical neural networks or state-space models with long-range dependencies, and the paper backs it with experiments showing faster convergence than traditional black-box VI on several benchmark models.

On the engineering frontier, TensorFlow Probability (Google Research) continues to mature as an industrial‑scale PPL backbone. The TensorFlow blog’s 2020 announcement [“Introducing TensorFlow Probability”](https://ai.googleblog.com/2020/05/announcing-tensorflow-probability.html) highlights how TFP integrates with Vertex AI to deliver production-ready probabilistic layers that scale across Google Cloud. Production teams there ship thousands of probabilistic forecasting and calibration workloads per day, relying on the library’s HMC, variational, and MCMC kernel optimizations so that the same probabilistic code can run in both research notebooks and inference servers.

That combination—programmable inference for complex models and production-hardened runtime for web-scale workloads—defines the state of probabilistic programming today. The research pushing the flexibility of the inference language and the engineering pushing the systems integration form a dialectic: programmable inference makes it easier to specify the models Vertex AI or NumPyro will execute, and production requirements (latency, memory, numerical stability) force future research to yield more efficient and debuggable compiler pathways.

## What's still open

Can we compile universal probabilistic programs with unbounded recursion and dynamic control flow into inference engines that provide low-variance, unbiased gradient estimators **without** requiring hand-crafted variational guides? Current techniques either restrict control flow, handcraft importance samplers, or accept high variance. A fully automated compiler that can transform any control-flow-intensive program into a stable inference objective would collapse a major barrier between expressive modeling languages and scalable inference.

Another question is how to bridge probabilistic programming and modern foundation models: can we embed a PPL trace inside a large language model so that the LLM’s latent state participates in HMC or VI updates, and what semantics guarantee consistent updates without locking the entire weight space? This might require a hybrid of amortized inference and symbolic tracing that respects the LLM’s token-level control flow.

Finally, how do we give non-expert analysts programmable diagnostics for inference failure? PPLs can already log divergences, but an “AI assistant” that tells you whether the posterior collapsed, whether the trace never visited a branch due to low prior mass, or whether the sampler failed to mix across hierarchical levels would turn probabilistic programming from a niche research tool into a widely trusted engineering asset.

## Where to read next

If you want the optimization picture under the hood, → [Variational Inference](variational-inference.md) unpacks the ELBO derivations—and the tricks like reparameterization gradients that PPLs compile automatically. The sampling counterpart is → *markov chain monte carlo* <!-- [[markov-chain-monte-carlo]] -->, which explains how HMC and Metropolis–Hastings work on the traces you just trusted to the runtime. To revisit the Bayesian foundations that make those traces meaningful, → [Bayesian inference](bayesian-inference.md) grounds the discussion in prior-posterior updates and predictive checks.

## Build it

The build proves that changing a single line of NumPyro model code (adding an outlier component) instantly updates a Hamiltonian Monte Carlo sampler, demonstrating the “write once, infer everywhere” decoupling from the PPL stack.

**What you're building:** A robust Bayesian linear regression with an explicit outlier mechanism trained on a synthetic noisy dataset and analyzed through HMC diagnostics in a free Google Colab notebook.  
**Why this is valuable:** It forces you to write the generative model, add a new latent indicating corruption, and watch NumPyro automatically adjust the HMC trace without touching gradient code.  
**Stack:**  
- **Model:** Custom NumPyro Bayesian regression with outlier indicator and Student’s t noise (no pretrained HF checkpoint).  
- **Dataset:** `ydata-synthetic/gaussian-mixture` [https://huggingface.co/datasets/ydata-synthetic/gaussian-mixture] — synthetic mixture data you can load quickly and augment with calibrated noise.  
- **Framework:** `numpyro==0.13.0`, `jax==0.4.20`, `optax==0.2.0`.  
- **Compute:** Free Colab T4 (16 GB RAM, 16 GB GPU memory); training runs in ~45 minutes with 1000 warmup + 500 samples.

**The recipe:**
1. Install and import: `pip install "jax[cpu]" numpyro optax datasets matplotlib arviz` (CPU mode works in Colab; switch to `jax[tpu]` only if you have TPU access).  
2. Data: load the `ydata-synthetic/gaussian-mixture` dataset, resample 2,000 points, add 5% uniform outliers by replacing their targets with large noise, and standardize the features.  
3. Train: define a NumPyro model with global slope/intercept priors, a Bernoulli outlier indicator that selects either Normal or Student’s t observation, and run `numpyro.infer.NUTS` with 4 chains, 1000 warmup, 500 samples; monitor the `rhat` and divergence counts reported by ArviZ.  
4. Evaluate: compute posterior predictive mean squared error on a held-out split; expect RMSE < 0.5 when outliers are modeled and divergence warnings drop compared to a model without the indicator.  
5. What you now have: a checkpointed `NumPyro` trace collection that captures both the clean-model parameters and the outlier probabilities, ready for calibration and deployment.

**Expected outcome:** A saved NumPyro inference result (posterior means, divergences) plus a Matplotlib/ArviZ plot showing how the outlier indicator cleanly separates corrupted datapoints, proving that a single code change rewired the HMC sampler.

- **CS student:** Run the same notebook on an RTX 4070 laptop by reducing chains to 2 and training for 500 warmup + 250 samples; this keeps the artifact identical but fits the compute budget.  
- **Applied engineer:** After training, export the posterior predictive function as a `jax.jit` callable, quantize the parameters to bfloat16, and serve via NVIDIA Triton with a 50 ms p95 latency target while monitoring divergence counts as a warmup signal.  
- **Applied researcher:** Swap the Student’s t likelihood for a Gaussian Process residual and test the hypothesis “a GP residual reduces posterior predictive RMSE by at least 10% compared to the baseline” on the same dataset; falsification criterion: the RMSE gap is within ±5%.  
- **Frontier researcher:** Extend the notebook with recursive control flow for an unknown number of latent clusters, then investigate whether existing NumPyro gradients still have low variance or whether you need the programmable transformations from Becker et al. (2024) to stabilize the updates.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*