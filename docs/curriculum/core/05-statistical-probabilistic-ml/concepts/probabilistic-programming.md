---
title: Probabilistic programming
slug: probabilistic-programming
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [bishop, bingham, mansinghka, hoffman]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bayesian-inference, probabilistic-modeling, markov-chain-monte-carlo, variational-inference]
tags: [probabilistic-programming, bayesian-inference, uncertainty, hmc, numpyro, pyro]
updated: 2025-05-20
has_mvb: true
---

# Probabilistic programming

Imagine you are tasked with estimating global CO2 emissions from a constellation of noisy, drifting satellite sensors. This is a classic inverse problem: you observe the data, but you need to infer the hidden physical processes—atmospheric wind patterns, sensor bias, and measurement error—that generated it. You could spend months manually deriving the complex Bayesian updates, which are the mathematical steps used to refine your beliefs about parameters as new data arrives. Alternatively, you could write a simple Python simulator that describes the physical process of gas diffusion and sensor observation, and then hand that simulator to a compiler that automatically runs the process in reverse to pinpoint the emission sources. This is the core promise of probabilistic programming: it decouples the generative story of data from the grueling math of inference. By using stochastic code—code that incorporates random variables to represent uncertainty—you can define domain knowledge and watch a posterior distribution emerge without ever re-implementing the underlying Bayesian machinery yourself.

## The territory

Traditional machine learning often forces practitioners to map inputs to outputs using rigid, off-the-shelf architectures. Probabilistic programming (PPL) shifts the focus to the generative process: "What latent story produced the observed data?" This approach belongs to the family of model-based machine learning, a philosophy championed by Bishop (2013) [Model-Based Machine Learning](https://www.cs.columbia.edu/~blei/fogm/2020F/readings/Bishop2013.pdf), which argues that models should be tailored to the specific structure of the domain rather than forcing data into a generic algorithm. When a new latent variable is added to a probabilistic program, there is no need to derive a fresh optimizer; the generative code is updated, and the inference engine adapts.

This field sits at the intersection of statistics, programming language theory, and high-performance computing. It borrows the rigor of Bayesian statistics while adopting the modularity of modern software engineering. The PPL stack typically consists of a modeling language—where priors, likelihoods, and stochastic dependencies are defined—and an inference backend that performs the heavy lifting of calculating the posterior. By treating inference as a programmable artifact, PPL allows for swapping between Hamiltonian Monte Carlo (HMC), variational inference, or sequential Monte Carlo without modifying the model definition. The consequence is a workflow where the modeler focuses on the "what" of the data generation, while the compiler handles the "how" of the reasoning.

## How it works

The mechanism of a probabilistic program is best understood as a transformation of a stochastic trace. When a model is written in a PPL, it defines a joint distribution \(p(\mathbf{x}, \mathbf{z})\) over observed data \(\mathbf{x}\) and latent variables \(\mathbf{z}\). The program executes as a series of stochastic operations, where each random variable is sampled from a prior distribution. The inference engine then seeks to compute the posterior \(p(\mathbf{z} \mid \mathbf{x})\), which is often intractable due to the high-dimensional integral in the denominator of Bayes' rule.

To make this tractable, modern PPLs like Pyro or NumPyro leverage the reparameterization trick to perform variational inference. This trick allows us to differentiate through random samples, which is necessary for gradient-based optimization. We move from the intractable posterior to an optimization objective by minimizing the Evidence Lower Bound (ELBO):

\[ \mathcal{L}(\phi) = \mathbb{E}_{q_\phi(\mathbf{z})} [\log p(\mathbf{x}, \mathbf{z}) - \log q_\phi(\mathbf{z})] \]

where \(\phi\) represents the variational parameters being optimized, \(p(\mathbf{x}, \mathbf{z})\) is the joint probability of the data and latents, and \(q_\phi(\mathbf{z})\) is the variational distribution. The term on the left captures the expected log-likelihood of the data under the model, while the term on the right penalizes the divergence between the approximation and the true posterior. Tran et al. (2017) [Deep Probabilistic Programming](https://www.cs.columbia.edu/~blei/papers/TranHoffmanSaurausBrevdoMurphyBlei2017.pdf) demonstrated that by using neural networks to parameterize \(q_\phi\), these models can scale to high-dimensional datasets.

The inference engine often employs Hamiltonian Monte Carlo (HMC) to explore the parameter space. HMC treats the negative log-posterior as a potential energy landscape and simulates the motion of a particle moving through this space using Hamiltonian dynamics. By calculating the gradient of the log-posterior with respect to the latent variables, the sampler can take large, informed steps. When the model is discrete or non-differentiable, the engine must fall back on techniques like surrogate modeling or program analysis to prune the search space, a frontier explored by Bowers et al. (2019) [Functional probabilistic programming for scalable Bayesian modelling](https://ar5iv.labs.arxiv.org/html/1908.02062), which showed that these methods can speed up inference by orders of magnitude.

## Where the field is now

The state-of-the-art in probabilistic programming has moved from academic prototypes to production-grade platforms. Early systems like BUGS (Spiegelhalter et al., 1996) and Stan (Carpenter et al., 2017) established the feasibility of MCMC, but modern frameworks like Pyro (Bingham et al., 2019) and NumPyro have integrated deep learning primitives directly into the PPL workflow. The 2017 work by Tran et al. (2017) [arxiv:1701.03757](https://arxiv.org/pdf/1701.03757) was a watershed moment, showing that variational inference could be unified with deep learning.

Today, the engineering frontier is defined by systems that can handle massive, streaming data. Google's TFX platform [TFX: A TensorFlow-Based Production-Scale Machine Learning Platform](https://research.google/pubs/tfx-a-tensorflow-based-production-scale-machine-learning-platform/) and specialized Bayesian neural network implementations (such as those described in Google's research on compositional Bayesian models [AutoBNN](https://research.google/blog/autobnn-probabilistic-time-series-forecasting-with-compositional-bayesian-neural-networks/)) demonstrate that probabilistic methods can scale to production environments where uncertainty quantification is a safety requirement. The research frontier is currently focused on "compiling" inference; rather than running a generic sampler, the PPL compiler analyzes the program's structure to generate specialized, high-performance code, as seen in recent work on inference compilation for simulators (Bowers et al., 2025).

## What's still open

Despite these advances, several fundamental questions remain. First, how can we automatically scale gradient-free inference to high-dimensional, non-differentiable simulators—such as global climate models—where Hamiltonian Monte Carlo fails due to discontinuous parameter spaces? Second, can we develop a universal "inference compiler" that automatically selects the optimal inference strategy based on the model's structural properties? Finally, is there a way to guarantee the convergence of variational inference in non-conjugate models without relying on the restrictive mean-field assumption? These are the barriers preventing probabilistic programming from becoming the default paradigm for scientific discovery.

## Where to read next

If you want the probabilistic foundation, → [[bayesian-inference]] gives the likelihood-based perspective that PPLs automate. The engineering counterpart is → [[variational-inference]] explaining how we approximate the posterior when exact calculation is impossible. For the next paradigm in scalable modeling, → [[probabilistic-graphical-models]] generalizes the structural dependencies that PPLs compile into efficient execution traces.

## Build it

**What you're building:** A Bayesian regression model with outlier detection using NumPyro.

**Why this is valuable:** It models heavy-tailed noise that standard MSE-based regression ignores, providing robust estimates in the presence of outliers.

**Stack:**
- **Model:** `numpyro` (JAX-based)
- **Dataset:** `scikit-learn` synthetic regression dataset (10% outliers)
- **Framework:** `numpyro` 0.15.0, `jax` 0.4.30
- **Compute:** Free Google Colab (T4 GPU)

**The recipe:**
1. Install dependencies: `pip install numpyro jax jaxlib scikit-learn matplotlib`.
2. Generate synthetic data: Create a linear relationship \(y = 2x + 1\) and inject 10% Gaussian noise with heavy-tailed Cauchy outliers.
3. Define the model: Use `numpyro.sample` to define priors for slope \(\beta\) and intercept \(\alpha\), and a Student-t likelihood to handle outliers.
4. Run inference: Use `NUTS` (No-U-Turn Sampler) with 1000 warmup steps and 1000 samples. Set `jax.random.PRNGKey(0)` for reproducibility.
5. Evaluate: Compare the posterior mean of \(\beta\) against the ground truth; plot the posterior distribution using `arviz`.

**Expected outcome:** A posterior distribution plot showing the true slope recovered despite the presence of outliers.

- **CS student:** Compare the posterior width of a model with Gaussian noise vs. Student-t noise; success = 20% reduction in bias on outlier-heavy data.
- **Applied engineer:** Export the model parameters to a JSON file and implement a lightweight JAX-based inference function for production serving; success = <10ms latency on CPU.
- **Applied researcher:** Compare the ELBO of a Student-t model vs. Gaussian; success = statistically significant improvement in log-likelihood on held-out data.
- **Frontier researcher:** Implement a custom gradient-free proposal distribution for a non-differentiable simulator using `numpyro.infer.mcmc.MCMC`; success = convergence on a 10D parameter space.

---

> *If this build worked for you — a ⭐ on the [NumPyro GitHub](https://github.com/pyro-ppl/numpyro) is the best way to support the community.*