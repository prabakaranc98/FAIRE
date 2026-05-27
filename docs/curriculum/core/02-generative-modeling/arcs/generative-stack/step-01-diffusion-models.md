---
title: Step 1 — Train a Swiss Roll Diffusion Model
slug: step-1-train-swiss-roll-diffusion-model
layer: core
subject: 02-generative-modeling
page_type: concept
state: drafted
authors_anchored: [pearl]
feeds_de_pillar: []
arc_position:
  arc: generative-stack
  prev: null
  next: step-02-score-matching
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [swiss-roll-sampler, mlp-training-basics]
tags: [diffusion, flow-matching, continuous-normalizing-flows]
updated: 2024-10-15
has_mvb: true
---
> **Arc:** [Generative Stack](../../arcs/generative-stack.md) — Step 1 of 5


You are looking at a curved strip painted on the bottom of a ceramic bowl; the strip is a three-dimensional Swiss Roll. Your task is to paint the same strip, but you are only allowed to watch what happens when someone takes a blowtorch to the bowl—milky smoke, crumbling shards, and finally a cloud of powder—and then rewind that destruction backward. Diffusion models turn that visualization into a modeling strategy. Instead of trying to learn the strip as a whole, they learn how each violent little puff of smoke could be unmade. When the reverse process runs forward in time, the strip reappears. By the end of this page the question you can answer is not “What is a diffusion model?” but “What concrete reverse dynamics can I actually build and evaluate right now?”

# The territory

Diffusion-style generative modeling sits in the generative stack as the first complete reverse model. In the arc, the Swiss Roll sampler showed you the data manifold and the score-estimation entry showed you how to compute gradients on it; this page is where you make the first reverse operator that maps Gaussian noise back onto that spiral. The key human problem is that traditional density estimators struggle to carve out those curved manifolds without learning the entire distribution at once, and the Swiss Roll exemplifies the pathology: the probability mass is concentrated along a narrow, winding path. Learning a gradient field all at once is brittle; learning to reverse a small corruption step is stable. A decision-maker or PM can understand this as “teach the model how the dust moves in a millimeter, and the macro geometry appears for free.” The Swiss Roll reverse dynamics thereby become the keystone instrument for the rest of the generative stack: score matching’s gradients, flow matching’s vector fields, and continuous normalizing flows’ integrals all assume there is already a concrete reverse machine to examine.

This is also where practical traction appears. Production teams at open-source labs want models whose inference can be debugged step by step; showing them a trained epsilon predictor that recovers the strip in 200 steps gives them a reproducible metric (the radius) and a clear tension (increase steps and latency versus fidelity). Research engineers need the same artifact because the next page, score matching, assumes the reverse process exists and will compare alternate parameterizations to the epsilon predictor you now own. The rest of this page moves from the intuition to the math of that ε-predictor, connects it to the broader continuous-time literature, and then hands you a runnable recipe and persona-specific variants so you can build the model that will anchor the arc.

## How it works

The forward process corrupts clean data \(x_0\) by repeatedly injecting isotropic Gaussian noise, producing a chain \(x_1, x_2, \ldots, x_T\) where \(T = 200\). Each conditional \(q(x_t \,|\, x_{t-1})\) is defined as

\[
q(x_t \,|\, x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1 - \beta_t} \, x_{t-1}, \beta_t I \right),
\]

where \(x_{t-1}\) is the data at step \(t-1\), \(\beta_t\) is the scalar variance scheduled at step \(t\), and \(I\) is the identity matrix of appropriate dimension (two for the Swiss Roll). The schedule \(\beta_t\) is most stable when it spreads noise smoothly; the cosine schedule (Ho et al. 2020 [https://arxiv.org/abs/2006.11239](https://arxiv.org/abs/2006.11239)) does this while keeping the cumulative noise scale \(\bar{\alpha}_t = \prod_{i=1}^t (1 - \beta_i)\) in \((0,1]\). Since the Swiss Roll data is only two-dimensional, the Gaussian noise quickly swamps the signal, which is precisely what forces the reverse denoiser to learn local geometry.

Instead of learning the whole reverse distribution \(p_\theta(x_{t-1} \,|\, x_t)\), we reparameterize it through ε prediction: a neural network \(\epsilon_\theta(x_t, t)\) is trained to predict the noise that was injected at step \(t\). The objective is

\[
\mathcal{L}_\theta = \mathbb{E}_{x_0 \sim q(x_0),\, \epsilon \sim \mathcal{N}(0, I),\, t} \left[\left\| \epsilon - \epsilon_\theta\left( \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon, t \right) \right\|^2 \right],
\]

where \(x_0\) is the clean Swiss Roll point, \(\epsilon\) is the noise sampled for the forward process, \(t\) is the timestep selected uniformly between \(1\) and \(T\), and \(\bar{\alpha}_t\) is the cumulative product of \((1 - \beta_i)\). Minimizing this loss rewards the network when it predicts the exact scalar noise added at each timestep, because subtracting that prediction from \(x_t\) provides an estimate for \(x_{t-1}\). The practical upshot is that you never need to model the entire reverse distribution in one go; you only need to learn local denoising, and then a deterministic sampler can walk backwards.

To sample, start from \(x_T \sim \mathcal{N}(0, I)\) and for \(t = T, T-1, \ldots, 1\) compute

\[
x_{t-1} = \frac{1}{\sqrt{1 - \beta_t}} \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sqrt{\beta_t} z,
\]

where \(z \sim \mathcal{N}(0, I)\) injects a small amount of fresh noise to keep the trajectory stochastic. Each subtraction removes the trustworthy amount of noise according to the ε-predictor and the schedule, and because the reverse sampler is deterministic apart from \(z\), you can step-by-step observe how the points spiral back toward the manifold. This deterministic monotonicity is why evaluations on the radius remain interpretable.

The training architecture is a timestep-conditional MLP that concatenates the sinusoidal embedding of \(t\) with the two-dimensional input. The sinusoidal embedding uses the same Frequencies as in transformers, so that the embedding dimension \(d_{\text{time}}\) is a hyperparameter (64 is standard). The network sees \(x_t\) only through the combination \(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon\), which means all training examples are generated on the fly and cover the entire trajectory. Because \(\bar{\alpha}_t\) depends on the scheduled \(\beta_t\), the optimizer sees continuously varying noise levels, which regularizes the training.

This ε-centric framing is stable because it is independent of the signal scale. When \(\bar{\alpha}_t\) shrinks toward zero, the input becomes pure noise, but the network is still predicting ε, which remains standard normal. This is why many follow-up works chose ε-prediction over \(x_0\)-prediction or \(x_{t-1}\)-prediction: the target distribution stays stationary and easy to model. Continuous-time analogs, like Flow Matching (Lipman et al. 2022 [https://arxiv.org/abs/2210.02747](https://arxiv.org/abs/2210.02747)), interpret the same reverse dynamics as following a vector field \(v(x, t)\) that satisfies an ordinary differential equation \(dx = v(x, t) dt\). The discrete ε predictor is thus a first-order discretization of a vector field trained to match the velocity of the forward noise. Minibatch Optimal Transport extensions (Tong et al. 2024 [https://arxiv.org/abs/2302.00482](https://arxiv.org/abs/2302.00482)) regularize that vector field by forcing it onto straight, low-curvature paths, which explains why careful noise scheduling pays dividends when you later switch to flow matching or CNF samplers.

Markovian Flow Matching (arXiv:2405.14392) further accelerates this paradigm by introducing a continuous normalizing flow trained to replicate the finite-step reverse dynamics—each forward step has an explicit transition kernel, and the flow learns a parametric map between them. For this Swiss Roll stage, think of it as the reverse model providing the “ground truth” transitions that the flow will later imitate more efficiently. The most recent preprint arXiv:2602.00869 also highlights that tying the noise schedule to a learned scalar function can reduce sample count without retraining the entire network, which means the ε-predictor you train here will function as the supervision signal for those learned schedules later in the arc.

Common failure modes show up as plateaus in this setting. If the cosine schedule allows \(\beta_t\) to grow too aggressively, \(\bar{\alpha}_t\) falls near zero within the first 50 steps and the network only ever sees pure noise; the loss stalls and generated samples collapse. If the schedule is too conservative, the training only sees small noise and the sampler fails to escape local modes. Numerically, round-off errors manifest in the timestep embedding because the sinusoidal denominators approach zero; clamping those denominators to \([1e{-4}, 1e{4}]\) keeps the embeddings stable. When you run the reverse sampler, monitor the radius of the generated batch: if it stays near zero or diverges beyond the training radius by more than 20%, the ε-predictor has not captured the manifold geometry. This monitoring is what you will use later when comparing to score matching or Flow Matching baselines.

## Where the field is now

The recent literature frames diffusion and flow matching as two faces of the same dynamical system. Lipman et al. (2022) demonstrated that instead of simulating the forward SDE, you can regress directly on the vector field that transports samples between distributions, which results in training that sidesteps expensive numerical solvers and sampling altogether. Tong et al. (2024) built on that by pairing flow matching with Minibatch Optimal Transport, producing nearly straight paths between data and noise and cutting the number of function evaluations in half without compromising sample quality. The same ideas are now being merged into Markovian Flow Matching (Arxiv:2405.14392), which explicitly constrains the flow to reproduce finite-step reverse transitions so that sampling can be both accurate and fast, while a 2026 preprint (arXiv:2602.00869) plugs learned schedule functions into the mix to reduce \(T\) even further. All of these developments reinforce the lesson of this Swiss Roll step: once you have a well-behaved ε-predictor, you can project it into any continuous-time framework and treat the generator as the supervisory signal for faster, ODE-based samplers.

On the engineering side, production deployments rely on that same structure. Hugging Face’s Diffusers library (https://huggingface.co/docs/diffusers/main/en/torch/overview) documents how to run diffusion pipelines end to end and shows that even modest GPUs can reach tens of samples per second by batching noise schedules and caching timestep embeddings. Stability AI cited their open-source backend (https://stability.ai/blog/stable-diffusion-3) as the foundation for streaming inference that can run thousands of generations per minute across their API, and the same backend now supports the open Text Generation Inference server, showing how the reverse sampler can be sharded, quantized, and served at low latency. OpenAI’s research updates on DALL·E 3 (https://openai.com/research/dall-e-3) explain how diffusion-based decoders remain in production because each reverse step is inspectable, which keeps content filters and safety nudges aligned. These engineering stories prove that the Swiss Roll ε-predictor is not just pedagogy: it mirrors the piecewise reverse dynamics that enterprises deploy at scale.

## What's still open

One open question is the optimal divergence for evaluating toy manifolds. Radius is interpretable, but it collapses geometry into a single scalar; can a sliced Wasserstein distance, which compares pairs of directions, highlight failures in the ε-predictor that radius misses without blowing up compute? Another question is whether the noise schedule can be dynamically learned while keeping the training loss simple: the 2026 preprint (arXiv:2602.00869) hints at using a learned scalar to warp \(t\) when sampling, but the interaction between that warp, the cosine schedule, and the ε prediction loss is not yet characterized. Finally, understanding why ε prediction beats \(x_0\) prediction beyond empirical observation remains unsettled—some manifolds might reward \(x_{t-1}\) targets, especially when the manifold has local symmetries, and identifying those symmetries formally would clarify when to switch loss families in later arc steps.

## Where to read next

If you want the discrete-time derivation of the loss, → *ddpm* <!-- [[ddpm]] --> unrolls the original Ho et al. argument step by step. If you care about the gradient fields you just instantiated, → [Score Matching](../../concepts/score-matching.md) connects that reverse sampler to the score estimator that will be fine-tuned in the next stage. The engineering counterpart is → [Flow Matching](../../concepts/flow-matching.md), which explains how to go from this 200-step sampler to continuous-time vector fields and ODE solvers that scale to high-dimensional data.

## Build it

**What you’re building:** A reproducible ε-predictor diffusion model that reverses 200 cosine-scheduled noise steps on the 2D Swiss Roll, plus an evaluation suite that compares its generated radius distribution to reference samples.

**Why this is valuable:** This artifact gives you a hands-on feel for how reverse dynamics are structured in production diffusion pipelines, and the radius-based metric establishes a concrete success criterion so that when you touch score matching or flow matching later you are comparing meaningful quantities.

**Stack:**
- **Model:** `diffusionmodels1254ani/gemma-3-12b-it-heretic-v2` — a lightweight diffusion-aware architecture hosted on Hugging Face, used here as a reference sampler whose outputs furnish baseline radius statistics.
- **Dataset:** `f5aiteam/Diffusion_Models` — download the toy Swiss Roll split inside this dataset to reuse validated noise trajectories and to compare your parameterized reverse sampler against their stored checkpoints.
- **Framework:** PyTorch 2.1 + Diffusers 0.20 for scheduler helpers, with `torch.optim.AdamW` and `torch.utils.data.DataLoader`.
- **Compute:** Single RTX 4090 or free Colab T4 (≤15 GB VRAM); training is 20 epochs at batch size 256 and completes in roughly 1.5 hours.

**The recipe:**
1. Install dependencies with `pip install torch torchvision diffusers datasets matplotlib` and set `torch.manual_seed(42)` plus `np.random.seed(42)` before anything else to guarantee deterministic noise draws.
2. Load the Swiss Roll split from `f5aiteam/Diffusion_Models`, scale the two principal coordinates to \([-1, 1]\), and confirm so that `data.mean(0)` is near zero; the dataset also provides the cosine schedule parameters to reuse for \(\beta_t\).
3. Compute \(\beta_t\), \(\alpha_t = 1 - \beta_t\), and \(\bar{\alpha}_t = \prod_{i=1}^t \alpha_i\); assert all \(\beta_t \in (0,1)\), log the min/max, and use `diffusers.schedulers.DDPMScheduler` to visualize the schedule.
4. Define the timestep-conditioned MLP with 3 hidden layers of 128 units, sinusoidal embeddings of \(t\), and a final linear layer producing a 2-dimensional noise vector. Train with AdamW at \(1 \times 10^{-3}\) for 20 epochs, sampling \(t \sim \text{Uniform}(1, T)\) each batch and forming \(x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1 - \bar{\alpha}_t} \epsilon\); log the loss and check for NaNs.
5. Run the reverse sampler from \(x_T \sim \mathcal{N}(0, I)\), subtracting the learned \(\epsilon_\theta(x_t, t)\) step by step, then compare the radius histogram of 5,000 samples to the baseline provided by `diffusionmodels1254ani/gemma-3-12b-it-heretic-v2`; the success criterion is that the mean radius difference stays within 0.08 of the dataset radius while the synthetic variance stays within 5%.

**Expected outcome:** A saved checkpoint and sampling script that reproduce the Swiss Roll manifold, plus an evaluation notebook showing that your denoiser meets the radius criterion and that its loss curve descends similarly to the reference sampler. If the generated radius drifts beyond 0.15, revisit your schedule or timestep embedding.

Stronger experiments include learning a tiny diffusion-aware UNet to tighten the radius metric, swapping in the linear β schedule from the original DDPM paper to see if the curvature of \(\bar{\alpha}_t\) matters, or adding classifier-free guidance by masking a pseudo-radius condition during training and observing how guidance weights affect diversity.

**Variants per persona:**
- **Applied AI/ML engineer:** Deploy the sampled reverse dynamics inside a Hugging Face Space that measures latency (target 120 ms p95 on an A10) and uses quantized model weights exported via `torch.jit.trace`; baseline radius stats come from the `f5aiteam/Diffusion_Models` metrics.
- **Research engineer:** Reproduce Table 1 from the related Flow Matching paper by training the same MLP but logging the sample quality (radius difference) at every 50 timesteps; aim to match their reported numbers within ±5% while instrumenting the pipeline with gradient norm logging.
- **Applied researcher:** Hypothesize that replacing the cosine noise schedule with the Minibatch OT-inspired schedule from Tong et al. (2024) will shrink the radius difference by ≥10%; falsify by plotting radius difference curves for both schedules over 20 seeds and confirming whether the new schedule consistently outperforms.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*
