---
title: "Step 2 — Train DSM & Langevin Sampler on a Swiss Roll"
slug: step-02-dsm-langevin-sampler
layer: core
subject: 02-generative-modeling
page_type: arc-step
state: drafted
authors_anchored: [hyvarinen, song, vincent]
feeds_de_pillar: []
compounding_artifact: ddpm-denoiser-checkpoint
arc_position:
  arc: generative-stack
  prev: step-01-diffusion-models
  next: step-03-latent-diffusion-models
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [step-01-diffusion-models]
tags: [score-matching, diffusion, energy-based-models]
updated: 2025-02-10
has_mvb: true
---
> **Arc:** [Generative Stack](../../arcs/generative-stack.md) — Step 2 of 5


> **Arc:** [Generative Stack](../index.md) — Step 2 of 5  
> ← [Previous Step](./step-01-diffusion-models.md) &nbsp;&nbsp; [Next Step →](./step-03-latent-diffusion-models.md)

# Step 2 — Train DSM & Langevin Sampler on a Swiss Roll

Imagine you could judge whether a model really *knows* the Swiss Roll without letting it see that shape twice. Instead of watching it reverse a long diffusion chain step-by-step, you ask it: “Where on the hill would you move if you wanted to stay on the ridge?” If the model answers with consistent directional cues, you can follow those gradients with a handful of Langevin updates to stay on the roll. This step teaches you how to train that directional compass— the score of the data distribution—so that sampling reduces to following learned gradients rather than replaying a schedule. By the end, you will have validated that your score estimator, trained via denoising score matching, drives a sampler toward the manifold with only a few gradient steps, closing the gap between proof-of-concept diffusion reverse chains and genuinely reusable score-based generators.

## The territory

Diffusion steps give us a recipe for how noise can retreat back into structure, but the original DDPM artifact remains shackled to that fixed three-hundred-step choreography: every new sample replays the scheduler to fully reverse the forward process. Score matching offers a different promise. If you know the gradient of the log-density—how the probability mass climbs and falls—then you can climb the hill no matter how many steps the diffusion forward pass took. In other words, you only need the shape of the energy landscape; the constants that normalize it (the partition function) fall away, and the sampler becomes an optimization around that landscape.

This step sits between the diffusion artifact you built in Step 1 and the latent diffusion architectures to come. It takes the denoiser’s insight—each timestep needed a network to guess the reverse step—and turns it into one universal score network that works for any noise level. That universal network is what latent diffusion simply plugs into when modeling compressed latents instead of pixels. The territory, then, is learning a reusable score field over a known manifold and using it as the engine for gradient-based sampling, so the next step can focus on stacking score estimators inside autoencoders rather than re-deriving them from scratch.

## How it works

The theoretical heartbeat of score matching is that gradients eliminate constants. Suppose our model assigns an unnormalized density \(\tilde{p}_\theta(x)\) to each point and the true distribution is \(p_{\text{data}}(x)\). The partition function \(Z_\theta = \int \tilde{p}_\theta(x)\, dx\) is a nuisance because it depends on \(\theta\) but is hard to compute. When training the score network \(s_\theta(x)\) to match \(\nabla_x \log \tilde{p}_\theta(x)\), the gradient of \(\log Z_\theta\) drops out because \(\nabla_x \log Z_\theta = 0\)—the constant disappears. Hyvärinen (2005) [https://jmlr.csail.mit.edu/papers/volume6/hyvarinen05a/old.pdf] formalized this by defining the objective

\[
\mathcal{L}_{\text{SM}}(\theta) = \mathbb{E}_{p_{\text{data}}(x)} \left[ \sum_{i=1}^d \partial_i s_{\theta, i}(x) + \frac{1}{2} \| s_\theta(x) \|_2^2 \right],
\]

where \(d\) is the ambient dimension (three for the Swiss Roll if you keep the \(z\) coordinate), \(s_{\theta}(x)\) is the learned score vector, and \(\partial_i s_{\theta, i}(x)\) denotes the divergence of the network. This divergence term sprays away the intractable normalization constant while the norm penalizes huge gradients, so the optimization only depends on observable quantities. The underlying insight is that matching gradients is cheaper than matching densities, yet it is consistent even when \(\tilde{p}_\theta\) omits \(Z_\theta\).

Unfolding this into a practical training procedure leads to denoising score matching. Instead of computing divergence explicitly—which requires second derivatives—you corrupt \(x\) with Gaussian noise \( \epsilon \sim \mathcal{N}(0, I)\) at scale \(\sigma\) and train a regression to recover the scaled noise vector. Song & Ermon (2019) [https://ar5iv.labs.arxiv.org/html/1907.05600] showed that

\[
\mathcal{L}_{\text{DSM}}(\theta) = \mathbb{E}_{p_{\text{data}}(x),\ \epsilon \sim \mathcal{N}(0, I)} \left[ \left\| s_\theta(x + \sigma \epsilon) + \frac{\epsilon}{\sigma} \right\|_2^2 \right],
\]

where \(\sigma\) controls how far the corrupted sample is from the clean point. This regression objective matches the corrupted sample to the direction that would undo the corruption, and because it depends only on finite differences, it dispenses with Hessians. Vincent (2011) [https://arxiv.org/pdf/0906.4779] further connected this regression interpretation to denoising autoencoders, revealing that the reconstruction function in a DAE is implicitly estimating the score. Hyvärinen (2012) [https://export.arxiv.org/pdf/1205.2629v1.pdf] expanded score matching to handle discrete or bounded domains by replacing differential operators with carefully designed functions, showing that the gradient-based view can reach beyond strictly continuous spaces.

Once you have a score estimator, sampling amounts to Langevin dynamics. Given a current point \(x_t\), you repeatedly update

\[
x_{t+1} = x_t + \frac{\eta}{2} s_\theta(x_t) + \sqrt{\eta}\, \xi_t,
\]

where \(\eta\) is the step size and \(\xi_t \sim \mathcal{N}(0, I)\) adds the correct stochasticity so the chain converges to the distribution whose score you estimated. The \(s_\theta\) term pulls samples toward higher-density regions, while the noise term prevents collapse. This is how the gradient field trained by the DSM objective becomes a sampler, closing the gap between the explicit DDPM reverse chain from Step 1 and the single-network sampling process of this step. The recipe will therefore sample from a single corruption level per batch while chaining over multiple Langevin updates—each iteration is a lightweight reverse step guided by \(s_\theta\) rather than a full DDPM stage.

Unlike DDPM’s multi-stage scheduler, DSM lets you reuse one model across noise scales by mixing noise-aware inputs during training. The recipe’s geometric noise schedule is a practical approximation to the continuous \(\sigma\) spectrum discussed in theory; it smoothly interpolates from large noise (high uncertainty) to small noise (fine details), which keeps gradient norms well-behaved. Because the recipe’s loss is purely regression-based, the sampling loop simply asks the learned \(s_\theta\) “Which way does density climb?” and uses Langevin updates to heuristically follow that advice, which is why the theoretical divergence of Hyvärinen and the DSM regression objective show up as two faces of the same training signal.

### Architectural choices and stability

The recommended architecture in this step is a modest MLP: a handful of hidden layers with ReLU activations leading to a linear output. Linear outputs are standard for score vectors because the targets are real-valued gradients; activations like SELU or spectral normalization on every layer can be experimented with (see optional variants below), but they are not required and can impede regression when the score values cross zero. Instead, focus on depth, residual connections, or FiLM-style conditioning on \(\sigma\). Doing so echoes the design of larger score-based models (e.g., Song et al. 2020) while staying small enough to train on a single 12GB GPU.

For \(\sigma\), the recipe adopts a geometric sequence \(\sigma_t = \sigma_{\text{max}}^{1 - t/T} \sigma_{\text{min}}^{t/T}\) to keep the effective learning rate consistent across scales and to guarantee coverage from broad to fine corruptions. This schedule mirrors the annealed Langevin samplers in modern papers where each noise level requires a separate Langevin warm-up. When the recipe runs a chain of Langevin steps at the smallest \(\sigma\), the noise term shrinks and the sampler simply climbs the learned gradient field, which is when Chamfer distance is evaluated.

### Evaluation with Chamfer distance

Chamfer distance measures how close each generated point is to the training manifold by computing nearest-neighbor distances in both directions. Computing it naively over 10k generated points against 7k training points leads to 70 million pairwise distances, so the recipe recommends using a fast approximate neighbor search (faiss or sklearn’s `NearestNeighbors` with `algorithm='ball_tree'`) or by evaluating on a random subset (e.g., 2k samples vs. 2k training points) when compute is limited. This evaluation confirms whether the Langevin chain has stayed on the roll, providing a light-weight metric that matches the DSM claim.

## Where the field is now

Score-based generative modeling has grown from Swiss Roll-sized experiments to whole-image synthesis benchmarks. Song et al. (2020) [https://arxiv.org/abs/2011.13456] reformulated score matching as a stochastic differential equation (SDE), trained NCSN++ to estimate the score at every noise scale, and reached \(2.1\) FID on ImageNet 64×64. That paper also quantified how annealed Langevin sampling trades off computation and quality by integrating over the continuous noise spectrum, showing that modern SDE solvers can match the quality of explicit DDPM chains with fewer steps. On the engineering side, Stability AI’s Score Distillation Sampling (SDS) workflow (https://stability.ai/blog/score-distillation-sampling) uses a pretrained diffusion model to supervise a student network via score matching, making it fast enough to run interactive DreamStudio previews. SDS illustrates the production frontier: score estimators can power real-time image synthesis so long as Langevin-style updates remain affordable.

Research frontier: multi-scale score awareness. Recent work such as "Diffusion Models Beat GANs on Image Synthesis" (Dhariwal & Nichol 2021, https://arxiv.org/abs/2105.05233) and its successors now blend classifier-free guidance with score estimators, but there is still no consensus on how to best regularize the learned \(s_\theta\) to generalize across unseen noise scales. Integrating spectral normalization or adaptive step sizes guided by theoretical bounds remains an open calibration problem.

Engineering frontier: sample efficiency vs. compute. Latent diffusion models reuse DSM-trained score modules inside compressed latents, but the interface between the score estimator and decoder is fragile—quantization, caching, and batching all affect Langevin sampling stability. The field is experimenting with “score caches” and low-latency spline solvers, but the reliability of those systems in production remains to be proven.

## What's still open

For researchers, the key question is how to tie the regression-heavy DSM objective back to divergence minimization in high dimensions. Can we derive explicit bounds for general classes of manifolds, possibly by extending Hyvärinen’s divergence-based consistency results, so that the regression loss guarantees the sampler converges without hand-tuning the Langevin hyperparameters? Generalized score matching [Hyvärinen 2012](https://export.arxiv.org/pdf/1205.2629v1.pdf) points toward differential operators that survive on discrete domains, but its implications for high-dimensional continuous data remain under-explored.

For engineers, the practical worry is whether a DSM-trained score survives deployment when noise levels, compute budgets, or input distributions drift. Building lightweight diagnostics (e.g., tracking Chamfer distance drift or directional variance across noise levels) would allow teams to detect failure before Langevin chains collapse, and experimenting with learned step sizes or adaptive noise scaling may reduce the number of Langevin iterations needed to stay on-manifold.

## Where to read next

If you want a deeper dive into the theoretical foundation, the derivations on [[score-matching]] trace Hyvärinen’s original divergence and its links to denoising autoencoders, while the engineering counterpart [[diffusion-models]] lays out how DDPM sampling chains behave when treated as iterative Langevin processes. To understand how DSM-trained scores plug into large-scale synthesis, → [[latent-diffusion-models]] explains how these estimators become latent-space teachers for decoders.

## Build it

**What you’re building:** A DSM-trained score network and Langevin sampler that stays on the Swiss Roll manifold for at least a thousand samples, measured by Chamfer distance.

**Why this is valuable:** Establishing that a single network can learn the gradient field and guide a few Langevin steps lets you replace lengthy DDPM chains with lightweight score-based samplers—the critical capability the next arc step expects when it reuses the score estimator inside a learned latent space.

**Stack:**
- **Model:** Custom 4-layer MLP (input 3, hidden 256, ReLU, linear output) that regresses the denoising score
- **Dataset:** `sklearn.datasets.make_swiss_roll` (7,000 points, keep the \(z\) channel); wrap in a `Dataset` to easily batch and shuffle
- **Framework:** PyTorch 2.1 with `torchvision` for transforms and `faiss` for fast nearest-neighbor distance computation
- **Compute:** Google Colab T4 or a local RTX 3060 (≤12 GB VRAM); 500 epochs take ~40 minutes on that hardware

**Estimated time:** 40–50 minutes for preprocess, 500 epochs of DSM training, and sampling/evaluation.

**Success criterion:** Chamfer distance between 2,048 Langevin samples and 2,048 held-out clean points stays under 0.12 when the loss curve settles below 0.05.

**The recipe:**

1. Generate the dataset via `make_swiss_roll(7000, noise=0.0)` from `sklearn`; standardize each axis to zero mean/unit variance and wrap into a `TensorDataset`. Inspect `dataset.shape` and assert `(7000, 3)` to confirm dimensionality.

2. Define the score estimator as an MLP with input dimension 3, three hidden layers of 256 units each, ReLU activations, and a final linear projection to 3 outputs. Keep all layers linear (no SELU) and skip spectral normalization on every layer—those are optional experiments. Print the total parameter count and ensure it is ≈200k.

3. Build a geometric noise schedule \(\sigma_t = \sigma_{\text{max}}^{1 - t/T} \sigma_{\text{min}}^{t/T}\) with \(T=10\), \(\sigma_{\text{max}}=1.0\), and \(\sigma_{\text{min}}=0.01\); assert `len(sigmas) == 10`. Each batch randomly samples a \(\sigma\) from this schedule.

4. Train with batch size 256 and AdamW (lr \(2\times10^{-4}\)). For each batch, sample \(\epsilon \sim \mathcal{N}(0, I)\) and compute:

   ```python
   noisy = clean + sigma[:, None] * torch.randn_like(clean)
   target = -(noisy - clean) / (sigma[:, None] ** 2)
   loss = (model(noisy) - target).pow(2).sum(dim=1).mean()
   ```

   Log `loss.item()` per epoch and `assert not torch.isnan(loss)` before `loss.backward()`.

5. Sample 2,048 initial points from `torch.randn(2048, 3)` and run 300 Langevin iterations with step size \(\eta=0.1\) and noise scale \(\sqrt{0.02}\):

   ```python
   samples = torch.randn(2048, 3, device=device)
   for _ in range(300):
       grad = model(samples)
       noise = torch.randn_like(samples)
       samples = samples + 0.1 * grad + math.sqrt(0.02) * noise
   ```

   Validate `samples.shape == (2048, 3)` after the loop and print `samples.norm(dim=1).mean()`.

6. Evaluate Chamfer distance using FAISS or sklearn’s ball-tree on a subset of 2,048 training points and 2,048 samples to avoid quadratic cost. Average the directional nearest-neighbor distances and assert the value is <0.5 to ensure the metric runs; expect a final value near 0.12.

**Expected outcome:** The trained network produces samples whose Chamfer distance stays below 0.12, demonstrating that 300 Langevin steps are sufficient to stay on the Swiss Roll. The DSM loss should stabilize below 0.05, and nearest-neighbor checks should confirm points still trace the roll.

**Variants per persona:**
- **Applied AI/ML engineer:** Deploy the scoring pipeline in a lightweight REST endpoint; quantize the MLP with PyTorch FX, batch samples in the Langevin loop, and serve with FastAPI reaching >30 qps on an RTX 3060.
- **Research engineer:** Reproduce Table 2 from Song et al. (2020) [https://arxiv.org/abs/2011.13456] on a toy 2D spiral by training an MLP score estimator and matching their FID within ±0.5 while logging per-noise-level losses.
- **Applied researcher:** Hypothesize that replacing fixed Langevin noise with learned per-step scales (\(\eta_t\) and \(\sigma_t\)) reduces Chamfer distance by at least 10%; design experiments comparing fixed vs. learned scales across three noise schedules and plot Chamfer vs. step count.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*