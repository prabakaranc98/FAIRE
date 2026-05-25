---
title: Flow Matching
track: 02-generative-modeling
tags: [flow-matching, continuous-normalizing-flows, optimal-transport, rectified-flow, cfm]
depth: all
prereqs: [normalizing-flows, diffusion-models]
updated: 2026-05-25
has_mvb: true
---

# Flow Matching
> **TL;DR:** A training framework for continuous normalizing flows that learns straight-line paths from noise to data — faster training and fewer inference steps than diffusion, now the backbone of FLUX, AlphaFold 3, and frontier audio models.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Run a flow matching model and understand why it's faster |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Understand what "flow matching" means and why it beat diffusion |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Derive the CFM objective and understand why OT paths are optimal |
| Researcher / frontier | [Current SotA](#current-sota) → [What's happening now](#whats-happening-now) | Know the open problems: discrete flow matching, scaling laws |

---

## What it is

Flow matching trains a neural network to learn a **vector field** v_θ(x, t) that transports samples from a source distribution p_0 (usually Gaussian noise) to a target distribution p_1 (the data). The flow is defined by an ODE: dx/dt = v_θ(x, t). Integrating this ODE from t=0 to t=1 transforms a noise sample into a data sample.

The key insight is **simulation-free training**: instead of integrating the ODE during training (expensive), you directly regress against a conditional vector field that you can compute analytically from pairs (noise, data). The conditional flow between a single noise sample x_0 and a single data point x_1 is just the straight line: u_t(x | x_1) = x_1 − x_0. This is why flow paths are straight: the model learns to interpolate linearly between noise and data.

**Why this beats diffusion:** Diffusion models add and remove noise in curved paths through high-dimensional space. Flow matching uses straighter paths, so the vector field is easier to learn and you need fewer ODE steps (8–20 vs. 50–1000) at inference to get the same quality.

## Why it matters at the frontier

Flow matching is now the preferred training paradigm at the image, audio, and molecular generation frontier. FLUX.1 (best open-weight image generator, 2024) uses Rectified Flow — a flow matching variant. AlphaFold 3 uses a flow matching-like diffusion module for all biomolecule structure prediction. Stable Audio and Voicebox use it for audio. The reason: same quality as diffusion, 4–8× faster inference, simpler training objective.

The theoretical unification is also compelling: flow matching, diffusion, and score matching are all special cases of the same probabilistic transport framework. Understanding flow matching means understanding the whole generative modeling landscape.

## Core concepts

- **Continuous normalizing flow (CNF)** — a generative model defined by an ODE: dx/dt = v_θ(x, t); integrating gives a deterministic map from noise to data
- **Vector field** — v_θ(x, t): a neural network that predicts the direction to move at position x and time t
- **Conditional flow matching (CFM)** — the core training trick: regress v_θ against a conditional vector field computed from individual (noise, data) pairs; avoids integrating the marginal ODE
- **OT paths (optimal transport)** — straight-line paths x_t = (1−t)x_0 + tx_1 between noise and data; minimize transport cost; the simplest and most compute-efficient choice
- **Rectified Flow** — a specific CFM variant: straight-line paths + reflow distillation to further straighten paths; backbone of FLUX and SD3
- **Simulation-free training** — train without integrating the ODE; sample t ∈ [0,1], compute x_t on the straight path, regress v_θ against (x_1 − x_0); simple and efficient
- **Stochastic interpolants** — a generalization unifying flow matching and diffusion: x_t = α(t)x_0 + β(t)x_1 + γ(t)ε with different schedules giving different models

## Mathematical foundations

Flow matching objective (marginal, intractable):
$$\mathcal{L}_{FM} = \mathbb{E}_{t,\, p_t(x)}\!\left[\|v_\theta(x,t) - u_t(x)\|^2\right]$$

Conditional flow matching (tractable — same gradient, efficiently computable):
$$\mathcal{L}_{CFM} = \mathbb{E}_{t,\, q(x_0),\, p(x_1)}\!\left[\|v_\theta(x_t, t) - u_t(x_t \mid x_1)\|^2\right]$$

where on OT (straight-line) paths:
$$x_t = (1-t)\,x_0 + t\,x_1, \qquad u_t(x_t \mid x_1) = x_1 - x_0$$

The key theorem (Lipman et al. 2022): **L_FM and L_CFM have the same gradient** — so training on the tractable conditional objective exactly optimizes the marginal objective.

Inference: integrate the learned ODE from t=0 to t=1 using any ODE solver (Euler, RK4, DPM-Solver):
$$x_1 \approx x_0 + \int_0^1 v_\theta(x_t, t)\, dt$$

## Key algorithms / techniques

- **Conditional Flow Matching (CFM)** (Lipman et al. 2022) — the core method; train on conditional vector fields; any path works in principle, OT paths work best
- **Rectified Flow** (Liu et al. 2022) — straight-line paths + reflow: train a second model to straighten paths further; enables 1-step generation after distillation
- **OT-CFM / Minibatch OT** (Tong et al. 2023) — use optimal transport coupling between mini-batch noise and data; better path quality than independent coupling
- **Stochastic Interpolants** (Albergo & Vanden-Eijnden 2023) — unified framework: α(t)x_0 + β(t)x_1 + γ(t)ε with different schedule choices recovering diffusion, flow matching, and DDPM
- **Discrete Flow Matching** (Gat et al. 2024) — extends flow matching to categorical / token data; enables discrete generation without autoregression

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | 2022 | Lipman et al. | Original CFM — the core theorem and method |
| [Flow Straight and Fast: Rectified Flow](https://arxiv.org/abs/2209.03003) | 2022 | Liu et al. | Straight-line paths; reflow distillation; backbone of FLUX |
| [Improving and Generalizing Flow-Matching with Minibatch OT](https://arxiv.org/abs/2302.00482) | 2023 | Tong et al. | OT coupling improves path quality |
| [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) | 2024 | Lipman et al. | Comprehensive tutorial by the original authors; best starting point |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | 2022 | Conditional flow matching — the foundational method |
| [Flow Straight and Fast: Rectified Flow](https://arxiv.org/abs/2209.03003) | 2022 | Straight-line paths; the basis for FLUX and SD3 |
| [Normalizing Flows for Probabilistic Modeling](https://arxiv.org/abs/1912.02762) | 2019 | Papamakarios et al. — classical antecedent; flow matching solves its key limitation |

## Current SotA
> *Updated: 2026-05-25*

**Image:** FLUX.1 (Black Forest Labs, 2024) — Rectified Flow + DiT backbone at 12B parameters — is the current best open-weight image generator on most benchmarks. **Audio:** Stable Audio (Stability AI) and Meta's Voicebox use flow matching for high-quality waveform generation. **Molecular:** AlphaFold 3 uses a flow-matching-like diffusion module for all biomolecule types.

## What's happening now
> *Research · Engineering · Systems*

**Research:** Discrete flow matching (Gat et al., ICML 2024) extends CFM to token-space, enabling flow-based language generation without autoregression. Consistency flow matching combines flow matching paths with consistency model distillation for 1-step generation.

**Engineering & Systems:** Production systems (FLUX) use flow matching with DDIM-style deterministic ODE solvers and 8-step inference. HuggingFace Diffusers added flow matching schedulers; most new image generation systems default to Rectified Flow or OT-CFM.

**Open problems:** How many ODE steps are truly needed for production quality? Can flow matching generalize to video (temporally consistent flows)? How do you scale discrete flow matching to large language models?

## In production
> *How top labs and companies have deployed this at scale*

- **Black Forest Labs (FLUX.1):** Rectified Flow + 12B DiT architecture; current best open-weight text-to-image model; available via HuggingFace and their API. [huggingface.co/black-forest-labs](https://huggingface.co/black-forest-labs)
- **Stability AI (Stable Diffusion 3):** OT-CFM-based training + multi-modal diffusion transformer; improved text rendering and composition over SD 1.5. [arxiv.org/abs/2403.03206](https://arxiv.org/abs/2403.03206)
- **Meta AI (Voicebox):** Flow matching for audio generation across languages; in-context learning for speech editing. [arxiv.org/abs/2306.15687](https://arxiv.org/abs/2306.15687)
- **DeepMind (AlphaFold 3):** Flow matching-style diffusion over atomic coordinates for joint protein-DNA-RNA-ligand structure prediction. [nature.com/articles/s41586-024-07487-w](https://www.nature.com/articles/s41586-024-07487-w)

## Minimum Valuable Build

**What you're building:** A minimal flow matching image generator trained from scratch on MNIST — you'll implement the CFM training loop, the ODE sampler, and generate handwritten digits from noise in 8 ODE steps.

**Why this is valuable:** Flow matching is the new default for generative AI. Implementing it from scratch gives you the intuition that code-reading never provides: why the vector field learns, what "straightness" means in practice, and why 8 steps beats diffusion's 1000.

**Stack:**
- **Model:** Small UNet or MLP denoiser (implement from scratch — ~100 lines)
- **Dataset:** [mnist](https://huggingface.co/datasets/ylecun/mnist) on HuggingFace Datasets
- **Framework:** PyTorch + `torchdiffeq` (for ODE solving) or manual Euler integration

**The recipe:**

1. **Set up the data:** Load MNIST, normalize to [−1, 1]. Each sample x_1 ∈ ℝ^784.
2. **Implement CFM training loop:**
   - Sample noise x_0 ~ N(0, I), data x_1, time t ~ U[0, 1]
   - Compute x_t = (1−t)x_0 + t·x_1 (OT straight path)
   - Compute target vector u_t = x_1 − x_0
   - Train v_θ(x_t, t) to minimize ‖v_θ − u_t‖²
3. **Implement Euler ODE sampler:** 8 steps from x_0=noise to x_1=generated digit
4. **Generate and evaluate:** Sample 100 images; compare quality vs. a 1-step vs. 8-step vs. 50-step ODE solver

**Expected outcome:** Generated MNIST digits after ~30 minutes of training, plus intuition for how the path "straightens" as the model trains and why fewer steps work.

**Stretch goals:**
- Run FLUX.1 via `diffusers` on a prompt, inspect the CFM scheduler, compare 4-step vs. 8-step quality
- Implement OT-CFM: use `scipy.optimize.linear_sum_assignment` to couple mini-batches optimally; observe cleaner paths

## Code & implementations

- [huggingface/diffusers](https://huggingface.co/docs/diffusers) — includes `FlowMatchEulerDiscreteScheduler` (FLUX scheduler)
- [Flow Matching Guide and Code — official repo](https://arxiv.org/abs/2412.06264) — reference implementation by the original authors
- [black-forest-labs/flux](https://huggingface.co/black-forest-labs/FLUX.1-dev) — FLUX.1 weights and inference code

## Connected topics

- [[diffusion-models]] — flow matching is the successor framework; same generation task, straighter paths
- [[normalizing-flows]] — classical antecedent; flow matching solves the simulation-during-training bottleneck
- [[score-matching]] — deep connection: the probability flow ODE of a score model is a special case of flow matching
- [[consistency-models]] — one-step distillation of diffusion/flow models

## Further reading

- [Flow Matching Guide and Code](https://arxiv.org/abs/2412.06264) — Lipman et al. 2024; definitive tutorial with proofs and code
- [Stochastic Interpolants: A Unifying Framework](https://arxiv.org/abs/2303.08797) — Albergo & Vanden-Eijnden 2023; unifies all continuous generative models
- [Scaling Rectified Flow Transformers for High-Resolution Image Synthesis (SD3)](https://arxiv.org/abs/2403.03206) — Esser et al. 2024; production scaling
