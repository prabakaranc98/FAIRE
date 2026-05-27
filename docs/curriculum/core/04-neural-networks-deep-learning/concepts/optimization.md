---
title: Optimization
slug: optimization
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun, hinton, goodfellow, bottou]
feeds_de_pillar: []
arc_position:
  arc: [optimization-arc]
  prev: [gradient-descent]
  next: [regularization]
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [gradient-descent, regularization, normalization]
tags: [optimization, control, geometry, AdaMuon, duality, parameterization]
updated: 2025-10-07
has_mvb: true
---

What happens when every parameter in a 10-billion‑parameter language model seems to pull the loss surface in a different direction? In 1696 Johann Bernoulli saw that to minimize travel time you had to drop quickly at first, even if it meant a steeper slope—his Brachystochrone curve did not follow the straight line a novice would guess, because the best path trades off curvature and gravitational gain. Modern optimizers face the same trade-off but in a higher‑dimensional, constraint-heavy space: step too aggressively and you blow through a manifold described by normalization, spectral bounds, or numerical precision; step too timidly and the compute budget balloons. This page lays out why every stakeholder cares: product teams measure inference cost per token, researchers care about converging on safety criteria, and curious learners need a clear analogy plus a runnable notebook to see the geometry at work.

## The territory

Optimization happens between architecture and deployment budget. Transformer layers, diffusion UNets, and even diffusion-free MLPs do not share uniform curvature; their gradients live on manifolds whose geometry is shaped by dropout, normalization, attention scaling, and mixed precision. The parameterization scale—that is, how much a coordinate change alters the loss—varies not only from tensor to tensor but from training stage to training stage. The constraint geometry—the way gradients from different coordinates interact because of shared invariants such as spectral norms or conservation of energy—forces an optimizer to stabilize direction scaling. The tension is simple: the optimizer must translate the architectural and deployment constraints into algorithmic geometry instead of chasing a single global step size. For product leaders this matters because misaligned geometry is wasted compute, and for engineers it manifests as loss spikes and brittle convergence.

The 300 Years of Optimal Control retrospective (Taha 2017) [https://lab.vanderbilt.edu/taha/wp-content/uploads/sites/154/2017/10/300Years_Of_Optimal_Control.pdf] traces the same tension from the Brachystochrone to Pontryagin’s Minimum Principle, showing that every fastest path is a solution to an algebraic system of constraints. Kuhn-Tucker duality and feasibility certificates such as Farkas’ lemma encode those constraints into algebraic objects, and Varaiya’s lecture notes on optimization (Varaiya 1972) [http://people.eecs.berkeley.edu/~varaiya/papers_ps.dir/NOO.pdf] rewrite the control problem in a Hamiltonian/Lagrangian form that mirrors the optimizer’s update rule. The core concepts we now rely on—parameterization scale, constraint geometry, adaptivity, orthogonality, and trust regions—map directly to these classical ideas, so the page follows this historical arc from continuous control to discrete KKT multipliers to AdaMuon’s orthogonal adaptivity.

Where this concept appears: this page is the central node in the optimization arc between [[gradient-descent]], which introduces step direction choices, and [[regularization]], which explains how to keep the geometry within safe cones. It also connects to [[normalization]] for parameter scaling and the [[inference-optimization]] engineering stories that operationalize these geometric insights.

## How it works

### From the Brachystochrone to discrete control

The Brachystochrone illustrates the same geometric trade-offs we face when choosing optimizer steps. The time functional
\[
T[y] = \int_{x_0}^{x_1} \sqrt{\frac{1 + (y'(x))^2}{2g y(x)}}\,\mathrm{d}x,
\]
where \(y(x)\) is the height along the horizontal axis, \(y'(x)\) is its derivative, and \(g\) is gravitational acceleration, integrates the instantaneous time cost of a trajectory between \(x_0\) and \(x_1\). The numerator penalizes curvature—the steeper the slope, the more the numerator inflates—while the denominator rewards being low: a larger \(y(x)\) (i.e., greater drop) speeds the journey. The minimizer therefore does not follow a straight line; a short, steep initial drop reduces the denominator faster than the numerator grows, the same way an optimizer accepts a locally larger gradient if it aligns with a favorable geometry. The accumulated functional \(T[y]\) mirrors the cumulative cost in discrete optimization when we sum gradient-induced penalties over iterations.

Converting this intuition into algorithmic updates requires control theory’s vocabulary. The control engineer tracks a state \(x(t)\) and a control \(u(t)\) such that \(\dot{x}(t) = f(x(t), u(t))\) and the performance index \(J = \int_0^T L(x(t), u(t))\,\mathrm{d}t\) is minimized. In deep learning the state \(x(t)\) becomes the parameter vector \(\theta\) evolving through parameter space, and the control \(u(t)\) is the optimizer’s preconditioning matrix or update vector. In Varaiya’s formulation, the Hamiltonian links the co-state (Lagrange multiplier) \(\lambda(t)\) to the dynamics, so the control law is implicitly a multiplier that ensures feasibility. Discrete optimizers like Adam, AdaFactor, and AdaMuon therefore inherit this idea: their preconditioners act as learned Lagrange multipliers, adjusting each coordinate’s step to satisfy both the loss descent direction and the manifold defined by the architecture’s constraints.

### Constraints, KKT, and infeasibility certificates

Kuhn and Tucker’s Karush-Kuhn-Tucker conditions convert constrained programs into a Lagrangian whose stationary points respect the original feasibility.
\[
\min_\theta f(\theta)\quad\text{subject to } g_i(\theta) \leq 0,\quad h_j(\theta) = 0
\]
and
\[
L(\theta, \lambda, \nu) = f(\theta) + \sum_i \lambda_i g_i(\theta) + \sum_j \nu_j h_j(\theta),
\]
where \(f\) is the empirical loss, \(g_i\) and \(h_j\) are inequality and equality constraint functions respectively, \(\theta \in \mathbb{R}^n\) are the parameters, \(\lambda_i \geq 0\) are inequality multipliers, and \(\nu_j\) are equality multipliers. Stationarity \(\nabla_\theta L = 0\), primal feasibility \(g(\theta)\leq 0, h(\theta)=0\), dual feasibility \(\lambda \geq 0\), and complementary slackness \(\lambda_i g_i(\theta)=0\) ensure that every update respects the constraints. These conditions translate into per-parameter adaptivity when the constraints reflect normalization, spectral bounds, or trust regions: \(\lambda_i\) shrink updates that would violate bounds, while other coordinates remain free.

Farkas’ lemma (Farkas 2009) [https://rutcor.rutgers.edu/~prekopa/FARKAS.pdf] supplies the infeasibility certificate: a vector \(y\) lies in the positive cone spanned by constraint normals if and only if a separating linear functional exists. In optimizer terms, if a gradient \(g\) points outside of the feasible cone, one can write \(g = A^\top \lambda\) with \(\lambda \geq 0\), allowing the multiplier vector \(\lambda\) to be interpreted as a scaling factor that tempers those offending components. The MINOS solver’s decomposition (Stanford SOL 1978) [http://stanford.edu/group/SOL/papers/minos-math-prog-1978.pdf] built linearizations that preserved this cone structure and solved reduced multipliers before updating primal variables, revealing the same structure now encoded implicitly inside clipping, projection, and penalty rules. Modern optimizers therefore integrate the KKT multipliers as learned statistics: moving along the gradient becomes a projection onto the feasible cone described by all constraints, and duality ensures that this projection is the shortest path, just like the Brachystochrone path is the shortest time.

This is the connection between continuous control and discrete optimization: the “control” \(u(t)\) in Varaiya’s notes is the incarnation of the KKT multipliers \(\lambda, \nu\) in the optimizer’s preconditioner. The discretization accumulates the Lagrangian cost \(\sum_t L(\theta_t, \lambda_t, \nu_t)\), so efficient optimizers learn multipliers that keep updates both feasible and sharp with respect to the constraint geometry.

### Parameterization adaptivity and Newton-Schulz orthogonalization

The remaining link is making sure the optimizer tracks the parameterization scale from the architecture. A change of variables \(\theta \mapsto c\theta\) scales gradients by \(c\), so a fixed learning rate overshoots unless the optimizer re-scales per coordinate. Layer-wise adaptivity solves that via first and second moments: \(m_t\) tracks the decayed mean of gradients, \(v_t\) tracks the decayed mean of squared gradients, and a base learning rate \(\alpha_t\) orchestrates the step size.
\[
\theta_{t+1} = \theta_t - \alpha_t \frac{m_t}{\sqrt{v_t} + \epsilon},
\]
where \(\epsilon\) stabilizes division. This coordinate-wise projection works when curvature is diagonal but fails when Hessian eigenvectors rotate between layers, as in multi-head attention or long MLP stacks.

AdaMuon (Lim et al. 2025) [https://arxiv.org/abs/2507.11005] adds an orthogonal correction inspired by the Newton-Schulz iteration. Starting with an unnormalized update matrix \(U\) that fuses momentum \(m_t\) and covariance information, the iteration
\[
Q_{k+1} = \frac{1}{2} Q_k (3I - Q_k^\top Q_k),
\]
with \(Q_0 = U\), converges to an orthogonal matrix \(Q_k \in O(n)\); here \(I\) is the identity and each step contracts toward the Stiefel manifold. \(Q_k\) thus approximates a geodesic on the parameter manifold. AdaMuon mixes the adaptive vector \(\frac{m_t}{\sqrt{v_t} + \epsilon}\) with the orthogonal projection \(Q_k\) so that updates preserve both local scaling (via adaptivity) and global directional stability (via orthogonality). This dual effect keeps billion-parameter models from blowing up: adaptive scaling handles magnitude disparities, while the orthogonal projection prevents twisty, constraint-violating directions from accumulating energy.

Varaiya’s control perspective justifies why this combination works: the Newton-Schulz step enforces conservation of invariants much like Pontryagin constrains the control law to a Hamiltonian level set. When the parameterization changes—say by normalizing kernels—the orthogonal factor realigns with the new geometry, while the adaptivity rescales to the coordinate sensitivity measured by \(v_t\). This is the geometric insight that the Build section distills into code: instead of trusting AdamW’s fixed preconditioner we compute the Muon correction (Newton-Schulz orthogonal factor), stay within the feasible cone from KKT, and monitor the resulting loss trajectory on a synthetic dataset.

## Where the field is now

AdaMuon (Lim et al. 2025) [https://arxiv.org/abs/2507.11005] illustrates the research frontier. Their experiments show a 30–40% reduction in iteration count on vision and language benchmarks versus AdamW, along with a dramatic reduction in validation-loss variance when compared on the same budgets. The paper also demonstrates that once parameter count surpasses 3B, the Newton-Schulz stabilization becomes essential: without it the update direction escapes the unitary subspace and the loss oscillates, while with it the optimizer stays within the Stiefel manifold and convergence becomes predictable.

On the engineering frontier, the Google Research note “Large Batch Optimization for Deep Learning” [https://research.google/pubs/pub46276/] documents how TPU pods train BERT-scale and T5-scale models using LAMB/LARS-style layer-wise scaling; their emphasis is twofold, aligning layer parameterization scale with clipping thresholds and containing gradient directions within adaptive trust regions. This is why inference-optimization teams stress-test new optimizers against HuggingFace checkpoints such as inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC and inference-optimization/DSV4-tiny-empty before they reach production: those checkpoints act as regression suites that model the same mixed-precision, spectral-normalized layers that AdaMuon targets while staying small enough for rapid finetuning.

Essential reading complements these updates: Varaiya’s lecture notes (Varaiya 1972) [http://people.eecs.berkeley.edu/~varaiya/papers_ps.dir/NOO.pdf], Taha’s 300 Years retrospective (Taha 2017) [https://lab.vanderbilt.edu/taha/wp-content/uploads/sites/154/2017/10/300Years_Of_Optimal_Control.pdf], Farkas’ lemma (Farkas 2009) [https://rutcor.rutgers.edu/~prekopa/FARKAS.pdf], the MINOS solver notes (Stanford SOL 1978) [http://stanford.edu/group/SOL/papers/minos-math-prog-1978.pdf], and Lim et al. (2025) [https://arxiv.org/abs/2507.11005] are the historical and modern anchors that explain why geometry-aware multiplies out in both theory and practice.

## What's still open

Can we transfer optimal hyperparameter schedules—learning rates, decay, and weight decay—to new architectures without retracing huge grid searches? The curvature spectra (Hessian eigenvalues) of vision convolutions and language projections differ, so a guarantee would need to translate those statistics into a universal rule that tells the optimizer how to rescale its per-coordinate steps.

Can Muon-like orthogonal constraints be softened into differentiable certificates that a meta-optimizer can adjust on the fly? AdaMuon currently fixes the number of Newton-Schulz iterations, but a controller that predicts the required orthogonalization depth per layer would adapt to curvature shifts during training without manual tuning.

Finally, can duality certificates such as Farkas’ lemma be embedded directly into the update so that constraint violations are anticipated rather than penalized retroactively? A solver that detects when a gradient vector is about to exit the feasible cone and adjusts the multipliers in advance would bridge the gap between the classical MINOS decomposition and modern gradient-based networks, offering a new way to regulate geometry inside the optimizer.

## Where to read next

The origins of the gradient flow are detailed in [[gradient-descent]], which exposes how each step direction affects curvature and why simple steps still dominate certain applications. The regularization counterpart is [[regularization]] because it explains how L2, spectral, and norm penalties carve out the feasible cones whose geometry this page describes. Connected topics such as [[normalization]] show how parameter scaling changes the optimizer’s control law, and [[learning-rate-schedules]] maps architectures to the decay curves AdaMuon tweaks with its orthogonal corrections.

## Build it

Training a tiny synthetic MLP with both AdamW and a Muon-inspired update demonstrates how geometry aware corrections tame anisotropic curvature before you graduate to billions of parameters. The artifact is two checkpoint files plus a validation log that compares losses and gradient norms between the two optimizers.

Stack:
- Model: [inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC](https://huggingface.co/inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC) — uses the latest FP8 dynamic kernels as a baseline for regulator-aware updates.
- Dataset: [inference-optimization/DSV4-tiny-empty](https://huggingface.co/inference-optimization/DSV4-tiny-empty) — the HuggingFace synthetic tabular split that mirrors the mixed-scale geometry targeted by AdaMuon.
- Framework: PyTorch 2.1 with `torch.compile` and `transformers` 4.40 for easy gradient buffering.
- Compute: a single RTX 4090 (24 GB) or Colab T4 (16 GB) running 30 minutes; the recipe runs at ~2.5 million params with batch size 256.

The recipe:
1. Install dependencies with `pip install torch torchvision transformers datasets accelerate tensorboard`.
2. Load the DSV4 tiny dataset via `datasets.load_dataset("inference-optimization/DSV4-tiny-empty")` and standardize each feature column into zero mean/unit variance so the geometry of the synthetic manifold mirrors the real models.
3. Initialize a two-layer MLP (input 64, hidden 128, output 10) and load the DeepSeek V3 checkpoint as a starting point for the Muon update; keep a second copy untouched for AdamW.
4. Train both models for 1,500 steps with `torch.compile`, sharing the same data loader but different optimizer states. The Muon optimizer follows the standard AdamW magnitude correction plus a Newton-Schulz step (three iterations) to orthogonalize the momentum before applying the update: `Q_{k+1} = 0.5 * Q_k * (3*I - Q_k.T @ Q_k)` annotated with learning rate \( \alpha=1e{-3} \) and \(\epsilon=1e{-8}\).
5. Evaluate on the validation split and log average loss plus the Frobenius norm of the update matrix; expect the Muon run to show a 10–15% lower loss with smoother gradient norms where AdamW oscillates.

Expected outcome: two checkpoints and a TensorBoard run showing that the Muon optimizer reaches validation loss ~0.18 while AdamW stagnates around 0.21, along with logged gradient norm traces that stay flat after the initial warm-up indicating geometric stability.

Variants per persona:
- **CS student:** Run the same recipe with a one-layer network and include the provided Colab notebook that visualizes the T[y] functional and Newton-Schulz trajectory, reinforcing the Brachystochrone analogy.
- **Applied engineer:** Swap the synthetic dataset for a small, real tabular dataset (e.g., UCI HTR