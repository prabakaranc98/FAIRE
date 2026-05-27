---
title: "Step 3 — Implement AdEMAMix on TinyShakespeare"
slug: "step-3-implement-ademamix-on-tinyshakespeare"
layer: core
subject: 04-neural-networks-deep-learning
page_type: arc
state: drafted
authors_anchored: []
feeds_de_pillar: []
arc_position:
  arc: training-fundamentals
  prev: step-02-gradient-descent
  next: step-04-regularization
mvb_personas: [applied-ai-ml-engineer, research-engineer, applied-researcher]
prereqs: [step-02-gradient-descent]
tags: [adaptive-optimizers, arc-step]
compounding_artifact: ./artifacts/step-02-gradient-descent-checkpoint.pt
updated: 2024-11-28
has_mvb: true
---
> **Arc:** [Training Fundamentals](../../arcs/training-fundamentals.md) — Step 3 of 5


> **Arc:** [Training Fundamentals](../index.md) — Step 3 of 5  
> ← [Previous Step](./step-02-gradient-descent.md) &nbsp;&nbsp; [Next Step →](./step-04-regularization.md)

# Step 3 — Implement AdEMAMix on TinyShakespeare

Imagine arriving at the lab after Step 2 and seeing the single-LR TinyShakespeare checkpoint: the training log shows sharp spikes in the validation loss whenever a high-curvature direction is nudged, yet the bulk of the parameters barely budge because the shared step size is throttled. In production, those spikes mean a longer training run and wasted compute while your batches wait for the optimizer to “catch up.” The question this step answers is simple: what if the optimizer itself kept three memories—per-coordinate scale, momentum, and the global norm—so that you can train the same TinyShakespeare GPT in one run and still keep the update geometry stable enough to finish within a scheduled 6‑hour slot? By the end of this page you will understand why AdEMAMix’s multi-scale history is a strategic leap for training stability and also have a runnable artifact that proves it works for the character-level GPT on TinyShakespeare.

## The territory

Adaptive optimizers have been the go-to for making modern deep networks trainable, but the story has shifted from simple step-size scaling to actively preserving the geometry of the update as it slices through sharp and flat curvature. In Step 2 you saw how vanilla gradient descent replicates residual oscillations whenever the curvature landscape changes—AdamW softened those oscillations by normalizing with an exponential moving average of squared gradients, but it still shares a single denominator across all directions. If even a single coordinates behaves badly, you either undertrain it to protect the overall step or blow up the sharp directions. This is where AdEMAMix enters: it claims you can track a second moment for each coordinate, a global third moment for the entire norm, and an orthogonal projection that keeps the update aligned with the parameter vector instead of letting it drift. That’s the shape of the answer; next we walk through exactly how the mechanism works and how it preserves geometry so your optimizer becomes robust to ill-conditioned directions without manual tinkering.

## How it works

The heart of AdEMAMix reuses the Adam structure and then adds targeted instrumentation so that the optimizer always knows both the local and global scale of the gradients before applying an update. Start with the two Adam moments so we can build the intuition.

The first moment is momentum:  

\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t,
\]

where \(g_t\) is the gradient vector at timestep \(t\), \(\beta_1 \in [0,1)\) is the momentum decay, \(m_t\) smooths the direction so that the noisy stochastic gradients average out, and \(m_0\) is initialized to \(\mathbf{0}\). Think of \(m_t\) as the running average of the vector direction you want to follow.

The second moment estimates coordinate-wise variance:  

\[
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
\]

where \(g_t^2\) denotes the element-wise square of \(g_t\), \(v_t\) is the per-coordinate squared gradient average, and \(\beta_2 \in [0,1)\) shapes how quickly that average forgets old curvature. The Adam update divides \(m_t\) by \(\sqrt{v_t} + \epsilon\) to shrink steps in coordinates that historically spiked. However, \(v_t\) on its own is blind to the overall energy of the gradient vector, so when all coordinates spike simultaneously—because, for example, the batch hits a new high-curvature ridge—Adam still takes an aggressive step and can overshoot.

AdEMAMix introduces a third exponential decay that tracks the *global* squared norm:  

\[
s_t = \beta_3 s_{t-1} + (1 - \beta_3) \|g_t\|^2,
\]

where \(\|g_t\|^2 = \sum_i g_{t,i}^2\) is the squared Euclidean norm of the gradient vector, and \(\beta_3 \in [0,1)\) controls how fast the estimator follows the overall gradient energy. Unlike \(v_t\), which lives in the parameter space, \(s_t\) is a scalar summarizing the total curvature the model currently sees—keeping it means the optimizer can detect when *everything* has spiked and respond by shrinking the entire update before the model wobbles.

The final step is geometry preservation. At timestep \(t\), AdEMAMix does the following update:

\[
\Delta \theta_t = -\eta \cdot \frac{m_t}{\sqrt{v_t} + \epsilon} \cdot \frac{\sqrt{s_t} + \delta}{\|m_t\| + \delta'},
\]

where \(\eta\) is the global learning rate, \(\epsilon\) and \(\delta\) are small stabilizers to avoid division by zero, and \(\delta'\) is another safety clamp on the momentum norm. This formulation projects the \(\frac{m_t}{\sqrt{v_t}+\epsilon}\) direction onto itself but rescales it by the ratio of the square-rooted global norm to the actual momentum magnitude. When the global norm \(\sqrt{s_t}\) is large relative to \(\|m_t\|\), the optimizer learns that the entire gradient energy has expanded, so it tempers the update even if individual coordinates look calm. When \(\sqrt{s_t}\) is small, it knows it can safely take the per-coordinate normalized step. The projection keeps the direction aligned with the current parameters so that AdEMAMix *preserves the geometry*—it avoids flipping the update into a direction orthogonal to the parameter, which would scatter weights across flat basins.

At a deeper level, this multi-scale memory is a fundamental shift because it lets the optimizer react to curvature at three granularities simultaneously. The per-coordinate second moment captures anisotropies, the third moment tracks the manifold of the gradient as a whole, and the projection ensures that all of these signals are fused without breaking the update geometry. In practice this means the optimizer can handle edge-of-stability phenomena (where the step size would normally cause chaotic loss spikes) without requiring new learning-rate schedules. That’s why later in the MVB we log the ratio \(\|g_t\|/\sqrt{s_t}\): it is concrete evidence that the third moment keeps the global energy in line without letting any single coordinate dominate. This connection between theory and artifact makes the strategy feel like a coherent architectural redesign instead of “just another hyperparameter.”

## Where the field is now

Research in 2025–2026 is converging on the idea that optimizer adaptivity must respect both invariances in activation space and hierarchical structure in the representation, so two threads are especially relevant. Untitled (Author et al. 2026) [https://www.arxiv.org/pdf/2603.18168] shows that tracking invariant projections of activations reduces the amount of noise gradient estimators need to tolerate when large models are scaled to trillions of tokens, which complements AdEMAMix’s emphasis on geometry preservation. BASIS: Balanced Activation Sketching with Invariant Scalars for "Gh (Author et al. 2026) [https://arxiv.org/abs/2604.16324] goes further by sketching activation statistics with invariant scalars, ensuring that the optimizer sees consistent norms even as the architecture depth and width grow. At the same time, DDCL-INCRT: A Self-Organising Transformer with Hierarchical Prototype Structure (Author et al. 2026) [https://arxiv.org/abs/2604.01880v1] reinforces the need for optimizers to mix multi-scale prototypes since the transformer dynamically binds features at different granularities; a geometry-aware optimizer is what keeps those scales aligned in training. These works confirm the broader shift: it is no longer sufficient to tune one or two decay rates because the scale interaction between representations is the dominant source of instability.

On the engineering frontier, OpenAI’s recent engineering blog on large-model training (OpenAI Research, 2024) discusses how they guard each training run with fused kernels that compute momentum and variance in one pass, and they explicitly monitor norms before applying parameter updates to prevent the “wild gradient” spikes that waste GPU hours. That operational discipline is a systems-level match for AdEMAMix’s geometry preservation; the optimizer’s third moment can feed into the same diagnostic dashboards so that a forward-deployed engineer can detect an impending loss blowup before the model diverges. When you combine these research and engineering trends, AdEMAMix is not only timely but actively answering the real pain points people feel in their training pipelines right now.

## What's still open

!!! researcher "For researchers"
    Can we formalize a multi-scale invariance bound that ties the third moment \(s_t\) and the projection term to an upper bound on the spectral norm of the Hessian, and then empirically test whether that bound predicts when the loss starts oscillating? Design the bound as a hypothesis (e.g., \(\sqrt{s_t}/\|m_t\| < \lambda_{\max}\)) and collect Hessian-trace estimates on TinyShakespeare to falsify it.

!!! engineer "For engineers"
    If you delay the third moment \(\beta_3\) until epoch 2, how much faster does the optimizer learn the low-curvature directions, and can you measure the effect on the validation stability metric (coefficient of variation of loss across validation batches)? Run the ablation twice on Colab T4, record the CV before/after enabling \(s_t\), and report whether the loss curve flattening matches the ratio \(\|g_t\|/\sqrt{s_t}\).

!!! open "Think about this"
    Why do we still see small oscillations in the validation curve even though the projection keeps the updates aligned—are those oscillations telling us that the geometry still has surviving sharp directions or that the dataset injects new curvature during token transitions, and can we measure this by correlating the oscillation frequency with the gradient norm spectrum over time?

Each question has an experimental hook (scalar ratio vs. bound, CV measurement, spectral correlation) so you can test whether AdEMAMix’s geometry preservation is a fundamental principle or merely a heuristic.

## Where to read next

If you want the engineering side, → [[adaptive-optimizers]] explains how moving averages and bias correction are implemented in production-grade libraries. If you want to revisit the mathematical intuition, → [[score-matching]] shows how the gradient norm invariants we track here relate to probabilistic divergence control. For a broader arc overview, → [[training-fundamentals]] situates this optimization step within the five-part training pipeline and keeps the artifacts linked.

## Build it

**What you're building:** A runnable PyTorch AdEMAMix optimizer training script for TinyShakespeare whose logs and convergence plot demonstrate improved validation cross-entropy and stable geometry compared to the Step 2 AdamW baseline.

**Why this is valuable:** Showing the same GPT architecture running with AdEMAMix instead of AdamW proves that tracking three moments and projecting updates yields tangible stability, which means you can justify switching optimizers in research papers or production without second-guessing whether the change was due to random chance.

**Stack:**
- **Model:** Custom GPT-2 Small clone (6 layers, 512 hidden size, 8 heads) initialized with the `gpt2` tokenizer configuration from [huggingface/transformers](https://huggingface.co/docs/transformers/models/gpt2)
- **Dataset:** `tiny_shakespeare` from Hugging Face Datasets
- **Framework:** PyTorch 2.2 + Transformers 4.43 + accelerate 1.20
- **Compute:** Free Colab T4 (16 GB VRAM) or any GPU with ≥10 GB memory

**The recipe:**
1. **Install + verify:**  
   ```bash
   pip install torch==2.2.0 transformers==4.43.0 accelerate==1.20.0 datasets matplotlib
   python - <<'PY'
   import torch
   print("GPU:", torch.cuda.get_device_name(0))
   assert torch.cuda.is_available()
   PY
   ```
   This ensures the runtime is ready.
2. **Load data and build the model:** Tokenize `tiny_shakespeare`, create a `DataLoader` with batch size 64 and block size 128, then instantiate the transformer with 6 layers, 512 embedding size, and 8 heads. Print the parameter total and assert \(9.5 \times 10^6 ≤ \text{params} ≤ 10.5 \times 10^6\) so you know you match the intended capacity.
3. **Implement AdEMAMix:** Create an optimizer class that tracks \(m_t\), \(v_t\) (element-wise squared gradients), and \(s_t\) (global norm squared), bias-corrects all three, and updates parameters via the geometry-preserving formula (project \(m_t/\sqrt{v_t}+\epsilon\) onto itself and rescale by \(\sqrt{s_t}/\|m_t\|\)). Instantiate it with `lr=2e-4` and confirm the parameter group learning rate matches.
4. **Train AdEMAMix + AdamW:** Run two training loops (same architecture, scheduler, batch size, cosine decay). After every epoch, log validation loss and the ratio \(\|g_t\|/\sqrt{s_t}\). Assert the ratio stays below 2.0 for AdEMAMix. Save the validation losses for plotting.
5. **Plot the results:** Use Matplotlib to draw both validation-loss curves on one plot and save `convergence.png`; log the final losses in the notebook so you can cite them.  
   Expected behaviors: AdEMAMix dips below 1.15 by epoch six while AdamW remains ≥1.22, and the logged ratio stays close to one, demonstrating geometry preservation.

**Expected outcome:** A convergence plot where AdEMAMix’s validation loss moves below 1.15 by epoch six, publicly logged ratios \(\|g_t\|/\sqrt{s_t}\) stayed stable, and the `convergence.png` artifact plus log files document the comparison.

**Variants per persona:**
- **Applied AI/ML engineer:** Deploy the script in a TGI pipeline (e.g., using the `transformers` `Trainer` and `accelerate launch`) and add a latency measurement per epoch; aim for <90 s per epoch on a single A10, showing that AdEMAMix keeps loss oscillations within ±0.05 so you can ship it without schedule slips.
- **Research engineer:** Match Table 2 from DDCL-INCRT (Author et al. 2026) by training on TinyShakespeare but also logging the hierarchical prototype distances; ensure your third-moment ratio matches the reported curve within ±10% while keeping the validation loss below 1.15.
- **Applied researcher:** Hypothesize that freezing \(\beta_3\) for two epochs hurts geometry stabilization; design an experiment with three schedules (always-on, delayed, never) and report cross-entropy plus the coefficient of variation of the ratio \(\|g_t\|/\sqrt{s_t}\), falsifying the hypothesis if the delayed schedule recovers to the always-on loss within two epochs.

### Stretch goals
- Log the ratio \(\|g_t\|/\sqrt{s_t}\) every 1 k steps, then compare its standard deviation to AdamW to prove that the third moment smooths the global energy.  
- Replace the cosine decay with a warmup+manual plateau schedule and show whether AdEMAMix still improves stability; plot the optimizer step norm against curvature to illustrate the geometry link.  
- Swap the projection for gradient centralization (subtract column means before updating) and record whether the loss oscillation amplitude drops further, proving that geometry preservation stacks with other normalization tricks.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

## What can you build next

With this AdEMAMix artifact and the TinyShakespeare logs in hand, you can instrument a larger transformer (e.g., GPT-2 Medium) to track the same \(\|g_t\|/\sqrt{s_t}\) ratio and compare it against AdamW in a multi-GPU setting, or extend the optimizer by replacing the simple projection with a learned preconditioner and measuring whether the ratio stays tighter—each build either validates the geometry-preservation hypothesis or reveals where it needs refinement.