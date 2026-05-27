---
title: "Step 2 — Gradient Descent Updates on MNIST"
slug: step-2-gradient-descent-mnist
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun, goodfellow, duchi]
feeds_de_pillar: []
arc_position:
  arc: training-fundamentals
  prev: step-01-backpropagation
  next: step-03-adaptive-optimizers
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [backpropagation]
tags: [optimization, mnist, gradient-descent]
updated: 2024-12-01
has_mvb: true
---
> **Arc:** [Training Fundamentals](../../arcs/training-fundamentals.md) — Step 2 of 5


# Step 2 — Gradient Descent Updates on MNIST

What do you do once backprop gives you a pile of gradients but the loss barely budges and the training accuracy flips between 10 % and NaN? Every beginner moment of deep learning is a story about movement: you have a local slope, you have a choice of how big a step to take, and you have to turn that slope into a stable rhythm of updates without skidding off the cliff. This page walks through the second act of the arc—turning those gradients into explicit parameter motion—so that by the end you can compare vanilla SGD and momentum on MNIST, understand why certain heuristics tame the “zig-zag” in high curvature, and feel confident that your own loop is the instrument of change rather than a library black box. You’ll see how the same gradient signal behaves under different scalings, why adaptive accumulators have been a touchstone since the 2010s, and where the frontier still slices into curvature estimation and large-scale stability.

## The territory

Gradient descent is the mundane magician behind every supervised deep learning run: it converts each backpropagated gradient—your estimate of the steepest ascent—into a change of weights that walks downhill. The whole problem of optimization is one of translation. Backprop gives you a direction; gradient descent must choose how far to go. Step size and curvature are the key partners. If the step is too small, you never leave the valley; if it is too large, you ignore the local quadratic approximation and bounce between walls.

This is why this step exists in the arc. Backpropagation (Step 1) handed you gradients but not the map that moves parameters. We now build that map ourselves with a tiny two-layer MLP on MNIST, because seeing the numerical updates lets you tease apart divergence, oscillation, and stability before you layer in the abstractions of library optimizers. Along the way you rehearse the ideas that have kept large-batch training stable since the mid-2010s: the field’s “Untitled” 2016 explorations of manual gradient control (Untitled 2016) and the companion follow-up on handling internal covariate shift before batch normalization became ubiquitous (Untitled 2016) both reaffirm that understanding raw updates pays dividends even in modern ResNet-scale training. The next section turns this intuition into concrete math so you can ask—not just assert—why the update behaves the way it does.

## How it works

### The basic update and the curvature tension

The canonical gradient-descent update rewrites a local linear approximation of the loss. Write the loss as \(L(\theta)\), where \(\theta\) collects all trainable parameters and \(t\) indexes the update step. The formula is

\[
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t),
\]

where \(\eta\) is the scalar learning rate, \(\nabla_\theta L(\theta_t)\) is the gradient computed by backprop, and \(\theta_t\) is the current parameter vector. This says that every parameter moves along the direction of steepest descent scaled by \(\eta\). Because we approximate the local loss landscape by its first-order Taylor expansion, the curvature—how much the slope changes—is captured in the Hessian, but we never compute the Hessian explicitly in this step. The only protection against curvature is the step size, so the entire game is about picking \(\eta\) small enough to avoid stepping over the local convex quadratic but large enough to exit the valley in reasonable time.

If you write this update for an ill-conditioned quadratic, you see the zig-zag: the gradient is steep along one axis and shallow along another, so the straight descent path bounces between the walls of the narrow valley. Without any filtering, the parameters oscillate back and forth instead of moving down the valley, which is why a constant \(\eta\) around 0.1 on MNIST often diverges in practice if the data gradients are poorly normalized. To observe this, implement the update loop yourself, log the parameter norms, and watch what “divergence” feels like in numbers.

### Momentum as smoothing and memory

Momentum modifies the update by introducing a velocity buffer that averages recent gradients and smooths oscillations. We keep a velocity vector \(v_t\) with the recursion

\[
v_{t+1} = \beta v_t + (1 - \beta)\nabla_\theta L(\theta_t), \qquad \theta_{t+1} = \theta_t - \eta v_{t+1},
\]

where \(\beta\) is the momentum coefficient (we use \(\beta=0.9\)) and the rest of the symbols follow from the previous equation. The velocity is an exponential moving average of gradients, so \(v_{t+1}\) retains a memory of past movement while still accreting the latest direction. The consequence is that the high-curvature axes, which flip sign every other step, are damped because they contribute less to the averaged velocity than the low-curvature axes, which consistently point downhill. The update thus feels like pushing a ball; once it starts moving along a stable direction, momentum carries it forward rather than letting it hover between walls.

This is the tension the build targets: with a fixed learning rate \(\eta=0.1\) the naive update either leaps past the bottom of the valley or never exits because of the bounce, while the momentum update learns to ride through a curved canyon by filtering noise. Implementing this manually not only exercises the numerical update logic but also prepares you for the more subtle variants such as Nesterov momentum and adaptive methods that will appear in the next step.

### Why per-coordinate scaling stabilizes the descent

The naive way of rescaling by the same \(\eta\) fails when different parameters see gradients of drastically different magnitudes. Adaptive subgradient methods collect the past squared gradients per coordinate to automatically shrink the step in steep directions. Let \(G_t\) be the element-wise accumulation of squared gradients:

\[
G_t = \sum_{i=1}^t \nabla_\theta L(\theta_i) \odot \nabla_\theta L(\theta_i), \qquad \theta_{t+1} = \theta_t - \eta \frac{\nabla_\theta L(\theta_t)}{\sqrt{G_t} + \epsilon},
\]

where \(\odot\) denotes element-wise multiplication and \(\epsilon\) is a tiny constant for numerical stability. Each coordinate of \(G_t\) is the sum of squared gradients seen so far, so higher-curvature directions (those with very large gradients) receive a smaller denominator. Duchi et al. (2011) showed that this accumulation gave online learning algorithms a kind of self-tuning by trading larger historical gradients for smaller steps, improving stability without expensive Hessian evaluations. In the MNIST build you keep the scalar learning rate fixed yet still observe how momentum already dampens the high-curvature axes; this foreshadows how spectral gradient methods (which add cheap curvature estimates) continue the same story of balancing memory, curvature, and step size.

### Putting it together in code

When you implement the loop, keep a tight separation between the two engines. First, run vanilla SGD for a fixed number of epochs while logging the training loss, the norm of the gradients, and the parameter norms. Then reset the parameters, initialize a velocity buffer \(v\) that mirrors each parameter tensor, and run the smoothed loop. The difference in how fast the log-loss decays, how the norms evolve, and whether you hit ≥92 % test accuracy within 15 epochs is what demonstrates the practical value of momentum.

Along the way you learn the danger of uncontrolled updates: in-place `param.data` modifications bypass Autograd’s built-in checks, so keep a comment that notes “These manual `param.data` writes are acceptable here because we are writing a pedagogical optimizer, but in production you should wrap custom optimizers with `torch.no_grad()` or register `torch.optim.Optimizer` subclasses to keep Autograd statistics consistent.” Similarly, instead of rigid asserts on exact batch counts or accuracies, treat those statements as sanity checks that help flag unexpected data loading or numerical issues while leaving room to adjust them when you switch datasets or architectures.

## Where the field is now

The research frontier still circles the same curvature tension we practiced in the build. Hestenes et al. (1952) proved that simple gradient descent zig-zags in poorly conditioned quadratics and introduced conjugate directions precisely to cut that oscillation at the root; their conjugate gradient method still informs Hessian-free approaches to training deep models today. Adaptive methods such as AdaGrad (Duchi et al. 2011) and Adam have since adopted per-coordinate scaling to modulate the effective step size without manual tuning, but the question of how much history to keep and how to compress it for massive models remains open. Meanwhile, the generative adversarial framework (Goodfellow et al. 2014) highlighted that simultaneous gradient descent in two-player games leads to even more pronounced oscillations, reinforcing that gradient translation must be engineered carefully to avoid divergence.

Engineering labs are still wrestling with the same themes. Large-scale training recipes pay careful attention to the interplay between learning rate, momentum, and normalization. Untitled 2016 (arXiv:1606.04474) documented that manual gradient control—without full-fledged optimizers—was still relevant when training convnets on ImageNet, and the companion work at arXiv:1602.04915 emphasized the need to stabilize internal statistics before the optimizer can meaningfully converge. At the same time, industry teams balance the opposite frontier: enabling cheap but stable gradient updates on massive clusters by combining adaptive heuristics with spectral approximations, low-precision arithmetic, and gradient clipping. The engineering priority is meeting wall-clock targets while guaranteeing that these handcrafted updates do not blow up when the batch size, sequence length, or model depth scales by an order of magnitude.

## What's still open

The tension between curvature, step size, and memory has clear instantiations that still lack crisp answers. One question is how to compute a cheap curvature proxy from minibatch gradients that preserves the damping effect of momentum without any additional buffers—a kind of “momentum-free” stabilization so the memory footprint stays constant even for trillion-parameter models. Another is whether the choreography between vanilla gradient descent and momentum could be dynamic: can we detect when the optimizer voyaged into a high-curvature canyon and automatically switch to a filtered update, then revert once the terrain smooths, without adding oscillatory artifacts? Finally, there is a gap in translating basic intuition to theory: how can we formally characterize the conditions under which the oscillations manifest, not just for quadratics but for the nonconvex losses of deep nets, so that engineers have the same confidence in their learning-rate choices as they do in backprop correctness?

Synthesis paragraph: The “why” behind this step is that we need to feel and control the curvature tension ourselves. That lived experience is the necessary bridge to these open questions—without a hands-on understanding of how raw gradients move parameters, we cannot confidently probe which approximations (momentum, adaptive scaling, spectral correction) are truly solving the tension versus merely masking it. The manual build keeps you honest about how much curvature is hiding behind the gradients that backprop hands you, and it is that honesty that supplies hypotheses worthy of experimentation on the frontier.

## Where to read next

If you want to double down on the calculus that produced the gradients you just used, → [[backpropagation]] breaks down how vector-Jacobian products travel through layers and what they assume about activation shapes. If you are more interested in the practical machinery that follows this build, → [[adaptive-optimizers]] traces the same exact gradient stream into Adam and RMSProp, explaining how the second moments reshape the update. For a broader overview of how optimization strategies stack across the curriculum, → [[gradient-descent]] summarizes the convergence guarantees, Hessian intuition, and the impact of curvature on step size.

## Build it

**What you’re building:** A transparent PyTorch loop that manually updates a two-layer MNIST MLP using both vanilla SGD and SGD with momentum so you can observe how curvature-driven oscillations are tamed.

**Why this is valuable:** This build forces you to translate each gradient tensor into motion, to document divergence, and to compare two update heuristics with a falsifiable empirical threshold—an experience that makes adaptive optimizers and large-scale training much easier to reason about.

**Stack:**
- **Model:** [hf-internal-testing/tiny-random-mlp](https://huggingface.co/hf-internal-testing/tiny-random-mlp) — tiny testing MLP (2 layers, linear-ReLU-linear) used to validate training loops.
- **Dataset:** [mnist](https://huggingface.co/datasets/mnist) — canonical handwritten digit dataset with standard train/test splits and `Normalize`-ready tensors.
- **Framework:** PyTorch 2.1 with CUDA 12.1 (plus `torchvision` 0.15 for dataset helpers).
- **Compute:** Single consumer GPU (RTX 4060 / A100) or Colab T4 (10.1 GB VRAM); runs in ~90 minutes on GPU, ~3 hours on CPU.

**The recipe:**
1. `pip install torch torchvision datasets matplotlib` and download the HuggingFace MODEL and DATASET IDs above. Instantiate the MNIST dataset with `torchvision.transforms.Compose([torchvision.transforms.ToTensor(), torchvision.transforms.Normalize(0.5, 0.5)])`, wrap it in a `DataLoader(batch_size=128, shuffle=True)`, and instantiate the tiny two-layer MLP by loading `hf-internal-testing/tiny-random-mlp`’s state dict into your sequential `torch.nn.Linear` stacks to match its architecture. Print the loader length and ensure the input shape is `(128, 1, 28, 28)` as a sanity check; if the batch count or shape deviates, adjust the loader to match your compute. These prints and shape checks are diagnostic, not immutable asserts.

2. Implement vanilla SGD manually: for each batch, flatten the images to `(128, 784)`, run a forward pass, compute cross-entropy loss, call `loss.backward()`, and update each parameter via `with torch.no_grad(): param -= 0.1 * param.grad`. Zero gradients after each update. Logging the first batch loss and the gradient norm (`torch.norm(param.grad)`) gives you immediate feedback without halting execution if the values are large; treat any “loss is NaN” as a pointer to scale the learning rate.

3. Reset the model weights using `model.load_state_dict(initial_state)` and introduce a velocity buffer `velocity_buffers = {name: torch.zeros_like(param) for name, param in model.named_parameters()}`. In the momentum loop compute `velocity = 0.9 * velocity + 0.1 * param.grad` before applying `param.data -= 0.1 * velocity`. After the update, store the velocity buffer back in the dictionary. Log `torch.norm(velocity)` to see smoothing; if the norm exceeds \(1e2\), consider gradient clipping via `torch.nn.utils.clip_grad_norm_`, but only after confirming the raw gradients actually blow up.

4. After 15 epochs of each run, evaluate on the HuggingFace MNIST test split with `torch.no_grad()` and `model.eval()`. Compute accuracy and log `vanilla_acc` and `momentum_acc`, but do not enforce hard asserts beyond using them as indicators (“I expect momentum_acc ≥ 0.92 when the run is healthy; if it is lower, check your learning rate and data normalization”).

5. Use plots (e.g., Matplotlib) to show the training loss curves and gradient norms for both runs. The artifact you now hold is a documented signal that momentum stabilizes updates where vanilla SGD diverges; you can attach these plots to any report or use them to compare future optimizers.

**Expected outcome:** The momentum run stabilizes quickly and exceeds 92 % test accuracy within 15 epochs while vanilla SGD either stagnates below 60 % or exhibits non-finite losses, providing a falsifiable proof that momentum is the practical remedy for oscillations in this setting.

**Variants per persona:**
- **Applied AI/ML engineer:** Deploy the momentum-enabled loop on the full HuggingFace MNIST dataset but with quantized weights (`torch.quantization.quantize_dynamic`) and batch the inference pipeline with `torch.jit.script` so you can ship an efficient scoring service on a CPU-only endpoint.
- **Research engineer:** Reproduce Table 2 from the Momentum paper in Goodfellow et al. (2014) by instrumenting gradient norms and velocity norms for both players in a GAN training run on MNIST and match the reported oscillation suppression within ±3 % of their loss curves.
- **Applied researcher:** Hypothesize that adding layer-wise adaptive learning rates (via per-layer scaling of the 0.1 base rate) can recover the momentum run’s convergence, then test three scalings (1×, 0.5×, 2×) and plot the resulting accuracy trajectories to see which approximates the smoothing effect of the global velocity buffer.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*