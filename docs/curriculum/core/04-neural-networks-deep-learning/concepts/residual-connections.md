---
title: Residual connections
slug: residual-connections
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [he, hochreiter, szegedy, karras, cheng]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [backpropagation, normalization, gradient-descent, multilayer-perceptron]
tags: [residual-connections, gradient-flow, skip-connections, architecture, stability, deep-learning]
updated: 2025-11-27
has_mvb: true
---

# Residual connections

Imagine a hundred people playing telephone where each person must translate the sentence into a new language before passing it on. By the tenth translator, the original message is gone; the meaning has been warped into noise. Deep neural networks face that same curse: every layer rewrites the signal, and the gradient signal back-propagated from the loss loses coherence. Residual connections solve the same puzzle by giving each layer a carbon copy of the original message alongside its translation. When the final person receives the pair, they can peek at the untouched text even if the translators went astray. By the end of this page you will know why that carbon copy is not a heuristic trick but a mathematical highway, how modern residual designs tune its tollbooth (scaling, normalization, memory), and how to prove the mechanism yourself with a concrete experiment that contrasts a 20-layer MLP trained on double moons with and without the identity path.

## The territory

Deep learning performance is a tale of two bottlenecks: the forward pass must compose non-linear features, and the backward pass must deliver meaningful gradients through that composition. Without intervention, stacking more than a handful of layers turns the backward signal into vanishing—and occasionally exploding—noise. Residual connections belong to the family of skip connections, but they differ from plain highway bypasses because they add rather than concatenate features, turning the model into an additive ensemble of “shallow experts.” The earliest residual blocks introduced by He et al. (2016) rewrote the layer transformation as an identity plus a small change, which let gradients split into two terms and dramatically extended trainability. Later works observed that improving training stability required not just the skip path but also careful scaling, normalization placement, and activation caching so that the identity path remains available at every depth. From transformers to vision models, residual streams are now the central scaffolding that keeps the gradient intact while allowing architecture teams to grow models in depth and width without catastrophic signal loss. How does it actually work under the hood, and what are the levers engineers now pull to keep the highway smooth?

## How it works

### The additive highway

The basic residual block replaces the usual layer output \(x_{l+1} = \mathcal{F}(x_l)\) with a sum:

\[
x_{l+1} = x_l + \mathcal{F}(x_l; \theta_l),
\]

where \(x_l\) is the input to layer \(l\), \(\mathcal{F}\) is the learned non-linear transformation parameterized by \(\theta_l\), and the addition is element-wise across the feature dimension. This identity shortcut means every layer passes forward not just the warped signal but also the untouched \(x_l\), which downstream layers can either trust directly or modify via subsequent \(\mathcal{F}\) blocks. The gradient of the loss \(\mathcal{L}\) with respect to \(x_l\) becomes

\[
\frac{\partial \mathcal{L}}{\partial x_l} = \frac{\partial \mathcal{L}}{\partial x_{l+1}}\left(I + \frac{\partial \mathcal{F}(x_l)}{\partial x_l}\right),
\]

where \(I\) is the identity matrix carrying the unmodified gradient, and \(\partial \mathcal{F}/\partial x_l\) encapsulates the new learned path. Because the identity matrix bypasses the potentially degenerate Jacobian of deep transformations, gradients arrive with the same magnitude they had at \(x_{l+1}\), plus a correction. This is the mathematical highway the hook promised: the backward signal can choose to stay on the identity term, effectively skipping all nonlinear layers, when the learned path would have diminished it.

### Gradient flow and the depth gap

The additive structure creates a sum of two gradient contributions that is not merely algebraic—it changes how the network perceives depth. In backpropagation through \(L\) layers, the identity contributions multiply to one, while the residual terms multiply through the chain rule as usual. Even if \(\mathcal{F}\) collapses to zero in some layers, the identity paths propagate gradients perfectly. This is why residual networks train with depths that would otherwise be impossible, yet behave like ensembles of shallower models: if the shortcut dominates, the network acts as if layers beyond a certain depth do not exist, and the optimizer only updates the necessary blocks.

Residual connections also change the interpretation of the objective. Consider a simple regression loss \( \mathcal{L} = \|f(x_0) - y\|^2\) where \(f\) is the network composed of \(L\) residual blocks. Because each block adds rather than replaces, the entire network can be rewritten as a telescoping sum:

\[
x_L = x_0 + \sum_{l=0}^{L-1} \mathcal{F}(x_l; \theta_l),
\]

where \(x_L\) is the final representation. The model now learns the total correction to the input \(x_0\) rather than the absolute mapping. This reframing implies that residual networks are biasing the function toward the identity map, which stabilizes training especially when the target is close to its initialization. The gradient of this sum naturally shares the same identity term at each step, reinforcing the highway metaphor: every block gets the original signal added back in.

### Scaling, normalization, and placement

Identity alone is not enough. As soon as \(\mathcal{F}\) grows too large, or normalization layers shift the scale of \(x_l\), the balance between the highway and the side path breaks. CompleteP et al. (2026) [arxiv:2603.18168](https://www.arxiv.org/pdf/2603.18168) shows that setting the residual scaling \(\alpha\) to 1 and carefully parameterizing the shortcut yields consistent depth-wise hyperparameter transfer and sizable FLOP savings compared to arbitrarily scaled residuals. Their core insight is that scaling the residual branch compresses the highway, so the optimizer must relearn globally when depth changes. With \(\alpha=1\), every new layer integrates seamlessly into the identity highway, reducing the re-training effort needed when stacking hundreds of blocks. This observation makes sense analytically because the derivative of \(x_l + \alpha \mathcal{F}(x_l)\) with respect to \(\alpha\) introduces an extra scalar factor; when \(\alpha\) deviates from 1, gradients no longer inherit the neat identity term that keeps earlier layers reachable.

Peri-LN (2024) [arxiv:2502.02732](https://arxiv.org/abs/2502.02732) extends this by showing normalization placement relative to the residual branch dictates mixed-precision stability. When LayerNorm or BatchNorm is applied after the addition (post-activation), the identity path loses amplitude due to re-centering and rescaling, so the gradient highway is no longer pure. By moving normalization to precede the addition—Norm(\(x_l\)) goes into \(\mathcal{F}\), and the sum is computed without a downstream norm—the identity preserves its scale, allowing FP16/BF16 training without the gradient spikes that previously crashed large transformers. Peri-LN also quantifies the gradient variance with and without this reordering, proving that the identity contribution is the only term that maintains low variance at depth when normalization is placed correctly. The result is a practical rule for engineers: place normalization inside \(\mathcal{F}\), keep the residual scaling at 1, and let the optimizer inherit the highway.

### Memory and activation reuse

Deep residual towers keep a lot of activations alive for the backward pass, which bloats memory. BASIS: Balanced Activation Sketching with Invariant Scalars for “Gh (2026) [arxiv:2604.16324](https://arxiv.org/abs/2604.16324) introduces Ghost Backpropagation, an activation sketching strategy that caches summary statistics instead of full activations while still preserving the identity highway information. The key idea is to maintain invariants of the residual stream—moments that do not depend on specific activations—so the backward pass can reconstruct gradients without storing intermediate \(x_l\) tensors. Ghost Backpropagation lets engineers fit deeper residual stacks in memory-constrained GPUs by compressing the stored residual contributions, while the identity path still supplies the exact gradient amplitude.

Another recent line of work represented by Untitled (2026) [arxiv:2602.07145](https://www.arxiv.org/pdf/2602.07145) investigates cross-layer residual adapters that merge signals from multiple horizons. The paper builds on the highway theme by showing that gating residual streams conditionally based on input semantics prevents redundant processing. Each layer receives an attention-weighted sum of the identity path, its local \(\mathcal{F}\), and the outputs of distant layers, and a stability term ensures that the residual highway never collapses even when gates shut off. The practice of gating hints at the open question asked later: the identity path does not need to be static; it can itself become a controller that decides whether the next block should execute.

These mechanistic insights explain why residual connections have become not just a good trick but a structural necessity. They preserve gradients via the identity highway, they run stably when scaling and normalization are aligned, and they remain feasible in production when activations can be sketched and gates can bypass unnecessary computation.

## Where the field is now

The basics above are now part of most mainstream models, but new research keeps refining the residual highway. CompleteP et al. (2026) [arxiv:2603.18168](https://www.arxiv.org/pdf/2603.18168) is currently the canonical reference for residual scaling policy, demonstrating experimentally that keeping \(\alpha=1\) across depths avoids the layer-wise tuning penalties that plagued earlier models. BASIS (2026) [arxiv:2604.16324](https://arxiv.org/abs/2604.16324) adds Ghost Backpropagation to that story by showing that storing invariant sketches of the identity path—means, variances, and a light quantized sample—lets modern GPUs hold 1.5× more depth without exceeding memory budgets. Peri-LN (2024) [arxiv:2502.02732](https://arxiv.org/abs/2502.02732) and its follow-ons have effectively enforced the “normalize before addition” rule in every production transformer, which is now the gold standard for mixed-precision training.

On the research frontier, DDCL-INCRT: A Self-Organising Transformer with Hierarchical Prototype Structure (2026) [arxiv:2604.01880v1](https://arxiv.org/abs/2604.01880v1) showcases a new residual topology where each layer maintains a hierarchy of prototypes that are aggregated through residual streams. The identity path in DDCL-INCRT carries not only the raw representation but also a prototype tree that regulates how much of each layer’s output enters the search. The self-organizing structure means gradients can flow through multiple prototype branches simultaneously, adding redundancy that improves stability while letting the network discover modular behaviours with minimal supervision.

Industrial production follows these research cues. Meta AI’s Llama 3 (2024) release on [ai.meta.com/research/llama-3](https://ai.meta.com/research/llama-3/) profits from residual scaling and normalization placement by using evenly scaled identity shortcuts across transformer decoder layers, which lets the 8.3B variant train in mixed precision without extensive tuning. The Llama 3 technical note highlights how residual paths avoid the “depth trap” when training on very long context windows, and how they allow the team to grow the model to 32 transformer layers while still hitting real-time inference p95 latencies in production. The same production trend is visible in Large Language Model deployments at Google and Anthropic: residual highways remain the plumbing that allows deeper, sparser, and more efficient models to ship.

## What's still open

Can the residual highway learn to gate itself at inference time so that redundant layers are bypassed without losing semantic coherence? A dynamic gate that switches the identity path on for “easy” inputs and hands off to deeper blocks for “hard” cases would let models adapt their computation graph on the fly, yet no method has demonstrated this without sacrificing accuracy. How can we extend current identity scaling rules to transformer variants with dynamic depth (such as Mixture of Experts) so the gradient highway remains stable when some residual branches are skipped altogether? Lastly, can activation sketching techniques like Ghost Backpropagation be combined with learned residual gating so that we only store the summary statistics of layers that flew over the highway instead of every block?

## Where to read next

If you want the probabilistic picture of why residuals act like likelihood-preserving skip connections, → [[backpropagation]] explains the calculus under the hood. The engineering counterpart is → [[normalization]] where the placement rules described above are fleshed out and connected to FP16 training. For the architecture fan, → [[transformer-architecture]] shows how decoders and encoders wrap the residual highway in multi-head attention without breaking the gradient path.

## Build it

This build proves that the identity highway is not just a theoretical construct: you can watch gradients collapse without it and remain stable when every layer receives a carbon copy of the input. The script in this recipe compares a 20-layer MLP with and without residual connections, records gradient norms per layer, and visualizes the difference on a synthetic double-moon task.

**What you're building:** a PyTorch experiment that trains two 20-layer MLP variants on the `openml/two_moons` dataset and plots the per-layer backward norm to show the collapse versus the highway.

**Why this is valuable:** the build forces you to implement both versions from scratch, log the gradient magnitudes at every residual step, and understand how the residual path rescues signal even in a tiny synthetic setting.

**Stack:**
- **Model:** scratch 20-layer MLP (two variants: plain and residual) implemented in PyTorch — each hidden block uses \(\text{Linear}\)-\(\text{ReLU}\) pairs with optional skip addition.
- **Dataset:** [openml/two_moons](https://huggingface.co/datasets/openml/two_moons) — the standard OpenML double-moons dataset accessible through Hugging Face.
- **Framework:** PyTorch 2.2.0 + scikit-learn 1.4 for dataset generation and evaluation.
- **Compute:** Free Colab T4 (16 GB VRAM) — training 20 layers for 100 epochs takes ~35 minutes per variant.

**The recipe:**
1. Install torch, scikit-learn, matplotlib, and datasets with `pip install torch==2.2.0 scikit-learn matplotlib datasets`.
2. Load `openml/two_moons` with `datasets.load_dataset("openml/two_moons")`, split into a 70/30 train/val split, scale inputs with `StandardScaler`, and create DataLoaders with batch size 128.
3. Implement two network classes: `PlainMLP20` with sequential Linear→ReLU layers, and `ResidualMLP20` where each block computes \(h = \text{Linear}(x)\), applies ReLU, then returns \(x + \tfrac{1}{\sqrt{2}} h\) (the \(\tfrac{1}{\sqrt{2}}\) scaling keeps variance controlled). Train both for 100 epochs with SGD (lr=0.1, momentum=0.9) and log the gradient norm of each Linear layer after every backward pass.
4. Evaluate using accuracy and the per-layer gradient norm curves; expect the plain MLP to show norms rapidly shrinking toward zero past layer 8, while the residual MLP keeps norms near the initialization range (~0.05–0.2).
5. What you now have: two checkpoints (plain and residual), a CSV of gradient norms per layer per epoch, and a matplotlib figure comparing the collapse versus the highway.

**Expected outcome:** a figure demonstrating that without residuals, gradients vanish after the tenth layer, but with the highway, gradients remain stable, paired with the checkpoints that can be loaded for future experimentation.

- **CS student:** Run the same recipe on an RTX 4070 (or Colab Pro T4) but reduce epochs to 50 and batch size to 64—focus on gradient logging and extend the plot to overlay training loss curves for extra insight.
- **Applied engineer:** Package the residual variant into a TorchServe handler, quantize the model to FP16 using `torch.fx`, and deploy it through a `vLLM` edge endpoint with p50 latency < 15 ms on the T4 instance; keep the gradient-log figure for internal monitoring.
- **Applied researcher:** Swap the residual block for a gated variant that interpolates between identity and \(\mathcal{F}\) via a learned sigmoid, hypothesize that gating helps on inputs farther from the manifold, and report whether gradient norms still stay stable on the out-of-distribution half-moons.
- **Frontier researcher:** Use the same data but train a dynamic routing controller that learns to skip layers during inference based on the norm of the residual stream, testing the open question "Can the residual highway gate itself without losing semantics?" by measuring accuracy degradation when the controller drops layers with \(\ell_2\) gating thresholds.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*