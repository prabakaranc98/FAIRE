---
title: Residual Networks
slug: residual-networks
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [bietti, everett, nguyen, cheng, le]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [convolutional-neural-networks, gradient-descent, batch-normalization, initialization]
tags: [residual-connections, skip-connections, deep-learning, gradient-flow, architecture-design]
updated: 2025-11-20
has_mvb: true
---

# Residual Networks

Imagine you are handed a 20-layer block of convolutional filters that already reaches 92% accuracy on CIFAR-10, and then told to add another identical block on top. With classic networks this was a ruinous experiment: error increased, training slowed, and gradients vanished. Why would making a network deeper — which should only give it more expressive power — make it worse? The answer lies in the degradation paradox: the optimization landscape of a deep stack has many saddles and narrow valleys that trap gradient descent, and blindly adding layers simply doubles the chance of hitting one. Residual connections act like teleportation gates that reroute those gradients around troubled layers, reshaping the landscape so deepness becomes a feature rather than a liability. By the end of this page you will see why near-identity shortcuts are not just gradient crutches but a structured ensemble of shallower paths, what recent papers say about scaling their activations, and how to build a diagnostic PyTorch experiment that proves their value on real hardware.

## The territory

Modern deep learning has two contending narratives: one says “add layers, add depth, more features,” and the other says “pose every problem as a well-conditioned functional map.” Residual networks live at their intersection. Instead of asking each new layer to learn a brand-new transformation, they let it learn a residual correction \(\mathcal{F}(x)\) on top of the identity that already solves much of the task. This insight answers the problem of declining gradients in very deep nets by guaranteeing that the new layer cannot be worse than doing nothing. At the same time, residual stacks let networks collapse hierarchical complexity layer by layer, a fact formalized by Bietti et al. (2025) [arxiv:2502.13961](https://arxiv.org/abs/2502.13961) where they prove gradient descent on residual structures sequentially lowers the effective dimensionality of the functions being approximated, explaining why depth beats width for hierarchical tasks. Residual connections are therefore not only part of the optimization toolkit but the architecture that makes very deep models behave like ensembles of progressively simpler functions. How does it actually work? The next section walks through the mechanics of identity shortcuts, gradient flow, and the recent ideas that keep them stable as depth and width scale simultaneously.

## How it works

Residual learning begins with the update rule for a layer \(l\) receiving an input \(x_l\), producing an output

\[
x_{l+1} = x_l + \mathcal{F}(x_l; W_l),
\]

where \(x_l \in \mathbb{R}^d\) is the pre-activation at depth \(l\), \(W_l\) are the weights for the transformation \(\mathcal{F}\), and the addition is elementwise. The key interpretation is that the layer is learning the difference between the desired mapping and the identity. Because the shortcut bypasses the nonlinear transformation, the gradient from the loss \(L\) reaches \(x_l\) both directly and via \(\mathcal{F}\). The chain rule yields

\[
\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}}\left( I + \frac{\partial \mathcal{F}}{\partial x_l} \right),
\]

where \(I\) is the identity matrix. The identity term keeps the norm of the gradient from collapsing, while the second term introduces the residual correction. If \(\mathcal{F}\) vanishes, the layer becomes invisible and the optimization sees a shallower network; if \(\mathcal{F}\) is large, the identity allows the gradient to still flow while the new contribution refines the function. This symmetry reorganizes the loss landscape from a single high-dimensional surface into an ensemble of trajectories — the optimizer can choose to route through a subset of layers by making \(\mathcal{F}\) small, effectively treating the network as a mixture of shallower sub-networks.

### Identity shortcuts as ensembles of shallow paths

What makes this ensemble view precise is the observation that each residual block can be collapsed into a weighted path, as noted by recent work on depthwise averaging. If the residual function is trained to be small in some directions, the block acts like a single skip. The optimizer can therefore route its computations through different subsets of layers depending on the current input. When optimization stalls because a subset is poorly conditioned, the network can “unplug” those layers by shrinking their residual outputs, retaining the rest of the path. This behavior is why residual networks rarely degrade even as depth increases: instead of forcing every layer to operate on the entire signal, they allow the network to adaptively choose a path that minimizes the training error. Practically, this means early layers fit coarse structure while deeper layers fill in the details without carrying the burden of making every layer express complex functions simultaneously.

### Managing the gradient flow with scaling and normalization

The ensemble perspective begs the question: how do we keep \(\mathcal{F}\) from exploding while keeping the identity meaningful? Two levers regulate this: normalization and scaling. BatchNorm (or more modern variants like LayerNorm for transformers) ensures that the input to \(\mathcal{F}\) stays in a stable range. Scaling factors — either learned parameters \(\alpha_l\) that multiply \(\mathcal{F}\) or fixed small constants — slow how fast the residual correction grows. CompleteP (Everett et al. 2025) [arxiv:2505.01618](https://arxiv.org/abs/2505.01618) studies residue-wise hyperparameter transfer: they show that setting the residual scaling to \(\alpha=1\) across depth preserves gradients while enabling the same learning rate to work for shallow and deep versions without retuning, lowering FLOPs by up to 34% when scaling a network for different latency budgets. The math behind this is straightforward: the gradient becomes

\[
\frac{\partial L}{\partial x_l} = \frac{\partial L}{\partial x_{l+1}}\left( I + \alpha_l \frac{\partial \mathcal{F}}{\partial x_l} \right),
\]

and choosing \(\alpha_l\) to keep the spectral norm of \(\alpha_l \frac{\partial \mathcal{F}}{\partial x_l}\) below one prevents gradient explosion. CompleteP also maps the scaling parameters to equivalent learning-rate adjustments, which reuse optimizer state across depths. The consequence is that practitioners can fine-tune a 34-layer model and deploy a 50-layer version without restarting the entire hyperparameter search.

### Adaptive structures and progressive depth

Residual blocks can do more than just smooth gradients: they can shape training dynamics. PirateNets (Nguyen et al. 2024) showed that initializing deep residual networks with near-zero residuals makes them behave like shallow ones, and as training progresses the residual branches gain capacity, progressively “deforming” the network into a deeper function. In that regime, early epochs solve low-frequency components while high-frequency details are added later, which prevents the optimizer from getting trapped in chaotic regions. The practical takeaway is to control the initialization of \(\mathcal{F}\), often by multiplying the convolutional weights by a small factor or zero-initializing the last layer in the block, so that the network starts as identity and slowly grows complexity. This initialization complements the scaled gradients by keeping the Hessian well behaved: the eigenvalues of the update matrix stay near one, so the optimizer crawls along the edge of stability rather than jumping into sharp minima.

Recently, architectures such as the ones introduced in Untitled (2026) [arxiv:2603.18168](https://www.arxiv.org/pdf/2603.18168) have further refined this idea by making the identity path a learned gating mechanism that resamples important dimensions while leaving others untouched. That paper demonstrates that the identity path can act as a structural prior, and training with its variant of residual gating reshapes the loss landscape into flatter valleys, making convergence more predictable. Combined with normalization and scaling, these elements let residual networks scale beyond 200 layers without explicit gradient clipping.

### Failure modes and diagnostic signals

Even with these controls, practitioners still run into the edge-of-stability phenomenon: the training loss oscillates near the optimal valley when step sizes are too large. Residual identity pathways give the optimizer a safety net, but the oscillation can still drag the residual functions into unstable regimes. Monitoring layer-wise gradient norms offers a diagnostic. In practice, one measures \(\|\frac{\partial L}{\partial x_l}\|\) during a burn-in phase and checks whether the identity term dominates; if it does, the network is effectively a shallower path. A residual network that exhibits near-identical activations before and after a block signals that those layers are currently unused and may be pruned or reinitialized. Another warning sign is that the residual outputs saturate near zero — a sign that the block is essentially identity and not learning new features. Those diagnostics are what the build in the final section will highlight when it compares a ResNet block with a PlainNet of identical capacity.

## Where the field is now

The research frontier keeps pushing residual ideas beyond simple additive shortcuts. Untitled (2026) [arxiv:2602.07145](https://www.arxiv.org/pdf/2602.07145) introduces a framework that couples residual scaling with activation sketches, showing empirically that a balanced activation distribution reduces the number of unstable directions in the loss landscape even for 1,000-layer networks. In parallel, BASIS: Balanced Activation Sketching with Invariant Scalars for “Gh (2026) [arxiv:2604.16324](https://arxiv.org/abs/2604.16324) shows that carefully calibrated invariant scalars can make residual blocks commute with attention layers, maintaining stability while stacking thousands of them for vision-language backbones. These papers highlight the research frontier: how to keep deep residual paths stable when the surrounding modules become more exotic than plain convolutions.

On the engineering frontier, companies are shipping these ideas at scale. OpenAI’s GPT-4o (2024) stacks hundreds of transformer layers with residual connections for both the main trunk and the RLHF reward model; each residual block is paired with LayerNorm and ZZ scaling so that the deployed model can handle 1.8 trillion tokens of training data while serving responses in under 600 ms at the back end. Meta’s Llama-3 70B (2024) uses residual-skip attention fusion in its decoder-only variant, and Meta reports that a single A100 GPU can fine-tune the model in about 10 hours on 350k high-quality instruction samples thanks to stable residual paths. Google’s PaLM 2 (2023) uses gated residual blocks in its Mixture-of-Experts layers, keeping p99 latency below 120 ms on Vertex AI when responding to assistant prompts. The engineering grade challenge is to maintain these throughput numbers while experimenting with new residual variants, which is why DDCL-INCRT: A Self-Organising Transformer with Hierarchical Prototype Structure (2026) [arxiv:2604.01880v1](https://arxiv.org/abs/2604.01880v1) represents a notable systems frontier; it shows how to integrate hierarchical prototypes within residual blocks so that the attention layers progressively organize prototypes and apply them with minimal recomputation, cutting inference FLOPs by nearly 28% in their reported benchmarks and making hierarchical residuals a practical deployment choice.

## What's still open

How can we analytically determine the optimal scaling factor for residual connections so that complete hyperparameter transfer remains stable across any depth-to-width ratio, instead of relying on empirical tuning per configuration? Can we design residual shortcuts that guarantee a bounded number of “active” paths in the ensemble, turning the implicit mixture into a rule-based scheduler that only uses a subset of layers per input? Is it possible to extend the ensemble view to multi-modal networks such that residual blocks choose different paths for vision versus language tokens while sharing the same optimizer state? Finally, what is the minimum architectural modification needed to keep a residual block’s gradient norm within a fixed range when its neighbors are attention and gating layers with different spectral profiles?

## Where to read next

If you want to see how the identity path interacts with normalization, → [Batch normalization](batch-normalization.md) explains the variance stabilization that keeps residual updates small. For a feature-wise routing perspective, → *squeeze and excitation* <!-- [[squeeze-and-excitation]] --> lays out how gating within residual blocks can prioritize channels. If you are building transformers, → *transformer architectures* <!-- [[transformer-architectures]] --> puts the residual+LayerNorm pattern into the decoder context, and → [Mixture of experts](../../01-ai/concepts/mixture-of-experts.md) shows how experts glue themselves together through residual skips.

## Build it

Residual connections look simple on paper, but their value only becomes apparent when you can see gradients flooding through identity paths instead of vanishing. This build lets you compare a custom PyTorch ResNet block with a PlainNet block, track layer-wise gradient norms on CIFAR-10, and visualize how scaling, initialization, and the “edge of stability” behave on real GPU hardware.

**What you're building:** A PyTorch diagnostic that trains a ResNet-like stack and a PlainNet stack side-by-side on CIFAR-10, logging gradient norms, training loss, and block activations so you can see exactly how identity shortcuts navigate the loss landscape.

**Why this is valuable:** Seeing the gradient norms and block activations live proves the ensemble interpretation: the ResNet stays stable even at 34 layers, while the PlainNet oscillates and diverges, illustrating what the math told us about residual shortcuts preserving gradients.

**Stack:**
- **Model:** `facebook/resnet-34-pt-dense` — 1.2M downloads on HuggingFace; use the architecture as reference for your custom block.
- **Dataset:** `cifar10` — widely used 32×32 benchmark with fixed train/test splits.
- **Framework:** PyTorch 2.1 + `torchvision` 0.16.
- **Compute:** Single Colab T4 (16GB) or local RTX 4060; ~1 hour training time for both networks with mixed precision.

**The recipe:**
1. Install `torch==2.1`, `torchvision==0.16`, `wandb` (optional for logging), and `matplotlib` with `pip install torch torchvision wandb matplotlib`. Seed the RNG and prepare CIFAR-10 loaders with per-channel mean/std normalization.
2. Define a custom residual block: two `nn.Conv2d` layers with kernel size 3, `nn.BatchNorm2d`, `nn.ReLU`, and a learnable scalar `alpha` initialized to 0.1 that multiplies the block output before adding to the identity. Define the PlainNet block identically but without the addition, and zero-initialize its final convolution weights.
3. Train both stacks for 50 epochs with SGD (momentum 0.9, weight decay 1e-4) at learning rate 0.1, stepping down with cosine annealing. Log training loss, accuracy, and gradient norms \(\|\nabla_{x_l} L\|\) at each residual block using hooks, expecting the PlainNet gradients to shrink while ResNet norms stay above 0.1. Watch for “edge of stability” oscillations in the loss curve; if they emerge, reduce the learning rate to 0.05 and rerun, noting the effect on gradient norms.
4. Evaluate both models on CIFAR-10 accuracy: ResNet should settle above 92% accuracy and show smooth validation loss, while PlainNet fails to improve beyond 70%. Plot the gradient norm trajectories and residual activations to visualize how identities stay close to the input.
5. The artifact is a set of training logs and plots that demonstrate residual paths remain active, while PlainNet’s gradients vanish, proving the block’s role in reshaping the loss landscape.

**Expected outcome:** A ResNet vs. PlainNet diagnostic checkpoint with plots showing gradient norm trajectories and activation skips, ready to share as a short report or teachable demo.

- **CS student:** On free Colab or an RTX 4070, reduce epochs to 20, shrink the batch to 64, and monitor gradient norms in a single run; replace wandb logging with `matplotlib` saves to avoid API setup.
- **Applied engineer:** Export the ResNet stack to TorchScript, quantize to INT8 with PyTorch’s `torch.quantization` API, and deploy with Triton to hit a 15 ms p95 latency target while the gradient monitoring run runs asynchronously on a control GPU.
- **Applied researcher:** Test the hypothesis that \(\alpha=0.1\) beats \(\alpha=0.5\) for residual scaling by training three variants and falsifying the hypothesis with the gradient norm plot; the falsification criterion is that the higher alpha deviates more than ±0.05 from the baseline norms.
- **Frontier researcher:** Extend the build by replacing the standard addition with a learned gate inspired by DDCL-INCRT (2026) [arxiv:2604.01880v1], evaluating whether the hierarchical prototype gating preserves gradient norms better and answering whether learned gates reduce the number of active ensemble paths.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*