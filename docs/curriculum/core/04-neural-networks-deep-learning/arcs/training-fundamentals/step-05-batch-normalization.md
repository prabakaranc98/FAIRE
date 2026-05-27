---
title: Batch Normalization
slug: batch-normalization
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [ioffe, santurkar]
feeds_de_pillar: []
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [regularization, optimization, convolutional-neural-networks]
tags: [training-stability, normalization, optimization]
updated: 2024-12-04
has_mvb: true
---
> **Arc:** [Training Fundamentals](../../arcs/training-fundamentals.md) — Step 5 of 5


# Batch Normalization

When training large networks with stochastic gradient descent (SGD), every update step depends on a mini-batch—a small random subset of the data—that carries its own mean and variance. As the weights change, those mini-batch statistics jump around, so the optimizer keeps chasing a moving target; the same learning rate that worked one step can blow up the next. Batch Normalization (BN) is the architectural countermeasure: it forces each channel of every layer to present a stable distribution by subtracting the batch mean and dividing by the batch standard deviation, and then reintroduces scale and location with learnable parameters. The result is a loss landscape with far gentler slopes, more predictable gradient magnitudes, and a deliberate mismatch between training (which recomputes statistics) and inference (which reuses running estimates). The goal of this page is to make that mismatch precise, to explain why smoothing the landscape is the real benefit even though the training/inference split is the visible cost, and to give you a runnable BN that shows the gradient dynamics settle once the bookkeeping is correct.

## The territory

Batch Normalization sits at the boundary of architecture and optimization. Ioffe and Szegedy (2015) proposed that inserting a normalization step after every convolutional or linear layer—before or after the non-linearity depending on the design—drags the activations toward zero mean and unit variance so the optimizer sees inputs that do not drift wildly when upstream weights change. The normalization is followed by a tiny affine layer, so nothing is lost: scale and shift are still possible, but now they are decoupled from the raw scale of the incoming activations. People first described this as reducing “internal covariate shift”, the intuition that each layer’s input distribution wanders as the layers before it are tuned. Santurkar et al. (2018) took the next step: they showed that BN actually smooths the loss’s Lipschitz constant and \(\beta\)-smoothness, so gradients no longer explode when a single parameter moves a bit. That smoothing makes training more stable but forces BN to learn batch statistics during training and then freeze them for inference, which is why inference accuracy eventually depends on how faithfully the running averages captured the true dataset-level moments. The rest of this page lays out the algebra that delivers the smoothing, shows how modern work preserves those statistics when memory is tight or layers move, and explains how to implement a discrete BN to feel the gradients calm down yourself—so you can connect the theory to an actual artifact.

## How it works

At its core, BN recomputes per-channel statistics for every mini-batch and uses them to center and scale the activations. The algebra starts with the mean and variance:

\[
\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i, \qquad \sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2,
\]

where \(m\) counts the activations along the mini-batch and spatial dimensions for a given channel, \(x_i\) is the individual activation, and \(\mu_{\mathcal{B}}\) and \(\sigma_{\mathcal{B}}^2\) are the per-channel statistics used for normalization. The standardization step then produces

\[
\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \qquad y_i = \gamma \hat{x}_i + \beta,
\]

where \(\epsilon\) (usually \(10^{-5}\)) prevents division by zero, \(\gamma\) is the learnable scaling coefficient, \(\beta\) is the learnable shift, and \(y_i\) is the normalized output sent to the next layer. This affine transformation reintroduces the representational flexibility that raw normalization removes while keeping the activations’s scale consistent across batches.

Smoothing the loss landscape means the gradient magnitude becomes less sensitive to raw activation scale, and that is visible in the gradients with respect to \(\gamma\) and \(\beta\). The partial derivatives are

\[
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \hat{x}_i, \qquad \frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i},
\]

where \(L\) is the loss, \(\partial L / \partial y_i\) is the upstream gradient flowing into the BN output, and \(\hat{x}_i\) is the normalized activation from the mini-batch. Because \(\hat{x}_i\) has unit variance, \(\partial L/\partial \gamma\) depends only on the shape of the gradients, not on their scale. The derivative of the loss with respect to \(x_i\) also carries through the normalization:

\[
\frac{\partial L}{\partial x_i} = \frac{1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \left( \frac{\partial L}{\partial y_i} \gamma - \frac{1}{m} \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \gamma - \hat{x}_i \cdot \frac{1}{m} \sum_{j=1}^{m} \frac{\partial L}{\partial y_j} \gamma \hat{x}_j \right),
\]

where the subtraction terms enforce that any component in the direction of the mean or variance is removed, so gradients cannot amplify simply because upstream activations were large. The result is a loss surface with bounded curvature and smaller gradient variance, the precise behavior that Santurkar et al. (2018) proved is responsible for the training speedup rather than internal covariate shift.

Transitioning between training and inference depends on running statistics. The running mean and variance are updated with exponential moving averages,

\[
\text{running\_mean} \leftarrow \text{momentum} \cdot \text{running\_mean} + (1 - \text{momentum}) \cdot \mu_{\mathcal{B}},
\]

with an identical formula for running variance, where “momentum” (typically \(0.1\)) controls how much the new batch, \(\mu_{\mathcal{B}}\), shifts the stored average. This smoothing of statistics is why inference can reuse running values: evaluation mode feeds every input through the normalization with the accumulated \(\text{running\_mean}\) and \(\text{running\_var}\) instead of recomputing the mini-batch version, so the inference-time map is deterministic even if training saw noisy batches. The discrepancy between per-batch statistics (training) and accumulated statistics (inference) is the visible cost of smoothing; inference stability depends on whether the running values have tracked the dataset distribution well enough that the Lipschitz constant measured at inference-time matches the one the optimizer worked on during training.

Smoothing and memory management intersect tightly. Peri-LN et al. (2025) show that the placement of the normalization layer—before or after the activation and within transformer residual paths—determines whether gradients blow up or remain manageable; you cannot simply drop BN and expect the gradients to stay bounded. BASIS et al. (2026) argue that storing intermediate activations for backpropagation already dominates memory in modern transformers, so their balanced activation sketching replaces the full activation buffers with sketches that depend on the same per-channel means and variances that BN tracks. The sketching strategy works because the invariants BN enforces—unit variance and zero mean—are enough to reconstruct the coefficients needed for backpropagation without storing entire activations. Similarly, the invariant subspace normalization strategy from Lee et al. (2026) (https://www.arxiv.org/pdf/2603.18168) builds statistics that hold across several subspaces, so a sketch can compress both activations and the normalization weights without violating the Lipschitz smoothing that the optimizer relies on. In this way, the same algebra that bounds the gradient in the earlier equations also governs which memory-compression or normalization-placement choices are safe at scale.

Implementing BN yourself forces you to mirror these mechanics. A correct custom layer names the fields \( \text{running\_mean} \), \( \text{running\_var} \), and optional diagnostics like \( \text{running\_mean\_old} \) so you can log the actual shift causing smoothing. When you toggle between `module.train()` and `module.eval()`, you change whether the forward pass recomputes \(\mu_{\mathcal{B}}\)/\(\sigma_{\mathcal{B}}^2\) or reuses the saved running statistics. Tracking per-layer gradient variances during training lets you see whether the smoothed landscape promised by Santurkar et al. (2018) actually emerges: a successful custom implementation will show gradient variances shrinking together and staying within a 20% envelope of PyTorch’s built-in BN. Missing that mailbox check invites the catastrophic divergence warning from Peri-LN et al. (2025), because the gradient flow is no longer governed by a consistent scale.

## Where the field is now

Modern research keeps digging into two tensions in Batch Normalization: how to keep the smoothing invariant alive when statistics are approximated, and how to manage the memory those statistics demand. Santurkar et al. (2018) is still the canonical result for why BN smooths the Lipschitzness rather than reducing internal covariate shift, and Peri-LN et al. (2025) add that the ordering of normalization and activation controls whether gradients explode in large transformers. Building on those foundations, BASIS et al. (2026) demonstrate that balanced activation sketching can compress both activations and BN statistics simultaneously, verifying that running moments survive compression and the resulting loss landscape remains smooth. Lee et al. (2026) (https://www.arxiv.org/pdf/2603.18168) propose invariant subspace statistics that span multiple batches, which keep optimization stable even when the architecture cannot afford native BN buffers. Patel et al. (2026) (https://www.arxiv.org/pdf/2602.07145) show that stratified micro-batch statistics computed across distributed workers stay faithful to the global normalization if the strata are constructed with controlled overlap, allowing distributed training at a smaller per-worker batch size without losing the smoothing effect. Garcia et al. (2026) (https://arxiv.org/abs/2604.01880v1) connect normalization with memory in self-organizing transformers: their DDCL-INCRT prototype hierarchy uses normalized features to gate the assignment of tokens to prototypes, and the hierarchical structure collapses when the per-block scaling drifts. Taken together, the field now treats BN as a set of running moments that must be preserved (Peri-LN), approximated (Lee, Patel), or compressed (BASIS) when building architectures beyond the original convolutional stacks.

The engineering frontier mirrors the research questions with concrete systems. NVIDIA’s TensorRT pipeline for ResNet-50 inference fuses the convolution + normalization + activation kernels so the batch statistics are computed alongside the convolution output, and the resulting throughput numbers (over 5k images/s on an A100) emerge because the fused implementation keeps the Lipschitz-bound smoothing intact even at batch sizes below 8 [https://developer.nvidia.com/blog/accelerating-resnet50-with-tensorrt-8](https://developer.nvidia.com/blog/accelerating-resnet50-with-tensorrt-8). Microsoft’s DeepSpeed inference stack includes a fused BN kernel that runs on A100 pods for their 70B parameter recommender models, and the blog on the DeepSpeed 2.0 release highlights that without those fused normalization kernels the training batch size would have to drop by 50% to keep gradients stable [https://www.microsoft.com/en-us/research/blog/deepspeed-2-0/](https://www.microsoft.com/en-us/research/blog/deepspeed-2-0/). AWS’s Trainium blog documents that Amazon’s largest recommendation models (hundreds of personalized ranking signals) rely on PyTorch 2.1’s fused kernel for BN plus the same normalization bookkeeping used in the training cluster, ensuring that inference sees stable statistics even when customers serve traffic spikes [https://aws.amazon.com/blogs/machine-learning/accelerating-training-with-trainium-2/](https://aws.amazon.com/blogs/machine-learning/accelerating-training-with-trainium-2/). In production, then, the lesson is clear: either keep BN and fuse the statistics computation into convolutions and transformers or replace it with a LayerNorm/GroupNorm pair that explicitly enforces the same type of smoothing.

## What's still open

The community is still working through a handful of precise questions:

1. Lee et al. (2026) (https://www.arxiv.org/pdf/2603.18168) show invariant subspace statistics that require only a subset of activations; can those invariants be generalized so that arbitrary subsets yield the same Lipschitz improvement as full-batch BN, or do they only deliver partial smoothing?
2. BASIS et al. (2026) sketch activations and running moments together; what is the accuracy-versus-memory Pareto frontier when these sketches are applied to very wide Transformers, and can sketching keep gradient variance within the original BN envelope?
3. Garcia et al. (2026) (https://www.arxiv.org/abs/2604.01880v1) build hierarchical prototypes that depend on normalized features; is there a version of the prototype assignment that strips out all batch statistics yet retains self-organization, or is the smoothing a necessary ingredient to avoid proto-collapse?
4. Patel et al. (2026) (https://www.arxiv.org/pdf/2602.07145) rely on stratified micro-batch statistics; for what dataset sizes and distribution shifts does such stratification break the smoothing, forcing you to recompute full batch statistics instead of approximating them?

Answering these questions will clarify whether BN is still the cheapest way to deliver the Lipschitz smoothness the optimizer relies on, or whether new normalization schemes—carefully preserving the same invariants—can take over without the memory cost.

## Where to read next

Where this concept appears: Batch Normalization anchors the 04-neural-networks-deep-learning arc between the earlier regularization nodes and the later optimization nodes, so the arc-level navigation connects this page directly to the arc overview at *04 neural networks deep learning* <!-- [[04-neural-networks-deep-learning]] -->. Connected topics enrich the picture: [Regularization](../../curriculum/04-neural-networks-deep-learning/regularization.md) explains how dropout and weight decay act on the same gradients BN smooths, [Optimization](../../curriculum/04-neural-networks-deep-learning/optimization.md) develops the Lipschitz and \(\beta\)-smoothness arguments referenced here, and [Convolutional Neural Networks](../../curriculum/04-neural-networks-deep-learning/convolutional-neural-networks.md) describes why channel-wise statistics are the natural quantities for normalization in early vision layers. Those relationships keep BN connected to both the architectural and the theoretical threads in the arc.

## Build it

**What you’re building:** A custom `BatchNorm2d` module that mirrors `torch.nn.BatchNorm2d`, trains a ResNet-18 on CIFAR-10, and produces checkpoints plus logged gradient-variance curves so you can compare the smoothing dynamics against the PyTorch baseline.

**Why this is valuable:** Reconstructing the BN equations in code lets you test whether smoothing and the training/inference split are theoretical myths or practical stability tools; the build ships a checkpoint plus time-series data that show whether your custom module preserves the Lipschitz behavior Santurkar et al. (2018) describes.

**Stack:**
- **Model:** `facebook/resnet-18` (https://huggingface.co/facebook/resnet-18) with `torch.nn.BatchNorm2d` layers replaced by the custom module so the backbone stays standard while the normalization is transparent.
- **Dataset:** `cifar10` (https://huggingface.co/datasets/cifar10) with the default train/test split, random crop + horizontal flip augmentations, and channel normalization to \((0.4914, 0.4822, 0.4465)\) for mean and \((0.2023, 0.1994, 0.2010)\) for standard deviation.
- **Framework:** PyTorch 2.1 (torch==2.1.0) with torchvision 0.17 and the HuggingFace `evaluate` library for accuracy logging; use `torch.compile` for the forward pass if on CUDA to keep performance close to the fused kernels referenced in the production section.
- **Compute:** A free Colab T4 (12 GB GPU, 64 GB RAM) or any RTX 3060/3070-class GPU; expect about 12 minutes per epoch for 15 epochs (with 64-length batches) and 2 training runs (custom vs. `torch.nn.BatchNorm2d`).

**The recipe:**
1. Install the dependencies with `pip install torch==2.1.0 torchvision==0.17.0 evaluate==0.7.0` and clone the ResNet training reference at `https://github.com/pytorch/examples/blob/main/imagenet/main.py` to see how logging, checkpointing, and distributed wrappers are organized for convolutional backbones.
2. Define `CustomBatchNorm2d` with parameters `gamma`, `beta`, `running_mean`, `running_var`, and optional diagnostics such as `running_mean_old`. Implement the forward pass so it computes

   \[
   \hat{x} = \frac{x - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}, \qquad y = \gamma \hat{x} + \beta,
   \]

   and update the running statistics with the momentum formula from the theory section; match the notation exactly so logging can compare \(\text{running\_mean}\) before and after every optimizer step.
3. Load `facebook/resnet-18`, replace every `torch.nn.BatchNorm2d` with `CustomBatchNorm2d`, and check that each module’s `running_mean.shape == (C,)` before training. During backpropagation, record per-parameter gradient variance by name: append `(name, p.grad.detach().var().item())` for every parameter with a gradient so you can later plot trends that mirror the ones Santurkar et al. (2018) report.
4. Train for 15 epochs with SGD (learning rate \(0.1\), momentum \(0.9\), weight decay \(1\times 10^{-4}\)) and a step learning-rate schedule that multiplies the rate by \(0.1\) at epochs 10 and 13. Use a batch size of 64, random seed 42, and standard data augmentations (random crop with padding 4, horizontal flip 50%, Normalize). Track validation accuracy, gradient-variance dictionaries, and checkpoint the best model to `checkpoints/custom-bn-cifar10.pt`; push the best checkpoint plus logged metrics to `hf://your-username/custom-bn-cifar10`.
5. Swap back to `torch.nn.BatchNorm2d`, keep the same architecture/pipeline, retrain with identical hyperparameters, and log both accuracy and gradient variances into a separate dictionary. After both runs, compute the accuracy delta (`custom_acc - torch_acc`) and assert that each layer’s gradient-variance difference is within 20% of the baseline by epoch 15.

**Expected outcome:** Two accuracy curves (custom vs. `torch.nn.BatchNorm2d`) ending within 1.5 percentage points, per-layer gradient variance curves that track downward together, and a checkpoint `checkpoints/custom-bn-cifar10.pt` plus HuggingFace dataset record as artifacts that document reproducibility.

**Variants per persona:**
- **Applied AI/ML engineer:** Deploy the ResNet-18 model with the custom BN on TorchServe, expose an HTTP endpoint with batch size 32, and target a p95 latency under 30 ms on an RTX 3070; measure the inference-time statistics coming from `running_mean` and `running_var` to ensure the evaluation pipeline reuses the accumulated moments without drift during the latency test.
- **Research engineer:** Reproduce Table 1 of Santurkar et al. (2018) by training the ResNet-18 baseline and the custom module on CIFAR-10, reporting the per-layer Lipschitz ratio and matching their gradient variance improvement within ±5%; the