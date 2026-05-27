---
title: Quantization-Aware Training
slug: quantization-aware-training
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [ke, kwun, garcia, chen]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [quantization-basics, differentiable-optimization, hardware-aware-training]
tags: [quantization, low-bit, efficiency, inference, qat]
updated: 2025-04-12
has_mvb: true
---

# Quantization-Aware Training

You are about to hand a high-accuracy transformer or vision model to a tiny accelerator that can only add, multiply, and accumulate 8-bit integers. The model has never seen those limits before, so the moment the compiler snaps on the quantization grid the loss jumps and the accuracy implodes. Quantization-aware training (QAT) exists to avoid that cliff: it forces the model to speak the quantized language throughout training so that the rounding, clipping, and fused-kernel rearrangements of inference become part of the loss landscape instead of a surprise down the road. This page explains how fake quantization sits inside the forward pass, how the straight-through estimator (STE) rescues gradients, how trainable step sizes keep the grid adaptive, and how recent compute-optimal analyses tie the whole practice back to the systems budget that began the story.

## The territory

Deploying large-scale models onto power-, latency-, or area-constrained accelerators is no longer an option; it is a necessity. Integer inference kernels consume less power than FP16, stretch across smaller caches, and can often hit lower latency so that every extra accuracy point matters only if the bit budget fits. The naive path—train in full precision, quantize weights and activations afterward, and pray—yields a model that behaves like a translator who never rehearsed the new vocabulary. Batch statistics shift, activation ranges saturate, and the minute the weights are rounded, the downstream layers stop cooperating.

QAT draws from two worlds. From [[quantization-basics]] it inherits scale/zero-point encodings and the insight that symmetric versus asymmetric ranges interact differently with signed integers. From [[differentiable-optimization]] it borrows surrogate gradients that let non-differentiable components survive backpropagation. What makes QAT different from static or post-training quantization is where the fake quantization happens: it wraps every quantized tensor (weights, activations, fused outputs) during training, and it applies the straight-through estimator so that the optimizer sees a loss landscape that already includes the quantization distortions. That way, the optimizer can shape both the weights and the grid parameters so that the final quantized inference matches the full-precision accuracy as closely as possible. How does this simulation work in detail?

## How it works

The mechanism has three interlocking pieces: (1) how tensors are mapped to the quantized grid, (2) how fake quantization keeps that mapping inside the forward pass, and (3) how STE lets gradients slip through the rounding walls. Each piece tightens the constraint, and their combination is why QAT works.

### Representing tensors for integer inference

The inference graph represents a real-valued tensor \(x\) by rounding to the nearest integer, clamping inside the bounds, and scaling back. The canonical formula is
\[
\hat{x} = s \cdot \mathrm{clip}\!\left(\left\lfloor \frac{x}{s} + z \right\rceil,\; q_{\min},\; q_{\max} \right),
\]
where \(x\) is the floating-point tensor before quantization, \(s > 0\) is the trainable step size (scale), \(z\) is the zero-point offset, \(q_{\min}\) and \(q_{\max}\) define the integer range (for signed 4-bit, \(q_{\min}=-8\), \(q_{\max}=7\)), and \(\lfloor \cdot \rceil\) denotes round-to-nearest-integer. The clip function keeps the rounded value within the representable integers, and the multiplication by \(s\) converts the integer back into a floating-point surrogate for the rest of the graph. Inference implementations store only the integers and reconstruct the floating point value \(s \cdot q\) before continuing the computation. If the model never saw that rounding during training, the accuracy falls off a cliff as soon as quantization is enabled.

To keep the training graph honest, QAT inserts a fake-quantization module after each tensor that will be quantized at inference: each weight tensor, each activation entering a convolution, and each output of a fused conv+bn kernel. The fake-quantizer uses the same \(s, z, q_{\min}, q_{\max}\) parameters as the inference kernel, so the forward activations become numerically identical to the quantized integers that run later. That simulation is what allows the optimizer to adapt not just the floating-point weights but the quantization grid itself.

### Fake quantization and trainable step sizes

Fake quantization is typically implemented with learnable scales. The formula is
\[
\hat{x} = \mathrm{clamp}\big( \mathrm{round}(x / s + z),\; q_{\min},\; q_{\max} \big) \cdot s,
\]
where \(s\) (and, in some cases, \(z\)) is a parameter that the optimizer updates. The optimizer therefore controls not only the raw weights but also the quantization grid: it can shrink \(s\) to get finer resolution around high-density regions, stretch it to avoid saturation, or even shift \(z\) when asymmetric ranges make sense. This trainable grid is how the network learns to adapt to low precision instead of being broken by it.

The key challenge is that \(\hat{x}\) depends on non-differentiable operations (round and clamp), so we cannot compute exact gradients. QAT solves this by adopting the straight-through estimator (STE) introduced by Courbariaux et al. (2016) [arxiv:1609.07061](https://arxiv.org/pdf/1609.07061) and further analyzed in the JMLR survey by Bengio, Léonard, and Courville (2017) [JMLR 18(2017):16-456](https://jmlr.csail.mit.edu/papers/volume18/16-456/16-456.pdf). STE simply pretends that round and clip are identity functions during the backward pass so gradients can flow through. That gives
\[
\frac{\partial \hat{x}}{\partial x} \approx 1,
\]
and for the step size we start with the exact derivative before applying STE:
\[
\frac{\partial \hat{x}}{\partial s}
= \mathrm{clip}\big(\mathrm{round}(x/s + z), q_{\min}, q_{\max}\big)
+ s \cdot \mathrm{clip}^\prime\big(\mathrm{round}(x/s + z), q_{\min}, q_{\max}\big) \cdot \mathrm{round}^\prime(x/s + z) \cdot \left(-\frac{x}{s^2}\right),
\]
where \(\mathrm{clip}^\prime(y; q_{\min}, q_{\max})\) is the derivative of the clip function: it is \(1\) when \(q_{\min} < y < q_{\max}\) and \(0\) otherwise, indicating that the quantization integer is inside the dynamic range and therefore the gradient does not vanish. Likewise, \(\mathrm{round}^\prime\) is the formal derivative of the rounding operation, which is zero almost everywhere but is replaced in STE with \(1\) so that the gradient can pass. Therefore, under STE the expression becomes
\[
\frac{\partial \hat{x}}{\partial s} \approx \mathrm{clip}\big(\mathrm{round}(x/s + z), q_{\min}, q_{\max}\big) - \frac{x}{s}.
\]
The clipped, rounded term is the integer that would be stored in inference; dividing by \(s\) gives \(\hat{x}/s\). If the quantized tensor stays close to the floating-point tensor (the goal of QAT), then \(\hat{x} \approx x\) and the difference reduces to
\[
\frac{\partial \hat{x}}{\partial s} \approx \frac{\hat{x} - x}{s} \approx -\frac{x}{s}.
\]
This last approximation implicitly assumes the quantized activation is centered on the real value and that saturation is rare; the \(-x/s\) term behaves like a soft penalty that keeps step sizes from growing unchecked. When the tensor saturates at \(q_{\min}\) or \(q_{\max}\), \(\mathrm{clip}^\prime\) drops to zero and the gradient no longer pushes \(s\) inward, which is exactly the guardrail you need to prevent exploding scales. The STE therefore provides a principled, adjustable gradient for \(s\) even though the forward pass is discontinuous.

Jacob et al. (2017) [arxiv:1712.05877](https://arxiv.org/pdf/1712.05877) pushed this idea into integer-only inference by showing that the trainable scales can be absorbed into affine quantization kernels, letting the entire network run with pure integer arithmetic once QAT converges. Their INT8 pipeline on MobileNet and Inception architectures used per-channel scales for convolutions, which gives each output channel its own \(s\) to match the per-channel activation distribution. Because the scale gradient behaves like \(-x/s\), the optimizer can shrink the scale where the activation is small and widen it where the activation spread is larger, which is why per-channel quantization often recovers more than 90% of the floating-point accuracy when compared to per-tensor quantization on ImageNet. That empirical observation sets the stage for tighter integration between compute budgets and the quantization process.

### Straight-through estimator in practice

STE implementations in PyTorch or TensorFlow typically subclass `torch.autograd.Function` so that the forward pass executes fake quantization (round + clamp + scale), and the backward pass simply copies the upstream gradients. Because the backward path ignores rounding, there is a persistent "gradient mismatch" between the actual discrete operation and the surrogate gradient; this mismatch is the tension that QAT must manage.

A practical implication of the mismatch is that the fake quantizer must appear everywhere the inference graph will quantize. If you fake-quantize only the weights but leave intermediate activations untouched or forget addition nodes before quantized successors, the optimizer never sees the noise those tensors will face in inference, so the resulting model still collapses when quantized. Frameworks like TensorFlow Model Optimization and PyTorch’s `torch.quantization` module therefore insert `QuantStub`/`DeQuantStub` modules around quantized sections and provide observers that continuously collect min/max statistics. QAT goes further by making those observers update trainable scales, so the statistics themselves become part of the gradient descent loop.

### Quantization-aware batch normalization and statistics

Quantization interacts with batch normalization and fused kernels in subtle ways. In inference, optimizers often fuse a convolution and its following batch norm (and sometimes ReLU) into a single kernel to reduce memory traffic, as outlined in Krizhevsky’s “One weird trick for parallelizing convolutional neural networks” (2014) [arxiv:1404.5997](https://arxiv.org/pdf/1404.5997). That fusion relies on executing the conv, BN, and activation in a single pass, so every participating tensor shares the same numeric precision. QAT therefore has to either (a) fake-quantize the running mean and variance before they are folded into the convolution weights or (b) fold the batch norm into the preceding convolution and insert the fake quantization after the fused kernel. Both approaches depend on consistent rounding semantics: if the fake quantizer on the batch norm statistics disagrees with the fused inference kernel, the activation distribution shifts, and the quantized model diverges.

Krizhevsky’s insight about operator fusion implies another constraint: all fused paths must maintain the same quantization grid so that the parallel threads that execute each fused operation see identical precision. In practice, this means the fake quantizer must respect the fused conv-bn-ReLU stack that TensorRT, Glow, or QNNPACK will emit later; otherwise, the fused inference kernel sees a distribution it never saw during training, and accuracy drops.

### Scaling laws and compute budgeting

The newest frontier in QAT is not rounding itself but deciding how much compute to spend in each phase. The Compute-Optimal QAT study (2024) [arxiv:2401.09322](https://arxiv.org/abs/2401.09322) fits scaling laws for the error reduction achieved by the QAT phase. For a total compute budget \(C_{\text{total}} = C_{\text{pretrain}} + C_{\text{QAT}}\), there exists an optimal split that keeps the quantized error low without wasting epochs on the QAT stage. The error follows an empirical power law:
\[
\text{Error}(C_{\text{QAT}}) \approx \alpha C_{\text{QAT}}^{-\beta},
\]
where \(\alpha\) and \(\beta\) are architecture-dependent constants, so returning to the computing budget, you backpropagate that the benefit of QAT saturates quickly once the pretraining stage already produced a smooth loss surface. Their experiments on ResNet-50 and ViT-B/16 further report that gradient instability crops up in sub-2-bit regimes because the STE mismatch becomes non-negligible unless you clamp the gradients or use smaller learning rates. In other words, the scaling law is only credible if the QAT phase avoids wandering into regions where the surrogate gradient diverges wildly from the true discrete gradient.

The conclusion is a synthesis: the STE gradient (with its approximate \(-x/s\) behavior) keeps the step sizes aligned, but the compute-optimal analysis tells you how long to let those gradients run before deploying the quantized model. Together, the fake-quantization loop, the STE surrogate, trainable scales, per-channel flexibility, and compute-optimal budgeting make QAT a reliable way to keep the model fluent when the hardware forces a reduced vocabulary.

## Where the field is now

On the research side, Compute-Optimal QAT (2024) [arxiv:2401.09322](https://arxiv.org/abs/2401.09322) quantifies how the STE-induced mismatch depends on the ratio between pretraining and fine-tuning compute. Their experiments on ResNet-50 and a ViT-B/16 seeded with ImageNet demonstrate that clamping the gradients during the QAT phase prevents divergence once the model moves toward sub-2-bit precision, and that the cheapest way to keep the mismatch small is to increase batch size and decrease learning rate rather than adding more epochs. Companion work by Jacob et al. (2017) [arxiv:1712.05877](https://arxiv.org/pdf/1712.05877) analyzes the bias introduced by the STE and shows empirically on MobileNetV2 and Inception-V3 that per-channel scaling recovers more than 90% of the FP32 accuracy when each output channel’s step size is tuned individually on ImageNet. This layer-wise tuning is the closest available recipe to closing the STE gap because it lets each channel absorb its own rounding noise instead of forcing a single parameter to cover a heterogenous activation field.

On the engineering side, NVIDIA’s quantization-aware YOLOv5 on Jetson Orin demonstrates that real deployments need per-layer calibration scripts and integer-only verification. The Jetson blog post (NVIDIA 2024) describes how TensorRT inserts per-channel observers, clamps gradients during QAT, and then uses the calibrated fake quantizers to export INT8 kernels that stay within the 15 W power envelope (developer.nvidia.com/blog/porting-yolov5-to-nvidia-jetson/). These deployments report that latency drops by tens of milliseconds relative to FP16 while mean average precision stays within one percentage point, proving that QAT is the default path for practical int8 inference on safety-critical edge sensors.

## What's still open

1. How can an analytical correction term to the STE reduce the gradient mismatch for sub-2-bit quantization without triggering gradient explosion? A closed-form term that keeps the surrogate gradient bounded would let QAT enter the 1-bit-plus regime with predictable stability.

2. Can we design adaptive schedules that assign fake quantization budgets per layer rather than globally? The current practice uses the same number of QAT epochs for all layers, yet sensitivity varies widely, so a layer-aware schedule could reduce wasted training compute.

3. What is the right fusion strategy to jointly optimize QAT and structured sparsity? The zero-point clamping that quantization imposes doesn’t interact cleanly with sparsity masks, and it remains unclear whether the optimizer should treat the sparse pattern as part of the quantization grid or keep them orthogonal.

## Where to read next

If you want the math that makes STE possible, → [[differentiable-optimization]] walks through surrogate gradients for non-differentiable activations. For the engineering system-level pressure that makes QAT necessary, → [[hardware-aware-training]] narrates how latency, power, and cost targets force quantization decisions. To tie these ideas back to simple kernels, → [[quantization-basics]] explains how scale, zero-point, and per-channel quantization are implemented inside inference graphs.

## Build it

QAT is only believable when you can watch a quantized model regain its original accuracy after fine-tuning. This build shows how a PyTorch STE fake-quantizer lets MobileNetV2 recover from 4-bit weight quantization on CIFAR-10 while remaining runnable on free Colab hardware, with pointers to the working notebook in the PyTorch quantization repository.

**What you're building:** A CIFAR-10 MobileNetV2 retrained with a custom STE fake-quantizer that outputs a 4-bit weight checkpoint whose integer-only evaluation matches the FP32 baseline within a few percentage points.

**Why this is valuable:** You run the full QAT loop—fake quantization, STE, trainable scales, and integer-only evaluation—so you can demonstrate to stakeholders that a low-bit checkpoint is valid for deployment.

**Stack:**
- **Model:** `pytorch/vision:v0.15.2` MobileNetV2 — 8.1M downloads, with quantization recipes in PyTorch’s quantization repo
- **Dataset:** `huggingface/cifar10` — normalized image benchmark that trains quickly
- **Framework:** PyTorch 2.1 + `torchvision` 0.15.2 + `torch.quantization` modules
- **Compute:** Free Colab T4 (16GB VRAM), ~2 hours for 20 epochs

**The recipe:**
1. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118` and add `torch.autograd.Function` to house the STE fake-quantizer; see the reference notebook at `https://github.com/pytorch/quantization/blob/main/examples/qat/mobile/qat_mobilenetv2.py` for glue code.
2. Load CIFAR-10 with standard normalization, insert `QuantStub`/`DeQuantStub` around the feature extractor, wrap each `Conv2d` weight with your fake-quantization layer (initialize \(s\) to the tensor’s standard deviation divided by \(\sqrt{\text{fan\_in}}}\)), and freeze the min/max observers but continue updating the scale via SGD with momentum.
3. Train for 20 epochs on CIFAR-10 with learning rate \(1\times10^{-3}\), cosine annealing, and weight decay \(1\times10^{-4}\); expect the fake-quantized validation accuracy to climb past 70% and the loss to decay smoothly.
4. After training, run `torch.quantization.prepare_qat(model, inplace=True)` earlier, then call `torch.quantization.convert(model.eval(), inplace=True)` to swap out fake quantizers for integer-only ops that perform round+clip+rescale; measure Top-1 accuracy and expect the quantized model to stay within 2 percentage points of the FP32 baseline.
5. You now have a deployable 4-bit MobileNetV2 checkpoint plus an inference script that reports the quantized accuracy without floating-point fallback.

**Expected outcome:** A deployable MobileNetV2 checkpoint quantized to 4 bits, an evaluation script that uses the integer-only graph, and documentation of how scale gradients evolve during QAT.

**Variants per persona:**
- **CS student:** Freeze the scale updates, run the same recipe on an RTX 4070 for only 8 epochs, and plot the fake-quantized accuracy gap compared to FP32 to demonstrate short-run quantization behavior.
- **Applied engineer:** After tuning, export the quantized MobileNetV2 to ONNX using `torch.onnx.export`, run TensorRT INT8 calibration on a Jetson-like edge device, and report p50 latency below 30 ms while keeping mAP within one percentage point.
- **Applied researcher:** Ablate per-channel versus per-tensor scales, logging the per-layer quantization error and accuracy to test whether channel granularity dominates the improvement on CIFAR-10.
- **Frontier researcher:** Implement a gradient mismatch metric \(\left\|\nabla_s^{\text{STE}} - \nabla_s^{\text{proxy}}\right\|_2\) that compares the STE estimate to a finite-difference proxy (the falsifier), and report how that metric behaves in sub-2-bit regimes to directly address the instability question raised in §What's still open.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*