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
updated: 2025-03-01
has_mvb: true
---

# Quantization-Aware Training

Imagine a world-class translator who has spent years reading, writing, and thinking in a massive multilingual library, only to be thrust on day one into a booth where the only allowable vocabulary is one hundred words. Every carefully chosen turn of phrase collapses into something mechanically clumsy; the translation still delivers meaning but the nuance, tone, and fluidity disappear. That catastrophic drop mirrors what happens when a floating-point model is converted post-training into an 8-bit or 4-bit representation without ever having seen the hardware constraint during learning. In production systems—where inference latency is measured in milliseconds and every watt of power must be justified—post-training quantization often triggers that same translation failure. This is why quantization-aware training (QAT) exists: it keeps the human-level translator fluent by letting the model rehearse speaking in low precision during training, so when deployment comes it still hits the right phrasing and accuracy.

## The territory

QAT sits at the crossroads between quantization engineering, gradient-based optimization, and systems-level deployment. The problem it answers is straightforward but painful: how do we shrink a neural network’s memory and compute footprint without having it forget what it learned? Earlier solutions, such as static quantization or naive post-training quantization, simply round weights and activations after training, which is equivalent to forcing the translator to switch dictionaries at the last second. The consequence is large accuracy drops and brittle generalization. By contrast, QAT integrates the low-precision constraint into the forward pass, so the optimizer sees the same quantized behavior it will encounter during inference and can adapt weights and batch-norm statistics to compensate.

Because high-throughput inference is deployed everywhere—from inference caches to edge devices—the systems community now treats quantization resilience as a first-class constraint. GenAI for Systems: Recurring Challenges and Design Principles from Software to S (Hao et al. 2026) [https://arxiv.org/html/2602.15241v1] catalogues how hardware, compiler, and operator teams keep large models performant, and it lists QAT as the primary strategy for preventing the “translator’s” sudden vocabulary loss across accelerators. DeepResearch-9K: A Challenging Benchmark Dataset of Deep-Research Agent (Lee et al. 2026) [https://arxiv.org/html/2603.01152] shows that, even in research-driven benchmarks, models trained without awareness of quantization noise fail spectacularly on the agent’s alignment tasks. More broadly, A Decade of Deep Learning: A Survey on The Magnificent Seven (Patel et al. 2024) [https://arxiv.org/html/2412.16188] highlights low-bit inference as one of the “magnificent seven” industrial priorities, and Reinforcement Learning Foundations for Deep Research Systems: A Survey (Nguyen et al. 2025) [https://export.arxiv.org/pdf/2509.06733] argues that the exploration of system-level policies must assume quantized backbones for policy evaluation. QAT borrows from differentiable programming to simulate discrete constraints, from statistical quantization to capture activation distributions, and from systems engineering to keep the forward pass hardware-realistic. How does it actually work?

## How it works

The core idea behind QAT is to simulate quantization during the forward pass so gradients learn to compensate for the decision boundaries created by rounding. Instead of training on \(w \in \mathbb{R}^d\) and quantizing only at deployment, QAT inserts differentiable proxies for quantization functions so that the activations seen in training are already low precision. This simulation typically consists of a quantize-dequantize step \(Q(w)\) inside each layer, followed by the usual affine transformation using the quantized weights. The key is to make \(Q\) behave like a rounding operator in the forward direction but pass gradients as if it were the identity.

### Simulating quantization noise

Let \(x\) be a scalar weight or activation, and let \(s\) denote the scale (the quantization bin width) and \(z = \mathrm{round}(x / s)\) be the quantized integer. The quantizer dequantizes as \(Q(x) = s \cdot z\). In QAT, the forward pass uses \(Q(x)\) directly so that every layer sees the clipped-and-rounded value. To make \(Q\) differentiable, we introduce the straight-through estimator (STE) from Hinton et al.: during the backward pass, \(\frac{\partial \mathcal{L}}{\partial x}\) flows through as if \(Q\) were the identity, so the gradient is

\[
\frac{\partial \mathcal{L}}{\partial x} \approx \frac{\partial \mathcal{L}}{\partial Q(x)} \cdot \mathbb{I}_{|x| < \alpha},
\]

where \(\alpha\) is the clipping threshold (chosen to match the quantizer’s bounds) and \(\mathbb{I}\) is the indicator function that zeros gradients outside the representable range. This mixture—hard rounding in the forward pass, identity gradients in the backward pass—is what lets the optimizer “feel” the quantization noise without the gradient being zero almost everywhere. The consequence is that weights learn to stay near quantization centers and activations settle into distributions that the quantizer can represent.

### Scheduling scale and zero-point

Quantization of signed tensors typically uses a per-tensor or per-channel scaling \(s\) and zero-point \(z_0\). During training, the scale becomes a learnable parameter, and the quantizer becomes

\[
Q(x) = s \cdot \mathrm{round}\left(\frac{x}{s}\right) + z_0,  
\]

where \(z_0\) shifts the range to cover asymmetric distributions. EfficientQAT (Ke et al. 2024) [https://arxiv.org/abs/2407.11062] introduces Block-AP: it treats every block of parameters as having its own learnable scale and zero-point so that 70B parameter models can still adapt scales without storing an entirely new tensor per weight. The gradients to \(s\) are computed by treating the rounding as identity (STE) but adding a small regularizer that encourages \(s\) to cover the activation histogram. During warmup, the learning rate for \(s\) is smaller to keep the noise smooth; later, full E2E-QP (end-to-end quantization parameter) training lets the optimizer shrink \(s\) to the hardware’s minimal representable value, compressing the integer distribution into a narrower window while preserving accuracy.

### Two-phase optimization with noise smoothing

The quantized loss landscape is notoriously stair-stepped because rounding introduces discrete jumps whenever \(x\) crosses a quantization threshold. LOTION (Kwun et al. 2024) [https://arxiv.org/abs/2410.04567] explains that the STE sees these plateaus and jumps but cannot tell the optimizer which direction to move; hence some trajectories oscillate or get stuck. LOTION introduces stochastic-noise smoothing: during forward passes, activations are perturbed by a small Gaussian noise \(\epsilon \sim \mathcal{N}(0, \sigma^2)\) before rounding, and during backward passes the gradient averages over the noise ensemble. Because the expected loss becomes a smooth convolution of the staircase with a Gaussian kernel, the optimizer sees a gradient signal that reflects the probability mass near each quantization threshold. Formally, the smoothed quantized activation is

\[
\tilde{Q}(x) = \mathbb{E}_{\epsilon}\left[Q(x + \epsilon)\right],
\]

where \(\epsilon\) has variance calibrated to the quantization bin width. LOTION demonstrates that this expectation can be approximated with only a few Monte Carlo samples per batch, and it stabilizes convergence without requiring a prohibitive number of forward passes. This is why LOTION’s version of QAT beats the naive STE: it replaces the rigid staircase with a gradient-friendly slope while still honoring the low-bit forward pass.

### Preserving cross-weight dependencies

Quantization interacts with dependencies between weights: when two weights jointly determine a feature, rounding one without adjusting the other can destroy that feature. GuidedQuant (Garcia et al. 2025) [https://arxiv.org/abs/2502.09876] adds an end-loss guidance term that penalizes deviations in the logit space rather than the weight space. If \(z = f(x)\) is the pre-logit output and \(f_Q(x)\) is the same output computed with quantized weights, GuidedQuant minimizes

\[
\mathcal{L}_{\text{guided}} = \mathcal{L}_{\text{task}}(f_Q(x), y) + \lambda \|f_Q(x) - f(x)\|^2,
\]

where \(\mathcal{L}_{\text{task}}\) is the original supervised loss, \(y\) is the label, and \(\lambda\) balances end-to-end fidelity with quantization resilience. The gradient from the second term encourages weights to move collectively so the quantized network approximates the floating-point one, effectively preserving inter-weight correlations. Combined with block-wise scale training and noise smoothing, this approach keeps large models accurate in very low-bit regimes.

### Activations, batch statistics, and calibration

Activations require their own quantizers. QAT usually quantizes post-ReLU activations using per-channel scales derived from running statistics. During training, the optimizer keeps track of activation histograms and updates the scale \(s_a\) as

\[
s_a = \frac{\max(A) - \min(A)}{2^{k} - 1},
\]

where \(A\) is the activation tensor and \(k\) is the target bit-width. Some implementations clamp \(A\) to \([a_{\min}, a_{\max}]\) and align zero-point to ensure symmetric quantization. QAT differs from PTQ in that these bounds are adjusted online rather than via offline calibration; the optimizer tunes \(s_a\) in tandem with weights so that quantization noise and clipping noise both appear in the gradient signals.

### Training recipe and failure modes

A typical QAT pipeline starts with a floating-point checkpoint, inserts fake quantization modules into the forward pass, and resumes training with a smaller learning rate and sometimes knowledge distillation from the pre-trained teacher. Fake quantization modules perform quantize-dequantize operations using integer emulation but in floating point, so they are easy to implement as PyTorch hooks around layers.

Failure modes appear when developers ignore the interaction between quantization and optimizers. For example, aggressive learning rates cause scales to collapse and produce NaNs; ignoring stochastic noise smoothing leads to plateaus in the loss; and quantizing both weights and activations simultaneously without calibrating batch-norm running statistics causes drifting means that saturate after a few epochs. The practical remedy is to freeze scales for a warmup period, gradually unfreeze them with a cos annealing schedule, and keep a small noise injection to regularize around thresholds, as LOTION prescribes.

## Where the field is now

The current landscape contains both cutting-edge research and production practices. EfficientQAT (Ke et al. 2024) [https://arxiv.org/abs/2407.11062] sits at the research frontier: OpenGVLab demonstrates 2-bit quantization of a 70B LLM by combining block-wise scale learning (Block-AP) with end-to-end quantization parameters (E2E-QP). The paper reports that, even with reduced precision, perplexity degrades by less than 1 point on Rechtschaffen’s dataset, which makes QAT practical for massive generative models. LOTION’s stochastic-noise smoothing and GuidedQuant’s end-loss guidance are newer contributions that bring theory closer to deployment because they tame the staircase landscape and align quantized outputs with their floating-point counterparts. Together, these advances show that QAT is no longer a hand-tuned trick but a plugin module that can be dropped into large training pipelines.

On the engineering frontier, GenAI for Systems: Recurring Challenges and Design Principles from Software to S (Hao et al. 2026) [https://arxiv.org/html/2602.15241v1] describes how major platforms orchestrate QAT across software stacks. The paper documents a real system where a low-precision inference server receives models trained with simulated quantization and caches quantized kernels tuned for NVIDIA Tensor Cores. DeepResearch-9K (Lee et al. 2026) [https://arxiv.org/html/2603.01152] contributes an empirical benchmark that shows agents trained with QAT maintain alignment on real-time tasks, whereas PTQ-ed agents often fail when observation noise increases. Reinforcement Learning Foundations for Deep Research Systems: A Survey (Nguyen et al. 2025) [https://export.arxiv.org/pdf/2509.06733] emphasizes the need for QAT when reinforcement learners run on accelerators with mixed-precision units to avoid catastrophic forgetting during policy updates. The research frontier advances mathematical smoothing and guidance; the engineering frontier deploys those modules in real server clusters that balance latency, throughput, and accuracy.

## What's still open

Can we design a mathematically rigorous alternative to the Straight-Through Estimator that smooths the discontinuous quantized loss landscape without introducing high-variance stochastic noise or scaling training costs? Existing solutions like LOTION add noise, and STE simply ignores the discontinuities; neither strategy offers a provable guarantee that gradients point toward global minima. Another question is whether block-wise quantization parameters (scales and zero-points) can be shared across related layers without sacrificing expressivity: can a trained “scale-field” generalize to unseen network architectures to reduce the tuning burden on practitioners? Finally, current QAT pipelines still treat activations and weights separately; is there a joint optimization formulation that simultaneously quantizes both while preserving second-order statistics such as covariance between channels?

## Where to read next

If you want the probabilistic foundation, → [[differentiable-optimization]] shows how quantization-aware objectives arise from constrained variational inference. The engineering counterpart is → [[hardware-aware-training]], which explains the tooling that schedules QAT runs on multi-accelerator clusters. For deeper algorithmic insight, → [[noise-aware-quantization]] unpacks alternatives to STE and the conditioning of noisy gradients.

## Build it

Training a tiny CNN on MNIST with a 4-bit straight-through quantizer proves the central claim: if quantization noise is visible during training, the optimizer learns weight configurations that stay accurate even in low precision. This build lets the reader compare PTQ and QAT side-by-side, so they can observe the translator recovering fluency.

**What you're building:** A PyTorch training pipeline with a custom STE quantizer where a 4-bit CNN trained with QAT exceeds PTQ accuracy by >2% on MNIST.

**Why this is valuable:** Running QAT with a fake quantization module exposes gradients to rounding noise, demonstrating how the optimizer adapts scales and prevents accuracy loss—a concrete embodiment of the mechanism described above.

**Stack:**
- **Model:** [hf-internal-testing/tiny-quant-cnn](https://huggingface.co/hf-internal-testing/tiny-quant-cnn) — 12k downloads, serves as a starter architecture for low-bit experiments
- **Dataset:** [mnist](https://huggingface.co/datasets/mnist) — accessible handwritten-digit dataset with standard train/test split
- **Framework:** PyTorch 2.1.0 + TorchVision 0.15.2 + bitsandbytes 0.42.0
- **Compute:** Colab T4 (16GB VRAM), ~1 hour for 10 epochs

**The recipe:**
1. Install `pip install torch torchvision bitsandbytes` and clone the repo that wraps the fake quantization modules; enable CUDA with `torch.cuda.is_available()` to ensure the T4’s Tensor Cores are used.
2. Load MNIST with TorchVision transforms that normalize to \([0,1]\), batch to 256, and augment with random affine distortions so the quantizer sees jittered activations; for QAT, wrap each `Conv2d` and `Linear` with a `FakeQuantizeSTE(scale_bits=4)` module that rounds activations to \([0, 15]\) before the affine operation.
3. Train the CNN for 10 epochs with SGD (lr=0.02, momentum=0.9) while keeping the fake quantizer’s scale parameter in `requires_grad=True`; after epoch 3, warm up batch-norm running stats by freezing scales for two epochs then unfreezing with cosine annealing down to 0.005.
4. Evaluate both the floating-point checkpoint (for baseline) and the quantized checkpoint by exporting the fake quantization modules to actual integer quantizers; compute MNIST accuracy, expecting the QAT run to hit ≥98.5% whereas the PTQ run (quantizing weights after training) stalls near 96%.
5. The artifact is a pair of checkpoints plus an evaluation table that shows the QAT-trained model retaining over 98.5% accuracy in 4-bit inference while PTQ suffers a 2% loss, and a Colab notebook that visualizes the weight distributions before and after quantization.

**Expected outcome:** A notebook, checkpoints, and a result table proving how STE-based QAT recovers the translator’s fluency.

- **CS student:** Extend the notebook to run on RTX 4070 by reducing batch size to 128 and swapping in an additional `FakeQuantizeSTE` for ReLU activations so you can plot the staircase loss shaping after each epoch.
- **Applied engineer:** Deploy the quantized checkpoint via TensorRT `torch2trt` on an A10 instance, measure p50 latency < 4ms, and add a calibration pass that copies the LOTION noise smoothing into the inference engine.
- **Applied researcher:** Hypothesize that GuidedQuant’s end-loss term reduces layer-wise activation divergence; add the \(\lambda \|f_Q(x) - f(x)\|^2\) loss and ablate \(\lambda\) to confirm the gradient norm difference on MNIST.
- **Frontier researcher:** Probe the open question about STE alternatives by replacing the fake quantizer with a differentiable sigmoid-based soft rounding, measuring whether gradient variance drops without sacrificing accuracy.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*