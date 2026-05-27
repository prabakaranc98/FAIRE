---
title: Mixed-precision training
slug: mixed-precision-training
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [micikevicius, huang, brown, wu, zhao]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [tensor-cores, numerical-stability, scaling-training]
tags: [mixed-precision, tensor-cores, loss-scaling, bf16, fp8, training-stability]
updated: 2025-05-12
has_mvb: true
---

# Mixed-precision training

Imagine you are trying to measure both the distance between stars and the width of a human hair with the same rigid, heavy ruler. Using FP32 for every neural network operation is that ruler: precise, slow, and overkill for most tensors. Switching everything to FP16 is like picking up a flimsy measuring tape that snaps whenever the numbers get small, so the optimizer stops receiving any signal because gradients drop below the smallest representable number. Mixed-precision training is the art of keeping the fast, narrow tape (FP16, BF16, or FP8) in the hot path while adding just enough metadata, scaling, and higher-precision copies so that the tiny, critical updates survive. By the end of this page you will understand how master weights, dynamic loss scaling, emitter-based autocasting, and modern BF16/FP8 recipes let you recover FP32-like stability even as the hardware runs at tensor-core speed, and you will be equipped to build a bare-metal PyTorch loop that exposes exactly when loss scaling rescues gradients from underflow.

## The territory

Large-scale generative stacks demand both colossal FLOPs and memory bandwidth that the hardware on your desk can barely keep up with. The survey GenAI for Systems: Recurring Challenges and Design Principles from Software to Science (2026) [arxiv:2602.15241v1](https://arxiv.org/html/2602.15241v1) frames this as a systems story: schedulers, compilers, and observability tools must turn every Tensor Core and data-movement pipeline into throughput. Mixed precision is the lever that coordinates these layers. Rather than a blunt, “use FP16 everywhere” hack, mixed-precision training carefully partitions tensors across precisions, amplifies the signal where FP16 would underflow, and keeps a high-precision backup so updates do not disappear. That choreography lives at the intersection of numerical stability, hardware-aware programming (Tensor Cores, DPUs, and FP8/Mamba units), and optimizer design.

From a modeling perspective, mixed precision belongs to the same family as numerical stability tricks such as gradient clipping and adaptive learning rates, but it borrows its language from hardware-aware compilers: “autocast” contexts, dtype promotion rules, and fused kernel launches. The foundational insight introduced by Micikevicius et al. (2017) — the triad of keeping master weights in FP32, running forwards/backwards in FP16, and applying loss scaling — anchors everything that follows. Yet modern practice has layered on adaptive scaling, BF16 baseline precision, and even hybrid FP8/FP16 schedules so that models with billions of parameters keep training without NaNs. The mechanism is best understood by starting with that 2017 triad and seeing how scaling, casting, and telemetry work together to keep gradients in view.

## How it works

### Master weights, half-precision arithmetic, and where the cliff lies

Micikevicius et al. (2017) [arxiv:1710.03740](https://arxiv.org/abs/1710.03740) first formalized the idea that you can maintain a master copy of every weight in FP32 while letting the forward and backward passes run entirely in FP16. The forward pass reads from a copy of the master weights cast to FP16, computes activations and losses in FP16, and during the backward pass, gradients arrive in FP16 too. The optimizer updates the FP32 master copy so it can accumulate small increments that the FP16 copy cannot represent. Without that FP32 master copy, the small momentum updates and Adam second-moment statistics would round to zero inside FP16, because FP16 has only 11 bits of mantissa.

That means two simultaneous dtypes exist: FP16 for arithmetic throughput and FP32 for accumulation. To keep them synchronized, after each optimizer step, the master weights are cast down to FP16 again and stored next to the tensors used in the current iteration. This cast is cheap relative to matrix multiplies, so the overall throughput remains high while the optimizer never loses fidelity.

### Loss scaling and gradient rescue

The real danger occurs in the gradients, not in the weights. Small gradients or small losses can fall below the FP16 minimum positive normal number (~6.10⁻⁸), causing NaNs after a few steps. Loss scaling rescues those gradients by multiplying the loss before backpropagation. Define the scaled loss as

\[
L_{\text{scaled}} = S \cdot L
\]

where \(L\) is the original scalar loss, \(S > 0\) is the scale factor, and \(L_{\text{scaled}}\) is what the backward pass actually differentiates. The resulting gradients are

\[
g_{\text{scaled}} = \nabla_\theta L_{\text{scaled}} = S \cdot \nabla_\theta L
\]

where \(\theta\) gathers the model parameters. After the FP16 gradient calculation, the gradients are descaled by dividing by \(S\) before the optimizer update, restoring their original magnitude:

\[
g = \frac{g_{\text{scaled}}}{S}
\]

This pair of scaling and descaling multiplies the signal fed into the FP16 arithmetic so it stays above the underflow threshold but returns it to its true magnitude before updating the FP32 master weights. Micikevicius et al. (2017) also introduced the practical workflow: multiply loss by a fixed scale \(S\), run the backward pass, detect overflows, and repeat. While that works, choosing \(S\) manually is brittle across datasets and architectures, which led to the next generation of work.

### Adaptive scaling and overflow detection

Zhao et al. (2019) [arxiv:1910.12385](https://arxiv.org/abs/1910.12385) introduced adaptive loss scaling to eliminate manual tuning. Their method keeps a scale factor \(S_t\) at each optimizer step \(t\), increases it when no overflows are detected for a few iterations, and halves it whenever an overflow (NaN or Inf) appears in the gradients. The goal is to stay at the largest safe \(S_t\) so that you get the maximum dynamic range enhancement without overflow.

Mechanically, after each backward pass the autograd graph inspects the gradients. If any gradient contains an Inf or NaN bit pattern, the optimizer skips the weight update, restores the previous FP32 weights, halves \(S_t\), and re-runs with the new scale. If no overflow occurs for \(k\) steps, \(S_t\) is multiplied by a growth factor (often 2). This simple feedback loop lets the scale chase the largest safe value, which is critical when gradient magnitudes change drastically during warmup or due to curriculum learning on difficult tokens.

Adaptive scaling also interacts with gradient accumulation. When gradients are accumulated across micro-batches, the scale must be coordinated: scale each micro-batch loss individually, accumulate the scaled gradients, and only after accumulation descaled before optimizer step. Libraries like PyTorch AMP make this automatic, but the core idea — track \(S_t\), detect overflows via bit patterns, grow/shrink the scale — remains the same.

### Autocasting, dtype policies, and building the mixed-precision pipeline

To relieve the engineer from manually casting every layer, frameworks expose “autocast” contexts that cast operations based on dtype policies and tensor properties. In PyTorch, `torch.cuda.amp.autocast` inspects the operators inside the context and promotes them to FP16 or BF16 depending on the data type and operator support. This context also ensures key operations such as layer norm, softmax, and cross-entropy remain in FP32 to avoid precision loss.

The forward pass therefore becomes:

1. Enter an autocast context.
2. Run the model; activations and matrix multiplies automatically use FP16 or BF16.
3. Compute the loss inside autocast, yielding a FP16 scalar.
4. Multiply by the current scale \(S_t\) before calling `loss_scaler.scale(loss)`.

Within autocast, each operator sees a `dtype_policy` that acts like a compiler hint: matrix multiplies run in FP16, elementwise addition stays FP16 for performance, and reductions default to FP32. When hardware supports BF16 (which keeps an 8-bit exponent like FP32 but only 7-bit mantissa), the policy uses BF16 as the default because BF16 already matches FP32 dynamic range, so the need for scaling diminishes. However, gradients are still cast to FP16/BF16 for accumulation when the hardware offers faster throughput.

### Modern precision formats: BF16, FP8, and hybrid schedules

Recent work extends the mixed-precision story beyond FP16. The 2024 blog “Mixed Precision Training - ADS” (2017) documented through the Harvard ADS portal how the IEEE 754 formats can be interpreted to tune exponent and mantissa bits, especially when hardware (e.g., NVIDIA Hopper, Cerebras) exposes FP8 units. Later theoretical work such as “Untitled” (2018) [http://arxiv.org/pdf/1807.11205v1](http://arxiv.org/pdf/1807.11205v1) explored how varying exponent bias can reshape training stability, showing that you can sometimes shrink the exponent width as long as contextual scaling guards are in place.

The modern SOTA (Nemotron-H 2025 [arxiv:2504.03624](https://arxiv.org/abs/2504.03624)) demonstrates the practical recipe: the forward activations and weights use NVIDIA’s E4M3 FP8 (4 exponent bits, 3 mantissa bits) while gradients use the more forgiving E5M2 FP8 configuration. Each layer keeps a FP32 master copy and engages a per-layer dynamic scaling factor, because the range of signals differs across depths. The reason for per-layer scaling is simple: the gradient distributions at the bottom layers are orders of magnitude smaller than at the top, and a global scale would either overflow the top or underflow the bottom.

CompleteP (2025) [arxiv:2505.01618](https://arxiv.org/abs/2505.01618) adds another twist: parameterization rules—such as using RMSNorm instead of LayerNorm or adjusting initialization gain—must change alongside precision scaling to keep hyperparameter transferability. When you shrink to FP8, the noise introduced by rounding interacts with the curvature of the loss landscape, so initialization and learning rates must adapt; CompleteP’s framework provides a mathematical recipe for updating those knobs. Without that, even the best scaling logic will produce divergences.

Operators therefore need to know not only what dtype to use, but also what scale factor to apply. Libraries handle this by pairing each tensor with a `GradScaler` object that remembers \(S_t\), counts consecutive safe steps, and exposes `scale(loss)`/`step(optimizer)`/`update()` helpers. Because the hardware difference between FP16 and BF16 is smaller than between FP32 and FP16, some teams choose BF16 as the default dtype (the `torch.float16` autopcast replaced with `torch.bfloat16`). BF16’s larger dynamic range reduces the frequency of scaling adjustments, but loss scaling still plays a role whenever training instabilities appear, such as during early warmup.

### Observability, telemetry, and failure modes

A mixed-precision pipeline must also ship telemetry: record the current scale, overflow counts, and unscaled gradient norms. When the scale keeps halving, it means overflow events occur too often, indicating either an incompatible operator is running in FP8 or the learning rate is too high. When the scale grows without bound, it means you keep missing small gradients and risk hitting the mantissa limit of FP16; this typically happens if gradient clipping is too aggressive or if your loss flattening indicates a plateau. Instrumentation cannot rely on average throughput numbers; you need per-layer histograms of gradient magnitudes to ensure the scaling factor makes sense.

Finally, remember that mixed precision is not a tuning knob you flip once and forget. New operators added to the model might not yet support the targeted dtype, causing the autocast context to fallback silently to FP32, which can drop throughput. Always run a dtype audit before training: log the dtype of each parameter, buffer, and gradient. PyTorch’s `torch.cuda.amp.autocast(enabled=True)` accepts a `dtype` argument (FP16 or BF16), so you can experiment with mixed settings: weights in BF16, activations in FP16, or even customizing per-layer dtype policies.

## Where the field is now

The research frontier has moved beyond simple FP16 scaling into fully hybrid precision regimes. Nemotron-H (2025) [arxiv:2504.03624](https://arxiv.org/abs/2504.03624) reports training hybrid Mamba-Transformer models with layer-wise FP8: E4M3 for weights and activations, E5M2 for gradients, BF16 for certain normalization layers, and FP32 master weights. The authors demonstrate that this recipe matches FP16 baseline quality across language modeling and reasoning benchmarks while cutting memory by 45% and multiplying tensor-core throughput by 2×. CompleteP (2025) [arxiv:2505.01618](https://arxiv.org/abs/2505.01618) picks up the trailing edge, showing that to keep hyperparameter transferability intact when pushing into sub-8-bit regimes, parameterizations must also evolve—e.g., replacing LayerNorm with additive RMSNorm and damping the learning rate schedule in the first 5% of warmup.

On the engineering front, NVIDIA’s developer blog “Automatic Mixed Precision” (2023) [https://developer.nvidia.com/blog/automatic-mixed-precision-applied-deep-learning](https://developer.nvidia.com/blog/automatic-mixed-precision-applied-deep-learning) documents how production teams ship AMP-enhanced PyTorch and TensorRT pipelines. Teams training state-of-the-art vision and language models on H100 and L40 GPUs rely on tensor cores for the bulk of computation while using BF16 for activations, FP16 for cross-layer matmuls, and FP32 master weights. The blog includes deployable benchmarking numbers: TensorRT 10 inference for LLaMA 3 Turbo reduces memory footprint by 40% and doubles throughput compared to straight FP32 kernels, demonstrating that the mixed-precision story extends from training into production inference. The real lesson: the same loss-scaling and casting logic that keeps training stable also stabilizes quantization-aware deployment.

## What's still open

Can we design a mathematically guaranteed, scale-free optimizer that makes loss scaling unnecessary in sub-8-bit regimes such as FP4 or dynamic mantissa/exp systems? Current adaptive scalers still hunt for a safe value \(S_t\), and when the signal-to-noise ratio drops sharply (e.g., when switching to curriculum samples with tiny gradients), the scaler dithers; a proof-backed optimizer that self-normalizes at the precision limit would remove that heuristics-driven loop.

Is there a principled way to decompose per-layer precision requirements so that each layer automatically selects between FP16, BF16, or FP8 based on quantized gradient statistics without manual policy files? Nemotron-H’s layer-wise recipe still requires the engineer to specify dtype transitions zone-by-zone; eliminating that manual policy with a data-driven selector would lower the entry barrier.

How should parameterization rules (initialization scale, normalization type, bias standardization) co-evolve with precision schedules to preserve zero-shot generalization across architectures? CompleteP hints that laziness and scaling interact, but the exact coupling between parameterization and precision remains unresolved.

## Where to read next

If you want the hardware details, → [[tensor-cores]] explains how fused multiply-add pipelines turn FP16/BF16 into throughput. If you want the numerical foundation, → [[numerical-stability]] walks through the Taylor expansions that underlie underflow and overflow diagnostics. If you want to see how mixed precision fits into the larger training story, → [[scaling-training]] shows how gradient accumulation, learning-rate warmup, and restarts interact with precision choices.

## Build it

This build turns the concepts above into a working training loop that runs on a free Colab T4: you will manually instantiate master weights, implement an adaptive loss scaler, and visualize the gradient magnitudes that the scaler rescues from underflow.

**What you're building:** A PyTorch Transformer trained on WikiText-2 using AMP-style autocasting and a custom adaptive loss scaler, along with plots of gradient norms and scale adjustments.

**Why this is valuable:** You directly witness how loss scaling changes the gradients that reach the optimizer, making the numerical rescue concrete rather than theoretical.

**Stack:**
- **Model:** `facebook/opt-125m` (Hugging Face) — 12K+ downloads, widely documented
- **Dataset:** `wikitext` (`wikitext-2-raw-v1`) — canonical language-modeling benchmark
- **Framework:** PyTorch 2.1.1 + `torchvision` 0.18 + `accelerate` 2.0 (for reproducible training)
- **Compute:** Single Colab T4 (16GB VRAM) or same-class GPU — full run ~90 minutes per pass

**The recipe:**
1. Install requirements via `pip install torch==2.1.1 accelerate datasets matplotlib`. Import `torch.cuda.amp` and wrap `model`/`optimizer` setup in a standard PyTorch training loop.
2. Load WikiText-2 using `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")`; tokenize with `AutoTokenizer.from_pretrained("facebook/opt-125m")`, batching 2K tokens per batch, and use a `DataCollatorForLanguageModeling`.
3. In each forward step, enter `torch.cuda.amp.autocast(dtype=torch.float16)` for the model; compute the loss; scale it with a custom `AdaptiveScaler` object that tracks \(S_t\), multiplies the loss, checks `torch.isfinite(grad)` after `scaler.scale(loss).backward()`, and halves/increases the scale based on overflow.
4. Clip gradients (norm 1.0) after unscaling (`scaler.unscale_(optimizer)`), then call `scaler.step(optimizer)` and `scaler.update()`. Record `S_t`, gradient norms, and overflow flags in a `deque` for plotting every 50 steps.
5. Evaluate on the WikiText-2 validation split using perplexity; expect perplexity around 30 after a few epochs with stable scaling behavior.

**Expected outcome:** A checkpointed OPT-125M model trained with mixed precision plus a notebook cell showing gradient norm vs. scale plots, demonstrating how the adaptive scaler rescues underflow.

- **CS student:** Run the same loop on an RTX 4070 or M1 Max by reducing batch size to 512 tokens and removing gradient clipping to focus on how the scaler behaves when gradients are noisier.
- **Applied engineer:** Extend the loop by quantizing the final checkpoint to BF16 weights, serve it with `torchrun --nnodes=1 --nproc_per_node=1` using a Triton backend, and target p50 latency < 120ms on an L4-instance inference pipeline.
- **Applied researcher:** Test the hypothesis that cosine annealing with warm restarts interacts poorly with adaptive scaling by keeping the scale growth factor fixed at 2 and comparing validation loss trajectories with and without restarts; success is a measurable divergence in `S_t` curves beyond 5% relative difference.
- **Frontier researcher:** Probe the open question from §What's still open by replacing the scaler with an optimizer that dynamically normalizes gradients based on their second moments, then falsify the conjecture if the optimizer diverges before epoch 3 for FP4-simulated tensors.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*