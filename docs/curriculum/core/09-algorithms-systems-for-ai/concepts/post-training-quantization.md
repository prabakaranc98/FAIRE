---
title: Post-Training Quantization
slug: post-training-quantization
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [frantar, egiazarian, guidetti, ho]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [precision-scaling, model-deployment, llm-inference]
tags: [quantization, compression, calibration, deployment, inference, llm]
updated: 2025-05-06
has_mvb: true
---

# Post-Training Quantization

Imagine you have a 70-billion-parameter reasoning model that needs to run inside a budget-aware product team, but every single inference begins with loading 140 GB of float32 weights—more memory than a single A100 and a payroll proposal that would haunt the CFO. You try a hopeful trick: convert the checkpoint to int8 in place. The load succeeds, but the model immediately loses its ability to chain reasoning steps; one or two outlier activations blow up the quantization scale, and the model hallucinates as badly as it did before you tuned anything. That failure is the kernel of post-training quantization (PTQ): it is not a rounding exercise that blindly shrinks a checkpoint but rather a localized optimization problem that asks, “Given a snapshot of activations on a small calibration set, how do we choose scales, zero-points, or even per-weight rounding offsets so the deployed model still behaves like the original?” What follows is a tour of that optimization picture—calibration metrics that behave like surrogate losses, second-order adjustments that borrow from Hessian approximations, the paradigm shift from uniform mapping to activation-aware protection—and finally a hands-on Colab script that quantizes `facebook/opt-125m` with Round-to-Nearest (RTN) versus a simplified calibration-aware solver so you can observe the perplexity gap yourself.

## The territory

When inference becomes the bottleneck, every float32 entry hides a cost: 4× the bytes, 4× the PCIe transfer time, and caches that thrash because the working set does not fit in the accelerator’s SRAM. PTQ sits on the compacting end of the pipeline, answering the question “How do you shrink a trained model to int8 (or lower) without access to gradients?” Over the decades this question has drawn from three families. The signal-processing view treats each weight matrix as a scale times quantized codes and asks how to minimize representational distortion; the deployment view adds hardware constraints such as symmetric ranges, fused biases, or mixed-precision blocks; and the calibration view borrows from optimization theory by introducing a surrogate reconstruction loss computed on a few hundred calibration samples so that the approximate inference path matches the FP32 one. PATQ sits where those worlds overlap: it assumes we can scan a small dataset, compute activations, and solve a low-dimensional optimization (often layer-wise) to choose scales, zero-point offsets, or even custom rounding functions that keep the downstream loss stable.

The naive picture of taking each weight tensor, computing a single scale as the maximum absolute value, rounding everything to the nearest integer, and moving on is the myth PTQ dispels. Instead, much as GPTQ (Frantar et al. 2022) [arxiv:2210.17323](https://arxiv.org/abs/2210.17323) showed, PTQ becomes a stack of local problems: for each block you collect activation statistics from a calibration batch, treat the quantized weights as a parametric function \(w_q = s \cdot \text{clip}\big(\text{round}(w / s), a, b\big)\) where \(s\) is the scale and \(a, b\) are integer bounds, and search for \(s\) that minimizes reconstruction error \(\|X W - X W_q\|\) for the sampled activations \(X\). Calibration noise, outlier handling, and second-order corrections take this from a rounding heuristic to an empirical guarantee of preserved accuracy, and that is the subject of the next section: how PTQ actually works.

## How it works

PTQ is a calibration-driven approximation of the original optimization path. To make that precise, start from how quantization maps a real-valued weight \(w\) to an integer code and a scale:
\[ q(w; s) = \text{clip}\left(\left\lfloor \frac{w}{s} + 0.5\right\rfloor, a, b\right) \cdot s, \]
where \(s\) is a positive scale scalar, \(a\) and \(b\) are the minimum and maximum representable integers (e.g., \(-127\) and \(127\) for signed int8), and \(\lfloor\cdot\rfloor\) denotes rounding to nearest. This formula captures two variables: the scale \(s\), which controls the dynamic range, and the rounding decision inside the floor. Standard RTN fixes the rounding to the nearest integer and searches for \(s = \max(|w|) / b\). The error surface \((w - q(w; s))^2\) can be large for “outlier” weights—those rare entries that exceed the bulk of the distribution—so PTQ instead turns the quantization into a constrained optimization.

### Calibration as surrogate loss

With a small calibration set \(\mathcal{C} = \{x^{(1)}, \dots, x^{(M)}\}\), PTQ measures how quantized weights \(W_q\) affect the network by comparing the activations before and after quantization. Let \(X\) be the matrix of a block’s activations on \(\mathcal{C}\); for a linear layer we record the input activations at the block’s entry and the reference output \(Y = X W\). PTQ then minimizes a surrogate loss
\[ \mathcal{L}(s) = \|X W - X W_q(s)\|_F^2 \]
where \(W_q(s)\) applies the quantization function \(q(\cdot; s)\) elementwise to \(W\), and \(\|\cdot\|_F\) is the Frobenius norm. \(X\) is fixed from the calibration pass, \(W\) is the pre-trained weight matrix, and the only variable is \(s\). This surrogate already explains why the calibration batch is crucial: it defines the geometry of the downstream task being preserved. When \(X\) spans high-loss regions, the minimization encourages scales that reduce perturbations where they matter. The PTQ optimizer can run a simple grid search over \(s\) or a gradient-based search, but the key is that it solves a local reconstruction problem rather than trusting a single global scale. Without this, outlier activations would push \(s\) such that the entire low-bit representation becomes useless.

### Learned rounding: AdaRound

AdaRound (Nagel et al. 2020) [arxiv:2006.10757](https://arxiv.org/abs/2006.10757) takes calibration one step further by learning per-weight rounding offsets. Instead of always rounding \(w/s\) to the nearest integer, the method introduces a variable \(\alpha\) for each weight and uses a soft approximation of the rounding operation. The quantized weight becomes
\[ w_q = s \cdot \big(\lfloor \frac{w}{s} \rfloor + \sigma(\alpha)\big), \]
where \(\alpha\) is learned by minimizing the same reconstruction loss \(\|X W - X W_q\|_F^2\), \(\sigma\) is a sigmoid that squashes the learned offsets into \([0,1]\), and \(s\) is either fixed or tied to a learned power-of-two scale. Because \(\alpha\) is continuous, AdaRound can be trained with gradient descent on the small calibration set until the quantized layer matches the full-precision outputs, and the final rounding flips depend on whether \(\alpha\) lies above or below 0.5. This turns round-to-nearest from a heuristic into a differentiable optimization: the algorithm can “choose” to round up or down per weight depending on how much it affects the surrogate loss. Annotating \(X\), \(W\), and \(W_q\) within the loss clarifies that AdaRound manipulates the representation error directly, letting the optimizer focus on the weights that contribute most to the downstream activations.

### Second-order compensation: GPTQ

Despite AdaRound’s success, it does not scale easily to large transformers where there are millions of weights. GPTQ (Frantar et al. 2022) reorganizes the layer-wise problem by leveraging a second-order approximation of the calibration loss. For a single layer output \(Y = X W\), GPTQ derives a quadratic approximation of the change in output when replacing \(W\) with \(W_q\):
\[ \Delta Y \approx X (W_q - W) \approx -X H^{-1} X^\top \delta W, \]
where \(H = X^\top X + \lambda I\) is the (regularized) Hessian of the surrogate loss with respect to \(W\), \(\lambda\) is a damping term, and \(\delta W = W - W_q\) is the quantization residual. GPTQ computes the inverse Hessian \(H^{-1}\) using a block-wise Cholesky decomposition with iterative updates, so that when it quantizes a single weight, the algorithm already knows how this reduction will ripple through the downstream activations. The update is performed in a greedy, sequential fashion: the algorithm iterates through the weights \(w_i\), quantizes \(w_i\) with the current scale, and immediately adjusts the remaining weights by subtracting \(\frac{H^{-1}_i}{H^{-1}_{ii}} (w_i - q(w_i))\) to compensate for the perturbation. Because GPTQ operates on the inverse Hessian, it avoids re-running calibration for every new candidate weight, which makes it practical for LLMs, and the Hessian captures the geometry of \(X\), so the resulting quantized layer preserves the activations \(X W\) much better than a naive rounding would.

### Activation-aware protection: AWQ

Even with Hessian correction, the most sensitive weights can still dominate the loss. AWQ (Lin et al. 2023) [arxiv:2303.14567](https://arxiv.org/abs/2303.14567) observes that only a small fraction of weights (often the top 1% ranked by activation magnitude) significantly affect the model output. The method splits weights into two sets: those with large activations that are protected (kept in higher precision or quantized later) and the rest that follow GPTQ-style compensation. The result is a hybrid scheme where the “important” weights retain FP16 precision, while the others go to int4 or int8. AWQ demonstrates that protecting the tiny subset of salient values is much more effective than uniform quantization and that the cost of keeping them at higher precision is negligible compared to the overall memory savings.

### Outlier handling and loss guarantees

Outliers—rare values that dominate the maximum absolute range—must be treated differently, either by applying clipping or by dedicating extra bits. ZeroQ (Cai et al. 2021) [arxiv:2106.08295](https://arxiv.org/abs/2106.08295) bypasses calibration data entirely and synthesizes activations matching the quantized statistics, but even when calibration samples exist, clipping the few outlier activations to a manageable threshold prevents the scale from inflating. The theoretical underpinning here relates to the guarantees provided by Post-training Quantization with Provable Guarantees (Jung et al. 2022) [arxiv:2201.11113](https://arxiv.org/abs/2201.11113), which shows that if the Hessian of the surrogate loss is well-conditioned, the quantization error is bounded by a function of the scale and the calibration noise. Their result treats PTQ as running projected gradient descent on the quantization scales and proves that, under certain assumptions about the activation covariance, the loss difference between the quantized and original networks can be made arbitrarily small.

The practical takeaway is that PTQ is a constrained optimization: the rounding decision is free to break but is regularized by the calibration loss, Hessian corrections, and outlier clipping. In Section 5 we will implement a simple version of this pipeline, compare RTN against calibration-aware search, and see how the surrogate loss predicts the final perplexity on WikiText-2.

## Where the field is now

The research frontier keeps pushing the boundaries of how little precision can still power accurate inference. GPTQ (Frantar et al. 2022) remains the go-to method for 4-bit quantization on LLMs, and its successor AWQ (Lin et al. 2023) extended the idea by protecting the few weights that dominate the activations in each layer, achieving near-FP16 perplexities for OPT-66B with mixed-precision strategies. The theoretical backdrop—Post-training Quantization with Provable Guarantees (Jung et al. 2022)—adds confidence that surrogate calibration losses can be tightened with Hessian-aware updates, and ZeroQ (Cai et al. 2021) keeps showing that even synthetic activations can inform reasonable scale choices when real calibration data is scarce. The very first wave of hardware-aware automation (HAQ, Wang et al. 2019) [arxiv:1911.07190](https://arxiv.org/abs/1911.07190) inspired the current crop of mixed-precision tools (the quantization-aware compiler in TGI, the vLLM quantization scheduler) by showing that search over bit widths and scales extrapolates to latency improvements on edge accelerators.

Engineering deployments are now mixing these research ideas into production systems. Salesforce, for example, partners with Amazon SageMaker AI to deploy incoming CRM prompts through models compressed via PTQ: the SageMaker blog describes how Salesforce uses mixture-of-methods strategies (weight clipping, GPTQ-style calibration, and int8 serving) to sustain enterprise-grade latency and throughput while leveraging large foundation models across millions of daily user sessions. That engineering frontier shows PTQ is not only about squeezing bits but about meeting hard latency and cost targets across global services. These research and engineering frontiers together underscore that PTQ remains the only practical bridge between enormous model checkpoints and sustainable deployment.

## What's still open

Can we extend PTQ to sub-2-bit quantization for Mixture-of-Experts (MoE) architectures without causing routing collapse, where gating decisions diverge catastrophically from the original model? Current techniques focus on dense or sparsely activated models, but the exponential scaling of MoE parameters means each expert’s quantization must preserve not just local activations but also the gating logits. The open question is whether we can design a calibration loss that simultaneously constrains the experts and the router so that the gating distribution produced by the quantized model stays close to the FP32 routing decisions.

How does PTQ behave on long-context autoregressive models when calibration batches come from a different token distribution than the production prompts? There is no theory yet describing how distribution shift between calibration and real inference affects surrogate losses, and empirical work tends to use WikiText or book corpora that may not capture user prompts. A targeted study could treat this shift as a covariate shift problem and develop a correction term for the calibration gradient that upweights rare sequences.

Is it possible to collapse the Hessian computation in GPTQ-like methods to a single shared statistic that transfers across layers, so the expensive inverse computation need only be run once per model rather than per block, without hurting accuracy? The current second-order algorithms recompute the Hessian block-wise; a shared representation would dramatically reduce PTQ latency and make real-time quantization feasible.

## Where to read next

If you want to understand how the dynamic range trade-offs PTQ juggles are set up, → [[precision-scaling]] traces the connection between FP16, bfloat16, and int8 representations across accelerators. The deployment side of the story lives in → [[model-deployment]], which explains how quantized checkpoints are served in latency-critical systems and how observability feeds back into calibration. For a practical grounding in the kinds of activation statistics collected during PTQ, → [[llm-inference]] describes the instrumentation needed to capture layer-wise activations and logit drift without slowing down inference.

## Build it

The build proves the central PTQ insight: when you compare a naive Round-to-Nearest (RTN) quantization to a calibration-aware scale search, the surrogate reconstruction loss on a small dataset predicts the perplexity gap you will see on a full validation split.

**What you're building:** a Colab-ready script that loads `facebook/opt-125m`, applies RTN quantization in place, reruns the same quantization but with a calibration-driven scale search per linear layer, and reports the perplexity on WikiText-2 to expose the impact of the calibration loss.

**Why this is valuable:** the build forces you to interact with the surrogate loss, see how calibration batches modify the scales, and connect the metric (Frobenius reconstruction error) to the final perplexity so you internalize why PTQ needs optimization, not just rounding.

**Stack:**
- **Model:** [`facebook/opt-125m`](https://huggingface.co/facebook/opt-125m) — 4.5M downloads
- **Dataset:** [`wikitext-2`](https://huggingface.co/datasets/wikitext) — standard calibration text
- **Framework:** `transformers` ≥ 4.40 + `bitsandbytes` 0.40 + PyTorch 2.2
- **Compute:** Google Colab T4 (16 GB VRAM) — expect ~40 minutes for the quantization runs

**The recipe:**
1. `pip install transformers==4.40 bitsandbytes==0.40 torch==2.2 datasets tqdm` and clone the Colab notebook that wraps the OPT inference pipeline, then load the `facebook/opt-125m` checkpoint along with its tokenizer.
2. Build calibration loaders with 256 WikiText-2 sequences (max length 512 tokens); for each linear module in the OPT transformer, capture the inputs \(X\) to that module on 100 calibration batches and cache them to disk.
3. Implement two quantizers. RTN quantization computes \(s = \max(|w|) / 127\) and applies `round_clip` per weight. The calibration variant iterates candidate scales around \(s\), computes \(\mathcal{L}(s) = \|X W - X q(W; s)\|_F^2\) for each, and picks the scale with the lowest loss. Replace the module’s weight with the quantized version and move to the next layer; the calibration search only touches scales, not rounding offsets.
4. Evaluate both quantized checkpoints on the WikiText-2 validation split and report perplexity; expectation: RTN perplexity rises by ~5–10% relative to the fp16 baseline, while the calibration-aware quantizer stays within ~1–2% of the baseline.
5. You now have two quantized artifacts: `opt-125m-RTN-int8.pt` and `opt-125m-calibrated-int8.pt`, plus logged reconstruction losses per layer so you can plot the correlation between loss and perplexity drift.

**Expected outcome:** two quantized OPT checkpoints with documented perplexities and a calibration-loss-to-perplexity plot, proving that PTQ’s optimization matters.

- **CS student:** Run the same recipe but reduce calibration batches to 32 sequences and quantize only the first four transformer blocks so the script fits inside a 30‑minute Colab session; still report the relative hit in perplexity to see the scale sensitivity.
- **Applied engineer:** After calibrating the checkpoint, convert the quantized model to `vLLM`’s format, serve it behind a SageMaker Endpoint, and target a p50 latency ≤ 220 ms with 16 concurrent requests while keeping throughput high; log the throughput/latency table as part of the deliverable.
- **Applied researcher:** Extend the calibration search to also learn a per-weight additive offset (a simplified AdaRound); hold the scale fixed and compare the reconstruction loss and perplexity to the scale-only variant to test the hypothesis that rounding flexibility matters more than scale tuning.
- **Frontier researcher:** Use the quantization script as a baseline and attempt to quantize a tiny MoE layer (e.g., 4 experts × 2 transformer layers) to 2 bits per expert while keeping router logits in float16; measure whether the router’s softmax distribution drifts by more than 5% KL divergence from the FP32 model, thereby addressing the MoE routing-collapse question.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*