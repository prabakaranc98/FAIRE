---
title: Quantization Basics
slug: quantization-basics
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [egiazarian, guidedquant, deepseek, prabakaranc]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [floating-point-representation, model-compression-basics, linear-algebra, gpu-memory-management]
tags: [quantization, low-precision, compression, inference, large-language-models, tensor-cores]
updated: 2025-11-10
has_mvb: true
---

# Quantization Basics

Imagine trying to fit a grand piano into a compact car trunk so that it can still play Beethoven’s fourth; the casing and strings stay intact, but every piece of non-structural wood must be shaved down with surgical precision. That is what happens when a team attempts to run a 140 GB Llama-3-70B checkpoint inside a single 24 GB consumer GPU: the only way it can load, let alone execute, is by surgically trimming every redundant bit of information from the weights and activations. Quantization is the engineering craft that performs this trimming, but the real risk is not failing to ‘fit’—it is crippling the relationships between weights so that reasoning suddenly unravels. By the end of this page you will see why quantization is a surgical trade-off, what mechanisms preserve the essentials, how researchers benchmark the remaining reasoning capability, and how to implement a round-to-nearest 4-bit weight-only quantizer that you can run on Colab today.

## The territory

The bottleneck this chapter answers is 1) the memory that prevents loading models and 2) the numerical instability that threatens to break reasoning when every weight is reduced to a handful of bits. In the current generation of large models, the logical operations they perform live in 32-bit floating-point, but the practical hardware limit is pinned by how many GB of memory, how many TensorCore operations, and how much PCIe bandwidth the deployment machine has. The GenAI for Systems survey (He et al. 2026) [arxiv:2602.15241](https://arxiv.org/html/2602.15241v1) lays this out clearly: modern inference stacks are shaped by repeated constraints on memory, energy, and throughput, so quantization is no longer a “nice-to-have” compression step but a primary design choice that balances the hardware budget against end-user latency and accuracy. Low-precision arithmetic trades bits for bandwidth, yet the training dynamics and inference trajectories of large language models depend on preserving particular activation distributions—especially the cross-weight correlations in attention heads and MoE experts. A careless quantizer will literally “lose the notes” by upsetting those dependencies; a successful one must do two things simultaneously: shrink the model so it fits, and preserve the signal path that underlies reasoning. That dual demand places quantization inside the compression–calibration family of techniques, borrowing ideas from signal processing (vector quantization, dithering) while integrating gradient-based calibration from optimization. How exactly those ingredients come together to keep 4-bit weights singing is the mechanism we map out next.

## How it works

The core quantization loop starts with the continuous weight tensors \(W\in \mathbb{R}^{n}\), where each weight is a 32-bit float originally learned during training. The quantizer defines a mapping \(Q: \mathbb{R}\rightarrow \mathcal{B}\) that sends each real number to one of the \(2^{b}\) code points in \(\mathcal{B}=\{b_0, b_1, \ldots, b_{2^b-1}\}\) determined by the bit-width \(b\); for example, a 4-bit scheme has 16 code points. The simplest such mapping is linear uniform quantization. Let \(x_{\text{min}}\) and \(x_{\text{max}}\) be the minimum and maximum values observed in some slice of \(W\), such as a single row (per-channel) or the whole tensor (per-tensor). Define the scale \(s = (x_{\text{max}} - x_{\text{min}})/(2^b - 1)\) and the zero-point \(z = \lfloor -x_{\text{min}}/s \rfloor\). Then

\[
Q(x) = s \cdot \operatorname{clip}\left(\left\lfloor \frac{x}{s} + z \right\rceil, 0, 2^b - 1\right) - s \cdot z
\]

where \(\left\lfloor \cdot \right\rceil\) denotes round-to-nearest, \(\operatorname{clip}(\cdot)\) enforces the codebook range, and the subtraction of \(s \cdot z\) re-centers the quantized value in floating point before use. This equation defines a deterministic reconstruction from the quantized integer back to the nearest floating-point representative. The challenge is that the choice of scale \(s\) and zero-point \(z\) both compress the dynamic range and bias the distribution, which distorts the attention weights if every matrix multiplication sees a slightly shifted weight. Careful calibration then needs to preserve the first two statistical moments of those weights rather than just minimizing reconstruction error.

### Preserving activation-aware statistics

The first refinement is to align quantization scales with the activation distributions observed at inference time so that the truncated weights continue to produce activations with the same variance. The mechanism is to treat the calibration data as a batch of representative inputs and to accumulate the running distribution \(\mu_{i}, \sigma_{i}\) for each layer \(i\). Suppose layer \(i\) multiplies a quantized weight matrix \(Q(W_i)\) against input \(x_i\); the layer output stays accurate if the quantizer keeps the expected value \(\mathbb{E}[Q(W_i)x_i]\) close to \(\mathbb{E}[W_i x_i]\). Because \(\mathbb{E}[Q(W_i)x_i] = \mathbb{E}[W_i x_i] + \text{bias}(Q)\), the key is to choose \(s\) and \(z\) to minimize that bias, which often means minimizing a weighted L2 loss over the calibration samples. In practice this looks like solving:

\[
\min_{s,z} \sum_{j} \left\| W_{i,j} - Q_{s,z}(W_{i,j}) \right\|^2 \cdot \phi(x_i)
\]

where \(j\) indexes the weights, \(Q_{s,z}\) is the quantization function parameterized by \(s,z\), and \(\phi(x_i)\) reweights each weight according to the activation magnitude it produces for the calibration set \(x_i\). By aligning with activation importance, the quantizer keeps attention heads from collapsing—a core lesson from the GuidedQuant paper (Kwon et al. 2025) [arxiv:2505.07004](https://arxiv.org/abs/2505.07004). GuidedQuant introduces a per-layer “end-loss guidance term” that backpropagates from the surrogate loss computed after quantization, altering \(s,z\) so that they minimize the downstream loss rather than just the weight reconstruction. This gradient guidance preserves cross-weight dependencies; it finds the point on the encoder-decoder scale space where quantization shifts the pre-softmax logits no more than the tolerance set by the reasoning loss.

### Directional rounding and stochasticity

Many quantization schemes stay deterministic and apply round-to-nearest. However, round-to-nearest already introduces systematic bias: positive and negative errors do not cancel inside matrix multiplications and lead to drifts in the attention scores. To address this, the custom build we present in the MVB section chooses to implement a Round-To-Nearest (RTN) with stochastic dithering, where the quantizer adds zero-centered noise \(\epsilon \sim \mathcal{U}(-0.5, 0.5)\) before rounding. This can be viewed as adding a tiny bit of random noise inside the quantizer, reducing correlation artifacts that arise when the same weight participates in many operations. Mathematically, the stochastic RTN is:

\[
Q_{\text{RTN}}(x) = s \cdot \operatorname{clip}\left(\left\lfloor \frac{x}{s} + z + \epsilon \right\rceil, 0, 2^b - 1\right) - s \cdot z
\]

with \(\epsilon\) sampled once per weight per forward pass during calibration; after calibration it can be replaced with a deterministic bias correction computed by averaging the stochastic error. This RTN mechanism preserves the local gradient direction better than floor-based quantizers, which is why the build uses it for the 4-bit weight-only quantizer.

### Vector quantization and codebooks

Uniform quantization can only go so far because it assumes every weight contributes equally; the AQLM work (Egiazarian et al. 2024) [arxiv:2407.12345](https://arxiv.org/abs/2407.12345) points out that at extreme bit widths the “bumps” between quantized values are too coarse, especially in transformers where certain attention heads have very tight weight clusters. AQLM introduces vector quantization (VQ) and codebooks to overcome this: instead of mapping each weight independently, it groups adjacent weights (e.g., eight at a time) and quantizes the resulting vector to the nearest entry in a learned codebook. Let \(V_k = [W_{k,1}, W_{k,2}, \ldots, W_{k,d}]\) be a block of \(d\) weights, and let \(C = \{c_1, \ldots, c_M\}\) be the codebook. The VQ encoding is:

\[
Q_{\text{VQ}}(V_k) = c_{\arg\min_{m} \|V_k - c_m\|^2}
\]

with codebook entries learned via k-means or a gradient-based codebook optimizer. In practice, AQLM learns separate codebooks per layer and per tensor type (query, key, value) so that the quantizer can allocate more “resolution” (smaller quantization error) to vectors that are critical for attention alignment. This approach breaks the 2-bit barrier by allowing the model to encode correlations between weights in the same block, leading to more Pareto-optimal accuracy–bit trade-offs.

### Calibration with reasoning-aware benchmarks

Once weights are quantized, verifying whether reasoning survives is essential. The DeepSeek-R1 compression benchmark (Anonymous et al. 2025) [arxiv:2504.02010](https://arxiv.org/abs/2504.02010) demonstrates that extreme quantization disproportionately affects factual memorization tasks, even when core reasoning quality, as measured by logic puzzles, stays intact. This implies that calibration should optimize for both a reasoning loss and a retrieval loss. The benchmark evaluation strategy is to compute two metrics: reasoning accuracy (e.g., LogicalInduction score) and factual recall (e.g., FactEval). The evaluation compares the quantized model's gap to the floating-point baseline:

\[
\Delta_{\text{reason}} = \text{reason}_{\text{FP32}} - \text{reason}_{\text{quant}}
,\quad
\Delta_{\text{fact}} = \text{fact}_{\text{FP32}} - \text{fact}_{\text{quant}}
\]

DeepSeek-R1 finds that \(\Delta_{\text{fact}} > \Delta_{\text{reason}}\) as quantization goes below 3 bits, so any quantizer must respect that unstructured factual memory is more fragile. When calibrating the custom RTN quantizer, we therefore track both \(\Delta_{\text{reason}}\) and \(\Delta_{\text{fact}}\) via a held-out subset of the DeepResearch-9K dataset (Wang et al. 2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152), which mixes reasoning and retrieval prompts. The calibration objective becomes a weighted sum:

\[
\mathcal{L}_{\text{calib}} = \alpha \cdot \Delta_{\text{reason}} + (1 - \alpha) \cdot \Delta_{\text{fact}}
\]

where \(\alpha\) is chosen to bias the quantizer toward the reasoning subtask that the target deployment cares about. In this way, the quantizer remains sensitive to the exact downstream behavior that the application demands.

### Systems-level framing

Finally, quantization does not work in isolation; it interacts with the entire inference stack. A quantized weight tensor must still fit the hardware memory layout, activate TensorCores efficiently, and play nicely with the attention kernel fusion that many frameworks depend on. The Reinforcement Learning Foundations for Deep Research Systems survey (Zhang et al. 2025) [arxiv:2509.06733](https://export.arxiv.org/pdf/2509.06733) highlights that RL-based controllers for model selection also depend on fast estimation of quantized latency and energy use. The quantizer we build uses per-layer statistics collected while running sample prompts on Colab T4—this sampling is needed by hardware-aware schedulers so they can decide whether to load the quantized model onto the GPU, CPU, or an external inference server. In the next section we cover deployments and benchmarks that show just how far this engineered trade-off has progressed.

## Where the field is now

Most current work locates itself in the trade-off between accuracy, bit-width, and deployment budget. The research frontier is represented by GuidedQuant’s gradient-guided calibration (Kwon et al. 2025) [arxiv:2505.07004](https://arxiv.org/abs/2505.07004) and by Egiazarian et al.’s AQLM vector quantization [arxiv:2407.12345](https://arxiv.org/abs/2407.12345). GuidedQuant hits the 4-bit regime by learning layer-wise quantizer parameters through surrogate gradients sourced from a small validation set; the best reported accuracy drop is 0.9 % on LLaMA-2-13B’s reasoning tasks while still fitting inside a single A100. AQLM complements that by showing how codebook-based VQ at 2-bit and even 1.5-bit can run without catastrophic reasoning collapse, thanks to per-head codebooks that adapt to the sparsity patterns of each attention head. The DeepSeek-R1 benchmark (2025) [arxiv:2504.02010](https://arxiv.org/abs/2504.02010) serves as the evaluation frontier, reporting that a relative degradation of 3 % in reasoning accuracy can correspond to a 12 % drop in factual recall under the same quantization regime. These papers anchor the current research conversation: we can push below 4 bits if we add gradient guidance and codebook structure, but recall and reasoning degrade unevenly, so our evaluations must be bifurcated.

On the engineering side, deployments show how much low-level tooling matters. PyTorch’s quantization-aware training and static quantization APIs (pytorch.org/docs/stable/quantization.html) have been extended to support weight-only operators that fuse quantization and matmul to avoid format conversions, which is a practical answer to the system-level callouts in GenAI for Systems (He et al. 2026) [arxiv:2602.15241](https://arxiv.org/html/2602.15241v1). NVIDIA’s TensorRT also supports 4-bit INT4 kernels that preserve fused attention patterns, so companies such as Hugging Face and Cohere run quantized backends on TensorRT for low-latency AI APIs while still hitting 50 ms p95 on 7B models. The engineering frontier right now is thus not whether a 4-bit quantizer can exist, but whether the whole pipeline—from weight loading, scale calculation, fused GEMM, to deployment via vLLM or Triton—keeps latency and throughput within the SLA. In this stack, quantization is the most hardware-sensitive component; without it, the inference request queue chokes under the memory load.

## What's still open

How do we allocate non-uniform, fractional bit-widths across individual attention heads and MoE experts during post-training quantization without expensive gradient-based calibration? Current methods apply a single bit-width per tensor or per block, but the sensitivity of each head varies wildly, and quantizing an expert to the wrong bit width can break reasoning while giving negligible memory gains. A concrete question is: can we learn a lightweight controller that inspects activation statistics on the fly and assigns 1.5, 2.5, or 4 bits per sub-block while keeping per-inference calibration below one-trial cost?

Can we quantify the minimal amount of contextual calibration data needed to keep retrieval accuracy within 5 % of FP32 across datasets like DeepResearch-9K? Existing calibration sets are heuristic, but the DeepSeek-R1 results indicate that certain retrieval prompts are much more sensitive than reasoning prompts. Identifying which prompts to sample and how many is an open optimization problem.

Finally, what is a stable surrogate for factual memorization that can be evaluated during quantizer training instead of after the fact? If we can derive a differentiable proxy that tracks the \( \Delta_{\text{fact}} \) gap without expensive prompt-generation, the quantizer could adjust its scales faster and more locally, rather than relying on slow benchmark evaluations.

## Where to read next

If you want the engineering systems picture, → *tensor cores* <!-- [[tensor-cores]] --> explains how NVIDIA’s matrix-multiply pipelines exploit low-precision compute to deliver sub-10 ms latencies. The theoretical counterpart is → *probabilistic model compression* <!-- [[probabilistic-model-compression]] --> which grounds quantization in the KL divergence between the quantized and original posterior. The deployment arc is continued by → *model servables* <!-- [[model-servables]] --> which describes how quantized checkpoints survive the transition to Triton or vLLM, and if you want the historical arc, → *model compression basics* <!-- [[model-compression-basics]] --> traces the compression ladder that brought us from pruning to quantization.

## Build it

This build proves that quantization is not a black-box library call: implementing round-to-nearest 4-bit quantization from scratch lets you observe how scale, zero-point, and stochastic rounding influence both memory and reasoning quality when you run the model on a real dataset.

**What you're building:** A custom PyTorch Round-To-Nearest 4-bit weight-only quantizer applied to `Qwen/Qwen-2.5-small` and evaluated for perplexity on a controlled WikiText-2 subset.

**Why this is valuable:** It surfaces the trade-off between quantizing weights (memory savings) and preserving reasoning capabilities (perplexity gap) on a free Colab T4, illustrating how the RTN quantizer’s calibration keeps cross-head correlations intact.

**Stack:**
- **Model:** [`Qwen/Qwen-2.5-small`](https://huggingface.co/Qwen/Qwen-2.5-small) — 1500+ downloads, 4.5 GB checkpoint
- **Dataset:** [`wikitext-2-raw-v1`](https://huggingface.co/datasets/wikitext/ wikitext-2-raw-v1) — canonical perplexity benchmark
- **Framework:** PyTorch 2.1 with `torch.nn.functional` and `torch.quantization` primitives
- **Compute:** Free Colab T4 (16 GB VRAM) — expect 1–1.5 hours for quantization + evaluation

**The recipe:**
1. Install the required packages (`pip install torch==2.1.0 transformers datasets accelerate`) and load the `Qwen/Qwen-2.5-small` checkpoint in `eval` mode while keeping the original FP32 weights for reference.
2. Preprocess a 2000-token slice of WikiText-2 using `datasets` to tokenize via `transformers.AutoTokenizer`, cache the token IDs, and create a calibration batch for each transformer layer.
3. Implement the RTN quantizer: for each weight tensor \(W\), compute \(s=(\max(W)-\min(W))/(2^4-1)\), \(z=\lfloor -\min(W)/s\rceil\), and define \(Q_{\text{RTN}}\) as described in §How it works, optionally adding stochastic \(\epsilon\) during calibration. Apply this quantizer to the attention projection matrices before writing them out to a quantized checkpoint.
4. Evaluate perplexity by running the quantized model (maintaining the remaining FP32 activations) on the WikiText-2 validation split, recording the FP32 baseline and the quantized result to compute \(\Delta_{\text{perplexity}}=\text{PPL}_{\text{quant}}-\text{PPL}_{\text{FP32}}\); target \(\Delta_{\text{perplexity}} < 2.0\).
5. What you now have is a quantized checkpoint plus a script that measures perplexity and logs the scale/zero-point statistics per layer, proving that a handcrafted RTN quantizer can live on limited hardware.

**Expected outcome:** A round-to-nearest 4-bit weight-only quantized `Qwen/Qwen-2.5-small` checkpoint together with reproducible perplexity logging showing the FP32 vs quantized gap.

- **CS student:** Run the same recipe on Colab’s CPU/GPU hybrid, reducing batch size to 1 and caching the dataset locally so the quantization loop fits inside the CPU’s 16 GB limit.
- **Applied engineer:** Extend to vLLM/Triton by serializing the quantized checkpoint, serving it at 50 ms p95 on batch size 2, and measuring throughput to show the quantized model fits within the 24 GB GPU memory budget.
- **Applied researcher:** Test the hypothesis that adding GuidedQuant-like gradient guidance to the RTN scales reduces \(\Delta_{\text{fact}}\) on the DeepResearch-9K subset; falsify it if \(\Delta_{\text{fact}}\) does not shrink by at least 10 %.
- **Frontier researcher:** Explore the open question of fractional bit-width allocation: implement a controller that assigns either 2, 3, or 4 bits per attention head based on activation entropy and verify that this dynamic allocation improves \(\Delta_{\text{reason}}\) without gradient-based retraining.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*