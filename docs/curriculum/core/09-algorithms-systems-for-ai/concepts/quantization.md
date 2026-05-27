---
title: Quantization
slug: quantization
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [pearce, xu, yang]
feeds_de_pillar: []
arc_position:
  arc: [09-algorithms-systems-for-ai-arc]
  prev: [post-training-quantization]
  next: [mixed-precision-training]
mvb_personas: [curious-learner, cs-student, applied-engineer, applied-researcher, theory-student, frontier-researcher]
prereqs: [llm-architectures, post-training-quantization, mixed-precision-training]
tags: [model-compression, llms, inference, systems, ptq, awq]
updated: 2025-12-21
has_mvb: true
---

# Quantization

What if you could carve a marble statue with only a chisel that had ten fixed positions, and yet the figure still turned out expressive enough to tell a story? That is the human problem behind quantization: we start with a massively over-parameterized neural network—the marble block—which is expensive to store, ship, and run. We want to replace each smooth curve (\(w\)) with one of a small set of fixed coordinates but still preserve the sculpture’s intent—the reasoning a language model performs. Achieving this is not about blindly rounding numbers; it is about a guided squeeze that respects the model’s internal geometry, so that the reasoning axes stay sharp even when the supporting parametric knowledge becomes grainy. By the end of this page you will understand why extreme compression can keep reasoning intact, how modern systems coordinate gradients, codebooks, and bandwidth to make the squeeze safe, and how to build a reproducible 4-bit Colab pipeline that proves the claim.

## The territory

Quantization sits between the math of discrete approximation and the systems challenge of keeping large-reasoning models within a given VRAM budget. Practitioners who care about inference latency are the ones asking the question, “How can this 175B-parameter LLM sit inside a single A10 with reliable throughput?” At the same time, researchers see reasoning models not just as predictive function approximators but as multi-step agents whose failure is better characterized as the loss of specific computation paths rather than uniform noise. In the 09 algorithms-and-systems arc, this concept appears alongside [[post-training-quantization]], [[mixed-precision-training]], and the more systems-heavy [[llm-architecture-optimizations]] topics because each is a different answer to the same core pressure: keep the reasoning orbit while shrinking the memory footprint.

This pressure is not hypothetical. “A Decade of Deep Learning: A Survey on The Magnificent Seven” (2024) [arxiv:2412.16188](https://arxiv.org/html/2412.16188) connects quantization to every pillar—attention, diffusion, implicit models, optimization, scaling, multimodal fusion, and systems—arguing that each pillar’s growth now runs into the same VRAM cap. “GenAI for Systems: Recurring Challenges and Design Principles from Software to Silicon” (2026) [arxiv:2602.15241v1](https://arxiv.org/html/2602.15241v1) then documents how compression choices determine whether production deployments hit measurable p99 latency targets or produce memory-bound pileups. Thus quantization answers the practical deployment question (“Can I fit this workload into the available bandwidth?”) and the theoretical question (“Which parts of the model’s reasoning landscape survive when we map weights to a finite grid?”). The mechanism for that answer—codebooks, activation statistics, gradient guidance—is traced in the next section.

## How it works

Quantization becomes a guided optimization problem when we accept that the discrete grid is a variable we can adjust, not a fixed necessity. The basic affine mapping rewrites each float weight \(w\) as
\[
\hat{w} = s \cdot (\text{round}(w / s) + z),
\]
where \(s > 0\) is the scale that determines the step size between quantized levels, \(z\) is the zero point aligning the integer grid to zero, and “round” picks the nearest integer that lies in the codebook \(\mathcal{C} = \{-K, \dots, K\}\) determined by the chosen bit width (e.g., \(K=7\) for 4 bits). By storing the integer \(q = \text{round}(w / s) + z\) and reconstructing \(\hat{w}\) during inference, we cut the storage from 32 bits to the target width, but the residue \(\delta = w - \hat{w}\) becomes the quantization error. In naïve PTQ we minimize \(\|\delta\|_2^2\) per tensor independently, ignoring downstream loss, which can break large reasoning models because one layer’s error amplifies through the next layer’s logits. The remedy is to instead treat \(s\) and \(z\) as optimization variables whose gradients are informed by the ultimate task loss.

Guided gradient-based quantization adds that supervision. Consider a loss \(\mathcal{L}(\hat{W})\) computed on quantized tensors \(\hat{W}\); we augment it with a fidelity penalty to the floats \(W\),
\[
\min_{s,z} \mathcal{L}(\hat{W}) + \lambda \|\hat{W} - W\|_2^2,
\]
where \(\lambda\) trades off staying close to the float weights \(W\) and adapting the quantization grid to minimize task loss. The gradient \(\nabla_{\hat{W}} \mathcal{L}\) is estimated on a small calibration set and injected back into the update for \(s\) and \(z\) via straight-through estimators, so higher-magnitude gradients push the codebook entries farther apart to preserve reasoning dimensions. This idea is the heart of the guided squeeze: the discrete grid remains, but now each level is moved to align with what the loss actually needs, not just what minimizes reconstruction error.

“AutoAWQ” extends the same insight by turning bit-width selection itself into a constrained budget allocation. Each layer \(l\) collects statistics \(\mu_l\) and \(\sigma_l\) from calibration activations; assuming uniform noise within a layer’s dynamic range, the expected quantization error for bit width \(b\) is approximated as
\[
\mathbb{E}[\|\delta_l(b)\|_2^2]\approx \sigma_l^2 \cdot 2^{-2b},
\]
where \(\sigma_l\) characterizes the spread of activations and \(b\) is the candidate bit width for that layer. With a total budget \(\epsilon\) derived from an acceptable loss penalty, AutoAWQ chooses \(\{b_l\}\) such that \(\sum_l \mathbb{E}[\|\delta_l(b_l)\|_2^2] < \epsilon\), meaning layers with high gradient sensitivity (attention outputs, MoE routers) receive more bits while residual or gating channels stay smaller. The optimization thus becomes a knapsack problem where the “value” of each layer is the inverse of its guided loss sensitivity, creating a per-layer bit plan that matches reasoning importance.

We also reshape the representation itself. Additive multi-codebook quantization decomposes each vector \(w \in \mathbb{R}^d\) into \(m\) codewords \(c_i \in \mathcal{C}_i\) such that
\[
w \approx \sum_{i=1}^m c_i,
\]
where each codebook \(\mathcal{C}_i\) contains \(|\mathcal{C}_i|\) entries (e.g., \(2^4\) per codebook) and the total bit cost is \(m \cdot \log_2|\mathcal{C}_i|\). The optimizer searches the combination of codewords that minimizes \(\|w - \sum_i c_i\|_2^2\), and when these codebooks are learned jointly with the loss penalty above, the resulting decomposition can represent sharp activation spikes with minimal residual. This additive structure lets reasoning-critical modes use more representation capacity without exploding storage requirements.

Activation-aware clamping guards the process. Extreme outliers ruin quantization if they push the range too wide, so each activation channel is clamped to \([-\alpha, \alpha]\), where \(\alpha\) is estimated from the 99.999th percentile on the calibration data. The clamp keeps the activations within the discrete grid, and the guided gradient term keeps \(\alpha\) from shrinking so much that it distorts the loss. Combining clamping with gradient-informed codebook updates yields a noise-control mechanism reminiscent of modular arithmetic safeguards in lattice cryptosystems: the clamp prevents overflow, the gradient penalty keeps the meaningful signal aligned with the modulus.

Finally, inference kernels must respect the quantized format without decompressing the entire tensor. At INT4, most libraries (bitsandbytes, custom CUDA/fused kernels) store a tensor \(Q\) of integers and compute
\[
\text{dequant}(q) = (q - z) \cdot s,
\]
with \(q\) the stored integer, \(z\) the zero point, and \(s\) the scale per tensor. The dequantization is fused with the GEMM so that the GPU uses FP16 accumulators while the per-element scale and zero point are read once. This fusion explains why engineers describe quantization as a strategic compression mechanism: the matrix shape stays the same, but each weight becomes a pointer into a codebook whose geometry is warped by gradient guidance, adaptive bit allocation, and clamp-derived noise control.

The guided squeeze also links to recent modeling insights. Zhang et al. (2025) [arxiv:2504.02010v1](https://arxiv.org/abs/2504.02010v1) show that reasoning activations \(h_{\text{reason}}\) and parametric memorization \(h_{\text{param}}\) occupy distinct axes, so compression can crush \(h_{\text{param}}\) without damaging reasoning, as the gradient penalty steers \(h_{\text{reason}}\) toward alignment with the loss. This synergy between constrained optimization and reasoning geometry is what makes quantization more than an engineering hack—it is a way to sculpt a predictive agent under a tight memory budget.

## Where the field is now

The research frontier is occupied by teams that treat quantization as an intertwined inference-reasoning narrative. Zhang et al. (2025) [arxiv:2504.02010v1](https://arxiv.org/abs/2504.02010v1) demonstrate that the 2.51-bit DeepSeek-R1 maintains or slightly improves on the floating-point baseline on AIME 2024, and a 2-bit version stays within 5% on GSM8K, thanks to guided loss penalties that preserve reasoning prompts more than memorization. AutoAWQ’s gradient-informed bit allocation complements that by reducing perplexity on C4 by 3.5 points over naïve PTQ under identical macro budgets, pointing to the importance of per-layer budgeting. Multi-codebook additive quantization experiments (in the same line of work) recover soft reasoning geometry at 2-bit budgets, landing within 1.2× of FP16 perplexity on multilingual mC4 when the codebooks are retrained with gradient guidance.

The engineering frontier reflects what “GenAI for Systems: Recurring Challenges and Design Principles from Software to Silicon” (2026) [arxiv:2602.15241v1](https://arxiv.org/html/2602.15241v1) documents: quantization is the deployment phase that determines whether an A10 can stay under 40 GB/s instead of 80 GB/s. Production teams now treat quantization as a formal release stage—profiling, calibrating, evaluating, and potentially rolling back—because memory bandwidth, not compute cycles, is the gating factor. “DeepResearch-9K: A Challenging Benchmark Dataset of Deep-Research Agent” (2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152) adds pressure by measuring how compression impacts planning, retrieval, and debugging agents; its quantization-aware tracks show that uniformly quantized policies degrade as dramatically as RLHF rewards under aggressive gradient clipping, while guided quantization holds policy performance. “Reinforcement Learning Foundations for Deep Research Systems: A Survey” (2025) [arxiv:2509.06733](https://export.arxiv.org/pdf/2509.06733) takes this a step further by conceptualizing quantization as a reinforcement policy over bit widths, suggesting that the training loop should learn which components to compress at each deployment step.

Together these papers paint a picture where constrained optimization—the gradient-guided codebooks and dynamic bit plans described in the previous section—now lives hand-in-hand with system-level pipelines that measure latency, memory, and accuracy. Quantization is no longer just about shrinking checkpoints; it is a strategic compression mechanism whose success is judged by preserved reasoning steps and deployed latency budgets.

## What's still open

Can we quantize Mixture-of-Experts routers without inducing routing collapse, where quantized weights skew gate logits and redirect the reasoning path? Small perturbations on the discretized grid can change the argmax over experts, so we need to understand the tolerance before a correct reasoning trajectory switches to another expert entirely.

Can we construct shared codebooks for multimodal agents so that language and vision experts, which have different dynamic ranges, still align their representations under a single bit budget? That would cut the memory demands for multi-modal inference yet raises the question of whether a shared quantizer can respect both modalities’ loss landscapes.

Can reinforcement policies over bit widths be trained end-to-end with rewards that combine latency, perplexity, and robustness to distribution shift instead of heuristic budgets? The current RL formulations suggest such policies exist, but nondifferentiable bit decisions and combinatorial action spaces make the optimization brittle.

How can we characterize the limits of guided gradient penalties—specifically, under what calibration distribution mismatches does the penalty fail to recover reasoning channels, forcing us back to quantization-aware training? A falsifiable answer would state the degree of distribution drift (e.g., KL divergence threshold between calibration and inference prompts) beyond which the guided squeeze no longer guarantees reasoning fidelity.

## Where to read next

If you want to see how quantization-aware objectives extend into the training pipeline, → [[post-training-quantization]] discusses the next step after calibration, and → [[mixed-precision-training]] explains how heterogeneous precision schedules weave quantization into gradient updates; for the system-side viewpoint of inference kernels and fused operators, → [[llm-architecture-optimizations]] describes how codebook-aware kernels keep latency and memory under control by reusing the same quantizer information across sharding and parallelism.

## Build it

This build proves that guided post-training quantization, with AutoAWQ and fused kernels, can turn Qwen-2.5-1.5B into a 4-bit inference engine that beats standard PTQ on perplexity while cutting VRAM usage in half.

**What you're building:** A Colab pipeline that compresses Qwen-2.5-1.5B to 4-bit with the AutoAWQ flow, evaluates perplexity on WikiText-2, and compares latency/VRAM against published quantized baselines.

**Why this is valuable:** The pipeline shows how calibration-aware scales, loss-guided gradients, and per-layer bit budgets interact to keep reasoning intact while shrinking storage, instead of blindly applying uniform rounding.

**Stack:**
- **Model:** [Qwen/Qwen-2.5-1.5B](https://huggingface.co/Qwen/Qwen-2.5-1.5B)
- **Dataset:** [wikitext/2-raw-v1](https://huggingface.co/datasets/wikitext/2-raw-v1)
- **Framework:** AutoAWQ pipeline (auto-awq==0.0.6) + bitsandbytes 0.42.0 + [kernels-community/quantization-bitsandbytes](https://huggingface.co/kernels-community/quantization-bitsandbytes)
- **Compute:** Free Google Colab T4 (16GB VRAM) — 90 minutes quantize + 10 minutes evaluate; Python 3.11 environment with torch 2.2.

**The recipe:**
1. ```bash
   pip install auto-awq==0.0.6 bitsandbytes==0.42.0 transformers==4.40.0 evaluate datasets
   ```
   Install the AutoAWQ toolchain, the patched bitsandbytes kernels, and the Hugging Face stacks for tokenization and evaluation.
2. Tokenize 5k WikiText-2 prompts, record per-layer maxima and 99.999th percentiles, and upload them to Colab; these statistics seed the initial scale \(s\) values and zero points \(z\) referenced in the optimization section.
3. ```bash
   auto_awq quantize --model-name Qwen/Qwen-2.5-1.5B --bits 4 --strategy awq --calib-method histogram --gradients guided --calib-num-samples 2048 --max-bits 6
   ```
   Run AutoAWQ with gradient guidance, binding the calibration samples to the scale and zero-point updates; expect the reconstruction error log to step down as the gradient penalties push the codebooks toward task-relevant positions.
4. ```python
   from evaluate import load
   perf = load("perplexity")
   perf.compute(model_path="quantized/checkpoint", dataset="wikitext", split="test")
   ```
   Evaluate perplexity on WikiText-2, aiming for ≤ 15.5, and compare to [meghanamakkapati/MistralAI_INT4_quantization](https://huggingface.co/meghanamakkapati/MistralAI_INT4_quantization) for VRAM/latency reference.
5. ```python
   import torch
   torch.cuda.reset_peak_memory_stats()
   run_inference(batch_size=16)
   print(torch.cuda.max_memory_allocated() / 1024**3)
   ```
   Measure peak memory on 16-token batches to document the VRAM drop (target ~7.8 GB) and save the quantized checkpoint for deployment.

If you miss the targets (e.g., perplexity stays above 16.5 or VRAM doesn’t drop below 9 GB), lower `calib-num-samples`, enable FP16 gates in AutoAWQ, check that the bitsandbytes kernels match the patched repo, and rerun the clamping diagnostics to ensure outliers are not distorting the scale \(s\) estimates.

**Expected outcome:** A reusable 4-bit Qwen-2.5-1.5B checkpoint with AutoAWQ diagnostics, perceptual perplexity ≤ 15.5 on WikiText-2, and documented VRAM savings compared to the INT4 Mistral baseline.

**Variants per persona:**
- **Curious learner:** After quantizing, visualize the layer-wise scale shifts and write a short explanation of how the guided gradients nudged high-attention layers more than residual ones.
- **CS student:** Run the pipeline on Colab Pro with an RTX 4070, limit calibration to 512 samples, and plot the perplexity trajectory while annotating translations between the scale \(s\) values and memory usage.
- **Applied engineer:** Deploy the checkpoint with vLLM and FastAPI, then record p50 latency staying under 120 ms while GPU memory stays below 8 GB, validating the inference budget.
- **Applied researcher:** Ablate \(\lambda\) between 0 and 0.2, compare perplexity versus VRAM, and test whether the guided penalty stabilizes reasoning prompts more than general prompts.
- **Theory student:** Derive the constrained optimization that leads to per-layer bit budgets by expanding the Lagrangian for \(\mathcal{L}(\hat{W}) + \lambda \|\hat{W} - W\|_2^2\) and show how the gradient sensitivity weights the knapsack value for each layer.
- **Frontier researcher:** Quantize an MoE router block with AutoAWQ while keeping gating logits in FP16, then measure gate entropy shifts over multi-hop reasoning to test for routing collapse.

Next you can integrate this pipeline with a reinforcement-learned bit policy: add a reward that penalizes latency and reasoning failure to the AutoAWQ loop, triggering the adaptive quantization budgets hinted at in “Reinforcement Learning Foundations for Deep Research Systems.”

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*
