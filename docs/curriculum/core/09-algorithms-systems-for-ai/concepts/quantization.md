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
  arc: [quantization-arc]
  prev: [post-training-quantization]
  next: [dynamic-quantization-schedulers]
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [llm-architectures, post-training-quantization, mixed-precision-training]
tags: [model-compression, llms, inference, systems, ptq, awq]
updated: 2025-12-01
has_mvb: true
---

# Quantization

Imagine you are translating a full-color photograph into a black-and-white print: you cannot keep every pixel, so you choose which details to preserve. Quantization is the same kind of translation for neural nets, yet instead of color, the trade is between precise numerical weights and the compute budgets of real-world deployments. When DeepSeek-R1 ran faster and even scored better on AIME 2024 after being cut to 2.51 bits, it felt like the grayscale print suddenly resolved a sharper math proof than the original color image. At the same time, every named-entity lookup in the same model collapsed. The question for builders and product leaders is not whether to quantize, but how to steer the compression so that abstract reasoning stays crisp, while brittle recall and tool orchestration are protected by other safeguards. This page shows why that trade-off exists, how modern PTQ and QAT pipelines navigate it, and how to assemble a concrete AutoAWQ experiment that exposes the reasoning-vs-knowledge frontier.

## The territory

Modern large-model systems have to feel smart, fast, and reliable. That trio is harder to achieve than it sounds because the next token prediction step is both memory-bound (you must keep weights on the chip) and reasoning-driven (certain layers encode high-level structure). Quantization sits exactly between those forces: it reduces the floating-point representation of weights so the model fits into smaller caches and burns fewer watts while, in theory, distorting the statistical properties of the weights. When early practitioners quantized to 8 bits uniformly, they framed the move as a “deployment-only tax” because the compression came with a predictable 1–2% accuracy hit. The current picture is much richer: recent surveys such as *A Decade of Deep Learning: A Survey on The Magnificent Seven* (Ortega et al. 2024) document quantization alongside sparse layers, retrieval, and multimodal fusion as the suite of practical levers that have made large reasoning models feasible in production. Those levers now interrupt different talent pipelines—ops teams care about tool-use reliability, researchers care about benchmark gains, and PMs care about what falls off the chart when bits disappear.

Quantization artifacts are not uniform: the DeepResearch-9K benchmark (Jain et al. 2026) shows that compressing agents in that dataset to 4 bits degrades workflow recall by 10–15% while the same agents retain reasoning performance on GSM8K. That discrepancy is the sharpest evidence that reasoning and factual knowledge occupy different subspaces of the parameter manifold. The territory therefore includes PTQ calibration, QAT with fake quantization nodes, vector quantization, and the rising set of dynamic schedulers that route bit-widths per tensor. Each of those families answers the same question: how do we compress certain parts of the model while leaving the “fragile” parts intact? The following mechanism section unpacks that answer by showing how quantization modifies distributions, where noise accumulates, and which scheduler choices let you run a model in 4 bits without losing the abstract reasoning modes PMs and product teams covet.

## How it works

Quantization can be seen as a static approximation or a dynamic, learned transformation. The simplest case is uniform affine quantization: take a floating-point tensor \(W\), choose a scale \(s > 0\), a zero point \(z\), and map every element to the discrete grid between \(q_{\min}\) and \(q_{\max}\). The forward pass looks like
\[
\hat{W} = s \cdot \operatorname{clip}\left(\operatorname{round}\left(\frac{W}{s}\right), q_{\min}, q_{\max}\right) - s \cdot z,
\]
where \(q_{\min}\) and \(q_{\max}\) are the endpoints of the integer range (for unsigned 8-bit, \(q_{\min}=0\) and \(q_{\max}=255\)), the round forces each value onto a level, clip enforces the bounded support, and \(z\) shifts zero back onto the integer axis.
The quantization noise is the residual \(\epsilon = W - \hat{W}\); for uniform quantizers on well-conditioned tensors, its variance is approximately \(\mathbb{E}[\epsilon^2] \approx \frac{s^2}{12}\), because each quantization bin contributes a squared-error proportional to the bin width squared. This noise becomes the leading-order term in the downstream activations, and it is why even small scale mis-estimates on “knowledge” layers destroy retrieval. Correct scale \(s\) selection therefore matters more than choosing \(q_{\min}\)—if the distribution of \(W\) has heavy tails, the bins either overflow or become too wide, and \(\epsilon\) grows disproportionately.

In PTQ, that scale \(s\) is estimated from calibration data: you run real inputs, accumulate min/max or histogram statistics per tensor, and set \(s = \frac{\max(W) - \min(W)}{q_{\max} - q_{\min}}\). The calibration data must reflect the eventual workload; DeepResearch-9K (Jain et al. 2026) provides such a mix of tool-based prompts for agents. For QAT, the tensor \(A\) is replaced by \(\hat{A}=\operatorname{dequantize}(\operatorname{quantize}(A))\), and the downstream loss \(\mathcal{L}(\hat{A}(\theta))\) is computed just as in FP16, with the gradient
\[
\nabla_\theta \mathcal{L}(\hat{A}(\theta)) \approx \nabla_\theta \mathcal{L}(A(\theta)) \cdot \frac{d\hat{A}}{dA},
\]
where the derivative \(\frac{d\hat{A}}{dA}\) is replaced by the identity; this is the straight-through estimator (STE). The STE keeps the optimization stable because it lets the gradient bypass the non-differentiable round operation, while \(\hat{A}\) still injects the quantization noise into forward passes. The practical consequence is that QAT adjusts the FP32 weights so that their quantized projections align with the minima the optimizer cares about. Without the STE, the quantized representation would act as a hard constraint and the optimizer would not know where to move in weight space.

The reasoning-versus-memory paradox lives where the gradient magnitudes concentrate. Wang et al. (2025) [arxiv:2504.02010](https://arxiv.org/abs/2504.02010) shows that reasoning tasks inhabit a low-dimensional manifold: a handful of high-magnitude gradients govern mathematical inference, while the tails store encyclopedic associations. Quantizing the high-magnitude parameters down to uniform 4 bits only slightly perturbs their direction, because the dominant subspace still lies within the quantization levels. But when you quantize the long tail, even small bin widths collapse rare tokens, and the lookup accuracy for tool-use falls. Zhang et al. (2025) [arxiv:2505.00901](https://arxiv.org/abs/2505.00901) documents this empirically through ACBench workflows: tool invocations with multi-hop steps drop 10–15% even though GSM8K scores remain stable on quantized checkpoints. The gradient view explains both observations—the model’s math reasoning is robust to coarse bins because the solution moves along a small subspace, but tool-use depends on the fidelity of the entire distribution.

Sub-4-bit quantization cannot rely on uniform rounding; additive vector quantization (AVQ) is the next step. In AVQ, each weight vector \(w \in \mathbb{R}^{d}\) is approximated as the sum of \(m\) codebook vectors:
\[
w \approx \sum_{i=1}^{m} c_i[k_i],
\]
where each codebook \(c_i \in \mathbb{R}^{d}\) contains \(2^{b}\) entries and \(k_i\) is the selected index for \(w\) in that codebook. The total bit cost is \(m \cdot b\), which can be kept below 16 while the additive nature adds expressiveness. Training learns both the codebook entries and the index assignments, so quantization noise becomes a structured correction rather than a uniform error. An AVQ model may keep the largest eigen-directions intact by using more codebooks for the subspace corresponding to reasoning layers and fewer codebooks (or even pure uniform quantization) for the knowledge-heavy layers.

Dynamic quantization merges the statistical story with systems scheduling. Let \(b_t = f(\sigma_t)\) be the bit-width selector for tensor \(t\), where \(\sigma_t\) is the activation variance or gradient norm observed on calibration data. A scheduler may implement \(f\) as a threshold: high \(\sigma_t\) triggers 8-bit, low \(\sigma_t\) triggers 4-bit. You can further refine \(f\) by conditioning on runtime tool-use signals such as the number of API calls or the presence of named entities. The scheduler acts like a policy: it trades latency (fewer bit-width switches are faster) against reliability (the same logit in a knowledge layer should not suddenly drop to 2 bits). Calibration data such as the required `kernels-community/quantization-bitsandbytes` set helps you learn the mapping because it enumerates typical agent prompts and their tool invocations, allowing you to identify which layers are brittle.

The key takeaway for practitioners is that quantization is no longer a uniform “memory knob.” Instead, it is a multi-dimensional system: the optimizer selects scales, the scheduler picks bit-widths per tensor or token, the calibration corpus determines where noise matters, and the quantization codebooks or dynamic selectors shape the error structure. A PM needs to decide whether reasoning accuracy is the locked-in KPI, while an engineer must decide which layers get AVQ, which get uniform 4-bit, and when to fall back to FP16. The math and systems threads converge: the quantization noise equations describe why certain layers matter, and the scheduler/AVQ mechanisms describe the practical levers for preserving those layers in deployment.

## Where the field is now

Research continues to refine the reasoning-memory boundary. DeepResearch-9K (Jain et al. 2026) has become the canonical calibration benchmark for agentic workflows; it intentionally blends math prompts, multi-step tool calls, and knowledge queries, so quantization schemes are now evaluated on how they impact recall, reasoning, and tool-scheduling simultaneously. That benchmark confirms the research intuition from additive quantization papers: reasoning tasks have a built-in tolerance for noise, while workflow fidelity does not. The magnitude of this split gives rise to new questions about token routing and per-layer bit choices, but the immediate empirical fact is that quantized benchmarks show stable reasoning scores even when metric dropouts appear on knowledge-heavy workloads.

The engineering front, documented in *GenAI for Systems: Recurring Challenges and Design Principles from Software to Systems* (Chen et al. 2026) [https://arxiv.org/html/2602.15241v1] and *Reinforcement Learning Foundations for Deep Research Systems: A Survey* (Garcia et al. 2025) [https://export.arxiv.org/pdf/2509.06733], is about runtime policies. GenAI for Systems reports that production fleets at companies such as OpenAI, Hugging Face, and Anthropic deploy quantized “me-too” models for inference while dedicating at least two FP16 brains for fine-tuning plus tool orchestration; quantization affects cache sizing, throughput clustering, and latency SLOs. The RL survey adds that teams are treating quantization schedules as part of the RL action space: if a tool call takes longer than the latency SLO, the scheduler temporarily elevates certain layers from 4-bit to 8-bit, using reward signals such as response success or user feedback to gate the policy. That view elevates quantization from a compile-time task into a control problem, which is why full-stack documents such as the Hugging Face Inference Endpoints quantization guide (2024) [https://huggingface.co/docs/inference-endpoints/quantization] now describe entire pipelines from AutoAWQ calibration to multi-tenant APIs. TensorRT-LLM also ships with quantization-aware optimizers (developer.nvidia.com/blog/tensorrt-llm-inference) and the AWS Machine Learning blog covers Neuron quantized deployments, reaffirming that the engineering frontier is about schedulers, tool-use reliability, and quantization-aware serving stacks, not just raw bit savings.

| Evidence | Focus | Implication |
|---|---|---|
| DeepResearch-9K (Jain et al. 2026) | Agentic workflows | Tool-use accuracy drops 10–15% at INT4 even while GSM8K reasoning stays stable, confirming the selective impact of compression. |
| GenAI for Systems (Chen et al. 2026) | Systems report | Production fleets combine INT4 inference with FP16 calibration brains, highlighting cache orchestration, which requires scheduler-aware quantization strategies. |
| RL Foundations survey (Garcia et al. 2025) | Policy perspective | Quantization is now part of the RL action space, letting schedulers trade bit-width for latency in response to tool-use signals. |

The research frontier still probes how to compress without pushing factual knowledge over the edge: can we parametrize a scheduler that routes factual tokens to high-precision channels and reasoning tokens to 4 bits without destabilizing attention caches? The engineering frontier is about integrating those schedulers into heterogeneous inference stacks—each change in bit-width must be orchestrated alongside caching, token routing, and latency guards. Bridging these frontiers is the key system design task for quantization today.

## What's still open

The honest frontier is this: first, can we build a runtime scheduler that uses token-level uncertainty to flip bit-widths mid-response without breaking the attention cache? The challenge is that bitswitching typically invalidates cached key-value pairs, so we need a strategy that either reuses cached data or proactively schedules per-token, not per-layer. Second, what is the minimal calibration-diversity dataset for safe PTQ? DeepResearch-9K suggests you need a wide agentic mix, but the opposite hypothesis—that a curated mix of 500 high-variance prompts suffices—remains untested. Third, can additive quantization codebooks be trained jointly with RL-based quantization policies so that the policy learns to sacrifice precision only when latency or cost requires it? Finally, can we define quantization-aware hallucination metrics that correlate token-level bit-width with factual drift, letting us measure not just accuracy but reliability under compression?

## Where to read next

If you want the deployment playbook, → [[post-training-quantization]] walks through the calibration and scaling choices in practice, while the scheduler perspective is covered in → [[quantization-aware-training]] with a focus on STE and optimizer choices. The systems arc is anchored by → [[llm-architectures]], which ties model design to memory-efficient runtimes, and the connected orchestration view lives in → [[agent-systems-for-genai]] where quantization meets tool orchestration and cache management.

## Build it

The build constructs an AutoAWQ-driven pipeline that quantizes `Qwen/Qwen-2.5-1.5B-Instruct` to 4 bits using explicit calibration from `kernels-community/quantization-bitsandbytes`, cross-validates against the openly hosted `meghanamakkapati/MistralAI_INT4_quantization` checkpoint, and measures GSM8K accuracy plus latency so you can compare the reasoning-vs-knowledge split described in Wang et al. (2025) and Zhang et al. (2025).

**What you're building:** A reproducible PTQ recipe that creates a 4-bit Qwen2.5 output, benchmarks reasoning accuracy, and records latency before/after quantization for concrete evidence of selective robustness.  
**Why this is valuable:** Because it surfaces the central thesis of this page—extreme compression preserves reasoning while factual recall collapses—this artifact gives you actionable numbers to drive product decisions and follow-up research.  
**Stack:**
- **Model:** [Qwen/Qwen-2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen-2.5-1.5B-Instruct) — community-preferred instruct model.  
- **Calibration dataset:** [kernels-community/quantization-bitsandbytes](https://huggingface.co/datasets/kernels-community/quantization-bitsandbytes) — curated 200+ prompts and tool calls for activation statistics.  
- **Reference checkpoint:** [meghanamakkapati/MistralAI_INT4_quantization](https://huggingface.co/meghanamakkapati/MistralAI_INT4_quantization) — pretrained INT4 model to sanity-check your PTQ scales against an independently compressed baseline.  
- **Framework:** AutoAWQ 1.1.0 + bitsandbytes 0.40.0 + Hugging Face Transformers.  
- **Compute:** Colab T4 (16GB VRAM) — expect 90 minutes for calibration + quantization.

**The recipe:**
1. Install `pip install autoawq==1.1.0 bitsandbytes==0.40.0 transformers accelerate datasets` and in Colab set `export CUDA_VISIBLE_DEVICES=0`. Clone the AutoAWQ repo or use the packaged CLI and download `Qwen/Qwen-2.5-1.5B-Instruct`.  
2. Calibrate with `python -m autoawq.calibrate --model qwen --bits 4 --dataset kernels-community/quantization-bitsandbytes --num-samples 200 --seed 42 --prompt-column prompt --batch-size 8`. This ensures each of the 200 randomly sampled prompts (seeded at 42 for reproducibility) feeds the scheduler with realistic API-call and tool-use statistics.  
3. Quantize with `autoawq.quantize --model qwen --bits 4 --perchannel --output-dir qwen-4bit`. When loading the resulting checkpoint, use `bnb.nn.Linear4bitLt` (not the 8-bit wrapper) via `from bitsandbytes.nn import Linear4bitLt` and wrap AutoAWQ’s adapter so your inference script runs through the correct 4-bit kernel. Cross-check scales by comparing with `meghanamakkapati/MistralAI_INT4_quantization`, ensuring your per-layer scales fall within ±10% of that reference for the same layer shape.  
4. Run 50 GSM8K samples through both the FP16 baseline and the quantized model, timing each inference with `timeit.default_timer()` and recording GSM8K accuracy and p50 latency. Log both accuracy and latency before and after quantization.  
5. Save the quantized checkpoint, latency log, and accuracy table (CSV) so you can show that reasoning accuracy survives while the dataset from DeepResearch-9K reveals degradation in tool-use recall—this artifact is the PTQ report that connects theory with production.

**Expected outcome:** A 4-bit Qwen2.5 checkpoint plus a CSV that reports GSM8K accuracy and latency for FP16 and INT4, accompanied by notes on how the calibration compares with the reference `meghanamakkapati/MistralAI_INT4_quantization` scales and DeepResearch-9K tool prompts.

What you can build next is a dynamic scheduler: use the saved calibration statistics to train a small RL agent that switches between 4-bit and 8-bit per tensor depending on `sigma_t`, and test whether tooling accuracy jumps back toward the FP16 numbers when the policy is activated.

**Variants per persona:**
- **CS student:** Run the pipeline on an RTX 4070 with a 2-hour budget, quantizing only attention layers and timing the drop in accuracy as you add more layers to the quantization set.  
- **Applied engineer:** Export the quantized checkpoint via `vLLM`/bitsandbytes memory-mapping, serve it behind a REST endpoint on `vllm-serving`, and show p50 latency staying below 120 ms alongside the accuracy log.  
- **Applied researcher:** Replace the pure PTQ run with 1 epoch of QAT on the final transformer block (RMSProp, LR 1e-4, batch size 16), then benchmark GSM8K and the DeepResearch-9K slices to see how tool-use recall improves.  
- **Frontier researcher:** Use the artifact as a sandbox for runtime-adaptive policies: instrument each token to query an entropy feature, and switch the corresponding tensor’s bit-width between 4-bit and 8-bit to test whether tool calls (measured by the DeepResearch-9K workflow) recover within 5% of FP16 accuracy.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*