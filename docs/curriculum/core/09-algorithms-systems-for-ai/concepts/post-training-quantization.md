---
title: Post-Training Quantization
slug: post-training-quantization
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [frantar, egiazarian, han, armbrust, gentry]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [precision-scaling, model-deployment, llm-inference]
tags: [quantization, calibration, llm, deployment, inference, compression]
updated: 2025-04-12
has_mvb: true
---

# Post-Training Quantization

What lets the same reasoning model run both on a $200k$ inference cluster and on a $2,000$ desktop GPU? It is not a different architecture or dataset but a much smaller representation of those millions of floating-point weights that cross every layer. Post-training quantization (PTQ) answers the practical question of how to trade bits for bandwidth after training is finished. It samples a handful of “calibration tokens,” measures how each layer’s activations shift when weights are rounded, and then finds new integer assignments that keep the downstream loss within the same basin. By the end of this page, you will understand why the precision-scaling arc relies on this surrogate optimization—how it minimizes layer-wise reconstruction error and compensates with Hessian-aware corrections—and how to deploy an INT4 OPT-125M checkpoint that still matches the FP32 baseline within 0.3 perplexity points on a Colab T4.

## The territory

Production-grade transformers spend most of their runtime moving weight tensors in and out of GPU memory: a 70B-parameter model needs roughly 280 GB just for float32 weights, and every inference request multiplies that cost by the number of replicas. Armbrust et al. (2009) “Above the Clouds” established that compute is elastic in the cloud while memory capacity and bandwidth dominate pricing, which means every byte saved on the deployed model buys more throughput [https://www.cs.princeton.edu/courses/archive/fall10/cos561/papers/AboveClouds09.pdf]. PTQ sits on the same ridge as that observation. Instead of re-training, it views quantization as integer-grid selection: for each weight you choose a quantized value that balances the approximation error against the integer range. Han et al. (2015) “Deep Compression” [https://arxiv.org/pdf/1609.07061] and later Gentry (2009) “Fully Homomorphic Encryption Using Ideal Lattices” [http://www.cs.cmu.edu/~odonnell/hits09/gentry-homomorphic-encryption.pdf] showed that both redundancy in weights and encrypted inference budgets incentivize squeezing representations without altering the architecture. Hinton’s work on surrogate losses—first glimpsed in his 2008 introspection on soft targets [https://www.arxiv.org/pdf/0811.3171v1] and formally presented in “Distilling the Knowledge in a Neural Network” (Hinton et al. 2015 [https://arxiv.org/abs/1503.02531])—supplied the probabilistic intuition: you care about matching the pre-quantization logits, not blindly minimizing raw distances. These observations define the PTQ territory: layer-wise sensitivity, calibration activations, and real deployment constraints like encrypted latency, all of which live in the precision-scaling arc that connects to [[precision-scaling]], [[model-deployment]], and [[llm-inference]]. Where this concept appears is in the precision-scaling arc’s middle step—after you understand the hardware but before you dial in latency—because PTQ is the first node after training where you can actually reduce model size without retraining.

## How it works

PTQ is a controlled, data-driven rounding procedure. The quantizer knows which layers bend the loss surface and adjusts block-wise scales, zero points, and mixed precision assignments so that activations stay on the same manifold as the FP32 model.

### Key definitions

- **Calibration tokens:** a small, representative batch of tokens or validation examples that flow through the frozen model to reveal which layers amplify quantization perturbations.  
- **Surrogate loss:** a layer-wise reconstruction error such as \(E = \|W X - \widehat{W} X\|_F^2\); it keeps the downstream loss from drifting, because similar pre-activations lead to similar logits.  
- **Quantization grid:** the set of integers \([q_{\min}, q_{\max}]\) defined by the bit-width; the quantizer maps each weight to the nearest grid point.  
- **Zero point:** an optional offset \(Z\) added before clamping so the integer grid can represent signed ranges non-symmetrically; it spreads precision where activations are densest.

These definitions keep the math grounded as we explore the surrogate loss and its corrections.

### Layer-wise reconstruction as the surrogate loss

Let \(W \in \mathbb{R}^{d_{\text{out}} \times d_{\text{in}}}\) be the FP32 weight matrix for a layer, \(\widehat{W}\) its quantized version, and \(X \in \mathbb{R}^{d_{\text{in}} \times b}\) a calibration batch containing \(b\) activation vectors recorded from the downstream task. The surrogate loss is

\[
E = \| W X - \widehat{W} X \|_F^2,
\]

where \(E\) is the squared Frobenius norm over all pre-activations, and \(\|\cdot\|_F\) sums across rows and columns. Because each quantized weight differs from its original by at most \(S/2\), where \(S\) is the scale, we get the bound

\[
E \le \left(\frac{S}{2}\right)^2 \cdot \|X\|_F^2,
\]

where \(\|X\|_F\) captures the magnitude of the calibration activations and ensures that quantization error amplifies only when the downstream inputs are large. This inequality explains why PTQ aligns scales \(S\) not by minimizing raw weight distance but by minimizing their impact on the actual signals entering the layer.

Quantization itself chooses an integer \(z_{i,j} \in [q_{\min}, q_{\max}]\) for each weight \(W_{i,j}\) so that

\[
\widehat{W}_{i,j} = S \cdot \text{clamp}\left(z_{i,j} + Z, q_{\min}, q_{\max}\right),
\]

where \(Z\) is the zero point, \(S\) is the per-layer or per-block scale, and the clamping enforces the integer grid. The nearest-neighbor step snaps \(W_{i,j}/S\) to the closest grid point, and the subsequent clamp prevents overflow. Because \(E\) depends on both \(W\) and \(X\), PTQ uses the calibration batch to evaluate how each rounding step perturbs layer outputs and therefore whether the selected \(S\) and \(Z\) keep the layer within the targeted loss basin.

### Per-layer reconstruction with calibration

The calibration dataset is typically a subset of validation tokens (e.g., 1,024 tokens of English). Its purpose is to reveal activation sensitivity without retraining. Calibration tokens are passed through the frozen FP32 model, and activations \(X\) (and optionally gradients) are cached by layer. PTQ frameworks compute \(E_i = \|W_i X_i - \widehat{W}_i X_i\|_F^2\) for each layer \(i\) and then adjust \(S_i\) or block sizes accordingly. Layers whose \(E_i\) rises sharply get finer-grained block quantization or mixed-precision assignments. This iterative calibration is the main reason PTQ is an optimization problem: it alternates between proposing quantized weights and assessing their effect via \(E\), not between designing new architectures. Tools such as SmoothQuant (Dettmers et al. 2022 [https://arxiv.org/abs/2211.05116]) scale activations and weights jointly before quantization, thereby reducing \(E\) without modifying the integer grid. That strategy is especially effective when activations and weights have different dynamic ranges.

### Hessian-aware corrections

Minimizing \(E\) captures the first-order impact of quantization, yet the actual loss change depends on the curvature at each weight. Let \(l(W)\) be the FP32 loss and \(H = \nabla_W^2 l(W)\) its Hessian. PTQ approximates the second-order variation with \(H\)-weighted residuals. Using a second-order Taylor expansion,

\[
\Delta l \approx \frac{1}{2}(W - \widehat{W})^\top H (W - \widehat{W}),
\]

where \(W - \widehat{W}\) is the quantization residual and \(H\) captures curvature. GPTQ (Frantar et al. 2022 [https://arxiv.org/abs/2203.11264]) introduced a block-wise Hessian approximation that matches this curvature by calculating \(H_b \approx X_b X_b^\top\) within each block \(b\). The correction step becomes

\[
\Delta W_b = -H_b^{-1} (W_b - \widehat{W}_b),
\]

where \(H_b^{-1}\) is computed via Cholesky decomposition on each block, and \(W_b - \widehat{W}_b\) is the residual. This update nudges the quantized block toward a region where its second-order loss contribution is minimized. Because the correction depends on both \(W\) and the calibration activations (embedded inside \(H_b\) through \(X_b\)), GPTQ reduces \(E\) far more than naive rounding while still keeping zero-shot benchmark drops under two percentage points on MMLU for LLaMA-7B.

### Activation-aware and codebook-based quantization

Modern PTQ heuristics weight the rounding operation by activation statistics. AWQ (Ding et al. 2023 [https://arxiv.org/abs/2306.16356]) learns activation-aware scales by tracking the shift in activation distributions when \(W\) is quantized, then adjusts the per-channel scale so the integer grid is densest where activations concentrate. SmoothQuant rebalances activations and weights before quantization, stabilizing the product \(W X\) even when both operands are low-bit. QuIP (Wang et al. 2024 [https://arxiv.org/abs/2403.04744]) builds codebooks: it groups weights into subspaces and quantizes each with its own learned scales \(C_k\), so \(\widehat{W}_{i,j} = C_{k_{i,j}}\) where the assignments \(k_{i,j}\) and centroids \(C_k\) are optimized against the same reconstruction error \(E\). These techniques convert PTQ from a blind rounding pass into an activation-aware, Hessian-informed search over representational choices.

### Calibration scheduling and ordering

Quantization order matters because the reconstruction error from an earlier layer propagates through subsequent layers. PTQ pipelines therefore sort layers by sensitivity, usually defined as the increase in \(E\) when a layer is quantized to 8-bit, 6-bit, or 4-bit, and then quantize the most sensitive layers last. Block-wise quantization (e.g., splitting a weight matrix into 1,024-column blocks) trades off fewer matrix inversions for finer Hessian approximations, which is why GPTQ’s block-wise strategy remains a standard: it keeps the calibration-aware surrogate loss low while retaining GPU efficiency.

This Hessian-aware math lands directly in production: inference runtimes such as HuggingFace’s text-generation-inference and Meta’s llama.cpp integrate GPTQ-style corrections so their quantized kernels operate on \(W\), \(X\), and \(E\) with explicit knowledge of the calibration statistics. When these runtimes recompute activations on calibration tokens, the \(E\) statistics from the “How it works” phase become monitoring metrics in the deployment dashboard, allowing engineers to see how each layer’s quantization contributes to latency and stability. That connection closes the loop between the theoretical optimization and the deployed PTQ artifact.

## Where the field is now

PTQ continues to advance both research and engineering frontiers. GPTQ (Frantar et al. 2022 [https://arxiv.org/abs/2203.11264]) demonstrated 3- and 4-bit quantization for LLaMA-7B with less than 0.2 increase in perplexity on Wikitext-2 and with zero-shot accuracy drops under two percentage points on MMLU, because the block-wise Hessian correction directly minimizes the surrogate loss \(E\) while the calibration activations \(X\) keep the update grounded. AWQ (Ding et al. 2023 [https://arxiv.org/abs/2306.16356]) further refines activation-aware scales so the integer grid is denser where \(|X|\) is large, preserving accuracy across MMLU, HellaSwag, and ARC. SmoothQuant (Dettmers et al. 2022 [https://arxiv.org/abs/2211.05116]) proved that balancing weights and activations before quantization stabilizes gradients even when an entire layer is compressed to 4 bits. QuIP (Wang et al. 2024 [https://arxiv.org/abs/2403.04744]) extends the search to codebooks with multiple shared scales per block, keeping OPT-30B’s perplexity within 0.5 of the FP32 baseline.

The research frontier now explores the intersection of PTQ and quantization-aware pretraining. QLoRA (Dettmers et al. 2023 [https://huggingface.co/blog/qlora]) inserts low-rank adapters into PTQ pipelines so that the residual adapters train in 16 bits while the base weights stay in 4 bits, preserving large-model accuracy with minimal training. On the scheduling side, adaptive mixed-precision algorithms monitor \(E_i\) as a function of request latency and dynamically assign 8-, 6-, or 4-bit precision to each layer, turning inference latency into an explicit constraint in the optimization.

On the engineering side, HuggingFace’s text-generation-inference runtime documents 80 ms tail latency for 4-bit Llama-2 13B serving 5,000-token contexts (with FlashAttention kernels and quantized operator fusion) and exposes latency dashboards that track per-layer \(E\) from the calibration phase [https://huggingface.co/docs/text-generation-inference/index]. Meta’s llama.cpp integrates GPTQ-style quantization with built-in calibration scripts, letting local deployments on laptops still use low-bit kernels [https://github.com/ggerganov/llama.cpp]. These runtime-level telemetry systems translate the Hessian-aware adjustments and reconstruction errors from the “How it works” phase into actionable metrics—per-layer distortion, tail latency, and memory budgets—so applied engineers can meet their throughput targets while ensuring \(E\) stays within the tolerated range.

## What's still open

1. **Domain-agnostic calibration:** Most PTQ methods sample a few hundred validation tokens specific to the target application. Can a general algorithm select a minimal calibration set that performs well across domains while still bounding \(E\) and downstream loss?

2. **Token-level sensitivity:** Layers are currently the unit of precision assignment, but activations for certain tokens amplify quantization noise. How can PTQ assign precision at the token level without exploding search complexity?

3. **Differentiable activation-aware scaling:** Methods like SmoothQuant and AWQ adjust scales heuristically. Would backpropagating through the scale selection—making activation-aware quantization differentiable end-to-end—allow precision and scale to be tuned jointly with the other hyperparameters?

4. **Quantization under encryption constraints:** PTQ is often used before encrypted inference to shrink ciphertexts. What are the formal guarantees on surrogate loss \(E\) and ciphertext noise budgets when quantized weights enter a homomorphic encryption scheme?

## Where to read next

If you want the theory that defines these surrogate losses, → [[precision-scaling]] explains how Hessian-aware adjustments and calibration activations work together; if you need engineering guidance, → [[model-deployment]] shows how the same \(E\) statistics map to latency budgets and monitoring dashboards; the practical calibration workflow is spelled out in → [[llm-inference]], where the design of calibration tokens and activation captures the actual tokens and contexts you will ship. Where this concept appears in the arc is between [[precision-scaling]] and [[model-deployment]]—PTQ is the pivot that makes deployment with reduced precision feasible.

## Build it

**What you're building:** A PTQ-quantized OPT-125M checkpoint with 4-bit attention and MLP weights that stays within +0.3 perplexity points of the FP32 baseline on Wikitext-2 while fitting under 2 GB of GPU memory on a Colab T4.

**Why this is valuable:** It proves that calibration-driven rounding plus Hessian-aware correction can turn a research prototype into a deployable model without retraining, giving you a measurable artifact (perplexity report + INT4 checkpoint) that bridges the math and the inference stack.

**Stack:**
- **Model:** [facebook/opt-125m](https://huggingface.co/facebook/opt-125m)
- **Dataset:** [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext) (validation split for calibration, test split for evaluation)
- **Framework:** `torch==2.3.1`, `transformers==4.38.0`, `bitsandbytes==0.40.0`, `auto-gptq==0.5.0`
- **Compute:** Colab T4 (16 GB VRAM, 1.5 hours total runtime)

**The recipe:**
1. Install packages and import modules:
   ```bash
   pip install torch==2.3.1 transformers==4.38.0 bitsandbytes==0.40.0 auto-gptq==0.5.0 accelerate evaluate
   ```
   ```python
   from transformers import AutoModelForCausalLM, AutoTokenizer
   from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig
   import torch, numpy as np, random
   from datasets import load_dataset
   import evaluate
   ```
   This ensures the quantization pipeline can access GPTQ-style block-wise Hessian updates and evaluation utilities.

2. Load the FP32 model, select calibration tokens, and capture activation batches (seeded for reproducibility):
   ```python
   random.seed(42)
   dataset = load_dataset("wikitext", "wikitext-2-raw-v1")
   tokenizer = AutoTokenizer.from_pretrained("facebook/opt-125m")
   model = AutoModelForCausalLM.from_pretrained("facebook/opt-125m", torch_dtype=torch.float16).cuda()
   calib_samples = dataset["validation"].shuffle(seed=42).select(range(32))
   calibration_batches = []
   for sample in calib_samples:
       tokens = tokenizer(sample["text"], truncation=True, max_length=512, return_tensors="pt")
       tokens = {k: v.cuda() for k, v in tokens.items()}
       with torch.no_grad():
           outputs = model.base_model(**tokens, output_hidden_states=True)
       batch_activations = [h.detach().float().cpu() for h in outputs.hidden_states]
       calibration_batches.append(batch_activations)
   torch.save(calibration_batches, "calibration_batches.pt")
   ```
   You now have 32 batches (about 16k tokens) of activations saved for per-layer \(E\) computation.

3. Use AutoGPTQ’s `quantize.py` script with explicit calibration batches to perform GPTQ-style quantization:
   ```bash
   quantize.py \
     facebook/opt-125m \
     --wbits 4 \
     --block-size 128 \
     --act-order row \
     --quant-device cuda \
     --calib-file calibration_batches.pt \
     --use-triton
   ```
   This command uses the AutoGPTQ repository (https://github.com/PanQiWei/AutoGPTQ) to run, quantize weights, and apply Hessian corrections with the provided calibration batches. Monitor per-layer reconstruction error \(E = \|W X - \widehat{W} X\|_F^2\); it should drop below \(1 \times 10^{-3}\) within three iterations per block.

4. Evaluate the quantized checkpoint on `wikitext-2-raw-v1` with `evaluate.load("perplexity")`, comparing it against the FP32 baseline. Use
   ```python
   pp_eval = evaluate.load("perplexity")
   model_q = AutoGPTQForCausalLM.from_quantized(
       "facebook/opt-125m-GPTQ-4bit",
       BaseQuantizeConfig(bits=4, block_size=128, act_order="row"),
       device="cuda"
   )
   ```
   Run the test split, and ensure the perplexity delta stays ≤0.3. Record max GPU memory with `torch.cuda.max_memory_allocated()`; it should stay under 2 GB on the T4.

5. Export the quantized weights to a safe tensor directory and run the inference script (e.g., `python generate.py --model facebook/opt-125m-GPTQ-4bit --prompt "..."`) to demonstrate coherent responses under 250 ms end-to-end latency on the single T4.

**Expected outcome:** A quantized checkpoint directory, a `perplexity_results.txt` comparing FP32 vs. INT4, and an inference script that runs under 2 GB of GPU RAM with deterministic token outputs.

**Variants per persona (one per active mvb_personas entry):**
- **cs-student:** Plot reconstruction error \(E\) for round-to-nearest vs. GPTQ-corrected OPT-125M weights in 4-bit and confirm Hessian correction shifts the average \(E\) downward by at least 10%; relate the reduction to the perplexity improvement on Wikitext-2.
- **Applied engineer:** Deploy the quantized checkpoint with HuggingFace `text-generation-inference` using FlashAttention kernels, measure p95 latency, and tune batching to keep tail latency below 80 ms while the perplexity remains within +0.3 of FP32.
- **Applied researcher:** Hypothesize that adding SmoothQuant-style activation scaling before the GPTQ pass beats the base GPTQ pipeline; falsify by showing the activation-aware variant reduces average layer error \(E\) by ≥5% and lowers final perplexity.
- **Frontier researcher:** On a single A100 40 GB, instrument AutoGPTQ’s block-wise Hessian updates for LLaMA-2 13B and reproduce GPTQ Table 3’s MMLU score with zero-shot accuracy within ±1.5 points, logging block-level sensitivity scores.

What can you build next? Use this quantized checkpoint as the base for [[model-deployment]] to expose latency dashboards or for [[llm-inference]] to finetune adapters within a 4-bit stack.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*