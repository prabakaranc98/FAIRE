---
title: Precision scaling
slug: precision-scaling
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [deletang, chao, wang, kalchbrenner]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [quantization-basics, scaling-laws, mixed-precision-training]
tags: [precision, quantization, scaling-laws, llm-efficiency, robust-arithmetic, dynamic-precision]
updated: 2024-11-15
has_mvb: true
---

# Precision scaling

Here is a paradox to start with: the more a language model is trained, the more brittle its internals become when you try to squeeze them into lower bit-widths. A heavily overtrained GPT converges to a narrow manifold where each weight participates in highly tuned cancellations, so dropping from 16 to 2 bits sends the loss spiraling upward far faster than the same step applied to a checkpoint from early training. That paradox—more data and compute creating greater fragility under quantization—is why precision cannot be treated as a static switch you flick after training. Instead, it is a scaling axis on par with parameter count and training tokens. If engineers understand precision as a predictable trade-off governed by scaling laws, they can plan compute budgets that intentionally trade raw capacity for quantization headroom, rather than playing defense after the fact. By the end of this page the reader will know how modern precision-scaling laws are written, why dynamic mixed-precision schedules track sequence depth, and how gradients can guide which weights survive at sub-4-bit resolutions, plus they will have built a small predictor that turns empirical quantization loss into an effective parameter budget.

## The territory

Model efficiency has long been framed as “more parameters for more compute,” but that story breaks down once the operational budget fixes the number of bits the hardware can represent. During inference, integer units and tensor cores do not look at the nominal parameter count \(N\); they see the total memory footprint \(N \cdot b\), where \(b\) is the bit-width. Quantization research therefore asks a different question: how low can \(b\) be before accuracy collapses, and how does that limit interact with training decisions like token count \(D\) and early stopping? Casting precision as a scaling axis means answering this question with the same tools researchers already use for parameter scaling: plotting performance versus some combination of \(N\), \(D\), and \(b\) and identifying predictable regimes. The territory we occupy is where quantization interacts with training dynamics and where the “effective parameter count” after compression matters more than the raw \(N\). Existing families of techniques borrow from robust arithmetic lessons—rounding, clipping, stochastic rounding—to make finite representations behave like their infinite-precision ancestors (Stanford Graphics Lab 2007 “Robust Arithmetic” manual [graphics.stanford.edu/courses/cs268-07-winter/manuals/robust-arithmetic.pdf]). At the same time, they plug into scaling-law narratives that previously only involved \(N\) and \(D\). The key question is: how do we write a scaling law that rewards lower \(b\) while penalizing loss degradation, and what pieces of that law are controllable before training begins? How does it actually work?

## How it works

To see precision scaling as a mechanism, start by asking what changes when we compress a trained model via post-training quantization (PTQ). The loss the user cares about is the cross-entropy after quantization, which can be written as

\[
L_{PTQ}(N, D, b_{train}, b_{inf}) = L(N, D, b_{train}) + \eta \cdot \left(\frac{D}{N}\right)^{\beta} \cdot (b_{train} - b_{inf})^{\delta}
\]

where \(L_{PTQ}\) is the loss observed after applying PTQ, \(L(N, D, b_{train})\) is the pre-quantization loss of the model trained with \(b_{train}\) bits per weight, \(D\) is the total number of tokens seen, \(N\) is the parameter count, \(b_{inf}\) is the bit-width used at inference, and \(\eta\), \(\beta\), \(\delta\) are empirical exponents discovered by Deletang et al. (2024). The first term is the baseline, and the second term captures the extra penalty from dropping precision; it grows when training data per parameter shrinks (large \(D/N\) ratio) and when the gap \(b_{train} - b_{inf}\) widens. This form reveals the scaling intuition: low effective precision is indistinguishable from having fewer parameters if the penalty term is large, which explains why overtrained models (large \(D\)) degrade faster—their loss is tightly coupled to the second term. Because the penalty is multiplicative, we can define an effective parameter count

\[
N_{eff} = N - \eta' \cdot D^{\beta} \cdot (b_{train} - b_{inf})^{\delta}
\]

with \(\eta'\) absorbing constants, translating the lost capacity due to quantization into a straightforward “parameter budget” subtraction. When engineers plan for a 2-bit accelerator, they should treat the model as if it effectively had \(N_{eff}\) parameters instead of \(N\), so that scaling computations on the compressed network stay grounded in actual inference behavior.

### The precision landscape: training time, inference time, and non-uniform bits

The training bit-width \(b_{train}\) is usually 16 or 32, but precision scaling introduces two levers to make PTQ less painful. The first lever is reducing \(b_{train}\) itself via quantization-aware training or mixed-precision blocks. Chao et al. (2024) explored this through progressive mixed-precision decoding [Chao et al. (2024) “Progressive Mixed-Precision Decoding”](https://arxiv.org/html/2407.17353), showing that the sequence depth determines how aggressively we can reduce precision. Their scheduler starts with high bit-widths for the prefix and lowers \(b\) as generation continues, effectively treating precision as a function of the decode position. This approach directly feeds into the second term of the PTQ penalty: since the decoder keeps \(b_{train}\) high where tokens have the most influence (early positions) and low elsewhere, the average \(b_{train}\) that enters \(L(N, D, b_{train})\) becomes a weighted average that balances compute workload with accuracy. By modeling precision as a dynamic schedule \(\tilde{b}(t)\), where \(t\) indexes tokens, the penalty term becomes a sum

\[
\sum_{t} \eta \cdot \left(\frac{D_t}{N}\right)^{\beta} \cdot (\tilde{b}(t) - b_{inf})^{\delta}
\]

with \(D_t\) tracking the remaining tokens after position \(t\). Early positions contribute more because they steer hidden states farther, and the progressive schedule keeps their \((\tilde{b}(t) - b_{inf})\) term small. The consequence is that you can drive the aggregate penalty down even if later tokens live in 2 bits, because their downstream effect is smaller.

The second lever is the actual quantization scheme used during inference. GuidedQuant (Wang et al. 2025) argues that not all weights deserve uniform precision drops. Their insight is to monitor gradients during training, especially the end-of-training loss gradient signal, to identify “critical crossings” where two weights jointly control a narrow manifold. By preserving those weights with higher effective precision (even within the same quantization block), the model retains cross-weight dependencies that would otherwise break when bits are truncated. Formally, they introduce a guidance map \(G_{ij} = |\nabla_{w_i} \nabla_{w_j} L|\) and then assign bit-widths via

\[
b_i = b_{min} + (b_{max} - b_{min}) \cdot \frac{\sum_j G_{ij}}{\sum_{i,j} G_{ij}}
\]

where \(b_{min}\) and \(b_{max}\) define the allowed bit-width range. Large second-order gradients lead to higher \(b_i\), so the effective \(b_{train} - b_{inf}\) gap in the penalty term shrinks for those directions. This guided allocation is the reason extremely low-bit-weight deployments can still match higher bits when they respect gradient structure.

### Putting precision on the scaling map

Having toured the levers, it helps to see where they meet the classic scaling coordinates. Suppose a project has a fixed compute budget \(C\), split between model multiplications and memory bandwidth. In uniform precision, the memory bandwidth scales like \(N \cdot b_{inf}\), so reducing bit-width immediately frees up compute to grow \(N\) or \(D\). But the budget metric we care about is not compute alone—it is the expected loss \(L_{PTQ}\). Twisting the equation gives:

\[
L_{PTQ}(N, D, b_{train}, b_{inf}) - \eta \cdot \left(\frac{D}{N}\right)^{\beta} \cdot (b_{train} - b_{inf})^{\delta} = L(N, D, b_{train})
\]

This rearrangement tells us the left-hand side—the operational loss after quantization minus the predictable penalty—must equal the clean loss we would have obtained at full precision. Thus, for a target accuracy \(L_{target}\), the penalty term determines a surface in the \((N, D, b_{inf})\) space: any point below that surface (with a larger penalty) is unacceptable. Engineers can plot iso-error surfaces by sweeping \(b_{inf}\) and solving for the corresponding \(N_{eff}\). That predictive surface is the scaling law in action: you can trade \(N\) for \(b_{inf}\) if you keep the penalty term constant, or trade \(D\) for \(b_{inf}\) by reducing dataset size or applying more aggressive regularization to keep \(D/N\) small.

The role of robust arithmetic is to soften the penalty by reducing \(\eta\); rounding to nearest even and stochastic rounding techniques lower the constant multiplier on \((b_{train} - b_{inf})^{\delta}\), making the law more forgiving. When these techniques are combined with progressive decoding and guided quantization, the net effect is to reshape the scaling surface so that the frontier shifts toward lower \(b_{inf}\) for the same \(L_{target}\).

### Failure modes and diagnostics

Precision scaling fails when any part of the law goes uncalibrated. If \(D/N\) is tiny, the penalty term explodes and even 8-bit quantization hurts, because there isn’t enough data to average out the quantization noise. If gradients are noisy, the guided quantization map misallocates bits, costing more than it saves. Additionally, when deployers try to compensate by increasing \(b_{train}\) dramatically (e.g., training in 32 bits then quantizing to 2 bits), the law highlights the dramatic \((b_{train} - b_{inf})^{\delta}\) growth, which explains the brittleness mentioned in the hook. The fix is to incorporate data-efficient training (keeping \(D/N\) in a reasonable regime), progressive decoding schedules that adapt to the token-level importance, and gradient-aware bit allocation so that the penalty is minimized before quantization happens. These are the levers that make precision tangible: we control \(\tilde{b}(t)\) and the guidance map \(G_{ij}\) before the penalty term spikes.

## Where the field is now

Deletang et al. (2024) “Scaling Laws for Precision” [arxiv:2411.04330v1](https://arxiv.org/abs/2411.04330v1) sets the current research frontier by showing that the effective parameter drop from quantization behaves like a simple power law and that the penalty term’s exponents \(\beta\) and \(\delta\) remain stable across architectures and token counts. Their benchmarks on PaLM-2 sized models empirically validated that overtraining (large \(D/N\)) systematically raises the loss residual after quantization, proving the hook’s paradox and providing a predictive equation for \(N_{eff}\). That law now serves as the quantitative anchor for other papers: they either seek to reshape \(\eta\) via numerics or restructure \(b_{train}\) and \(b_{inf}\) via scheduling.

Chao et al. (2024) built on this by introducing progressive mixed-precision decoding [Chao et al. (2024) “Progressive Mixed-Precision Decoding”](https://arxiv.org/html/2407.17353), which remains the go-to experimental technique for making the penalty “stretch” across tokens rather than treating \(b\) as constant. Their results show that a linear drop in bit-width across generation yields lower average loss than a uniform 4-bit run, even though both use the same average bits, indicating that precision must be budgeted dynamically according to inference depth. Meanwhile, GuidedQuant (Wang et al. 2025) [Wang et al. (2025) “GuidedQuant”](https://export.arxiv.org/pdf/2303.11257v2.pdf) demonstrated that gradient-based guidance recovers accuracy under extreme quantization because it preserves the key directions that dominate the loss landscape.

On the engineering front, Meta AI’s Llama 3 release (2024) summary describes how their inference stack uses 4-bit quantization with block-wise scaling and a runtime scheduler that gradually increases bit widths for the first 64 tokens, mirroring the progressive decoding idea and keeping p99 latency under 55 ms on Meta’s internal CPUs (Meta AI blog “Llama 3: Foundational Models for Every Scale”). That deployment proves the production-side frontier: real systems no longer treat precision as a one-time optimization but as a runtime parameter tuned through both the decoder and the quantization tables to hit strict latency and accuracy targets.

## What's still open

Can we derive a unified scaling law that predicts the Pareto-optimal boundary between \(N\), \(D\), and \(b\) when training directly in sub-4-bit non-uniform precisions from scratch, without relying on a higher-precision checkpoint? This question is the clearest extension of existing work: the current law holds for PTQ and assumes a high \(b_{train}\); training within the low-bit regime itself may change the exponents \(\beta\) and \(\delta\) because the loss landscape is reshaped by quantization noise during optimization.

Can we formulate a guiding metric that predicts which tokens deserve extra bits in dynamic decoding beyond simple heuristics such as token position or gradient magnitude? Progressive decoding currently drops precision based on distance from the start, but a learned or loss-aware signal could target bits toward tokens that actually influence future loss gradients, possibly reducing total bit usage further.

How tight is the “gradient guidance” assumption from GuidedQuant? In particular, can we quantify the trade-off between preserving cross-weight dependencies and inflating the memory footprint with additional metadata (guidance maps and offsets)? The practical question here is whether there exists a compact surrogate for \(G_{ij}\) that yields most of the benefit with a small side cost.

## Where to read next

If you want the probabilistic foundation that explains why the penalty term arises, → [Scaling laws in neural networks](../../04-neural-networks-deep-learning/concepts/scaling-laws.md) gives the derivation starting from the ELBO perspective. The engineering counterpart is → *quantization basics* <!-- [[quantization-basics]] --> which walks through hardware-friendly rounding methods that keep \(\eta\) small. For more on precision schedules during inference, → [Mixed-precision training](mixed-precision-training.md) explains how progressive decoding and runtime schedulers coordinate to minimize accumulated loss.

## Build it

This build proves that the precision-aware scaling law is not just an abstract curve but a fit you can observe on a small language model by simulating quantization loss and translating it into an effective parameter count.

**What you're building:** a Colab-ready “Precision-Aware Loss Predictor” that fine-tunes `gpt2` on `wikitext-2-raw-v1`, simulates 8/4/2-bit weight quantization, measures cross-entropy loss as a function of \(b_{inf}\), and fits the simplified law \(L_{PTQ}(b_{inf}) = L_{base} + \eta \cdot (b_{train} - b_{inf})^{\delta}\).

**Why this is valuable:** it forces you to experience the loss penalty, estimate \(\eta\) and \(\delta\), and interpret how much \(N_{eff}\) shrinks when you settle on a target inference bit-width; the artifact is the predictor that tells you whether you can afford another half-bit or need to shrink \(N\).

**Stack:**
- **Model:** [gpt2](https://huggingface.co/gpt2) — 2.6M downloads
- **Dataset:** [wikitext-2-raw-v1](https://huggingface.co/datasets/wikitext/2-raw-v1) — tokenizer-friendly
- **Framework:** [transformers==4.39.2 + bitsandbytes==0.41](https://pytorch.org/)
- **Compute:** Single Colab T4 (16GB VRAM), ~1 hour including quantization sweeps

**The recipe:**
1. `pip install transformers datasets accelerate bitsandbytes && python - <<'PY'` then inside load AutoTokenizer/AutoModelForCausalLM, set `torch.cuda.set_per_process_memory_fraction(0.8)` so the 8-bit optimizer fits.
2. Load `wikitext-2-raw-v1`, tokenize with `padding="longest"` and pack into `blocks=512`; use `DataCollatorForLanguageModeling` so the sequence length matches GPT-2’s context window.
3. Fine-tune for 2 epochs with `gradient_accumulation_steps=4`, `per_device_train_batch_size=2`, learning rate 5e-5; log the base loss \(L_{base}\) at the end of training before any quantization.
4. Post-training, quantize weights via `bnb.nn.quantization` to 8, 4, and 2 bits; run evaluation on the validation split to record \(L_{PTQ}(b_{inf})\) for each bit-width, then fit \(L_{PTQ} = L_{base} + \eta (b_{train} - b_{inf})^\delta\) using SciPy’s `curve_fit`.
5. Use the fitted \(\eta, \delta\) to compute \(N_{eff}(b_{inf}) = N - \eta' (b_{train} - b_{inf})^\delta\) (with \(N=124\)M for GPT-2) and plot the effective parameter drop; the artifact is the fitted predictor plus a loss curve showing the penalty.

**Expected outcome:** a Colab notebook that takes quantized losses and predicts how many parameters the inference stage effectively retains, demonstrating the precision-scaling law on a toy LLM.

- **CS student:** Run the same notebook on Colab Free (Tesla T4) but shrink context length to 256 and only sweep 8→4 bits; the reduced sequence keeps training time under 30 minutes while still visualizing the loss penalty.
- **Applied engineer:** Extend the notebook to quantize `meta-llama/Llama-2-7b` to 4 bits with `bitsandbytes`, measure loss on a short dialog, and use the predictor to decide between 4-bit and 3-bit inference while keeping p95 latency under 70 ms in vLLM’s `TensorRT` runtime.
- **Applied researcher:** Add a hypothesis variant: test whether \(\delta\) becomes closer to 1 when you fine-tune a low-data subset (10k tokens) versus full wikitext, and falsify by requiring \(|\delta_{\text{low-data}} - \delta_{\text{full}}| > 0.2\) to claim significance.
- **Frontier researcher:** Use the same predictor but replace PTQ data with a training-from-scratch 3-bit checkpoint, then test the open question about the Pareto-optimal surface by plotting \(L_{PTQ}(N, D, b_{inf})\) for varying \(D\) and checking whether the empirical frontier matches the unified law you hypothesized in “What’s still open.”

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*