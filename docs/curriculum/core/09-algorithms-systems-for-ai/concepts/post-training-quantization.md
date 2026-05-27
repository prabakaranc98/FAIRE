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
updated: 2025-04-10
has_mvb: true
---

# Post-Training Quantization

Imagine a team that has finally trained a 70B-parameter reasoning model, only to discover that loading it requires some 140 GB of dense weights in float32—more memory than a single A100 and a price tag that pushes the deployment into a $100,000 cluster upgrade. The only other option is to shrink those 140 gigabytes without touching the training run, yet still expect the model to behave like the original, especially when it has to work as the core of a reasoning chain whose intermediate states must survive long search trees. That real-world hardware pinch is why post-training quantization (PTQ) exists: no more retraining, just numbing those 140 GB down to tens of gigabytes while keeping accuracy losses small. By the end of this page the reader will see PTQ as a constrained optimization problem, learn how calibration data plays the role of a surrogate loss, understand how second-order methods compensate for the greedy rounding of quantized weights, and build a working pipeline that pits naive round-to-nearest against calibration-guided scale search on a 1B-parameter Llama variant.

## The territory

Large language models are an edge-of-the-budget luxury for anyone who cannot field multi-GPU training clusters, yet their inference quality depends on the precision of the weight matrices stored across many gigabytes. The sharp trade-off is that each float32 parameter not only consumes 4× the storage of an int8 representation but also quadruples the working set that must fit inside accelerator caches and PCIe transfers. PTQ answers the question: what happens if we take a trained model and, without any additional gradient steps, compress its weights to low-bit representations while keeping the downstream loss almost unchanged? The solution sits at the intersection of quantization theory from signal-processing and LLM deployment engineering: we treat each weight matrix as a collection of scalar values that can be represented as integer code times a scale, but we also bake in the idea that some weights hurt task performance much more when perturbed than others. This approach borrows calibrations from activation-driven scaling, invariance principles from low-bit signal representations, and the careful layer-wise accounting popularized by GPTQ (Frantar et al. 2022) [arxiv:2210.17323](https://arxiv.org/abs/2210.17323). These ingredients keep PTQ far from the naive picture of "just round the weights and pray" and instead cast it as a surrogate minimization that calibrates weights against a representative data slice. The mechanism is best understood by starting from the quantization mapping itself and then seeing how optimization equipment—Hessian approximations, scale search, gradient guidance—keeps the reconstruction loss small.

## How it works

PTQ proceeds weight tensor by weight tensor. The simplest operation is to replace each full-precision scalar \(w_i\) in a layer with a quantized counterpart \(q_i = s \cdot \text{round}(w_i / s)\), where \(s\) is a positive scale factor shared across the tensor or a sub-block of it. This means every quantized value lies on a uniform grid of spacing \(s\). The obvious loss to minimize is the squared error between the quantized tensor \(Q(s)\) and the original \(W\), written as
\[
L(s) = \|W - Q(s)\|_F^2 = \sum_i (w_i - s \cdot \text{round}(w_i / s))^2,
\]
where the Frobenius norm sums over every scalar in the tensor. The scale \(s\) trades off quantization noise: smaller \(s\) allows more granular representation but increases the magnitude of integers and therefore their bit width, while larger \(s\) limits the integer range but raises rounding error. The ingenuous idea of PTQ is to tune \(s\) not just with this local metric but with an approximation of how the entire model’s outputs change.

Calibration data—a small dataset like 128 WikiText-2 samples—gives us that approximation. Each sample \(x\) produces activations \(a(x)\) at the quantized layer. When the weights are perturbed to \(Q(s)\), the downstream activation becomes \(a'(x; s)\). We compute a calibration loss
\[
L_{\text{cal}}(s) = \frac{1}{N}\sum_{x\in\mathcal{D}_{\text{cal}}} \|a(x) - a'(x; s)\|_2^2,
\]
where \(\mathcal{D}_{\text{cal}}\) is the calibration set and \(N\) is its size. The calibration loss tells us whether a scale \(s\) preserves the activations seen during inference more faithfully than raw quantization error. In practice, this involves running the network forward on the calibration samples after each candidate quantization, which is feasible because \(|\mathcal{D}_{\text{cal}}|\) is small. This is why modern PTQ pipelines alternate between tuning scales and evaluating the resulting loss over calibration data before accepting a quantization for a layer.

### Layer-wise optimization and Hessian compensation

Not all layers are equally sensitive. GPTQ (Frantar et al. 2022) [arxiv:2210.17323](https://arxiv.org/abs/2210.17323) introduced the key insight that the quantization error in one layer can be compensated by adjusting the remaining unquantized weights using a second-order approximation. Consider quantizing weights \(W\) in layer \(L\), while layers after \(L\) remain in float32. If \(f(W)\) is the network output (logits) as a function of the quantized layer and \(g(\Delta W)\) approximates the change when we replace \(W\) with \(W + \Delta W\), the local Taylor expansion is
\[
f(W+\Delta W) \approx f(W) + \nabla f(W)^\top \Delta W + \frac{1}{2} \Delta W^\top H \Delta W,
\]
where \(H = \nabla^2 f(W)\) is the Hessian with respect to the layer weights. PTQ keeps track of a diagonalized or block-wise approximation to \(H^{-1}\) so that when a weight gets quantized, the remaining weights in the same layer are updated by \(\Delta W = -H^{-1} (Q(W) - W)\). This update distributes the quantization noise in a way that minimizes the change in logits rather than the change in raw weights. Since computing the full Hessian is intractable, GPTQ approximates \(H^{-1}\) with a set of low-rank or diagonal blocks that are accumulated from the calibration data, which is enough to steer the rounding process.

The second-order remapping keeps track of the inverse Hessian incrementally: as each weight is quantized, the Hessian approximation is updated, so the next weight sees a different conditioning. This makes PTQ a greedy optimization that is still guided by curvature rather than being blind. The calibration data feeds into the Hessian by estimating the gradients; the key trade-off is compute vs. accuracy: a richer calibration set yields a more accurate Hessian at the cost of extra forward passes.

### From scalar rounding to multi-codebook quantization

When we push bit widths below 4, uniform scalar quantization faces severe signal distortion. AQLM (Egiazarian et al. 2024) [arxiv:2403.04198](https://arxiv.org/abs/2403.04198) shows how to stretch PTQ to the 2-bit regime by abandoning uniform scalar grids entirely. Instead, each weight vector is split into blocks, and each block is represented by a small dictionary (codebook) of learned vectors. The quantization now becomes
\[
q_i = C_{b(i)},\quad b(i) \in \{1,\dots,K\},
\]
where \(C_k\) is the \(k\)th codebook entry and \(b(i)\) picks the codebook index assigned to weight group \(i\). The codebooks are trained such that \(b(i)\) minimizes the reconstruction loss on calibration activations while also taking into account which blocks contribute most to the loss. AQLM solves this by alternating between updating codebooks and clusters via a vector quantization objective that mirrors k-means, but weighted by the gradients of the calibration loss. The outcome is a multi-codebook quantization that approximates the original weight vectors more faithfully than scalar rounding, while the training-free nature of PTQ remains because the codebooks are optimized through calibration alone.

### End-loss guidance and task-aware prioritization

GuidedQuant (GuidedQuant Research Team 2025) changed the story again by pointing out that calibration losses on activations are only proxies for the real task loss. Instead of looking at layer activations, GuidedQuant attaches a small probe head or task-specific loss evaluator after each quantized section and measures how weight quantization affects the final loss. The guidance term takes the form
\[
L_{\text{task}} = \frac{1}{N} \sum_{x\in\mathcal{D}_{\text{cal}}} \ell(f(Q(W); x), y_x),
\]
where \(\ell\) is the true task loss (e.g., cross-entropy on next-token prediction), \(f\) is the inference function, \(Q(W)\) is the quantized weight set being evaluated, and \(y_x\) is the true label. GuidedQuant then computes gradient signals from \(L_{\text{task}}\) with respect to each weight to determine which dimensions should be quantized later in the greedy pass. We keep the greedy, layer-wise structure, but add an ordering heuristic: weights that lead to a large increase in \(L_{\text{task}}\) when quantized are postponed until after their neighbors have been adjusted through Hessian compensation.

This choice recognizes that in large reasoning models the reasoning chain is fragile; an early quantization mistake can amplify through subsequent attention spans. Task-loss gradients implicitly encode the multi-step computation graph, which is why GuidedQuant outperforms purely activation-based PTQ on multi-stage tasks like chain-of-thought.

## Where the field is now

The 2024 survey "A Decade of Deep Learning: A Survey on The Magnificent Seven" (Li et al. 2024) [arxiv:2412.16188](https://arxiv.org/html/2412.16188) explicitly named PTQ among the pillars that enable deep learning’s industrial scaling—without it, operators still wrestle with float32 modellings that cannot leave the datacenter. The space has since bifurcated into two frontiers. On the research side, calibration-sensitive second-order methods like GPTQ and multi-codebook schemes like AQLM remain the baseline, but work now trains the models to tolerate PTQ by altering their loss landscapes during training, so that the post-training pass sees flatter minima. This is the setting highlighted by the benchmarks leveraging DeepResearch-9K (Zhou et al. 2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152), a challenging dataset tailored to deep-research agents performing long-horizon reasoning. When quantized models are evaluated on DeepResearch-9K, the gap between 8-bit PTQ and float32 becomes visible, casting the calibration procedure as central, not optional. The same paper shows that HPC-style calibration schedules that see entire reasoning episodes in a single batch give reliable guidance for the Hessian approximation.

Engineering teams, on the other hand, are following the design principles outlined by "GenAI for Systems: Recurring Challenges and Design Principles from Software to Systems" (Garcia et al. 2026) [arxiv:2602.15241v1](https://arxiv.org/html/2602.15241v1). The survey enumerates three recurring pains: memory management, latency spikes due to quantized kernels, and robustness to prompt distribution shifts. Expert teams counter these by pairing PTQ with kernel fusion, low-latency caching, and aggressive layer pruning inside the quantized stack. NVIDIA’s TensorRT 10 deployment story (developer.nvidia.com/blog/tensorrt-10) exemplifies this synthesis: it ships a quantized inference path that combines PTQ quantization with optimized kernels, reducing the VRAM footprint of 70B decoders to under 70 GB while maintaining sub-100 ms p95 latency on H100 at batch size 1. The blog explicitly credits calibration-aware scale tuning as the technique that unlocks 4-bit inference at acceptable accuracy.

A third axis is the reinforcement learning layer that orchestrates LRMs, as surveyed in "Reinforcement Learning Foundations for Deep Research Systems: A Survey" (Karthik et al. 2025) [https://export.arxiv.org/pdf/2509.06733](https://export.arxiv.org/pdf/2509.06733). Here the PTQ problem is not only weight compression but also the preservation of online adaptation loops; when policy networks are quantized, the downstream reward model must still operate on precise activations so that RL feedback signals remain stable. The survey highlights how quantized models can be inserted into RL pipelines via retrofitted critic networks that monitor quantization drift, and how the actor-critic duo can be retrained with frozen quantized backbones.

Taken together, these directions paint the current frontier: PTQ is no longer a standalone knob but part of a broader system that includes dataset-aware calibration, hardware-aware kernel fusion, and RL-based feedback loops that guard reasoning chains.

## What's still open

1. Can post-training quantization co-design compensate for the multi-step reasoning paths in large reasoning models without the expensive calibration datasets currently required by Hessian-based methods? Specifically, can a small set of tree-structured reasoning traces yield the same guidance as the tens of thousands of WikiText samples used today?

2. Is it possible to quantify and preserve the internal search-tree generation capacities of LRMs after PTQ? The open question is whether PTQ can be made sensitive to the branching structures generated during inference so that quantizing does not collapse rare but critical reasoning branches.

3. How do RL-based feedback signals interact with quantization noise? Can we design a feedback loop that detects quantization-induced reward drifts before they corrupt policy updates, and if so, what metric should govern the detection?

4. How much extra compute can calibration-aware PTQ afford on the inference-critical path before latency budgets are breached? The systems frontier needs a clear accounting of PTQ overhead vs. latency gain when fused with kernels like those in TensorRT 10.

## Where to read next

If you want the mathematical grounding in turning quantization into an optimization, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) walks through the Gaussian likelihood and how its gradients translate into PTQ-style surrogate losses. The engineering counterpart is → [Flash Attention](flash-attention.md), which explains the kernel-level fusion that makes those quantized tensors run at low latency in practical deployments. For broader context on model deployment and hardware-aware compression, → [[llm-hardware-deployment]] shows the end-to-end stack from quantized checkpoint to inference endpoint.

## Build it

The build proves that PTQ is an optimization process: you will compare naive round-to-nearest quantization with a calibration-guided scale search that mimics AWQ-style procedures, and you will see how the calibration loss can rescue accuracy.  
**What you're building:** a PTQ pipeline in PyTorch that quantizes `meta-llama/Llama-3-2-1B` weights, comparing raw RTN quantization against a calibrated scale search using a 128-sample slice of `wikitext-2-v1`.  
**Why this is valuable:** the build turns the conceptual calibration loss into code and lets you measure whether Hessian-like compensation and task-aware scale tuning recover perplexity close to the float32 model with only a few forward passes.  
**Stack:**
- **Model:** [meta-llama/Llama-3-2-1B](https://huggingface.co/meta-llama/Llama-3-2-1B) — 10k+ downloads, open weights
- **Dataset:** [wikitext-2-v1](https://huggingface.co/datasets/wikitext/wiki40b) (limit 128 samples) — documented small corpus
- **Framework:** PyTorch 2.2 + `bitsandbytes` 0.41 + `transformers` 4.44
- **Compute:** Google Colab T4 (16 GB VRAM) / ~1.5 hours

**The recipe:**
1. Install `pip install torch==2.2.0 transformers==4.44 bitsandbytes==0.41 avalanche-dl` and download the model weights via `AutoModelForCausalLM.from_pretrained`. Keep the original float32 copy for baseline inference.
2. Load 128 samples from `wikitext-2-v1` and batch them to produce calibration forward passes. Record activations for the target layer (e.g., the last feed-forward matrix) before quantization to reuse as targets.
3. For RTN quantization, compute \(s = \max(|W|)/2^{b-1}\) for \(b=8\) and set \(Q(W) = s \cdot \text{round}(W/s)\). For calibration-guided scale search, sweep scales around \(s\) and evaluate \(L_{\text{cal}}(s)\) by running the calibration samples through the partially quantized model, then pick the scale with the lowest loss.
4. For both methods, perform a single downstream evaluation pass to compute perplexity on the 128-sample slice; record both losses plus the change relative to the float32 perplexity (~5.5). Expect the calibrated variant to be within ~0.1 perplexity of full precision, while RTN should degrade more significantly.
5. After evaluation, you now have two quantized checkpoints and a measurement script that compares their perplexities. The artifact is a pair of quantized weights plus the recorded calibration loss curve demonstrating the value of scale search.

**Expected outcome:** a PTQ pipeline that ships both `rt~n` and `calibrated` quantized checkpoints along with a plot of calibration loss vs. scale and their respective perplexities.

- **CS student:** Run the calibration steps on an RTX 4070 with the subset of `wikitext-2-v1` limited to 64 samples to cut runtime while still comparing RTN vs. scale search; the evaluation metric remains perplexity.
- **Applied engineer:** Extend the pipeline to export the calibrated checkpoint to vLLM via `vllm.export_to_json` and measure p50 inference latency on an A10, aiming for <120 ms per token with the quantized weights serving through FastAPI.
- **Applied researcher:** Use the calibration loop to test the hypothesis that cosine-scaled schedules (scale search across logarithmic steps) surpass linear scaling for the last transformer block; report the perplexity gap and the calibration loss trajectories.
- **Frontier researcher:** Probe the open question about reasoning chains by using DeepResearch-9K episodes as calibration samples and measuring whether the quantized model maintains the token-level decision tree branching (quantify branch divergence before and after quantization); the falsifier is if divergence increases by more than 5% on critical decision points.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*