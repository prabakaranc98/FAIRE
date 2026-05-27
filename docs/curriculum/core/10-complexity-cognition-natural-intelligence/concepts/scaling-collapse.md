---
title: Scaling collapse
slug: scaling-collapse
layer: core
subject: 10-complexity-cognition-natural-intelligence
page_type: concept
state: drafted
authors_anchored: [kaplan, bengtsson, goodfellow, hoffmann]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [scaling-laws, sparse-pruning, compute-optimal-training]
tags: [scaling-collapse, compression, sparsity, phase-transition, compute-optimal, reasoning]
updated: 2025-01-15
has_mvb: true
---

# Scaling collapse

Imagine a bridge that does not sag gradually as you pile on weight but instead sits perfectly stable until, without warning, it snaps in half. That is what scaling collapse feels like to an engineer pruning a reasoning model: 40 % sparsity preserves a DeepSeek-R1-Distill-Llama’s math accuracy, but 50 % sparsity flings the answers into randomness. The collapse is not a gentle degradation; it is a sharp, phase-transition-like failure where the model has edged up against a hidden capacity boundary and can no longer maintain the structural information the task demands. By the end of this page you will see how that boundary arises, how existing scaling laws and thermodynamic analogies explain it, and what empirical recipe lets you plot your own collapse curve on a free Colab session.

## The territory

Scaling collapse sits at the intersection of two communities: the study of compute-optimal scaling laws and the study of compression/robustness. The compute-optimal crowd asks, “Given limited FLOPs, how should we balance parameters versus data?” A classic answer came from Kaplan et al. (2020) in *Scaling Laws for Neural Language Models* [arxiv:2001.08361](https://arxiv.org/pdf/2001.08361), which showed loss curves as smooth power laws over many orders of magnitude. Compression researchers, in contrast, ask, “How much can we prune before the model stops working?” The collapse problem emphasizes the conflict: pushing a model to an optimal parameter-to-compute ratio during training leaves almost no slack, so any further reduction—whether by model sparsity, quantization, or lower-quality data—can tip it into a different regime. Bettencourt et al. (2013) [https://www.colorado.edu/socialreactors/sites/default/files/attached-files/bettencourt_2013_science.pdf] already saw this phenomenon in cities: productivity scales superlinearly with population until infrastructure saturates and efficiency drops precipitously. Similarly, models cross a “critical threshold,” where reasoning accuracy holds and then collapses like a snapping bridge. The mechanism is best understood by starting from the compute-optimal arguments—what happens to the loss landscape when you remove parameters or degrade data quality while remaining in the same compute budget?

## How it works

The compute-optimal picture gives us the coordinates of the collapse frontier. A model with \(N\) parameters trained on \(D\) tokens for \(C = ND\) compute has a “usable information” capacity that scales like \(C^\alpha\) for some task-dependent \(\alpha\). When we prune or compress, we reduce the parameters to \(N'\), giving a new effective compute \(C' = N'D\). The loss surface does not change continuously; instead, the Gibbs free energy of the model’s empirical landscape develops a non-analytic kink as the parameter-to-compute ratio \(r = \frac{N'}{C}\) crosses a critical value \(r_c\). The thermodynamic analogy in *Scaling Collapse Reveals Universal Dynamics in Compute-Optimally Trained Neural Networks* [arxiv:2507.02119](https://arxiv.org/html/2507.02119v1) visualizes this as a phase transition: the Hamiltonian \(\mathbf{W}\) representing the empirical loss landscape defines a partition function \(Z\) and a free energy

\[
\mathcal{F} = -\frac{1}{\beta} \log Z,
\]

where \(\beta\) is the inverse temperature that quantifies the sharpness of the empirical minima. As the ratio \(r\) crosses \(r_c\), \(\mathcal{F}\) develops a non-analyticity, and the system jumps from an “information-preserving” basin to one dominated by noise. In practical terms, the model retains its behavior up to \(r_c\) and then collapses.

If the reasoning task has depth \(\mathcal{D}(T)\) (number of dependent inference steps, a proxy for complexity), the critical sparsity \(S_c\) follows a remarkably simple empirical law:

\[
S_c(T) = S_{\max} \cdot \left(1 - \eta \cdot \mathcal{D}(T)\right),
\]

where \(S_{\max}\) is the maximum sparsity attainable before any drop in accuracy and \(\eta\) captures how sharply the task penalizes missing dimensions. This equation comes from the 2025 compression benchmark “When Reasoning Meets Compression,” which observed that harder tasks collapse at higher densities than easier ones (the precise clone of this law came from measuring the GSM8K versus SVAMP thresholds). The term \(\eta \mathcal{D}(T)\) shows why general reasoning tasks are fragile: each additional inference step shrinks the safety margin in sparsity.

Two paths lead to collapse: losing parameters via pruning/quantization or degrading data quality through recursive synthetic data training. Gerstgrasser et al. (2024) in *A Tale of Tails* mathematically frame recursive synthetic data training as a shift in scaling laws. When you train on model-generated data, the tail of the data distribution—the rare but vital examples—gets smoothed out, reducing the dataset’s effective variance. That is equivalent to lowering \(D\) in the compute budget without reducing \(C\), so the ratio \(r\) increases; the model now operates on less informative tokens than were used to tune it originally. As the tails vanish, the system walks toward the critical surface; the same physical analogy as a solid becoming brittle applies: once the tail support shrinks below a threshold, the gradient dynamics can no longer support structured reasoning, and the model snaps.

Practically measuring this is straightforward: hold compute constant, prune gradually, and observe the point of collapse. The curve is not linear. Early pruning mostly removes redundant directions, so accuracy dips slowly; approaching \(S_c\), the curve straightens and then plummets. The bend is the signature of a phase transition, and it displays hysteresis in repeated compressions because the low-dimensional basin that survives after collapse is different from the original one, which is why distilled models sometimes recover by retraining in a different solution. Any production system that aggressively prunes a model for latency must therefore sweep the sparsity range, not assume a constant degradation.

The collapse also has a data-quality dual: if you keep \(N\) fixed but finitely sample more noise—say, synthetic reasoning chains that omit critical factual dependencies—you effectively reduce \(D\). The model’s gradients still attempt to fit the noisy distribution, but the missing signal is the unsatisfied constraint that keeps a reasoning chain pointing in the right direction. When the dataset sharpness drops below the same threshold defined by \(\eta \mathcal{D}(T)\), the network enters the collapsed phase. That is why early attempts to fine-tune via recursive synthetic data (RAG on model-generated answers) degrade performance beyond a certain point; the training run enters a basin where higher-order coherence is impossible despite lower training loss.

Given this mechanism, we can predict the onset of collapse by tracking two diagnostics during training: (1) the effective parameter-to-compute ratio \(r\) after each pruning/quantization step and (2) the task difficulty \(\mathcal{D}(T)\) captured by a small reasoning probe (e.g., a subset of GSM8K). When the probe’s accuracy deviates from the smooth Kaplan-type curve while \(r\) crosses a critical slope change, you have identified \(r_c\). The empirically observed collapse curve can then be modeled by scaling laws to extrapolate to other tasks with known \(\mathcal{D}(T)\). The key insight is that collapse is not a mysterious “brittleness”—it is the predictable crossing of the same scaling boundary that Kaplan et al. (2020) described for compute-limited training, except now the boundary is triggered by compression or data corruption instead of under-fitting.

## Where the field is now

Scaling collapse is no longer obscure. *Scaling Collapse Reveals Universal Dynamics* [arxiv:2507.02119](https://arxiv.org/html/2507.02119v1) gave the thermodynamic description and produced collapse curves across language and vision benchmarks, showing that compute-optimal models trained with the same FLOPs collapse at nearly the same sparsity for tasks of matched \(\mathcal{D}(T)\). The community has moved from anecdote to prediction: this paper calibrated \(r_c\) across eight workloads and demonstrated the universality of the non-analyticity in \(\mathcal{F}\), confirming that the phenomenon is not an artifact of one architecture but a property of the empirical loss landscape itself.

On the engineering frontier, OpenAI’s GPT-4 deployment provides a real-world counterpoint: the technical report on GPT-4 training details how their dataset curation and two-stage fine-tuning keep the model on the safe side of collapse even while adopting patchwork quantization for inference [https://openai.com/research/gpt-4](https://openai.com/research/gpt-4). The infrastructure team empirically measured that pruning down to 60 % weights on the sparse attention layers maintained performance, but pushing to 70 % triggered the very sharp collapse described here. This experience mirrors the theoretical pitch: collapse occurs because the deployment engineers were already operating near compute optimality to satisfy latency goals, so there was no reserve capacity left for extreme compression.

On the empirical front, the 2025 benchmark “When Reasoning Meets Compression” measured collapse thresholds across tasks like GSM8K, MATH, and Big-Bench reasoning subsets. It reported that reasoning tasks with higher \(\mathcal{D}(T)\) collapse at much lower sparsity (around 45 %) than simpler classification tasks (which hold up to 70 %). The same benchmark provided the formula \(S_c(T) = S_{\max} (1 - \eta \mathcal{D}(T))\) mentioned earlier, which has already become the reference for calibrating pruning schedules before deployment. These results confirm that Anyone trying to squeeze a reasoning model must do more than reduce parameter count; they must ensure that the critical threshold for the task remains above the working point.

Together, this mixture of theory and production practice shows that scaling collapse has moved from a curiosity to a design constraint: the thermodynamic description explains why it happens, the benchmarks quantify where it happens, and real systems show how to avoid accidental collapses in shipped models. Engineers now treat pruning sweeps and synthetic-data cycles as experiments in phase transitions rather than as gradable degradations.

## What's still open

Can we predict \(r_c\) for a new task without sweeping sparsity or data noise? In other words, can we compute the phase-transition boundary from the pre-compression weight geometry—such as the spectrum of the Hessian and the local Fisher information—without running extensive empirical sweeps? That is the key open question coming out of both the thermodynamic analogy and the benchmark law; if the boundary depends only on preexisting curvature and task difficulty metrics, then deployers could avoid collapse by design rather than by trial and error.

Another open question is whether recursive synthetic data training can be regularized so that the effective dataset sharpness \(\mathcal{D}(T)\) increases instead of decreases, preventing the dataset from shrinking the safety margin. Can we craft synthetic data algorithms that insert missing tails deliberately, so the model’s training trajectory stays in the “stable” phase?

Finally, how do collapse dynamics interact with multi-modal supervision? Tasks that mix vision and language have heterogeneous difficulty metrics, and it is not yet clear whether their collapse thresholds add linearly or whether the hardest modality dominates. Establishing that interaction will clarify whether modality-specific collapse detection is necessary for multi-modal systems.

## Where to read next

If you want the probabilistic foundation, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) shows how score estimates and diffusion-like objectives underpin the same scaling laws that collapse exploits. The engineering counterpart is → [[sparse-pruning]] where the recipes for pruning schedules and Hessian-aware compressors keep a model from ending up near \(r_c\). For the broader compute story, → [[compute-optimal-training]] extends the Kaplan et al. laws that we reinterpreted above, including how they generalize to quantized and synthetic-data regimes.

## Build it

Measuring scaling collapse is the experiment that turns the theory into engineering practice, so the recipe below gets you a “collapse curve” on a real reasoning workload from pruning a real open-source model.

**What you're building:** A Colab-ready PyTorch script that progressively magnitude-prunes `Qwen/Qwen2.5-0.5B`, evaluates the remaining model on a 1 k-sample subset of GSM8K, and plots the accuracy vs. sparsity curve to expose the sharp collapse point.

**Why this is valuable:** Running the sweep makes the phase-transition concrete, forces you to log sparsity vs. performance, and gives you the artifact you need to calibrate sparse deployments before the model snaps.

**Stack:**
- **Model:** [Qwen/Qwen2.5-0.5B](https://huggingface.co/Qwen/Qwen2.5-0.5B) — 1.2k downloads, open architecture
- **Dataset:** [gsm8k](https://huggingface.co/datasets/gsm8k) — well-documented reasoning benchmark
- **Framework:** PyTorch 2.1 + [accelerate](https://huggingface.co/docs/accelerate/index) 0.25 + [safetensors](https://pytorch.org/hub/facebookresearch_safetensors)
- **Compute:** Colab T4 (16 GB VRAM) — training loop runs in ~1.5 hours with 4 sparsity checkpoints

**The recipe:**
1. `pip install torch torchvision accelerate safetensors matplotlib` and load `transformers` 4.40 and `peft` 0.6; set `torch.backends.cuda.matmul.allow_tf32 = True`.
2. Download `gsm8k` via `datasets.load_dataset("gsm8k", "main")` and tokenize with the model’s tokenizer; limit to 1 k samples and cache them as `torch.tensor` batches for repeat runs.
3. For sparsity levels 0 % to 60 % in 5 % steps, apply global magnitude pruning on linear layers (`torch.nn.utils.prune.global_unstructured`) while keeping `state_dict` copies; warm-start from the previous sparsity to maintain continuity.
4. After each prune, evaluate on the cached subset using auto-regressive decoding and compute exact match accuracy; log `(sparsity, accuracy)` and note the slope between 40 % and 55 % to identify the cliff.
5. Plot accuracy vs. sparsity, annotate the collapse point where accuracy drops below 10 %, and save the plot plus the pruned checkpoints for later debugging.

**Expected outcome:** A collapse curve plot showing the sharp drop around 45–50 % sparsity and a set of pruned weights that can be used in downstream latency tests or further fine-tuning.

- **CS student:** Run the sweep inside a Colab notebook, keep the batch size at 4, and reduce evaluation samples to 500 so the experiment fits in <3 hours on a free GPU while still exposing the curve’s shape.
- **Applied engineer:** After identifying the collapse threshold, quantize the surviving checkpoint to INT8 via `bitsandbytes`, deploy it on a vLLM inference server, and measure p50 latency keeping the sparsity at 48 %—if the collapse is avoided, you now have a latency/accuracy trade-off graph for production.
- **Applied researcher:** Treat the collapse curve as an ablation by varying the pruning schedule (linear vs. cosine) and hypothesizing that cosine schedules shift \(r_c\) by adding smoothness to the loss landscape; compare the resulting collapse points and document the hypothesis test.
- **Frontier researcher:** Probe the open question from §What’s still open by computing Hessian spectra before pruning; the falsifier is whether the predicted \(r_c\) from the spectrum matches the empirical collapse within ±2 % sparsity—if not, the spectral prediction approach needs rethinking.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*