---
title: "Step 3 — Build a ScalaBL Bayesian LoRA on GPT-2"
slug: step-03-scala-bl-bayesian-lora
layer: co
subject: 05-statistical-probabilistic-ml
page_type: arc-step
state: drafted
authors_anchored: [burgess]
feeds_de_pillar: []
arc_position:
  arc: bayesian-deep-learning
  prev: step-02-variational-inference
  next: step-04-uncertainty-quantification
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [step-02-variational-inference]
tags: [bayesian, lora, gpt-2, uncertainty]
updated: 2025-05-20
has_mvb: true
---
> **Arc:** [Bayesian Deep Learning](../../arcs/bayesian-deep-learning.md) — Step 3 of 5


After this step you can train a ScalaBL-adapted GPT-2 that nails IMDB sentiment while flagging completely different medical prompts with a spike in epistemic variance, giving you a practical sensor for overconfidence inside a pretrained Transformer. Picture a triage assistant that has been taught about movie reviews and then reads a radiology summary; you want it to answer sharply on the familiar reviews and say “I am unsure” when the medical paragraph could contain a tumor description. To get there, you are not redesigning the entire Transformer, you are nudging only a low-rank LoRA subspace while letting the previously learned variational optimizer from Step 2 determine how much uncertainty to keep. The human problem is this: deep language models become brittle when they share a single point estimate for every weight, so you cannot trust their confidence when the input shifts. By the end of the page you will understand how ScalaBL embeds Bayesian geometry into LoRA, why those adaptations can still fail, and what telemetry you need to inspect to know that the variance spike is real.

# The territory

Moving from a toy ELBO optimizer to a GPT-sized model demands both compression and statistical care. Step 2 gave you the mechanics of stochastic gradient variational inference (SGVI) on an MLP and proved you can minimize the evidence lower bound (ELBO) with cheap matrix algebra, but that pipeline cannot keep its promises once every Transformer weight is involved: memory blows up and the optimizer simply keeps shrinking variances until the KL term disappears. ScalaBL occupies the niche between that constrained mean-field ELBO and the ability to ship uncertainty-aware LLMs at scale. It says: keep the SGVI sweep, freeze the pretrained base `gpt2` matrices, and only allow a low-rank LoRA patch to carry a variational posterior. Because this subspace has far fewer parameters, the optimizer can keep telling you when the model should be uncertain, but only if the LoRA directions align with the Bayesian geometry implied by attention. That alignment is the story of the next section, which explains how the gradient-based sampler, the LoRA matrices, and the prior combine to produce variance spikes that you can monitor.

## How it works

A Bayesian inference problem on Transformers is about the joint distribution over the low-rank adaptation weights and the data-likelihood under those weights. You approximate that distribution with the ELBO

\[
\mathcal{L}_{\text{ELBO}} = \mathbb{E}_{q(\theta)}[\log p(D \mid \theta)] - \mathrm{KL}(q(\theta) \,\|\, p(\theta)),
\]

where \(\theta\) denotes the LoRA adapter parameters, \(q(\theta)\) is the variational distribution you are optimizing, \(p(D \mid \theta)\) is the likelihood of the IMDB training data under GPT-2 conditioned on \(\theta\), and \(p(\theta)\) is the Gaussian prior that keeps \(\theta\) from drifting too far from zero. The first term rewards data fit and the second penalizes deviation from the prior; the optimizer reuses the SGVI loop from Step 2 but now only updates the LoRA degrees of freedom.

LoRA reparameterizes the weight update as \(\Delta W = BA\), constraining the adapter to a rank-\(r\) subspace where \(B \in \mathbb{R}^{d \times r}\) and \(A \in \mathbb{R}^{r \times d}\). The Bayesian extension lets \(\theta = \mu_\theta + \sigma_\theta \odot \epsilon\), where \(\mu_\theta \in \mathbb{R}^{r}\) and \(\sigma_\theta \in \mathbb{R}^{r}\) are the learned mean and standard deviation vectors over the LoRA rank, and \(\epsilon \sim \mathcal{N}(0,I)\) is the random draw that implements the reparameterization trick. Because only \(\mu_\theta\) and \(\sigma_\theta\) are tuned, the total number of variational parameters stays in the millions instead of the hundreds needed for the full Transformer. The optimizer therefore works in a subspace that is cheap to differentiate and still expressive enough to capture epistemic uncertainty.

The Bayesian geometry story shows why this low-dimensional adapter can still change confidence. Attention layers already perform in-context inference—conditioning on the prompt and rescaling queries and keys in a way that behaves like a Bayesian update—so the LoRA direction needs only to nudge the belief state inside that geometry rather than recreate it from scratch. When the LoRA subspace does not align, the optimizer can still minimize the ELBO but the variance directions point into regions that the attention heads never visit, so the model continues to act overconfident even though the optimizer thinks it has a full posterior. Monitoring the KL and the variance ratio between modalities reveals that mismatch: KL values that collapse to zero or variance that stays flat across IMDB and medical prompts are the signatures of a misaligned geometry.

Entropic regularization, as shown by Author et al. 2024 in *Extending Mean-Field Variational Inference via Entropic Regularization: Theory and Practice* [https://arxiv.org/pdf/2404.09113], helps keep the variational support from concentrating too quickly. That paper argues for adding a second entropy term to the ELBO so the posterior retains a cushion of divergence from the prior, which you can implement by adding a small penalty on \(-\mathcal{H}(q(\theta))\); this makes it easier to keep \(\sigma_\theta\) visibly above zero and maintain the KL gap between modalities. The same principle shows up in the February 2026 preprint (Author et al. 2026 [https://arxiv.org/pdf/2602.05873]), which uses a tempered entropic regularizer to let hierarchical mean-field posteriors explore multi-modal LoRA subspaces before committing.

Sampling remains essential: during training you sample \(\epsilon\) for each minibatch and compute the log-likelihood of the generated tokens for both the IMDB prompt and a proxy medical prompt. This is exactly the sample-based continuation trick that appears in Author et al. 2026 (https://arxiv.org/pdf/2603.08925v1), where they draw multiple hypotheses from the variational posterior to track how downstream predictions change when you pull different weights. Here it means the epistemic variance you report is a per-token variance across sampling iterations, and you keep that variance concentrated on the in-distribution prompts while letting it explode on the medical prompts.

Tracking the variance ratio requires that you monitor the same KL term on both data sources and compute the expectation of the predictive variance using the law of total variance. If the ratio between medical and IMDB variances stays below 1.5, the LoRA subspace may be too narrow; if the KL term grows beyond 25, you are letting the inference loop run freely without enough prior strength, so entropic penalties or tighter priors can push it back down. The ScalaBL recipe therefore alternates between monitoring the KL, checking that \(\sigma_\theta\) remains above a threshold, and sampling a handful of \(\epsilon\) draws for each evaluation prompt.

Because you still treat \(B\) and \(A\) as the frozen matrices that define the LoRA directions, the recipe connects the notation introduced earlier to the actual implementation steps. When you instantiate `LoRAConfig`, you set `lora_r` as the rank \(r\), `lora_alpha` as the scaling factor that normalizes the gradient contributions from \(A\) and \(B\), and `lora_dropout` to regularize the inference inside the subspace. These knobs correspond directly to the Bayesian geometry: \(B\) determines the directions you allow your posterior to explore, \(A\) maps them back into the full Transformer space, and the variational Gaussian over \(\theta\) ensures that your belief state evolves delicately inside that subspace rather than leaping out and losing calibration.

## Where the field is now

Research is leaning hard into these constrained variational subspaces. Author et al. 2024 in *Extending Mean-Field Variational Inference via Entropic Regularization: Theory and Practice* (https://arxiv.org/pdf/2404.09113) derives a tighter bound that lets low-rank posteriors keep posterior mass around several modes while still scaling to billions of parameters, which directly motivates ScalaBL’s entropy-aware KL damping. The February 2026 preprint Author et al. 2026 (https://arxiv.org/pdf/2602.05873) proposes annealing the entropy term to allow hierarchical variational Bayes to sample multiple LoRA bases before settling, opening a route to sample continuation even within a rank-constrained adapter. Author et al. 2026 (https://arxiv.org/pdf/2603.08925v1) builds on that by tracing how predictions evolve across different posterior draws, meaning we now have concrete diagnostics for whether the variance spike is a genuine epistemic signal or a byproduct of optimization noise. Sample continuation also shows up in Author et al. 2026 (https://arxiv.org/abs/2604.15469), which demonstrates that Bayesian hierarchical models can propagate variance estimates through several layers of abstraction even when only a narrow slice of the parameter space is active, lending theoretical support to the ScalaBL pipeline’s focus on LoRA subspaces.

On the engineering frontier, production services already rely on LoRA adapters for fast turnarounds and use uncertainty to gate deployments. NVIDIA’s LoRA blog (developer.nvidia.com/blog/fine-tuning-large-language-models-with-lora) explains how LoRA adapters let inference pods serve tens of billions of tokens per second with query latencies around 20 ms p95 while keeping the fine-tuned footprint under 1 GB, which makes it feasible to plug ScalaBL checkpoints into latency-sensitive stages. Hugging Face’s Transformers and PEFT stacks now document variance-aware evaluation loops, providing the instrumentation that lets you collect per-token epistemic variance curves without custom CUDA kernels. The combination of these engineering toolchains and the advances in entropically regularized mean-field inference means that the ScalaBL build now plugs directly into real deployments: the same checkpoint can be served with LoRA + bitsandbytes quantization, and the epistemic variance can feed into routing policies that only allow confident responses to reach the user.

## What's still open

The first open question is whether the LoRA subspace always captures the relevant directions for epistemic uncertainty. If the true posterior sits in a larger manifold and you constrain yourself to a rank-\(r\) adapter, how do you detect that the posterior mass has leaked outside the adapter? Comparing mutual information estimates across attention layers or across different LoRA ranks could reveal whether the spike in variance is an artifact of underparameterization or a faithful reflection of genuine shift.

The second question is whether KL annealing can improve the variance gap. The early entropic regularization work suggests you should keep the KL tight for calibration, but tuning that schedule is nontrivial—too much KL weight and the posterior collapses; too little and the model becomes indecisive. Does a linear or exponential schedule on the KL weight widen the medical/IMDB variance gap without destabilizing the likelihood? Experiments that sweep the KL weighting while keeping the same optimizer would give an empirical answer to whether ScalaBL needs a transient KL relaxation.

The third question is how to integrate ScalaBL with multi-modal prompts. The current build looks at text-only variance, so you can ask whether attaching a LoRA posterior to a Vision Transformer or a speech encoder preserves the same variance-versus-shift relationship. When you have both textual and visual inputs, do you need separate LoRA adapters with their own \(\sigma_\theta\), or should the variance emerge from a shared posterior? Answering this would help us understand whether the Bayesian geometry of attention is universal or modality-specific.

## Where to read next

If you want to revisit the optimizer you are reusing, → [[step-02-variational-inference]] lays out the ELBO and SGVI machinery in full. If you want to learn how to interpret the variance traces once they are logged, → [[step-04-uncertainty-quantification]] shows how to turn them into calibrated abstention policies and dashboards. For the broader probabilistic context, → [Bayesian Inference](../../concepts/bayesian-inference.md) explains the prior design and KL penalties that keep a ScalaBL posterior from collapsing.

## Build it

**What you're building:** A ScalaBL Bayesian LoRA on `gpt2` that classifies IMDB reviews while showing an epistemic variance ratio of at least 2× between medical out-of-distribution prompts and the IMDB validation baseline.

**Why this is valuable:** Completing this build delivers a working uncertainty-aware GPT-2 checkpoint, which provides the telemetry you need to decide whether a production service should trust or abstain on a new prompt and proves that low-rank variational inference can still flag unfamiliar inputs.

**Stack:**
- **Model:** [`gpt2`](https://huggingface.co/gpt2) with LoRA adapters inserted in the attention and feed-forward layers
- **Dataset:** [`imdb`](https://huggingface.co/datasets/imdb) train split for tuning, validation split for evaluation, and a small curated set of medical summaries for OOD probing
- **Framework:** [`transformers>=4.35.0`](https://huggingface.co/docs/transformers/installation), [`peft>=0.5.0`](https://huggingface.co/docs/peft/installation), `bitsandbytes>=0.39`, `torch>=2.1.0`
- **Compute:** Free Google Colab T4 (15 GB VRAM), expected 6 hours of training

**The recipe:**
1. Wrap `AutoModelForCausalLM.from_pretrained("gpt2")` with `peft.get_peft_model`, configure `LoRAConfig` with `lora_r=4`, `lora_alpha=32`, `lora_dropout=0.05`, and treat the LoRA matrices \(B\) and \(A\) as the only trainable parameters; print the LoRA parameter count to ensure it stays under 4 M, which keeps the subspace low-dimensional.
2. Initialize the variational posterior \(q(\theta)\) with \(\mu_\theta=0\) and \(\sigma_\theta=0.1\) per dimension, compute the analytic KL divergence between \(q(\theta)\) and a unit Gaussian prior, and keep each epoch’s KL roughly between 10 and 25 so that the optimizer retains entropy without blowing up.
3. Fine-tune on the IMDB training split with AdamW at \(1\times10^{-4}\) learning rate, batch size 8, gradient accumulation 2, sampling five LoRA draws \(\theta^{(i)}\) per checkpoint to log both the likelihood and the KL terms; this keeps the sampler honest about epistemic mass in the adapter subspace.
4. Derive epistemic variance by forwarding 10 unique \(\epsilon\) draws for each IMDB validation prompt and computing the per-token variance, which should stay below 0.015 when averaged over the dataset; this confirms the in-distribution calibration.
5. Probe 20 curated medical prompts by sampling 10 distinct \(\epsilon\) draws each, compute their per-token variance, and report the ratio between the medical average and the IMDB average—the ratio should exceed 2.0 to demonstrate that out-of-distribution text triggers higher uncertainty.
6. Persist the LoRA checkpoint and tensorboard logs, keeping the checkpoint size under 120 MB by exporting only the LoRA weights; this artifact is already ready for Step 4 and for shipping to latency-sensitive inference pods.

**Expected outcome:** A LoRA checkpoint that keeps IMDB variance near or below 0.015 while capturing at least a 2× variance gap on medical prompts, plus logged KL and variance diagnostics that tie the Bayesian geometry to the observed uncertainty.

**Variants per persona:**
- **Applied AI/ML engineer:** Replace Colab with an A10 instance, quantize the LoRA tensors with `bitsandbytes` 4-bit quantization, and deploy the checkpoint to a TGI endpoint with a 20 ms p95 latency target, using the variance signal to gate responses through an abstention policy.
- **Research engineer:** Reproduce Figure 3 from the February 2026 preprint by sampling 20 posterior draws per prompt and matching the reported variance curves within ±5%, instrumenting the training loop to log both KL and entropy for ablation.
- **Applied researcher:** Hypothesize that annealing the KL weight from 1.0 to 0.1 over 10 epochs widens the medical-IMDB variance gap; falsify the hypothesis if the gap either stays the same or shrinks while likelihood drops more than 1% relative.

**Stretch goals:**
1. Replace `lora_r=4` with a layer-wise schedule that uses \(\{2,4,6\}\) ranks and observe whether higher-capacity layers capture extra epistemic mass without collapsing the KL.
2. Swap the medical prompts with adversarial IMDB paraphrases to test whether the variance spike tracks true distribution shift or merely lexical novelty.
3. Train three ScalaBL checkpoints with different seeds, aggregate their epistemic variance curves, and show whether the 2× gap is consistent across posterior modes.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---