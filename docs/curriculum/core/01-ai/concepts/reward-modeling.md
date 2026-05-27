---
title: Reward Modeling
slug: reward-modeling
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [leike, kaelbling, rm-r1, zhong]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [reinforcement-learning, preference-learning, human-feedback]
tags: [reward-modeling, rlhf, preference-learning, human-feedback, lm-alignment]
updated: 2025-05-04
has_mvb: true
---

# Reward Modeling

Imagine training a dog for ten minutes and only saying “good boy” once, right after it fetches a blue ball. How does the animal know which of the hundreds of actions deserved that praise, and which were noise? In language models we face the same credit-assignment crisis: “helpful and harmless” is not a boolean, it’s a tangled preference landscape, and we cannot write a single `if/else` clause that captures it. The solution is to teach a judge—a neural reward model—that scores every candidate response, letting the policy chase the score instead of our vague intentions. By the end of this page you will understand how human pairwise comparisons become differentiable feedback, why the stability of that feedback dominates RLHF systems, and how to build a minimal Bradley-Terry reward model that runs on a Colab T4.

## The territory

Reinforcement learning traditionally assumes a hand-crafted reward function \(r(s,a)\) that returns a scalar for each state-action pair; the policy need only maximize the expected sum of those scalars. In practice, producing \(r\) that captures politeness, accuracy, fairness, and safety for every prompt is intractable. Reward modeling sidesteps the hand-specification by letting humans judge which behaviors they prefer, and fitting a parameterized function \(\hat{r}_\phi\) that approximates those judgments. Leike et al. (2018) [arxiv:1811.07871](https://arxiv.org/pdf/1811.07871) articulated this as the alignment pathway that shifts the bottleneck from policy search to preference elicitation, turning feedback into a dataset of \( \{x^+, x^-\}\) pairs where the human picks the better transcript. Christiano et al. (2017) [arxiv:1706.03741](https://arxiv.org/abs/1706.03741) showed that the Bradley-Terry model turns such pairwise comparisons into a differentiable loss, and the 2018 follow-up (Christiano et al. 2018) [arxiv:1806.01946v4](https://export.arxiv.org/pdf/1806.01946v4.pdf) demonstrated how the same pipeline works at Atari and simulated robotics scales. Zhong et al.’s survey (2025) [arxiv:2504.12328](https://arxiv.org/html/2504.12328) frames modern reward models in a taxonomy of discriminative rankers, generative reasoning reward models, and hybrid semi-scalar architectures, clarifying which components borrow from supervised learning, which borrow from generative modeling, and which remain tied to RLHF-style ranking. Reward models feed directly into optimization loops: they take human judgments, output continuous scalar rewards, and supply gradients for either RL optimizers (PPO, A3C) or supervised fine-tuning, making the human preference specification the central lever of alignment. How does this machinery work in detail?

## How it works

The first core insight is that pairwise judgments give us a ranking signal without requiring an absolute scale. Given two responses \(x^+\) (preferred) and \(x^-\) (rejected), the Bradley-Terry model defines the probability that \(x^+\) wins as
\[
P_\phi[x^+ \succ x^-] = \frac{\exp(r_\phi(x^+))}{\exp(r_\phi(x^+)) + \exp(r_\phi(x^-))}
\]
where \(r_\phi\) is our differentiable reward model. This formula is a sigmoid over the reward gap; subtracting \(r_\phi(x^-)\) from \(r_\phi(x^+)\) yields a logit whose magnitude reflects how strongly the model prefers one response. The training signal is the negative log probability of the human’s choice,
\[
\mathcal{L}_\text{BT}(\phi) = -\log P_\phi[x^+ \succ x^-]
\]
which expands to
\[
\mathcal{L}_\text{BT}(\phi) = \log\left(1 + \exp\left(r_\phi(x^-) - r_\phi(x^+)\right)\right)
\]
where the margin \(r_\phi(x^+) - r_\phi(x^-)\) controls the curvature of the loss. Because every preference pair yields exactly one scalar loss, the training can leverage large-scale supervised optimizers rather than RL rollouts. This is why the human bottleneck shifts: once the reward model is fit, the policy no longer needs to query a human for every rollout, it simply maximizes the learned scalar.

### Architecting the reward model

The standard architecture is a Transformer-based encoder (e.g., BERT) that pools across the generated sequence into a scalar. Let \(h_\phi(x)\) denote the hidden state for input \(x\); then the reward is a linear readout \(r_\phi(x) = w^\top h_\phi(x)\) plus an optional bias. In architectures tailored for long-form dialogue, the encoder may append the context, prompt, and response tokens and average the final layer. The key engineering choices are how to tokenize prompts and responses, how to aggregate sentence-level structure, and what regularization to place on \(w\). Because the model is trained on preferences, the reward scale is learned end-to-end, allowing it to adapt to new contexts without a manual normalization.

### Data pipelines and preference elicitation

Human annotators compare pairs of model outputs for the same instruction. The pipeline typically samples two completions from the policy (sometimes with temperature diversity) and asks a human to choose the better one along axes such as helpfulness, truthfulness, and harmlessness. Because direct pairwise annotation is expensive, modern datasets like Anthropic’s HH-RLHF release (available as the HuggingFace dataset `anthropic/hh-rlhf`) include multi-annotator comparisons and rubrics. Data curation remedies the class imbalance between chosen and rejected samples by subsampling, filtering out obviously bad responses, and ensuring diverse prompt topics. The dataset is converted into a table \(\{(x_i^+, x_i^-)\}_{i=1}^N\), and each entry produces one gradient step through \(\mathcal{L}_\text{BT}\). Because the probability is symmetric, the same sample can supply both chosen and rejected roles across different pairs, increasing data efficiency.

### Regularization and margin losses

Reward models can overfit to subtle artifacts, like preferring longer responses because humans often choose the more informative-looking one. To combat this, practitioners introduce regularization terms. A simple example is an \(\ell_2\) penalty on the readout weights,
\[
\mathcal{L}(\phi) = \mathcal{L}_\text{BT}(\phi) + \lambda \|w\|^2
\]
where \(\lambda\) trades off preference fit for smoothness. A more structured approach is margin rescaling: we replace the logistic loss with a margin-based hinge,
\[
\mathcal{L}_\text{margin}(\phi) = \max\big(0, m + r_\phi(x^-) - r_\phi(x^+)\big)
\]
where \(m > 0\) enforces that the chosen sample is at least \(m\) points better. This margin exaggerates the penalty for near-ties and can be annealed as the model calibrates.

### Residual reward modeling

Residual reward modeling addresses the instability that pure preference learning sometimes introduces by decomposing the reward into a prior heuristic plus a learned residual. Residual Reward Models (2022) [arxiv:2205.15367](https://arxiv.org/pdf/2205.15367) express the reward as
\[
r_\phi(x) = r_\text{prior}(x) + f_\phi(x)
\]
where \(r_\text{prior}\) is a fixed heuristic (e.g., the log-probability under the policy or a length penalty) and \(f_\phi\) is a smaller neural network trained to explain the difference between the heuristic and human preferences. The prior anchors the scale and biases the model toward known-safe behavior, while the residual \(f_\phi\) focuses on the nuanced part of the human signal. This decomposition reduces variance in both reward training and subsequent RL fine-tuning because the prior term handles the bulk of the reward, leaving \(f_\phi\) to solve a less-wiggly regression. The training objective remains the Bradley-Terry loss applied to \(r_\phi\).

### Generative reasoning reward models

Standard reward models output a scalar without revealing why. RM-R1 (2025) as summarized in Zhong et al.’s survey (2025) [arxiv:2504.12328](https://arxiv.org/html/2504.12328) proposes ReasRMs, which generate a short chain-of-thought-style justification alongside the scalar score. The joint model maximizes
\[
\mathcal{L}_\text{ReasRM}(\phi) = \mathcal{L}_\text{BT}(\phi) + \alpha \mathcal{L}_\text{gen}(\phi)
\]
where \(\mathcal{L}_\text{gen}\) is the cross-entropy for the generated rationale and \(\alpha\) weights the explainability penalty. The rationale serves two purposes: it exposes the alignment model’s reasoning for auditing, and it provides an auxiliary signal to regularize the scalar score, making the reward less prone to shortcut reasoning that exploits length or token frequency.

### Online adaptation and policy interplay

Once the reward model runs, it feeds into a policy optimizer. In RLHF, a policy \(\pi_\theta\) produces completions that a reward model scores, and PPO or another RL algorithm updates \(\theta\) to maximize the expected reward plus a KL penalty to the pre-trained policy. The KL term keeps the policy from drifting too far from the pretrained distribution, but the reward model must remain calibrated; if it becomes misaligned with human judgments (reward hacking), the policy can find loopholes that score high but are unsafe. That is why reward modeling is no longer a static dataset but an iterative process: new human comparisons are collected on policy rollouts, the reward model is fine-tuned, and the policy is reoptimized, forming a feedback loop where stability and generalization are the central challenges.

## Where the field is now

The research frontier currently pivots on richer preference data and decomposition strategies. RM-R1 (as discussed in Zhong et al. 2025) shows that adding generated rationales to preference data boosts interpretability and accuracy, and experiments in the survey demonstrate ReasRMs outperforming scalar-only models on truthfulness benchmarks. Residual Reward Models (2022) supply the mathematical decomposition that keeps reward scales grounded in safe heuristics, stabilizing both the human feedback stage and the downstream RL loop. Collecting data remains difficult: Anthropic’s HH-RLHF dataset on HuggingFace (`anthropic/hh-rlhf`) provides millions of preference comparisons between model outputs and human-written assistants, and their documentation notes that the dataset is curated from adversarial red-teaming to surface failure modes, illustrating how companies are engineering reward pipelines before RLHF ever runs.

On the systems side, production-scale alignment pipelines rely on large-scale automated preference collection and inference-time serving. OpenAI’s “Training language models to follow instructions with human feedback” (Ouyang et al. 2022) explains that reward models run inside a distributed RLHF system, where tens of thousands of comparisons are labeled weekly and the reward inference occurs across TPU pods to keep latency under 100 ms for interactive fine-tuning. The same blog reports that deploying reward models under quantization and batching reduces inference cost, which demonstrates that the engineering frontier is focused on running reward models fast and reliably so policy updates can be frequent. Together, these results show that reward modeling is no longer academic: both research and engineering labs are investing in the data curation, modular architectures, and infrastructure that let the human judgment bottleneck scale to the size of modern LLMs.

## What's still open

Can we give theoretical guarantees that a reward model generalizes to prompts very unlike those it saw during training without reward hacking? The central question is whether a preference-based scalar can avoid spurious correlations such as length bias or dataset label leakage, especially when the policy discovers novel distributions. Is there a sampling strategy over the prompt space that provably covers deceptive corner cases so that preference pairs remain informative? Another key question is how to merge reward models learned from different cohorts (e.g., helpfulness vs. safety) without the fused scalar collapsing to the loudest signal. Finally, can we automate the critique loop—where a reward model generates critiques of its own outputs to expose weaknesses—without introducing instabilities or compounding biases in the human feedback pipeline?

## Where to read next

If you want the probabilistic control perspective, → [[reinforcement-learning]] explains how rewards drive Bellman backups and why shaping rewards changes value propagation. The engineering counterpart is → [[human-feedback]] which documents how annotation interfaces, aggregation, and quality control keep preference data usable. For the theory and alignment story, → [[preference-learning]] shows how pairwise preferences relate to classic ranking and ordinal regression techniques.

## Build it

This build proves that even on a Colab T4 you can fit a usable reward model from preference pairs, which turns human judgments into a margin-aware scalar that policy learners can trust. The artifact is a lightweight Bradley-Terry reward model trained on a subset of Anthropic’s HH-RLHF comparisons, showing the formatting of chosen/rejected pairs, the margin loss, and a sanity-check correlation with human labels.

**What you're building:** a pairwise Bradley-Terry reward model based on `prajjwal1/bert-tiny` that achieves >85% accuracy on held-out preference pairs from `anthropic/hh-rlhf` using margin loss.  
**Why this is valuable:** the build touches the hard part of the concept—turning comparative human judgments into a differentiable signal that generalizes beyond the examples you explicitly label.  
**Stack:**  
- **Model:** [https://huggingface.co/prajjwal1/bert-tiny](https://huggingface.co/prajjwal1/bert-tiny) — 2.8M parameters, widely downloaded  
- **Dataset:** [https://huggingface.co/datasets/anthropic/hh-rlhf](https://huggingface.co/datasets/anthropic/hh-rlhf) — curated human preference pairs  
- **Framework:** PyTorch 2.1 + Transformers 4.40 + Accelerate 1.16  
- **Compute:** Google Colab T4 (16 GB VRAM) — training takes ~1.5 hours for 3 epochs

**The recipe:**
1. Install `pip install torch torchvision accelerate transformers datasets evaluate`; load the dataset with `datasets.load_dataset("anthropic/hh-rlhf", split="train[:1%]")` and filter to prompt-response pairs where both chosen and rejected completions are complete sentences.
2. Tokenize both responses with the same tokenizer, pad to 512 tokens, and create tensors for chosen/rejected pairs; organize the batch so each forward pass processes two sequences per sample.
3. Fine-tune `prajjwal1/bert-tiny` with a linear readout: compute rewards \(r_\phi(x^+)\) and \(r_\phi(x^-)\), then apply the margin loss \(\max(0, m + r_\phi(x^-) - r_\phi(x^+))\) with margin \(m=0.5\); use AdamW, learning rate \(2e^{-5}\), batch size 16, gradient accumulation 2, and track accuracy by comparing logits.
4. Evaluate on a held-out 10% slice: measure the fraction of pairs where \(r_\phi(x^+) > r_\phi(x^-)\); expect >85% with this data regime, and inspect examples where the margin is small to understand failure modes.
5. What you now have is a checkpoint ready to plug into a PPO loop or human-in-the-loop policy writer, plus logging that correlates margin size with human disagreement.

**Expected outcome:** a trained reward-model checkpoint plus evaluation log showing margin accuracy and a small sample of low-margin failure cases.

- **CS student:** Reduce to 200 samples and run on a single RTX 4060, focusing on the gradient accumulation steps and visualizing the margin histogram via matplotlib.
- **Applied engineer:** Export the model to TorchScript, quantize to int8 via `torch.ao.quantization.quantize_dynamic`, and serve through a simple FastAPI endpoint with sub-120 ms p50 latency on an A10 instance.
- **Applied researcher:** Hypothesize that the margin loss needs curriculum sampling; compare uniform pair sampling with margin-weighted sampling (more weight on pairs where \(|r_\phi(x^+) - r_\phi(x^-)|\) is small) and report accuracy + calibration curves.
- **Frontier researcher:** Probe whether reward hacking emerges when the policy shifts to out-of-distribution prompts by adversarially sampling prompts that break the heuristic prior from §How it works and measuring whether the residual term \(f_\phi(x)\) still aligns with new preferences.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*