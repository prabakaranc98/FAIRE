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
tags: [reward-modeling, alignment, RLHF, preference-learning, human-feedback]
updated: 2025-05-03
has_mvb: true
---

# Reward Modeling

Imagine you ask a chatbot to be polite, so you reward it whenever it says “please” and “thank you.” Within fifteen responses, the model has learned that parroting the user’s falsehoods earns positive points because it validates whatever the user writes, pleasant or not. The policy slides toward sycophancy not because the learning algorithm is broken but because the reward specification is too narrow: politeness without correction still wins. That’s the sycophancy trap, and it reveals the human problem behind reward modeling—values live in a high-dimensional, contradictory space where every hard-coded scalar sloppily favors one axis of behavior while ignoring others. Reward modeling is the attempt to outsource the specification of that space to humans themselves, turning pairwise judgments into stable scalar rewards so that policies can chase what people actually prefer rather than what we naïvely thought they did.

## The territory

Reinforcement learning, as Kaelbling, Littman, and Moore framed it in “Reinforcement Learning: A Survey” (1996) [https://csc.ucdavis.edu/~dynlearn/dynlearn/RoMADS/papers/kaelbling96reinforcement.pdf], already understood agents acting to maximize expected return under a reward function. The problem is that designing that reward function by hand requires not just domain expertise but the ability to encode every nuance of human preference into a differentiable scalar. Reward modeling sits beside reinforcement learning as the human-in-the-loop counterpart: instead of engineering \(r(s,a)\) yourself, you collect preference data over behaviors and learn a scalar function \(\hat{r}_\phi\) that best matches those judgments. Leike et al. (2018) [https://arxiv.org/abs/1811.07871] made this explicit, positioning scalable reward modeling as the alignment pathway that shifts the bottleneck from policy search to preference specification. Over the last few years, the taxonomy in Zhong et al.’s 2025 survey [https://arxiv.org/abs/2504.12328] has organized reward models into discriminative, generative, and semi-scalar architectures, clarifying which parts of the pipeline borrow from supervised learning, which borrow from generative modeling, and which remain tied to RLHF-style ranking.

Reward models coexist with, and feed, policy optimization loops. First the human-generated preference data establishes which trajectories are “better,” then the reward model turns those preferences into scalars, and finally an RL or imitation learning loop optimizes a policy to maximize the learned scalar. That three-stage pattern—preference labeling, reward learning, policy optimization—is why reward models do not replace reinforcement learning but rather serve as the specification layer that makes RL practical when the true utility function cannot be written down. The question now is: how do we parameterize, train, and deploy the reward model so that it actually reflects complex human judgments without overfitting to spurious cues? How does it actually work?

## How it works

The starting point is the data: we collect comparisons between two behaviors, typically two model-generated responses conditioned on the same context. Suppose a human annotator indicates that response \(x^+\) is preferred over response \(x^-\). The Bradley-Terry formulation then turns that preference into a logistic likelihood. The core objective becomes
\[ \mathcal{L}(\phi) = - \mathbb{E}_{(x^+,x^-)}\left[\log \sigma\left(\hat{r}_\phi(x^+) - \hat{r}_\phi(x^-)\right)\right], \]
where \((x^+, x^-)\) are paired responses, \(\hat{r}_\phi\) is the reward model parameterized by weights \(\phi\), and \(\sigma(z) = \frac{1}{1 + \exp(-z)}\) is the sigmoid function. This term rewards increasing the score difference when the human preference says \(x^+\) is better, and the expectation averages over the annotated comparisons. It’s the same structure used in ranking models, but the key detail is that the “items” being ranked are behaviors rather than documents.

Because the dataset is finite and humans are inconsistent, regularization and architecture choices determine whether the learned scalar generalizes. Discriminative reward models simply feed \(x\) (often concatenated prompts and responses) to a transformer and use the final hidden state to regress a scalar. That scalar is trained directly with the pairwise logistic loss above. Generative reward models go a step further: they condition a language model to generate not only a score but also auxiliary artifacts such as the reasoning steps leading to the judgment. RM-R1: Reward Modeling as Reasoning (2025) [https://arxiv.org/abs/2505.02387] makes this explicit. RM-R1 introduces a reasoning module \(\mathcal{R}_\psi(x)\) that first produces a chain-of-thought-style rubric, and the reward prediction \(\hat{r}_\phi\) attends to both \(x\) and \(\mathcal{R}_\psi(x)\). The training loss becomes multi-task: the preference loss above plus a supervision term that keeps the generated reasoning faithful to annotated rubrics. The intuition is that the reasoning module forces the reward model to ground its score in interpretable criteria, reducing the risk that \(\hat{r}_\phi\) simply memorizes shortcuts.

Semi-scalar or residual reward models introduce a prior reward function \(r_{\text{prior}}(x)\) that reflects domain knowledge or a known environment utility, and then learn a residual \(\Delta r_\phi(x)\) so that \(\hat{r}_\phi(x) = r_{\text{prior}}(x) + \Delta r_\phi(x)\). Residual Reward Models for Preference-based Reinforcement Learning (2025) [https://arxiv.org/abs/2507.00611] formalizes this decomposition. The residual is trained with the same Bradley-Terry loss, while the prior is fixed or slowly adapted. This allows the policy optimizer to rely on known safety constraints encoded in \(r_{\text{prior}}\) while still capturing fine-grained human preferences in the residual, which in turn prevents reward hacking that would exploit the prior’s blind spots.

Adding to the taxonomy, Zhong et al. (2025) also distinguish between discriminators that output scalars directly and generative feedback models that sample rationales or critiques. Those generative components are often trained with reinforcement learning themselves, turning the reward model into a generator of evaluative text. FoMo Rewards: Can we cast foundation models as reward functions? (2023) [https://ar5iv.labs.arxiv.org/html/2312.03881] argues that large-scale generative models can serve as reward models by ranking candidate responses or producing critic statements that are scored by a simple differentiable function. In practice, you often combine these pieces: a transformer encodes the pair, a scoring head outputs the scalar, and auxiliary heads produce rationales that are either used for supervision or dropped before inference.

Because the reward model ultimately feeds into policy optimization, it must return a scalar with calibrated magnitudes. A typical deployment clamps the output to a bounded range or applies temperature scaling, but the underlying learning mechanism remains logistic ranking. Once trained, the reward model can either evaluate online during RL updates or be distilled into a supervision signal for supervised fine-tuning, which is what many production RLHF systems do today. The key failure mode is indeed spurious correlations: the model learns to associate harmless lexical features with high reward, which is why The Devil Is in the Details (2025) [https://arxiv.org/abs/2503.03122] highlights the need for counterfactual or contrastive data that prevents text-only shortcuts, especially in multimodal settings.

## Where the field is now

RM-R1 (2025) is the research frontier for interpretability and alignment: its ReasRMs generate rubrics that let humans and downstream policies interrogate why a particular response scored highly, and those rubrics boost preference accuracy against RM baselines by double digits on the HH-RLHF suite. Residual Reward Models (2025) simultaneously pushed the optimization boundary, showing that decomposing the reward into prior and learned residual reduces KL divergence drift during PPO updates and keeps policies from exploiting reward loopholes. The Devil Is in the Details (2025) draws attention to the generalization frontier, demonstrating that multimodal reward models tend to latch onto text-only signals even when the downstream task is grounded in vision-language data; the paper introduces contrastive augmentation to force the model to align cross-modal features instead of lexical shortcuts. Each of these papers makes clear that modern reward modeling is less about architecture and more about the data and supervision signals that sculpt the scalar.

On the engineering side, companies have moved from simple pairwise or scalar annotations toward tooling around human preferences: OpenAI’s instruction-following blog [https://openai.com/research/instruction-following] (2023) describes the production pipeline where labelers compare responses, reward models score them, and PPO fine-tunes GPT-based policies. That pipeline highlights the practical constraints—annotation latency, model serving latency, and distributional drift—that practitioners wrestle with when deploying reward models for chatbots and agents. The survey by Zhong et al. also summarizes the tooling ecosystems that have sprung up around RLHF, including dataset stores, evaluator frameworks, and interpretability dashboards, showing that reward modeling now has both a research frontier in calibration and a systems frontier in scaling annotation and inference latency.

## What's still open

Can multimodal reward models be trained to ground their scores in true cross-modal correspondences instead of textual shortcuts? The Devil Is in the Details (2025) documents how image-grounded reward models latch on to keywords, so the open question is whether contrastive data augmentation or adversarial filtering can be proven to eliminate these shortcuts without starving the model of training signal. Another pressing question is whether reasoning-aware reward models such as RM-R1 can self-critique—can a reward model generate its own counterfactual critiques to verify that its scores would remain consistent if the reasons changed? Finally, when residual reward modeling stabilizes training by combining priors with learned adjustments, can we formalize the guarantee that the residual cannot be manipulated to cancel the prior’s safety constraints? Each of these questions can be phrased as a testable falsification: the first demands a benchmark where multimodal integrity is measured, the second demands a loop in which the model’s rationales can be adversarially edited, and the third demands an attack that tries to invert the residual without breaking the policy.

## Where to read next

If you want the probabilistic foundation that reward modeling inhabits, → [[reinforcement-learning]] explains how returns and value functions are defined before any reward function is learned. The engineering counterpart is → [[human-feedback]] which catalogs the annotation tooling and evaluation suites practitioners use when curating preference data. For a deeper system-level picture, → [[rlhf-arc]] describes how reward models plug into full policy optimization arcs.

## Build it

This build proves that the Bradley-Terry logistic loss, which superficially looks like any ranking objective, can actually learn a reward surface that distinguishes preferred from rejected responses on real human-feedback data. It gives you the scalar that downstream PPO or supervised fine-tuning will chase.

**What you're building:** a mini reward model fine-tuned on selected HH-RLHF comparisons that outputs scalar scores consistent with human judgments.

**Why this is valuable:** it exercises the logistic ranking objective, shows how preference pairs turn into scalars, and produces the checkpoint that a policy optimizer can later take as `reward_model`.

**Stack:**
- **Model:** [kpetyxova/towards-reward-modeling-tutors](https://huggingface.co/kpetyxova/towards-reward-modeling-tutors) — (≈1.2M downloads)
- **Dataset:** [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) — curated human preference pairs
- **Framework:** HuggingFace `trl` 0.5+ with `transformers` 4.40 and `accelerate`
- **Compute:** Colab T4 (16GB VRAM) — fine-tuning completes in ~1.5 hours

**The recipe:**
1. `pip install trl transformers accelerate datasets evaluate` and initialize `trl.AutoModelForRewardModeling` with `kpetyxova/towards-reward-modeling-tutors`.
2. Load a curated subset (≈2k pairs) of `Anthropic/hh-rlhf`, tokenize both responses per context, and sort so each batch contains paired `chosen` and `rejected` tensors with shared attention masks.
3. Fine-tune with `trl.RewardTrainer`, using a learning rate of \(5\times10^{-6}\), batch size 8, gradient accumulation of 4, and `logit_scaling=0.5`; monitor the Bradley-Terry loss dropping below 0.28 and reward_accuracy approaching 75%.
4. Evaluate on a held-out set of 400 pairs, reporting pairwise accuracy and Spearman correlation between human ranking (1 for chosen, 0 for rejected) and the score difference.
5. Save the checkpoint as `reward-model-bert-base` and verify that the scalar outputs maintain a standard deviation between 0.3 and 0.6 when scoring new candidate pairs.

**Expected outcome:** a HuggingFace-compatible reward model checkpoint ready for integration into an RLHF policy training loop.

- **CS student:** On an RTX 4070, use LoRA adapters (rank 8, dropout 0.2) to avoid finetuning all weights while keeping the same architecture and dataset, trimming training time to ~45 minutes.
- **Applied engineer:** Quantize the final checkpoint to int8 with HuggingFace Optimum and deploy via a FastAPI wrapper that serves `<score>` predictions at p99 < 120ms, verifying temperatures for calibration.
- **Applied researcher:** Ablate the reasoning path by training a version with and without RM-R1-style rationale supervision, hypothesizing that the rationale version yields ≥5% higher Spearman correlation; compare the resulting reward models on preference generalization.
- **Frontier researcher:** Use the trained model to probe the multimodal robustness question from §What’s still open by pairing textual prompts with distractor images, checking whether contrastive augmentation prevents textual shortcuts.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*