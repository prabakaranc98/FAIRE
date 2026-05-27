---
title: "Train a reasoning reward model"
slug: train-reasoning-reward-model
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [leike, chen]
feeds_de_pillar: []
arc_position:
  arc: agentic-rlvr-reasoner
  prev: step-01-chain-of-thought
  next: step-03-rlhf
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [chain-of-thought, preference-modeling]
tags: []
updated: 2024-11-05
has_mvb: true
---
> **Arc:** [Agentic Rlvr Reasoner](../../arcs/agentic-rlvr-reasoner.md) — Step 2 of 4


Imagine you are the person grading a long-form answer where correctness is not in a single number but in a sequence of moves: one student argues about edge cases, another rewrites the whole proof mid-argument, and every step could hide a critical oversight. You cannot judge success by the final line alone, because a clever student might finish with the right conclusion while having built the argument on a faulty premise. You need to replay the chain of reasoning you wish the student had followed, then reward them only when that replay matches your own rubric. That is exactly the problem reasoning reward models try to solve for agents: how do you turn a reward signal that used to be a scalar judgment into something that rehearses the line of thought you care about before it ever spits out a score?

# The territory

Reward modeling became a cornerstone of alignment after researchers realized manually writing reward functions leads to unintended shortcuts. Deep Reinforcement Learning from Human Preferences (Christiano et al. 2017) [https://export.arxiv.org/pdf/1806.01946v4.pdf] formalized the pairwise preference dataset: humans compare two rollouts, and a small head is trained to explain which rollout is “better.” Leike et al. (2018) [https://arxiv.org/pdf/1811.07871] extended that idea beyond toy environments, arguing that agents ought to learn rewards from humans wherever hand-crafting objectives would either be brittle or dangerous. The original pipeline, though, still produced a single scalar per comparison, so researchers soon realized the reward model itself could be hacked if it cared only about correlation with the preference label and not about the underlying reasoning steps.

The new frontier asks: can a reward model behave like the grader rather than the grade? If it first rehearses its own chain of rubrics and only then emits a numeric score, we get a signal that can be audited, edited, and paired with the policy’s reasoning before the policy ever sees feedback. The survey “A Comprehensive Survey of Reward Models: Taxonomy, Applications, Challenges, and…” (2025) [https://arxiv.org/html/2504.12328] charts this shift—classifying a spectrum from scalar-only critics to reasoning-aware generative critics—and concludes that the more structured end of the spectrum is where reward hacking yields to inspectable alignment checks. The rest of this page shows how that structured end works, how it trains, and what you can build with it.

## How it works

We break down the mechanism into three stages: modeling human preferences, generating internal reasoning, and turning that reasoning into a scalar while keeping the whole process differentiable and verifiable.

### Modeling human preferences

Each pair of responses \((x_i, x_j)\) to a prompt becomes a binary outcome \(y_{ij}\in\{0,1\}\), with \(y_{ij}=1\) when the annotator prefers \(x_i\) and \(y_{ij}=0\) when they prefer \(x_j\). The probability assigned by a reward model \(r_\theta\) to the human ordering is

\[
P[y_{ij}=1\mid x_i,x_j] = \sigma\left(r_\theta(x_i) - r_\theta(x_j)\right),
\]

where \(\sigma(z)=1/(1+e^{-z})\) is the sigmoid, \(r_\theta(x)\) is the reward model’s scalar output for response \(x\), and the difference \(r_\theta(x_i)-r_\theta(x_j)\) is interpreted as the model’s confidence that \(x_i\) is better. The sigmoid rescales that difference into a probability between zero and one, so the model can be trained with a standard binary log-likelihood: when the human label favors \(x_i\), the loss penalizes the model if \(r_\theta(x_i)\) is not sufficiently larger than \(r_\theta(x_j)\).

Expanding that into the cross-entropy objective gives

\[
\mathcal{L} = -\sum_{(i,j)} \left[ y_{ij} \log P[y_{ij}=1] + (1-y_{ij}) \log \left(1 - P[y_{ij}=1]\right)\right],
\]

where \(\mathcal{L}\) averages the loss across all preference pairs and the \(P[y_{ij}=1]\) term is replaced with the sigmoid expression above. This penalizes any ordering where the reward scalars contradict the annotator and pushes the model to fit ordinal data rather than raw scores. At this level, the training procedure is identical to classical reward modeling, but it still conceals what the reward depends on. That is where the reasoning component enters.

### Generating internal reasoning before the scalar

To prevent the reward model from becoming a surface-level scalar regressor, we interpose a token-level reasoning sequence \(c_\theta(x)\) between the response and the final scalar. The model no longer computes \(r_\theta(x)\) directly from the response but instead autoregressively generates \(c_\theta(x)\), a chain of rubrics or explanations that describe why \(x\) deserves a particular score. The scalar head then consumes both \(x\) and \(c_\theta(x)\) to produce

\[
r_\theta(x) = h_\theta(x, c_\theta(x)),
\]

where \(h_\theta\) is a learned function (implemented as a projection or attention readout) that fuses the raw response tokens with the generated reasoning tokens. Because \(c_\theta(x)\) is part of the computation graph, gradients pass from the scalar loss through the reasoning as well, meaning the model must learn to pick reasoning chains that lead to high agreement with the human preference. This enforces that the reward depends on the same structure we care about when crafting rubrics.

Training the reasoning portion uses teacher forcing: the target sequence includes the ground-truth reasoning text followed by a separator token, then the scalar label expressed as a tokenized float. We concatenate these targets onto the original response, so the input looks like “prompt + agent trace + reasoning target + <scalar 0.73>.” During inference, the model first samples or greedily decodes a reasoning string, then a scalar value, and we decode that scalar into the same float range used during training.

This approach has two immediate benefits. First, the scalar head cannot ignore the reasoning tokens because the float is conditioned on them. Second, those reasoning tokens are inspectable, so we can confirm whether the reward model is replaying plausible rubrics or simply parroting the dataset. If the reasoning output looks incoherent, we can audit and intervene before the policy is updated.

### Training regimen and dataset shaping

Training follows the preference competition pipeline but with reasoning-aware targets. We sample batches of comparison triples \((x_i,x_j,y_{ij})\), tokenize them with a context window that keeps the reasoning and scalar tokens in a fixed place (after the response), and apply LoRA or another parameter-efficient fine-tuning technique so that the bulk of the reasoning LM remains fixed while the heads adapt. The training loss is the binary cross-entropy above, computed on the scalar tokens, but logged alongside auxiliary metrics: reasoning token perplexity, reasoning agreement with human rubrics, and Spearman correlation between generated scalars and human labels across validation pairs.

Because many reasoning targets are long, we use gradient accumulation to keep the effective batch size small (≤16 sequences) while still processing long prefixes, and we monitor the range of generated scalars to ensure they stay in \([0,1]\); token mappings like `scalar_00`–`scalar_99` help the tokenizer treat each float as a discrete token. Early stopping is triggered either by plateauing Spearman or by seeing repeated reasoning chains, which is one of the failure modes described later.

The reasoning LM base can be as large or small as resources allow; even a 2.5B parameter base like Qwen-2.5 can deliver a meaningful reward head when combined with reasoning targets, because the model is not re-trained from scratch. This keeps the build within practical compute while still delivering interpretable reward outputs.

### Synthesis: why the shift matters

Moving from scalar-only reward models to this generative, reasoning-aware formulation matters for both trust and capability. Scalar critics can learn to correlate with human labels while ignoring internal reasoning, which is sufficient for surface-level tasks but brittle under distribution shift. The reasoning head imposes a bottleneck: the only way the scalar can change is if the reasoning string changes in a direction that improves the logits. That means the reward model agrees with humans not just at the final score but across an inferred rubric. Papers like RewardBench (2022) [https://arxiv.org/pdf/2205.15367] show that scalar reward models generalize poorly to adversarial or out-of-distribution traces, but generative models that inspect intermediate reasoning degrade more gracefully. The dedicated reasoning tokens are both a guardrail and a diagnostic output—they let you ask, “Why did the reward change?” instead of only seeing “how much.” The rest of this page shows how to train such a model, how to evaluate its outputs, and what open questions remain once you can read the reward as a reasoning chain.

## Where the field is now

Research frontier: The latest wave of papers keeps widening the gap between surface-level critics and reasoning-aware reward models. RM-R1 (Chen et al. 2025) introduced the idea of reasoning reward modeling, reporting that a reasoning-augmented critic trained on 20K preference pairs reaches a Spearman correlation of 0.32 on a human-annotated validation set while a scalar-only model falls below 0.25 on the same compute budget. These results are collected and contextualized in the 2025 survey “A Comprehensive Survey of Reward Models…” (2025) [https://arxiv.org/html/2504.12328], which highlights how taxonomy now divides reward models by their introspective capabilities instead of just their loss functions. RewardBench (2022) [https://arxiv.org/pdf/2205.15367] complements that narrative by stressing how scalar-only critics break under adversarial perturbations—a pathology the reasoning step is designed to expose and correct.

Engineering frontier: Deployment teams are still largely dependent on scalar reward heads for RLHF because they are cheaper to store and faster to evaluate, but reproducible datasets and tooling are emerging to cross that chasm. The Hugging Face dataset `kpetyxova/towards-reward-modeling-tutors` packages pairs of tutor-style critiques with reasoning rubrics, making it easy to fine-tune an off-the-shelf reasoning LM with LoRA peft techniques. As a result, applied engineers can now produce reasoning-aware reward checkpoints in a few hours on a single Colab T4, validating that the reasoning output is coherent and the scalar correlates with human rankings. This engineering push proves that the gap between research ideas (RM-R1, RewardBench) and production practice is shrinking, and the reproducible dataset is the bridge that keeps the forward-deployed models accountable.

## What's still open

The central scientific question is causal: does intervening on the generated reasoning change the scalar reward in predictable ways, or is the reasoning just post hoc rationalization? A formal intervention study would manipulate one piece of the reasoning string while keeping the rest constant and then measure scalar drift—if the reward liquidly follows the reasoning intervention, the head is truly dependent on the rubric.

Another mystery is generalization: can a reasoning reward model trained on tutoring traces sustain Spearman correlation above 0.32 when confronted with agent traces far outside its training distribution? The adversarial splits in RewardBench are a start, but we still need controlled experiments on types of hallucination (fabricated facts vs. ethical lapses) and how much rerouting the reasoning head requires before the scalar degrades.

On the engineering side, we do not yet know what the optimal mix is between frozen reasoning LMs and trainable scalar heads. Some teams freeze the LM after a warm-up epoch, while others keep the reasoning generation open to affinity with the scalar. The question is whether freezing the reasoning head degrades adaptability to new tasks, or whether it is necessary to keep the scalar head from drifting into reward hacking when the reasoning network is overfit.

## Where to read next

If you want the engineering side, → [[rlhf]] explains how the reward signal is applied to the policy loop and what infrastructure keeps models safe during deployment. The theoretical counterpart is → [[preference-modeling]], which breaks down pairwise logits and gradient flows without assuming you have seen RM-R1. For the prompting context that feeds the reasoning tokens, → [[chain-of-thought]] details how to harvest coherent internal rubrics from a scaffolded policy.

## Build it

**What you're building:** A LoRA-fine-tuned “ReasRM” checkpoint that generates an autoregressive chain-of-rubrics and a scalar reward whose Spearman correlation with held-out human preferences exceeds 0.32 on the RM-R1 validation split.

**Why this is valuable:** It turns reward modeling from an opaque scalar regression into an auditable, reasoning-based evaluator that the downstream RLHF policy can trust, which is crucial when agents execute long agentic trajectories and one misaligned reward step can lead to reward hacking.

**Stack:**
- **Model:** `Qwen/Qwen-2.5-1.5B-Instruct` with PEFT LoRA wrappers and 4-bit quantization
- **Dataset:** `kpetyxova/towards-reward-modeling-tutors` (reasoning-annotated tutor comparisons derived from RM-R1-style preferences)
- **Framework:** `transformers==4.41.0`, `datasets==2.15.1`, `accelerate==0.21.0`, `peft==0.5.0`
- **Compute:** free Colab T4 (≤15 GB GPU RAM) using gradient accumulation (batch size 8, accumulation 2) to keep memory within limits

**The recipe:**
1. Install the stack, authenticate with Hugging Face, and confirm GPU availability (`torch.cuda.is_available()`); this primes Colab and ensures you can download the reasoning dataset without hitting rate limits.  
2. Use the Qwen tokenizer to encode each comparison, appending the tutor trace, the ground-truth reasoning target, a separator, and the scalar token (mapped to `scalar_00`–`scalar_99` to keep floats discrete); assert that every sample stays under 2048 tokens so you never truncate the reasoning.  
3. Wrap the Qwen base with LoRA (rank 8, alpha 32) and print the trainable parameters—this keeps the main LM frozen while letting reasoning and scalar heads adapt to the RM-R1 data.  
4. Structure the forward pass so the model first autoregressively emits the reasoning string and then the scalar token; compute the binary cross-entropy loss on the scalar outputs using the logistic model described in the theory section and log both Spearman correlation and reasoning token perplexity every 1,000 steps.  
5. Train for 2 epochs with AdamW (learning rate 1e-5, weight decay 0.01, gradient accumulation steps 4) and monitor whether gradient norms stay below 1.0; if Spearman stalls below 0.32, add LoRA rank to 16 before continuing.  
6. Evaluate on 2,000 held-out pairs from the dataset, decoding scalars back to floats, computing Spearman between generated scalars and human labels, and inspecting a random set of reasoning chains for diversity; the expectation is that reasoning tokens vary across inputs and scalars align with the ranking.

**Expected outcome:** A checkpoint that produces coherent reasoning strings (~80–120 tokens) followed by scalars that uphold the ordering of human preferences, demonstrating Spearman ≥0.32 on the validation split. If the scalar collapses to a constant or the reasoning tokens copy the dataset verbatim, the build surfaces those failure modes so you can adjust LoRA rank, dropout, or the tokenizer mapping.

**Variants per persona:**
- **Applied AI/ML engineer:** Deploy the checkpoint behind an API on an A10 serving container, use it as the reward critic for a live assistant, and ensure the end-to-end latency stays under 120 ms p95 while logging reasoning outputs for audit trails.  
- **Research engineer:** Reproduce Table 2 of Chen et al. (2025) RM-R1 by matching the Spearman correlation of ≥0.32 on a single H100 node, instrumenting the training loop to log reasoning token perplexity for each gradient update, and submitting your logs to a benchmark repo for comparison.  
- **Applied researcher:** Hypothesize that longer reasoning prompts (150 tokens vs. 80 tokens) increase Spearman correlation significantly; run two LoRA experiments with the same compute budget, plot correlation vs. reasoning length, and treat the difference as a falsifiable effect.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---