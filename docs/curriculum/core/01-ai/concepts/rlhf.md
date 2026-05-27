---
title: Reinforcement Learning from Human Feedback
slug: rlhf
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [stiennon, ziegler, leike, ouyang]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [supervised-finetuning, reward-modeling, reinforcement-learning-basics]
tags: [rlhf, grpo, reward-modeling, alignment, reasoning, tool-use]
updated: 2025-04-05
has_mvb: true
---

# Reinforcement Learning from Human Feedback

Most explanations of RLHF begin with the Bayesian posterior over human preferences, but the question that actually drove its adoption is much more pedestrian: why do large language models sometimes agree with people even when what they agree with is wrong? Supervised fine-tuning (SFT) trained on ChatGPT-style assistants trains models to mimic friendly, human-sounding answers, so they learn to be “very helpful” in the same tone as the dataset even when that helpfulness lands them in a sycophantic trap—apologizing for mistaken facts, inventing details just to maintain politeness, and hiding uncertainty behind canned hedging. Reinforcement Learning with Human Feedback steps away from mimicry by replacing imitation with outcome-based learning: the policy now receives a preference signal that rewards actually useful reasoning paths rather than the agreeable phrasing that appeared in the dataset. By the end of this page you will understand the loop that takes a reward model’s judgment back into policy updates, how modern systems engineer those signals to coax self-correction, and how to run your own GRPO-based callback that aligns a 2.5B policy on reasoning tasks.

## The territory

The history of RLHF is a story of increasing agency. Annotated History of Modern AI and Deep Learning (2022) [arxiv:2212.11279v1](https://arxiv.org/abs/2212.11279v1) shows the first wave of RLHF emerging as a soft safety layer: reward models simply scored helpfulness, and reinforcement learning was a means to avoid repetitive rejection sampling over all SFT outputs. That early work borrowed from the policy gradient math codified in the mid-2000s, with Anonymous et al. (2007) [arxiv:0708.4311](https://arxiv.org/abs/0708.4311) providing the stability guarantees for gradient estimators that still underlie the optimization in RLHF. As transformer-scale models became capable of complex reasoning, the paradigm shifted. Rather than just policing style, practitioners began to treat RLHF as a way to elicit tool use, hierarchical planning, and truth-seeking by explicitly rewarding the outcome of a generated chain of thought. This family of techniques straddles supervised imitation (the SFT warm start), reward modeling (scoring outcomes), and reinforcement learning (GRPO, PPO, or KL-constrained policy updates). The mechanism is best understood by starting with how the preference signal is obtained and distilled back into the policy parameters—how do we actually turn judgments about reasoning into incentives?

## How it works

The RLHF loop has three actors: the policy, the reward model, and the preference aggregator. The policy \(\pi_\theta(a \mid s)\) is the language model with parameters \(\theta\), the reward model \(R_\phi(s,a)\) is a network with parameters \(\phi\) that maps state-action trajectories to scalar scores, and the preference aggregator introduces a baseline \(b(s)\) so the policy has a reference point for each conversational context \(s\). The policy rollout generates a completion \(a\) (or multi-turn sequence), the reward model scores it, and the optimizer updates \(\theta\) to increase the expected reward relative to the baseline.

The core RL objective becomes
\[
L(\theta) = -\mathbb{E}_{s \sim \mathcal{D}, a \sim \pi_\theta(\cdot \mid s)}\left[(R_\phi(s,a) - b(s)) \log \pi_\theta(a \mid s)\right]
\]

where \(s\) is a prompt sampled from your preference dataset \(\mathcal{D}\), \(a\) is the policy’s generated completion conditional on \(s\), \(R_\phi\) is the reward model’s scalar score for the whole trajectory, \(b(s)\) is the baseline (often a value head or reference policy score), and \(\log \pi_\theta\) is the log probability of the generated sequence. This objective is a standard REINFORCE-style policy gradient with a learned baseline, so the variance of the gradient depends heavily on how precise \(R_\phi\) and \(b(s)\) are. That precision is the lever modern RLHF pulls: the reward model must itself reason about the output, not just check a string match, as RM-R1 (2025) [arxiv:2505.02387](https://arxiv.org/abs/2505.02387) demonstrates through its Chain-of-Rubrics approach, where the reward model’s own chain-of-thought is matched against rubric criteria before scoring the policy output. This reflexive reasoning dramatically sharpens gradients because the reward model can penalize partial reasoning steps rather than waiting to see whether the final answer is superficially plausible.

To make that sharper signal practical, practitioners train the reward model on pairwise preferences derived either from human labels or synthetic rule-based evaluators. When a dataset is small or the domain is structured, a deterministic verifier—such as a regex that checks whether the final line ends with a numerically precise answer—can provide a reproducible signal. The reward training objective minimizes the cross-entropy over pairwise comparisons:

\[
\mathcal{L}(\phi) = -\mathbb{E}_{(s,a^+,a^-)}\left[\log \sigma(R_\phi(s,a^+) - R_\phi(s,a^-))\right]
\]

where \(a^+\) and \(a^-\) are preference-labeled completions for the same prompt, \(\sigma\) is the sigmoid function, and \(R_\phi\) must score the preferred trajectory higher. RM-R1’s Chain-of-Rubrics adds intermediate supervision by forcing \(R_\phi\) to align with scoring steps that mirror human reasoning, which also makes it easier to generalize to new problem templates.

Once the reward model is trained, we have to decide how to actualize the policy update. GRPO—Group Relative Policy Optimization—extends the PPO framework by batching preferences from several prompts and computing a group-level baseline that reduces variance across heterogeneous tasks (Beyond Accuracy, 2025) [arxiv:2506.04723](https://arxiv.org/abs/2506.04723). The GRPO update moment is:

\[
L_{\text{GRPO}}(\theta) = -\mathbb{E}_{g \sim \mathcal{G}} \mathbb{E}_{s \in g, a \sim \pi_\theta}\left[\min\left(r(\theta)\hat{A}_g, \text{clip}(r(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_g\right)\right]
\]

where \(\mathcal{G}\) is a group of prompts with shared characteristics, \(r(\theta) = \frac{\pi_\theta(a\mid s)}{\pi_{\theta_{\text{old}}}(a\mid s)}\) is the probability ratio between current and previous policies, \(\hat{A}_g\) is the group advantage (aggregate reward minus baseline), and \(\epsilon\) is the clipping hyperparameter. Beyond Accuracy shows that optimizing this grouped advantage preserves the diversity of reasoning strategies by keeping the baseline local to the problem archetype—groups can be as simple as “math word problems” versus “common-sense explanations”. The result is that the policy learns not just to mimic a clipping style but to change its internal strategy formulation: when the group reward increases after a chain-of-thought update, it is because the policy is exploring new ways of decomposing the reasoning steps.

Integrating action spaces beyond tokens is where ReSearch (2025) [arxiv:2503.19470](https://arxiv.org/abs/2503.19470) shows RLHF’s second act. In addition to generating tokens, the policy can choose actions such as “issue a web search query,” “invoke a calculator tool,” or “use a specialized API.” These actions become part of the trajectory tackled by the reward model. ReSearch extends the reward model to consider both the chain of thought and the external observations retrieved via those search actions. The reward function \(R_\phi(s,a,q)\) now takes the prompt \(s\), the generated completion \(a\), and the appended search query \(q\), and the training data includes annotations that say both “this set of queries led to a better answer” and “these queries are redundant.” That richer action space is what lets RLHF systems purposefully ask for information rather than waiting for a human prompt to supply it.

Implementing the RLHF loop therefore splits into three tracked phases: (1) curriculum SFT warm-start, (2) reward modeling with high-quality preference data, and (3) policy optimization with GRPO or PPO under KL constraints to avoid catastrophic forgetting. The warm-start ensures the policy covers the domain so that RL gradients have a signal to amplify; the reward model shapes a precise objective; and the RL optimizer executes the policy update while respecting the SFT prior. Each phase carries its own failure mode: warm-start data can encode sycophancy, reward models can overfit to heuristics, and RL trainers can collapse if the reward estimates are miscalibrated. The appeal of reward-model-as-reasoner and group-aware policy updates is that they attack those failure modes head-on.

## Where the field is now

Modern RLHF systems are no longer just about safety at the margin; they are the mechanism by which generative models internalize complex reasoning flows. RM-R1 (2025) proved that reward models must match the architectural depth of the policy by reasoning over intermediate rubric steps, and its deployment path has become a reference for building “rewards that explain themselves.” Beyond Accuracy (2025) put numbers to the intuition that reinforcement learning yields fundamentally different behaviors: by measuring strategy divergence across generations, the paper showed that GRPO training yields more stable chain-of-thought structures compared to simple KL-penalized policy updates. ReSearch (2025) operationalized prompting for retrieval control, adding search actions to the policy so that RLHF can optimize both the text generated and the information acquired. Combined, these papers show a research frontier where reward quality, strategy diversity, and tool-use are the core axes.

On the engineering frontier, the anonymous systems engineering report (2026) [arxiv:2603.14664](https://arxiv.org/pdf/2603.14664) documents how a large lab uses RLHF to tune a “planning agent” that keeps a log of intermediate steps, quantifies uncertainty via a separate critic, and enforces budgeted retrieval by penalizing excessive search actions. In production, companies such as Anthropic and OpenAI use RLHF loops that mix human feedback with automated reward models; the same report underscores that the main operational challenge is the tooling around reward annotation pipelines and the monitoring of distribution shift from training to deployment. This engineering frontier—deployment robustness, reward instrumentation, and rollout monitoring—is the complement to the research frontier’s focus on modeling assumptions.

| System | Reward innovation | Benchmark signal |
|---|---|---|
| Anthropic Constitutional AI | Multi-critic consensus + rule-based paraphrase checking | Human evaluation jump from 70 to 85 average helpfulness (2024 internal) |
| OpenAI GPT‑4 RLHF (public) | KL-constrained PPO with pairwise human labels | Human preference win rate 82% vs. SFT baselines (2024) |
| ReSearch prototype | Search action tokens scored by reward model | 15% higher factual accuracy on unseen knowledge graphs |

This table shows that current deployments add resilience either through consensus-based reward evaluation or by linking RLHF to retrieval actions; future systems are likely to merge both.

## What's still open

Can we design reward signals for open-ended, non-verifiable tasks (such as system architecture design or policy drafting) where rule-based checkers fail and the reward model itself is prone to subtle reasoning errors? The central risk is that the reward stretches the policy toward an ungrounded reasoning loop that still satisfies the reward model but not the real-world standard. Another question is how to debug and quantify the generalization gap between the reward model’s Chain-of-Rubrics training tasks and the policy’s deployment prompts—if the reward model loses its reasoning depth out-of-distribution, GRPO will amplify the wrong behavior. Finally, could search-augmented RLHF systems (ReSearch-style) learn to “query for their own correction” by treating annotation requests as actions, and if so, what is the falsification criterion that shows the policy hasn’t simply learned to game those queries?

## Where to read next

The theoretical foundation lives in [[reward-modeling]], which explains the pairwise preference loss that anchors every gradient step, while the engineering counterpart is → [[tool-augmented-llms]] describing how actions such as search, calculators, or APIs become part of the trajectory. For the optimization community, → [[policy-gradients]] lays out why GRPO’s clipping combined with group baselines produces low-variance updates, and for a broader arc of reasoning the papers collected in [Chain-of-thought](chain-of-thought.md) show how RLHF can strengthen self-verification.

## Build it

The build proves that a GRPO-style RLHF loop with a lightweight reward function can coax a 2.5B policy into producing structured reasoning on math word problems without relying on human preference labels. *What you're building:* a GRPO reinforcement loop on Colab T4 that aligns `Qwen/Qwen-2.5-0.5B-Instruct` using a deterministic regex-based reward on GSM8K answers. *Why this is valuable:* the artifact forces you to implement the three-phase pipeline (SFT warm start, reward model, GRPO update) and to measure how the reward signal steers the policy’s chain-of-thought instead of its tone. *Stack:*
- **Model:** `Qwen/Qwen-2.5-0.5B-Instruct` for the policy, with inference weights cached locally; the reward model uses `RLHFlow/ArmoRM-Llama3-8B-v0.1` so you don’t have to train a RM from scratch.
- **Dataset:** `gsm8k` filtered to 4,000 training prompts with symbolic answers; the auxiliary preference dataset is bootstrapped from `RLHFlow/Llama3.1-8B-PRM-Deepseek-Data` completions labeled via regex.
- **Framework:** `trlx` 0.5.2 + `accelerate` 0.20 for offloading to the Colab T4, plus `einops` for batching.
- **Compute:** free Colab T4 (16 GB VRAM, 2h per run).

**The recipe:**
1. `pip install trlx accelerate einops datasets tensorboard` and load the `Qwen/Qwen-2.5-0.5B-Instruct` tokenizer; download `RLHFlow/ArmoRM-Llama3-8B-v0.1` once to use as the reward model.
2. Preprocess GSM8K: keep the question + answer pairs, convert answers to LaTeX-free digits, and define a regex reward that checks the final line for the digits present in the ground-truth answer; generate synthetic “wrong” completions using the SFT policy with temperature 1.2 for preference pairs.
3. Train the reward model by fine-tuning `RLHFlow/ArmoRM-Llama3-8B-v0.1` on pairwise comparisons (preferred vs. non-preferred completions) with the cross-entropy loss above; expect reward loss to drop below 0.5 after 1.5 epochs.
4. Run the GRPO loop: load the warmed policy, compute the group baseline as the mean reward per prompt bucket (e.g., arithmetic vs. algebra), apply the clipped objective with \(\epsilon=0.2\), and update for 3 epochs; monitor that the regex reward average increases while the KL to the SFT policy stays under 0.08.
5. Evaluate by generating 256 GSM8K questions and verifying the regex matches the final token; expect final reward-match accuracy above 75%, and inspect chains of thought to ensure they include explicit subtraction/division steps rather than hallucinated heuristics.

**Expected outcome:** a Colab checkpoint of a GRPO-aligned Qwen 2.5B policy plus TensorBoard traces showing reward stabilization and chain-of-thought improvement.

Variants per persona:
- **CS student:** Run the same recipe on an RTX 4070 by halving the dataset to 2,000 samples and using gradient accumulation, which lets you visualize the reward curve in the notebook.
- **Applied engineer:** Export the aligned checkpoint to ONNX, quantize to INT8, and serve through vLLM with a 50 ms latency budget while logging regex reward hits.
- **Applied researcher:** Modify the reward signal to compare regex-based scoring with a small Transformer critic trained on handwritten rubric labels, testing the hypothesis that the critic improves factual accuracy by ≥5%.
- **Frontier researcher:** Probe the open question of non-verifiable rewards by replacing the deterministic regex with a learned reward model that judges “design clarity,” and declare the falsifier as a decrease in human-rated coherence despite stable reward calibration.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*