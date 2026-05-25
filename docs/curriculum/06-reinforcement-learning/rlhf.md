---
title: Reinforcement Learning from Human Feedback (RLHF)
track: 06-reinforcement-learning
tags: [rlhf, reward-model, ppo, alignment, dpo, grpo, preference-learning]
depth: all
prereqs: [ppo, mdp, transformer]
updated: 2026-05-25
has_mvb: true
---

# Reinforcement Learning from Human Feedback (RLHF)
> **TL;DR:** The post-training paradigm that transformed raw pretrained LLMs into helpful assistants — train a reward model from human preferences, then optimize the LM with PPO to maximize it; the foundation of ChatGPT, Claude, and Gemini.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Fine-tune a model with DPO using TRL |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Understand what alignment actually means mechanistically |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Derive the DPO objective from the RLHF reward model |
| Researcher / frontier | [Current SotA](#current-sota) → [What's happening now](#whats-happening-now) | Know why GRPO and RLVR are replacing RLHF for reasoning |

---

## What it is

RLHF fine-tunes a pretrained language model to produce outputs humans prefer. It works in three stages:

**Stage 1 — Supervised Fine-Tuning (SFT):** Fine-tune the base LLM on demonstration data (human-written conversations, instructions, answers). This gives the model a "warm start" with the right output format and basic instruction-following.

**Stage 2 — Reward Modeling (RM):** Present annotators with pairs of model responses; collect which one they prefer. Train a reward model (another LLM with a linear head) on these pairwise preferences via the Bradley-Terry model. The RM learns to predict human preference scores.

**Stage 3 — RL Optimization:** Run PPO to optimize the LLM's outputs to maximize the learned reward — subject to a KL divergence penalty that prevents the model from drifting too far from the SFT policy. The KL term prevents reward hacking (the model finding degenerate ways to game the reward model).

The result: a model that is helpful, harmless, and honest — where those qualities are operationalized as "what humans preferred when given pairs of outputs."

## Why it matters at the frontier

RLHF is what made GPT-3 → ChatGPT. Without it, base LLMs produce raw completions — grammatically good but often unhelpful, off-topic, or unsafe. RLHF is the mechanism that aligns the model with human intent. Understanding it is understanding how models like Claude, ChatGPT, and Gemini actually work.

The broader significance: RLHF established that human preferences can be operationalized into a training signal. The entire post-training landscape — DPO, GRPO, Constitutional AI, DAPO — is built on or as alternatives to this insight. If you understand RLHF, you can understand every alignment paper published in the last three years.

## Core concepts

- **Reward model (RM)** — an LLM fine-tuned to predict which of two responses a human prefers; output is a scalar reward
- **Bradley-Terry model** — pairwise preference model: P(y_w ≻ y_l | x) = σ(r(x, y_w) − r(x, y_l)); the likelihood used to train the RM
- **SFT policy (π_SFT)** — the LLM after supervised fine-tuning; the starting point for RL optimization
- **KL divergence penalty** — regularization term that penalizes the RL policy for deviating too much from π_SFT; prevents reward hacking
- **PPO** — Proximal Policy Optimization; the RL algorithm used in RLHF; updates are bounded by a clipping ratio
- **Reward hacking** — the policy finds ways to exploit the reward model (e.g., generating very long responses that score well but aren't actually helpful)
- **DPO** — Direct Preference Optimization; reformulates RLHF as a supervised classification problem; bypasses reward model and PPO
- **GRPO** — Group-Relative Policy Optimization; uses group of responses as reference; no critic network; designed for verifiable rewards

## Mathematical foundations

Reward model training (Bradley-Terry pairwise loss):
$$\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim \mathcal{D}}\!\left[\log\sigma\!\left(r_\phi(x, y_w) - r_\phi(x, y_l)\right)\right]$$

RLHF objective (maximize reward, penalize KL divergence from SFT):
$$\max_{\pi_\theta}\; \mathbb{E}_{x \sim \mathcal{D},\, y \sim \pi_\theta(y|x)}\!\left[r_\phi(x, y)\right] - \beta\, D_{\text{KL}}\!\left(\pi_\theta \| \pi_{SFT}\right)$$

DPO objective (closed-form, no RM or PPO needed):
$$\mathcal{L}_{DPO}(\pi_\theta; \pi_{SFT}) = -\mathbb{E}_{(x, y_w, y_l)}\!\left[\log\sigma\!\left(\beta\log\frac{\pi_\theta(y_w|x)}{\pi_{SFT}(y_w|x)} - \beta\log\frac{\pi_\theta(y_l|x)}{\pi_{SFT}(y_l|x)}\right)\right]$$

The key DPO insight: the optimal RLHF policy has an analytic form in terms of log-ratios. DPO directly optimizes this without needing an explicit reward model.

## Key algorithms / techniques

- **RLHF with PPO** (Christiano et al. 2017; Ouyang et al. 2022) — original formulation; three-stage pipeline; most resource-intensive but most general
- **Direct Preference Optimization (DPO)** (Rafailov et al. 2023) — convert RLHF to supervised learning; same objective, no RM, no PPO; simpler and stable
- **Constitutional AI (CAI)** (Anthropic 2022) — use AI feedback instead of human feedback; scale preference signal with AI critique + revision
- **GRPO / Group-Relative Policy Optimization** (DeepSeekMath 2024) — no critic; uses advantage estimated from group of sampled responses; designed for math/reasoning with verifiable rewards
- **DAPO** (ByteDance 2025) — decoupled clipping + dynamic sampling + token-level policy gradient; state-of-the-art on AIME reasoning benchmarks
- **RLVR (RL with Verifiable Rewards)** — use symbolic verifiers (code execution, math grader) instead of a reward model; no reward model error; the paradigm behind DeepSeek-R1
- **Reward model ensembles** — train multiple RMs; use their disagreement to detect out-of-distribution inputs; reduce reward hacking

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | Christiano et al. | Original RLHF paper; the framework |
| [Training language models to follow instructions (InstructGPT)](https://arxiv.org/abs/2203.02155) | 2022 | Ouyang et al. (OpenAI) | RLHF applied to LLMs at scale — the paper behind ChatGPT |
| [Direct Preference Optimization (DPO)](https://arxiv.org/abs/2305.18290) | 2023 | Rafailov et al. | DPO — closed-form alternative that makes RLHF accessible |
| [DeepSeekMath: Pushing the Limits of Mathematical Reasoning (GRPO)](https://arxiv.org/abs/2402.03300) | 2024 | Shao et al. (DeepSeek) | GRPO — the algorithm behind DeepSeek-R1's reasoning |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | RLHF framework — proved humans can supervise complex behavior |
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 2022 | Applied RLHF to LLMs; proved the paradigm works at GPT scale |
| [Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) | 2022 | Replaced human feedback with AI feedback; scaled alignment |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 2023 | Removed the reward model; made RLHF accessible to everyone |

## Current SotA
> *Updated: 2026-05-25*

The frontier has moved to **RLVR** for reasoning tasks: DeepSeek-R1 (2025) uses pure RL with a code/math verifier as reward signal — no human preferences at all — and matches o1 on reasoning benchmarks. DAPO (ByteDance, 2025) achieves 50 points on AIME 2024. For chat/instruction-following, DPO remains the practical default — stable, accessible, no critic required.

## What's happening now
> *Research · Engineering · Systems*

**Research:** Reward model-free alignment is the active frontier: DPO variants (IPO, KTO, SimPO) optimize different behavioral objectives. Process Reward Models (PRMs) provide step-level rewards for reasoning — catching errors in intermediate steps, not just final answers.

**Engineering & Systems:** TRL (Hugging Face) is the production library for DPO/PPO; supports PEFT/LoRA for memory-efficient training on consumer hardware. vLLM powers the rollout generation phase of PPO at scale. Flash Attention + gradient checkpointing are required for training 7B+ models on RLHF.

**Open problems:** How do you prevent reward hacking without a costly KL penalty? Can verifiable rewards fully replace human preferences for open-ended tasks? How do PRMs scale to multi-step tool use and agent tasks?

## In production
> *How top labs and companies have deployed this at scale*

- **OpenAI (InstructGPT → ChatGPT → GPT-4):** Three-stage RLHF pipeline at scale; reward models trained on millions of preference pairs; the paper that started the alignment era. [arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
- **Anthropic (Claude via CAI + RLHF):** Constitutional AI + RLHF; the model critiques and revises its own outputs; reduces need for human preference labeling. [arxiv.org/abs/2212.08073](https://arxiv.org/abs/2212.08073)
- **DeepSeek (R1 via GRPO + RLVR):** Pure RL with verifiable math/code rewards; no human preference data; achieves GPT-4-level reasoning at a fraction of the cost. [arxiv.org/abs/2501.12948](https://arxiv.org/abs/2501.12948)
- **Meta AI (Llama 3 Instruct):** DPO + RLHF hybrid post-training pipeline; DPO for general alignment, RLHF for safety. [arxiv.org/abs/2407.21783](https://arxiv.org/abs/2407.21783)
- **Hugging Face (Open RLHF ecosystem):** TRL library powers accessible DPO/PPO training; >10k repos have used it to fine-tune open models. [huggingface.co/docs/trl](https://huggingface.co/docs/trl)

## Minimum Valuable Build

**What you're building:** A DPO-fine-tuned chat model — take an open-source model (Qwen2.5-1.5B or Llama-3.2-1B), apply DPO on a preference dataset, and compare the aligned vs. unaligned model on helpfulness.

**Why this is valuable:** DPO is now the default alignment method in industry and research. Building it teaches you what alignment actually means mechanically: not injecting values, but making the model prefer responses that look like ones humans chose. You'll see the log-ratio objective in action.

**Stack:**
- **Model:** [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) — small enough for consumer GPU, large enough to be interesting
- **Dataset:** [Anthropic/hh-rlhf](https://huggingface.co/datasets/Anthropic/hh-rlhf) — human preference pairs (chosen vs. rejected responses)
- **Framework:** [HuggingFace TRL](https://huggingface.co/docs/trl) `DPOTrainer` + PEFT/LoRA

**The recipe:**

1. **Load the model and dataset:** `Qwen2.5-1.5B-Instruct` + `hh-rlhf`; format as (prompt, chosen, rejected) triples.
2. **Configure DPO training:** `DPOTrainer(model, ref_model, β=0.1, ...)`; add LoRA adapters (rank 8) to reduce memory to ~6GB VRAM.
3. **Train for 1 epoch:** ~2-4 hours on a T4 GPU (Google Colab free tier). Watch the log-ratio objective decrease.
4. **Evaluate before/after:** On 20 prompts, compare outputs from the base model vs. DPO-aligned model. Use a stronger model (Claude/GPT-4) to judge which is more helpful.

**Expected outcome:** An aligned model you trained yourself, with measurable improvement in helpfulness on held-out prompts, and a concrete understanding of what β (KL weight) does.

**Stretch goals:**
- Train a reward model from scratch on the same dataset; evaluate its accuracy as a preference predictor
- Run GRPO on math problems: use a simple math dataset (GSM8K), use `sympy` to verify answers, use group-relative advantage — no reward model needed
- Compare DPO (β=0.1 vs. β=0.5) — see how the KL weight affects output diversity

## Code & implementations

- [huggingface/trl](https://huggingface.co/docs/trl) — `DPOTrainer`, `PPOTrainer`, `GRPOTrainer`; production RLHF library
- [OpenRLHF](https://github.com/OpenRLHF/OpenRLHF) — scalable PPO RLHF training for 70B+ models with Ray
- [DeepSpeed-Chat](https://github.com/microsoft/DeepSpeed) — Microsoft's three-stage RLHF pipeline at scale

## Connected topics

- [[ppo]] — the RL algorithm used in RLHF Stage 3
- [[dpo]] — the simplified alternative; no RM, no PPO, just supervised learning
- [[grpo]] — Group-Relative Policy Optimization; the algorithm behind DeepSeek-R1
- [[constitutional-ai]] — AI-generated feedback replaces human preference labeling
- [[instruction-tuning]] — the SFT stage that precedes RLHF

## Further reading

- [A General Language Assistant as a Laboratory for Alignment (Anthropic)](https://arxiv.org/abs/2112.00861) — Askell et al. 2021; early alignment research
- [RLHF Book](https://rlhfbook.com) — Nathan Lambert 2024; accessible and comprehensive
- [DAPO: An Open-Source LLM RL Alignment Recipe](https://arxiv.org/abs/2503.14476) — Yu et al. 2025; state-of-the-art reasoning alignment
- [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) — DeepSeek AI 2025
