---
title: "Arc: Language Models — Pretraining to Frontier LLMs"
arc: language-models
super_domain: D-Perception
tracks: [07-attention-memory-reasoning, 06-reinforcement-learning, 04-neural-networks-dl]
estimated_depth: "7-9 weeks, ~28 papers"
prereqs: [transformer, self-supervised-learning, backpropagation]
---

# Arc: Language Models
> **What this arc builds:** How large language models are built, trained, and aligned — from the masked language modeling objective through RLHF to the current frontier of reasoning models and multimodal systems.

## Why this arc exists

Language models are the current center of gravity for frontier AI. But "LLM" is not a single thing — it's a product of at least three distinct training stages: pretraining (SSL on tokens), instruction tuning (SFT on demonstrations), and alignment (RLHF, DPO, or GRPO). Each stage requires different understanding.

This arc makes each stage legible — from why next-token prediction produces useful representations, to how scaling changes what emerges, to how preference learning shapes behavior, to the current frontier of reasoning via RL.

## Prerequisites

Understand the transformer architecture (or complete the MLP → Transformer arc first). Familiarity with self-supervised learning and basic probability is helpful.

## The sequence

**Pretraining**

1. **Language modeling as next-token prediction** (foundational) — causal LM; softmax over vocabulary; why this objective produces useful representations.
2. **BERT: masked language modeling** (applied) — bidirectional encoder; MLM + NSP; fine-tuning for classification. [→](https://arxiv.org/abs/1810.04805)
3. **GPT series: decoder-only autoregression** (applied) — GPT-2 as unsupervised multitask learner; GPT-3 as few-shot learner.
4. **Scaling laws** (theoretical) — Kaplan 2020 (model-size dominant); Chinchilla 2022 (data matters equally). [→](../../curriculum/04-neural-networks-dl/scaling-laws.md)
5. **Data quality & the Pile** (applied) — what's in the training data; deduplication; filtering.
6. **Emergent capabilities** (theoretical) — discontinuous ability gains at scale; the debate. [→](../../curriculum/10-complexity-cognition/emergent-capabilities.md)
7. **Efficient transformers for LLMs** (applied) — RoPE, RMSNorm, SwiGLU, GQA; modern architecture choices.

**Instruction Tuning & Alignment**

8. **Instruction following & SFT** (applied) — fine-tuning on demonstration data; FLAN, T5, InstructGPT.
9. **RLHF** (applied) — reward model from human preferences; PPO + KL penalty; InstructGPT. [→](../../curriculum/06-reinforcement-learning/rlhf.md)
10. **Constitutional AI** (applied) — AI feedback replaces human feedback; self-critique for alignment. [→](https://arxiv.org/abs/2212.08073)
11. **DPO** (applied) — closed-form preference optimization; NeurIPS 2023 Outstanding Paper. [→](https://arxiv.org/abs/2305.18290)
12. **GRPO & RLVR** (frontier) — RL with verifiable rewards; group-relative advantage. [→](https://arxiv.org/abs/2402.03300)

**Reasoning & Frontier**

13. **Chain-of-thought prompting** (applied) — eliciting step-by-step reasoning; few-shot CoT. [→](https://arxiv.org/abs/2201.11903)
14. **Inference-time scaling** (frontier) — best-of-N, MCTS, process reward models; more compute → better answers.
15. **DeepSeek-R1** (frontier) — pure RL produces reasoning traces; no human reasoning labels needed. [→](https://arxiv.org/abs/2501.12948)
16. **Long context & memory** (applied) — extending context: RoPE scaling, Titans, RAG. [→](../../curriculum/07-attention-memory-reasoning/state-space-models.md)

**Multimodal**

17. **CLIP & contrastive vision-language** (applied) — joint embedding of images and text. [→](../../curriculum/07-attention-memory-reasoning/vision-language-models.md)
18. **LLaVA & visual instruction tuning** (applied) — connect vision encoder to LLM; visual QA.
19. **Multimodal reasoning** (frontier) — GPT-4V, Gemini, Claude; current capabilities and limits.

## Key figures

- **Alec Radford** — GPT series
- **Jacob Devlin** — BERT
- **Tom Brown** (OpenAI) — GPT-3
- **John Schulman** — PPO, RLHF (InstructGPT)
- **Rafael Rafailov** — DPO
- **DeepSeek AI** — R1, GRPO
- **Jason Wei** — Chain-of-thought prompting

## Essential reading sequence

1. [BERT](https://arxiv.org/abs/1810.04805) — Devlin et al. 2018 — masked LM
2. [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) — Brown et al. 2020
3. [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — Kaplan et al. 2020
4. [Training Compute-Optimal LLMs (Chinchilla)](https://arxiv.org/abs/2203.15556) — Hoffmann et al. 2022
5. [InstructGPT](https://arxiv.org/abs/2203.02155) — Ouyang et al. 2022 — RLHF for LLMs
6. [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — Rafailov et al. 2023
7. [Chain-of-Thought Prompting Elicits Reasoning](https://arxiv.org/abs/2201.11903) — Wei et al. 2022
8. [DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via RL](https://arxiv.org/abs/2501.12948) — DeepSeek AI 2025

## Current frontier anchors
> As of 2026-05-25

- **DeepSeek-R1** — pure RL training produces reasoning ability comparable to o1 without human traces
- **DAPO** — 50 pts on AIME 2024; state-of-the-art on hard math reasoning benchmarks
- **Inference-time scaling** — compute at inference via MCTS, process reward models shows diminishing returns
- **GPT-4o, Claude 3.5, Gemini 1.5 Ultra** — multimodal frontier systems

## What you'll know when done

1. Explain what GPT-3's "few-shot learning" actually is technically (it's not gradient updates)
2. Walk through the three stages of InstructGPT's training pipeline
3. Explain the mathematical relationship between DPO and the reward model PPO is optimizing
4. Describe what makes GRPO different from PPO and why it scales better for reasoning tasks
5. Explain the key open question: whether LLMs can "reason" or are retrieving patterns

## Branch points to other arcs

- **→ RL arc**: Post-training algorithms (RLHF, GRPO, DPO) are detailed in the RL arc
- **→ Systems for Scale arc**: Serving LLMs efficiently; KV cache; speculative decoding
- **→ Mechanistic Interpretability** (future arc): What LLMs actually compute internally

## Where to go next

[RL arc →](../reinforcement-learning/index.md) — Post-training alignment in depth

[Systems for Scale arc →](../systems-for-scale/index.md) — Efficient training and inference
