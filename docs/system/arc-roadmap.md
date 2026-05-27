---
title: Arc roadmap (research-grounded)
description: Human-curated arc proposals derived from a live 2026 frontier scan of each track. Each arc names a specific frontier destination the reader builds toward — not a reading list of the track's first 5 concepts. Distinct from /system/arc-proposals.md which is the auto-generated cycle-time view.
updated: 2026-05-27
---

# Arc roadmap

> Each arc below was designed against the **current frontier state** of its track (as of 2026-05-27) — sourced from live research, not picked from existing concept ordering. The destination is what makes an arc an arc: the reader can name what they will have built by the end.

This doc is the **source of truth** for which arcs the autonomous loop should spin. The auto-generated `arc-proposals.md` is the per-cycle view; this roadmap is the durable design that the proposer reads from.

## How to read this doc

Each arc has:

- **arc_id** — kebab-case slug used in frontmatter and the arc-step paths.
- **destination** — the named frontier capability at the end. Specific enough to recognise when the reader gets there.
- **diagonal spine** — the 5 steps in order. **Diagonal** means at least one step lives in a different track than the home track — the arc crosses domains.
- **frontier evidence** — what's happening at this frontier right now, with named systems/papers, so the destination isn't aspirational.
- **status** — `ready` (all 5 step slugs exist as substantive concept pages today) or `needs-seeds` (one or more steps need concept pages first).

Cardinality cap per `agents/SCHEMA.md`: ≤5 arcs per track. The proposals below stay at 3 per track to leave headroom for the retro to add more as the field moves.

---

## 01-ai — General AI and agentic systems

### 1. `agentic-rlvr-reasoner` — ready

- **Destination:** A small LLM (1–7B) fine-tuned with RLVR on verifiable math/code rewards, evaluated on a held-out reasoning benchmark — measured to recover the R1-Zero-style behaviour at ≥60% of the published gain.
- **Diagonal spine:** `chain-of-thought` (01) → `in-context-learning` (07) → `reward-modeling` (01) → `rlhf` (01) → `mixture-of-experts` (01)
- **Frontier evidence:** DeepSeek R1-Zero went from 15.6% → 77.9% on AIME 2024 using GRPO + verifiable rewards alone, without an SFT phase. The 2026 standard post-training pipeline is SFT → DPO → GRPO/DAPO. Reasoning models like o3 / o4-mini class are now agentic-RL-trained.
- **Why diagonal:** crosses 01 (LLMs) ↔ 07 (attention / context) — reasoning is RL learning to manipulate context windows.

### 2. `mechanistic-interpretability-with-saes` — needs-seeds

- **Destination:** A working sparse-autoencoder trained on a small open-weight LLM's MLP activations, extracting features that match a published circuit (induction heads / IOI / greater-than) on the same layer.
- **Diagonal spine:** `mechanistic-interpretability` (01) → *sparse-autoencoders* [seed] → *feature-circuits* [seed] → `attention` (07) → `mixture-of-experts` (01)
- **Frontier evidence:** SAEs are now the standard feature-extraction tool. Lorsa (Low-Rank Sparse Attention) at ICLR 2026 extracts attention-superposition features at scale. SAE-based feature steering gives more atomic control than raw activation steering.

### 3. `alignment-via-cot-monitoring` — needs-seeds

- **Destination:** A working CoT monitor that flags deceptive reasoning in a frontier model's outputs, calibrated to OpenAI's published rate on the chain-of-thought-monitoring benchmark.
- **Diagonal spine:** `chain-of-thought` (01) → `alignment-safety` (01) → `mechanistic-interpretability` (01) → *cot-monitoring* [seed] → `reward-modeling` (01)
- **Frontier evidence:** OpenAI's 2025 paper "Monitoring Reasoning Models for Misbehavior and the Risks of Promoting Obfuscation" introduces the technique. Currently active research at OpenAI, Anthropic, GovAI.

---

## 02-generative-modeling

### 1. `generative-stack` — ready

- **Destination:** Five trained generative models (DDPM, score-based, latent diffusion, flow-matching, consistency distillation) compared head-to-head on the same dataset with reported FID, sample diversity, and inference latency.
- **Diagonal spine:** `diffusion-models` → `score-matching` → `latent-diffusion-models` → `flow-matching` → `consistency-models`
- **Frontier evidence:** Consistency models enable single-step or n-step sampling for ~10× speedup. Rectified flow learns the velocity field directly. Modern view: all are special cases of learning flows that transport simple distributions to data.
- **Note:** Intentionally vertical inside 02 — teaches the unified flow-based view. The diagonal arc for generative is `world-models-from-video` below.

### 2. `world-models-from-video` — needs-seeds

- **Destination:** A small video diffusion model conditioned on actions, trained on a driving or game dataset, producing 16-frame rollouts that respect physical conservation on a held-out test set.
- **Diagonal spine:** `diffusion-models` (02) → `flow-matching` (02) → *video-generation* [seed] → `world-models` (06) → *physical-consistency* [seed]
- **Frontier evidence:** OpenAI's Sora positions video generation as world simulation. PhyWorld uses flow matching + DPO on physical principles. HEAT is a trajectory-guided world model for autonomous driving. The standing gap: 10–15% of frontier model outputs violate Newtonian mass conservation.

### 3. `controllable-and-distilled-generation` — needs-seeds

- **Destination:** A finetuned latent diffusion model with ControlNet-style conditioning on edge maps, distilled to single-step sampling at <100 ms per image on an A10 GPU.
- **Diagonal spine:** `latent-diffusion-models` (02) → `consistency-models` (02) → *controlnet* [seed] → *lora-finetuning* [seed] → `variational-autoencoders` (02)
- **Frontier evidence:** Distillation at diffusion + VAE layers yields ~10× speedup with minimal quality loss. Custom conditioning (ControlNet, T2I-Adapter, LoRA) is the production reality at most image-gen labs.

---

## 03-representation-learning

### 1. `self-supervised-vision-foundations` — ready

- **Destination:** A vision encoder trained without labels on a 100k-image dataset (SimCLR + MAE hybrid) that transfers to ImageNet linear-probe ≥75% top-1.
- **Diagonal spine:** `simclr` → `contrastive-learning` → `data-augmentation` → `masked-autoencoders` → `representation-learning`
- **Note:** Intentionally vertical inside 03; the diagonal arc is below.

### 2. `world-model-representations` — needs-seeds

- **Destination:** A JEPA-style world model on a video sequence, where the latent representations support a planner reaching a goal state with measured success rate vs a reactive baseline.
- **Diagonal spine:** `jepa` (03) → `masked-autoencoders` (03) → `world-models` (06) → `model-based-reinforcement-learning` (06) → *latent-dynamics* [seed]
- **Frontier evidence:** V-JEPA 2024, LeCun's prediction that JEPA-style representations are the path to common-sense AI. Dreamer-V3 uses latent imagination for RL planning.

### 3. `multimodal-encoders` — needs-seeds

- **Destination:** A CLIP-class encoder trained on a 1M image-text dataset, evaluated by zero-shot transfer on three downstream tasks (CIFAR-10, food-101, custom domain).
- **Diagonal spine:** `contrastive-learning` (03) → `simclr` (03) → *clip-architecture* [seed] → *vision-language-pretraining* [seed] → `representation-learning` (03)
- **Frontier evidence:** CLIP, OpenCLIP, SigLIP, EVA-CLIP at progressively larger scale.

---

## 04-neural-networks-deep-learning

### 1. `training-fundamentals` — ready

- **Destination:** A from-scratch CNN trained on CIFAR-10 to ≥85% test accuracy with documented loss curves, normalization choices, and a learned schedule — written up like a small lab notebook.
- **Diagonal spine:** `backpropagation` → `gradient-descent` → `adaptive-optimizers` → `regularization` → `batch-normalization`
- **Note:** Vertical foundations arc. Keeps the entry point to the field intact.

### 2. `scaling-and-emergence` — ready

- **Destination:** A small empirical scaling study across 3 model sizes (10M, 100M, 300M params) on a fixed token budget, fitting Chinchilla-style scaling laws and reporting where emergence appears on a target task.
- **Diagonal spine:** `scaling-laws` (04) → `optimization` (04) → `emergence` (10) → `double-descent` (10) → `mixture-of-experts` (01)
- **Frontier evidence:** Chinchilla 2022 reset compute-optimal training. 2026 frontier: how MoE shifts scaling laws; whether reasoning-RL has its own emergence boundary.
- **Why diagonal:** crosses 04 (training) ↔ 10 (complexity / emergence) ↔ 01 (MoE).

### 3. `efficient-large-model-training` — ready

- **Destination:** A 1B-parameter model trained across 8 GPUs with mixed-precision, ZeRO-style sharding, and gradient bucketing — reporting throughput in tokens/sec/GPU and convergence curves vs the single-GPU baseline.
- **Diagonal spine:** `gradient-descent` (04) → `adaptive-optimizers` (04) → `mixed-precision-training` (09) → `data-parallelism` (09) → `tensor-parallelism` (09)
- **Frontier evidence:** ZeRO-3, FSDP, Megatron-DeepSpeed are the current production stacks. NVIDIA H200 / Blackwell hardware is the 2026 reality.

---

## 05-statistical-probabilistic-ml

### 1. `bayesian-deep-learning` — ready

- **Destination:** A Bayesian neural network on a real production-like dataset, reporting calibration error and showing the uncertainty is actionable (predictive entropy correlates with held-out errors).
- **Diagonal spine:** `bayesian-inference` → `variational-inference` → `bayesian-neural-networks` → `uncertainty-quantification` → `gaussian-processes`

### 2. `probabilistic-programming-end-to-end` — ready

- **Destination:** A hierarchical Bayesian model fit in Pyro / NumPyro on a real dataset, with full posterior inference (MCMC or VI), posterior predictive checks, and a credible interval that beats a frequentist baseline.
- **Diagonal spine:** `probabilistic-programming` → `bayesian-inference` → `variational-inference` → `mcmc` → `gaussian-processes`

### 3. `causal-bayesian-inference` — ready

- **Destination:** A Bayesian instrumental-variables model fit on an observational dataset, recovering a causal effect with credible interval, compared against a naive regression baseline.
- **Diagonal spine:** `bayesian-inference` (05) → `variational-inference` (05) → `instrumental-variables` (08) → `causal-discovery` (08) → `counterfactuals` (08)
- **Frontier evidence:** Industry surveys (Gartner) put ~70% of AI-driven orgs projecting Causal AI adoption by 2026.
- **Why diagonal:** crosses 05 (Bayesian) ↔ 08 (causal).

---

## 06-reinforcement-learning

### 1. `rl-for-post-training` — ready

- **Destination:** A 7B open-weight LLM finetuned via GRPO with verifiable rewards on a math/code task, measured to recover ≥60% of DeepSeek R1's published gain on AIME 2024 starting from the same base.
- **Diagonal spine:** `policy-gradient` (06) → `ppo` (06) → `actor-critic` (06) → `reward-modeling` (01) → `rlhf` (01)
- **Frontier evidence:** GRPO (DeepSeek 2025) is the standard. RLVR (Reinforcement Learning with Verifiable Rewards) avoids reward-model training entirely for math/code. DAPO scales it further. Most frontier labs run GRPO/DAPO post-training pipelines as of 2026.
- **Why diagonal:** crosses 06 (RL) ↔ 01 (LLM alignment). **The single most important arc in the wiki right now.**

### 2. `world-models-and-imagination` — ready

- **Destination:** A Dreamer-class agent that learns a latent world model from environment rollouts and plans in imagination, beating a model-free baseline by ≥30% sample efficiency on a control task.
- **Diagonal spine:** `mdp` → `policy-gradient` → `model-based-reinforcement-learning` → `world-models` → `q-learning`
- **Frontier evidence:** Dreamer-V3 (2023–24), GenRL, Reinforcement World Model Learning (RWML 2026). NVIDIA NeMo Gym provides interactive RL environments for training world-model-based agents.

### 3. `agentic-rl-with-tools` — needs-seeds

- **Destination:** A small LLM agent that calls a web-search + a calculator tool, learns from execution outcomes via GRPO, measured on a multi-step QA benchmark vs the no-RL baseline.
- **Diagonal spine:** `ppo` (06) → `reward-modeling` (01) → `rlhf` (01) → *tool-use* [seed] → *multi-turn-rl* [seed]
- **Frontier evidence:** Search-R1, NVIDIA NeMo Gym for tool-using agents, ICLR 2026 paper "Agentic Reinforcement Learning". The hardest open problem in 2026 RL: multi-turn credit assignment.

---

## 07-attention-memory-reasoning-continual

### 1. `long-context-attention` — needs-seeds

- **Destination:** A small Transformer trained with FlashAttention 2 + ring attention reaching 128K context with sub-quadratic memory, evaluated on a long-context retrieval task (needle-in-haystack ≥95%).
- **Diagonal spine:** `attention` (07) → `multi-head-attention` (07) → `long-context` (07) → `flash-attention` (09) → *ring-attention* [seed]
- **Frontier evidence:** Ring attention (Liu 2023), StreamingLLM, YaRN, position interpolation. 1M-context models (GPT-5.4, Claude 3.5 Sonnet, Gemini 1.5 Pro) are 2025-26 production.

### 2. `retrieval-and-memory` — needs-seeds

- **Destination:** A RAG pipeline with hybrid retrieval (BM25 + dense embeddings) feeding a long-context LLM, evaluated on a domain-specific QA benchmark — measured against pure-long-context and pure-retrieval baselines.
- **Diagonal spine:** `retrieval-augmented-generation` (07) → `in-context-learning` (07) → `long-context` (07) → *vector-search* [seed] → *embedding-models* [seed]
- **Frontier evidence:** Production RAG is the dominant LLM deployment pattern. ColBERT, splade, vector DBs at scale.

### 3. `reasoning-and-test-time-compute` — needs-seeds

- **Destination:** A small reasoning model fine-tuned with CoT + self-verification, evaluated on MATH-500 — comparing test-time compute regimes (greedy / beam / majority vote / R1-style self-verify).
- **Diagonal spine:** `chain-of-thought` (01) → `in-context-learning` (07) → `attention` (07) → *self-verification* [seed] → *test-time-compute* [seed]
- **Frontier evidence:** o3, o4-mini, R1, R1-Zero. "Test-time reasoning and the rise of reflective agents" is the 2026 trend.

---

## 08-causal-statistical-inference

### 1. `causal-deep-learning` — ready

- **Destination:** A neural net trained to predict counterfactual outcomes under a hidden confounder, recovering the true treatment effect within 10% on a semi-synthetic dataset.
- **Diagonal spine:** `structural-causal-models` → `do-calculus` → `counterfactuals` → `causal-representation-learning` → `potential-outcomes`
- **Frontier evidence:** NeurIPS 2023+ identifiability papers, 2026 work showing causal representations can be learned with logarithmic intervention budgets.

### 2. `causal-rl` — needs-seeds

- **Destination:** An RL agent in a structural causal environment that learns to intervene (not just observe) — measured by causal effect recovery vs an observational baseline.
- **Diagonal spine:** `counterfactuals` (08) → `do-calculus` (08) → `policy-gradient` (06) → `world-models` (06) → *causal-intervention* [seed]
- **Frontier evidence:** Causal RL is the named frontier intersection in `agents/skills/arc-anatomy.md`. Pearl + Schölkopf 2022 vision. Specific 2025 work: Causal-MuZero, Counterfactual RL.

### 3. `causal-discovery-in-practice` — ready

- **Destination:** Run NOTEARS / PC algorithm on a real observational dataset (e.g. gene expression), recover a causal graph, validate against held-out interventions, report identifiability limits.
- **Diagonal spine:** `causal-discovery` → `structural-causal-models` → `instrumental-variables` → `mediation-analysis` → `potential-outcomes`
- **Frontier evidence:** NOTEARS (Zheng 2018), DAGMA, CausalDisco at scale. Production deployments in biology, epidemiology.

---

## 09-algorithms-systems-for-ai

### 1. `serve-an-llm-efficiently` — ready

- **Destination:** A quantized 7B model served behind an endpoint with measured p95 latency under 100 ms, with kv-cache + FlashAttention + INT8 quantization wired and benchmarked at batch sizes 1, 4, 16.
- **Diagonal spine:** `flash-attention` → `kv-cache` → `kv-cache-management` → `quantization` → `llm-inference`
- **Frontier evidence:** vLLM, TGI, SGLang are the 2026 inference stacks. INT8/FP8 quantization is production-default. p95 latency is the KPI most teams optimise.

### 2. `train-at-scale` — ready

- **Destination:** Distributed training of a 1B-parameter model across 8 GPUs with mixed precision, ZeRO-3 sharding, and pipeline parallelism — reporting tokens/sec/GPU vs the single-GPU baseline and identifying the dominant cost (compute vs communication).
- **Diagonal spine:** `distributed-training` → `data-parallelism` → `tensor-parallelism` → `pipeline-parallelism` → `mixed-precision-training`
- **Frontier evidence:** ZeRO-3 (DeepSpeed), FSDP (PyTorch), Megatron-LM. Frontier labs running 100K-GPU clusters; same primitives in miniature.

### 3. `compiler-and-kernel-fusion` — needs-seeds

- **Destination:** Take a transformer block, profile it, identify the bottleneck, write a fused kernel (Triton or CUDA) for it, and measure the speedup against the PyTorch eager baseline.
- **Diagonal spine:** `flash-attention` (09) → `compiler-optimizations-for-ml` (09) → `automatic-differentiation` (09) → *triton-kernels* [seed] → `tensor-cores` (09)
- **Frontier evidence:** Triton (OpenAI), torch.compile, FlashAttention 3 (Tri Dao 2024).

---

## 10-complexity-cognition-natural-intelligence

### 1. `scaling-laws-empirical` — ready

- **Destination:** Empirically fit Chinchilla-style scaling laws on a 3-point series of model sizes (10M / 100M / 300M) on a fixed dataset, reporting the compute-optimal ratio you recover vs the published one.
- **Diagonal spine:** `scaling-laws` (04) → `emergence` (10) → `double-descent` (10) → `generalization` (10) → `scaling-collapse` (10)
- **Frontier evidence:** Chinchilla (Hoffmann 2022), Kaplan (2020). 2026 question: do reasoning-RL gains have their own scaling law?

### 2. `emergence-and-double-descent` — ready

- **Destination:** An empirical demo of double descent on a small classifier (vary model width across the bias-variance frontier), AND show one emergence-style task where capability appears suddenly with scale.
- **Diagonal spine:** `double-descent` (10) → `generalization` (10) → `scaling-laws` (04) → `emergence` (10) → `compositionality` (10)
- **Frontier evidence:** Belkin et al. (2019), Schaeffer et al. (2023) on emergence-as-measurement-artefact, Wei et al. on capability emergence.

### 3. `compositionality-and-generalization` — needs-seeds

- **Destination:** Train a small Transformer on a compositional task (SCAN, COGS, or CFQ), measure systematic generalization to held-out compositions — vs a recurrent baseline.
- **Diagonal spine:** `compositionality` (10) → `generalization` (10) → *systematic-generalization* [seed] → `attention` (07) → `in-context-learning` (07)
- **Frontier evidence:** SCAN (Lake & Baroni 2018), COGS (Kim & Linzen 2020), CFQ.

---

## Summary

| Track | Ready arcs | Needs-seeds arcs | Total |
|---|---|---|---|
| 01-ai | 1 | 2 | 3 |
| 02-generative-modeling | 1 | 2 | 3 |
| 03-representation-learning | 1 | 2 | 3 |
| 04-neural-networks-deep-learning | 3 | 0 | 3 |
| 05-statistical-probabilistic-ml | 3 | 0 | 3 |
| 06-reinforcement-learning | 2 | 1 | 3 |
| 07-attention-memory-reasoning-continual | 0 | 3 | 3 |
| 08-causal-statistical-inference | 2 | 1 | 3 |
| 09-algorithms-systems-for-ai | 2 | 1 | 3 |
| 10-complexity-cognition-natural-intelligence | 2 | 1 | 3 |
| **Total** | **17** | **13** | **30** |

**17 arcs are ready to spin today** with the existing concept corpus. The other 13 need 1–2 seed concepts each — the retrospective auto-seeder picks those up naturally over the next 2–3 cycles, then those arcs unlock too.

## What changes next (code)

The autonomous arc-proposer should:

1. **Read this roadmap as the canonical seed list** — replaces the generic `<track>-foundations` placeholders that the old `_suggest_track_arcs` emitted.
2. **Mirror the research methodology** demonstrated above (Exa scan → 3 distinct destinations per track → diagonal spine → frontier evidence) when proposing additions.
3. **Refresh this roadmap each cycle** as the frontier moves (e.g. when GRPO is superseded, swap `rl-for-post-training`'s destination).

For now, this doc is the source of truth. The next code change replaces `_suggest_track_arcs` to parse this file rather than emit hardcoded canonicals.

---

*Frontier sources used in this 2026-05-27 scan:*

- [Post-Training in 2026: GRPO, DAPO, RLVR & Beyond — llm-stats](https://llm-stats.com/blog/research/post-training-techniques-2026)
- [RL Posttraining for Tool-Using Agents — Zylos](https://zylos.ai/research/2026-04-10-rl-posttraining-tool-using-agents-grpo-async-rl)
- [Agentic Reinforcement Learning — ICLR 2026](https://openreview.net/pdf/a1e5782d4488277b71a1a194ebecc820d981e887.pdf)
- [Video generation models as world simulators — OpenAI](https://openai.com/index/video-generation-models-as-world-simulators/)
- [Mechanistic Interpretability — ICLR 2026 paper](https://arxiv.org/pdf/2510.02917)
- [AI Reasoning Models 2026 — Zylos](https://zylos.ai/research/2026-01-24-ai-reasoning-models)
- [Causal AI: Current State-of-the-Art — Alex G. Lee](https://medium.com/@alexglee/causal-ai-current-state-of-the-art-future-directions-c17ad57ff879)
