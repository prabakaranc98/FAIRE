---
title: Arcs — the wiki's USP
description: Every arc is a diagonal path from a tool you already touch to a named frontier capability you can build. Curated against the 2026 frontier; backed by the arc-roadmap design doc.
---

# Arcs

> An arc is a **diagonal** learning path: from a tool you already touch, through a broader frame, to a synthesised capability, landing at the intersection of two active research areas. Each arc names a **specific frontier destination** you build toward. The MVB at each step is the recipe; the arc is the journey.

**0 live · 17 ready to spin · 13 need concept seeds · 30 total**

See the [arc roadmap](../system/arc-roadmap.md) for the full design rationale and frontier evidence behind each one. The autonomous loop spins the `ready` ones; the `needs-seeds` ones unlock as the retro auto-seeds the missing concepts.

---

## AI

### **`agentic-rlvr-reasoner`**

🟡 ready to spin · **track:** [AI](core/01-ai/index.md)

**Destination —** A small LLM (1–7B) fine-tuned with RLVR on verifiable math/code rewards, evaluated on a held-out reasoning benchmark — measured to recover the R1-Zero-style behaviour at ≥60% of the published gain

`chain-of-thought` → `in-context-learning` → `reward-modeling` → `rlhf` → `mixture-of-experts`


### **`mechanistic-interpretability-with-saes`**

🟠 needs seeds · **track:** [AI](core/01-ai/index.md)

**Destination —** A working sparse-autoencoder trained on a small open-weight LLM's MLP activations, extracting features that match a published circuit (induction heads / IOI / greater-than) on the same layer

`mechanistic-interpretability` → `sparse-autoencoders` → `feature-circuits` → `attention` → `mixture-of-experts`


### **`alignment-via-cot-monitoring`**

🟠 needs seeds · **track:** [AI](core/01-ai/index.md)

**Destination —** A working CoT monitor that flags deceptive reasoning in a frontier model's outputs, calibrated to OpenAI's published rate on the chain-of-thought-monitoring benchmark

`chain-of-thought` → `alignment-safety` → `mechanistic-interpretability` → `cot-monitoring` → `reward-modeling`


---

## Generative Modeling

### **`generative-stack`**

🟡 ready to spin · **track:** [Generative Modeling](core/02-generative-modeling/index.md)

**Destination —** Five trained generative models (DDPM, score-based, latent diffusion, flow-matching, consistency distillation) compared head-to-head on the same dataset with reported FID, sample diversity, and inference latency

`diffusion-models` → `score-matching` → `latent-diffusion-models` → `flow-matching` → `consistency-models`


### **`world-models-from-video`**

🟠 needs seeds · **track:** [Generative Modeling](core/02-generative-modeling/index.md)

**Destination —** A small video diffusion model conditioned on actions, trained on a driving or game dataset, producing 16-frame rollouts that respect physical conservation on a held-out test set

`diffusion-models` → `flow-matching` → `video-generation` → `world-models` → `physical-consistency`


### **`controllable-and-distilled-generation`**

🟠 needs seeds · **track:** [Generative Modeling](core/02-generative-modeling/index.md)

**Destination —** A finetuned latent diffusion model with ControlNet-style conditioning on edge maps, distilled to single-step sampling at <100 ms per image on an A10 GPU

`latent-diffusion-models` → `consistency-models` → `controlnet` → `lora-finetuning` → `variational-autoencoders`


---

## Representation Learning

### **`self-supervised-vision-foundations`**

🟡 ready to spin · **track:** [Representation Learning](core/03-representation-learning/index.md)

**Destination —** A vision encoder trained without labels on a 100k-image dataset (SimCLR + MAE hybrid) that transfers to ImageNet linear-probe ≥75% top-1

`simclr` → `contrastive-learning` → `data-augmentation` → `masked-autoencoders` → `representation-learning`


### **`world-model-representations`**

🟠 needs seeds · **track:** [Representation Learning](core/03-representation-learning/index.md)

**Destination —** A JEPA-style world model on a video sequence, where the latent representations support a planner reaching a goal state with measured success rate vs a reactive baseline

`jepa` → `masked-autoencoders` → `world-models` → `model-based-reinforcement-learning` → `latent-dynamics`


### **`multimodal-encoders`**

🟠 needs seeds · **track:** [Representation Learning](core/03-representation-learning/index.md)

**Destination —** A CLIP-class encoder trained on a 1M image-text dataset, evaluated by zero-shot transfer on three downstream tasks (CIFAR-10, food-101, custom domain)

`contrastive-learning` → `simclr` → `clip-architecture` → `vision-language-pretraining` → `representation-learning`


---

## Neural Networks & Deep Learning

### **`training-fundamentals`**

🟡 ready to spin · **track:** [Neural Networks & Deep Learning](core/04-neural-networks-deep-learning/index.md)

**Destination —** A from-scratch CNN trained on CIFAR-10 to ≥85% test accuracy with documented loss curves, normalization choices, and a learned schedule — written up like a small lab notebook

`backpropagation` → `gradient-descent` → `adaptive-optimizers` → `regularization` → `batch-normalization`


### **`scaling-and-emergence`**

🟡 ready to spin · **track:** [Neural Networks & Deep Learning](core/04-neural-networks-deep-learning/index.md)

**Destination —** A small empirical scaling study across 3 model sizes (10M, 100M, 300M params) on a fixed token budget, fitting Chinchilla-style scaling laws and reporting where emergence appears on a target task

`scaling-laws` → `optimization` → `emergence` → `double-descent` → `mixture-of-experts`


### **`efficient-large-model-training`**

🟡 ready to spin · **track:** [Neural Networks & Deep Learning](core/04-neural-networks-deep-learning/index.md)

**Destination —** A 1B-parameter model trained across 8 GPUs with mixed-precision, ZeRO-style sharding, and gradient bucketing — reporting throughput in tokens/sec/GPU and convergence curves vs the single-GPU baseline

`gradient-descent` → `adaptive-optimizers` → `mixed-precision-training` → `data-parallelism` → `tensor-parallelism`


---

## Statistical & Probabilistic ML

### **`bayesian-deep-learning`**

🟡 ready to spin · **track:** [Statistical & Probabilistic ML](core/05-statistical-probabilistic-ml/index.md)

**Destination —** A Bayesian neural network on a real production-like dataset, reporting calibration error and showing the uncertainty is actionable (predictive entropy correlates with held-out errors)

`bayesian-inference` → `variational-inference` → `bayesian-neural-networks` → `uncertainty-quantification` → `gaussian-processes`


### **`probabilistic-programming-end-to-end`**

🟡 ready to spin · **track:** [Statistical & Probabilistic ML](core/05-statistical-probabilistic-ml/index.md)

**Destination —** A hierarchical Bayesian model fit in Pyro / NumPyro on a real dataset, with full posterior inference (MCMC or VI), posterior predictive checks, and a credible interval that beats a frequentist baseline

`probabilistic-programming` → `bayesian-inference` → `variational-inference` → `mcmc` → `gaussian-processes`


### **`causal-bayesian-inference`**

🟡 ready to spin · **track:** [Statistical & Probabilistic ML](core/05-statistical-probabilistic-ml/index.md)

**Destination —** A Bayesian instrumental-variables model fit on an observational dataset, recovering a causal effect with credible interval, compared against a naive regression baseline

`bayesian-inference` → `variational-inference` → `instrumental-variables` → `causal-discovery` → `counterfactuals`


---

## Reinforcement Learning

### **`rl-for-post-training`**

🟡 ready to spin · **track:** [Reinforcement Learning](core/06-reinforcement-learning/index.md)

**Destination —** A 7B open-weight LLM finetuned via GRPO with verifiable rewards on a math/code task, measured to recover ≥60% of DeepSeek R1's published gain on AIME 2024 starting from the same base

`policy-gradient` → `ppo` → `actor-critic` → `reward-modeling` → `rlhf`


### **`world-models-and-imagination`**

🟡 ready to spin · **track:** [Reinforcement Learning](core/06-reinforcement-learning/index.md)

**Destination —** A Dreamer-class agent that learns a latent world model from environment rollouts and plans in imagination, beating a model-free baseline by ≥30% sample efficiency on a control task

`mdp` → `policy-gradient` → `model-based-reinforcement-learning` → `world-models` → `q-learning`


### **`agentic-rl-with-tools`**

🟠 needs seeds · **track:** [Reinforcement Learning](core/06-reinforcement-learning/index.md)

**Destination —** A small LLM agent that calls a web-search + a calculator tool, learns from execution outcomes via GRPO, measured on a multi-step QA benchmark vs the no-RL baseline

`ppo` → `reward-modeling` → `rlhf` → `tool-use` → `multi-turn-rl`


---

## Attention, Memory, Reasoning, Continual

### **`long-context-attention`**

🟠 needs seeds · **track:** [Attention, Memory, Reasoning, Continual](core/07-attention-memory-reasoning-continual/index.md)

**Destination —** A small Transformer trained with FlashAttention 2 + ring attention reaching 128K context with sub-quadratic memory, evaluated on a long-context retrieval task (needle-in-haystack ≥95%)

`attention` → `multi-head-attention` → `long-context` → `flash-attention` → `ring-attention`


### **`retrieval-and-memory`**

🟠 needs seeds · **track:** [Attention, Memory, Reasoning, Continual](core/07-attention-memory-reasoning-continual/index.md)

**Destination —** A RAG pipeline with hybrid retrieval (BM25 + dense embeddings) feeding a long-context LLM, evaluated on a domain-specific QA benchmark — measured against pure-long-context and pure-retrieval baselines

`retrieval-augmented-generation` → `in-context-learning` → `long-context` → `vector-search` → `embedding-models`


### **`reasoning-and-test-time-compute`**

🟠 needs seeds · **track:** [Attention, Memory, Reasoning, Continual](core/07-attention-memory-reasoning-continual/index.md)

**Destination —** A small reasoning model fine-tuned with CoT + self-verification, evaluated on MATH-500 — comparing test-time compute regimes (greedy / beam / majority vote / R1-style self-verify)

`chain-of-thought` → `in-context-learning` → `attention` → `self-verification` → `test-time-compute`


---

## Causal & Statistical Inference

### **`causal-deep-learning`**

🟡 ready to spin · **track:** [Causal & Statistical Inference](core/08-causal-statistical-inference/index.md)

**Destination —** A neural net trained to predict counterfactual outcomes under a hidden confounder, recovering the true treatment effect within 10% on a semi-synthetic dataset

`structural-causal-models` → `do-calculus` → `counterfactuals` → `causal-representation-learning` → `potential-outcomes`


### **`causal-rl`**

🟠 needs seeds · **track:** [Causal & Statistical Inference](core/08-causal-statistical-inference/index.md)

**Destination —** An RL agent in a structural causal environment that learns to intervene (not just observe) — measured by causal effect recovery vs an observational baseline

`counterfactuals` → `do-calculus` → `policy-gradient` → `world-models` → `causal-intervention`


### **`causal-discovery-in-practice`**

🟡 ready to spin · **track:** [Causal & Statistical Inference](core/08-causal-statistical-inference/index.md)

**Destination —** Run NOTEARS / PC algorithm on a real observational dataset (e.g. gene expression), recover a causal graph, validate against held-out interventions, report identifiability limits

`causal-discovery` → `structural-causal-models` → `instrumental-variables` → `mediation-analysis` → `potential-outcomes`


---

## Algorithms & Systems for AI

### **`serve-an-llm-efficiently`**

🟡 ready to spin · **track:** [Algorithms & Systems for AI](core/09-algorithms-systems-for-ai/index.md)

**Destination —** A quantized 7B model served behind an endpoint with measured p95 latency under 100 ms, with kv-cache + FlashAttention + INT8 quantization wired and benchmarked at batch sizes 1, 4, 16

`flash-attention` → `kv-cache` → `kv-cache-management` → `quantization` → `llm-inference`


### **`train-at-scale`**

🟡 ready to spin · **track:** [Algorithms & Systems for AI](core/09-algorithms-systems-for-ai/index.md)

**Destination —** Distributed training of a 1B-parameter model across 8 GPUs with mixed precision, ZeRO-3 sharding, and pipeline parallelism — reporting tokens/sec/GPU vs the single-GPU baseline and identifying the dominant cost (compute vs communication)

`distributed-training` → `data-parallelism` → `tensor-parallelism` → `pipeline-parallelism` → `mixed-precision-training`


### **`compiler-and-kernel-fusion`**

🟠 needs seeds · **track:** [Algorithms & Systems for AI](core/09-algorithms-systems-for-ai/index.md)

**Destination —** Take a transformer block, profile it, identify the bottleneck, write a fused kernel (Triton or CUDA) for it, and measure the speedup against the PyTorch eager baseline

`flash-attention` → `compiler-optimizations-for-ml` → `automatic-differentiation` → `triton-kernels` → `tensor-cores`


---

## Complexity, Cognition & Natural Intelligence

### **`scaling-laws-empirical`**

🟡 ready to spin · **track:** [Complexity, Cognition & Natural Intelligence](core/10-complexity-cognition-natural-intelligence/index.md)

**Destination —** Empirically fit Chinchilla-style scaling laws on a 3-point series of model sizes (10M / 100M / 300M) on a fixed dataset, reporting the compute-optimal ratio you recover vs the published one

`scaling-laws` → `emergence` → `double-descent` → `generalization` → `scaling-collapse`


### **`emergence-and-double-descent`**

🟡 ready to spin · **track:** [Complexity, Cognition & Natural Intelligence](core/10-complexity-cognition-natural-intelligence/index.md)

**Destination —** An empirical demo of double descent on a small classifier (vary model width across the bias-variance frontier), AND show one emergence-style task where capability appears suddenly with scale

`double-descent` → `generalization` → `scaling-laws` → `emergence` → `compositionality`


### **`compositionality-and-generalization`**

🟠 needs seeds · **track:** [Complexity, Cognition & Natural Intelligence](core/10-complexity-cognition-natural-intelligence/index.md)

**Destination —** Train a small Transformer on a compositional task (SCAN, COGS, or CFQ), measure systematic generalization to held-out compositions — vs a recurrent baseline

`compositionality` → `generalization` → `systematic-generalization` → `attention` → `in-context-learning`


---

*This page is auto-rebuilt from `docs/system/arc-roadmap.md` by `scripts/build_arc_catalog.py`. Refreshed each cycle.*
