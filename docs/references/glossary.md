---
title: Glossary
tags: [glossary, definitions, terminology]
updated: 2026-05-25
---

# Glossary
> Key terms used across the Frontier Wiki — defined precisely, not verbosely.

> 🚧 Agent-generated content pending. Core terms seeded manually; agents will expand this glossary as they generate new topic pages.

---

## A

**Attention** — a mechanism that computes a weighted sum over input elements, where weights are learned based on pairwise similarity between query and key vectors. The core mechanism in transformers.

**ELBO** — Evidence Lower Bound. The variational lower bound on log p(x) optimized in VAEs and other variational models. ELBO = E[log p(x|z)] − KL(q(z|x) || p(z)).

**Autoregressive model** — a model that generates sequences by predicting one token at a time, conditioning each prediction on all previous tokens. GPT family models are autoregressive.

## B

**Backpropagation** — the algorithm for computing gradients of a loss function with respect to all parameters in a neural network, using the chain rule of calculus. Makes gradient descent practical.

**Bandwidth (memory bandwidth)** — the rate at which data can be read from or written to memory. The primary bottleneck for many neural network operations on GPUs; see [[roofline-model]].

## C

**Causal masking** — a mask applied to attention weights in decoder-only models (GPT family) that prevents tokens from attending to future positions. Enables autoregressive generation.

**CFG (Classifier-Free Guidance)** — an inference technique for conditional diffusion/flow models that trades sample diversity for quality by interpolating between conditional and unconditional score predictions.

**Confounding** — when a third variable (confounder) causally influences both the treatment and the outcome, creating a spurious correlation. Controlling for confounders is the central challenge of observational causal inference.

## D

**Do-operator / do(X=x)** — Pearl's intervention operator: P(Y | do(X=x)) is the distribution of Y when X is externally set to x, unlike P(Y | X=x) which conditions on observing X=x. The difference captures causation vs. correlation.

**Diffusion model** — a generative model that learns to reverse a gradual noising process. Forward: add Gaussian noise across T steps. Reverse: learn to denoise; at inference, start from noise and iteratively denoise.

## E

**ELBO** — see above.

**Equivariance** — a function f is equivariant to transformation g if f(g(x)) = g(f(x)). In molecular modeling, SE(3)-equivariant networks preserve 3D rotation/translation symmetry.

## F

**Flow matching** — a training framework for continuous normalizing flows that learns a vector field transporting noise to data along simple paths, avoiding expensive simulation during training.

**Fine-tuning** — adapting a pretrained model to a specific task or domain by continuing training on task-specific data. PEFT methods (LoRA, adapters) fine-tune only a small fraction of parameters.

## G

**GAN (Generative Adversarial Network)** — a generative model consisting of a generator and discriminator trained in a minimax game. Generator tries to fool discriminator; discriminator tries to distinguish real from fake.

**Gradient checkpointing** — a memory optimization: recompute activations during the backward pass instead of storing them. Trades compute for memory; essential for training large models.

## H

**HBM (High Bandwidth Memory)** — the main memory on modern GPUs (e.g., H100 has 80GB HBM3). Slower than SRAM (the on-chip cache) but much larger. Most neural network operations are HBM-bandwidth-limited.

## I

**In-context learning** — a model's ability to perform new tasks from a few examples provided in the prompt, without any gradient updates. GPT-3 demonstrated this emerges from scaling.

## K

**KL divergence** — Kullback-Leibler divergence: KL(P || Q) measures how different distribution P is from Q. Always ≥ 0; equals 0 iff P = Q. Used as the regularization term in VAEs and RLHF.

**KV cache** — storing the key and value tensors for all previously generated tokens during autoregressive decoding, avoiding recomputation. Memory grows O(sequence_length × layers).

## L

**LoRA (Low-Rank Adaptation)** — a PEFT method that adds low-rank update matrices ΔW = AB (A ∈ R^{d×r}, B ∈ R^{r×k}, r ≪ d) to pretrained weights. Fine-tunes ~0.1% of parameters with minimal quality loss.

**Loss landscape** — the surface defined by a model's loss as a function of its parameters. Neural networks have complex non-convex landscapes with many saddle points and local minima.

## M

**MDP (Markov Decision Process)** — a mathematical framework for sequential decision-making: states S, actions A, transition function T(s'|s,a), reward function R(s,a). The foundation of RL.

**Mixed precision** — training with FP16 or BF16 for activations/gradients, FP32 for master weights. Reduces memory 2×, increases throughput 2-4× on modern hardware.

## N

**Neural operator** — a neural network that learns maps between function spaces (infinite-dimensional), rather than between finite-dimensional vectors. DeepONet and FNO are neural operators.

**Next-token prediction** — the training objective for autoregressive LLMs: maximize log p(x_t | x_1, ..., x_{t-1}) over all positions t. Simple objective that produces rich representations.

## O

**Off-policy** — in RL, learning a policy from data collected by a different (behavior) policy. Enables learning from historical data without online interaction.

**Operator learning** — learning a map from one function to another (e.g., from initial conditions to PDE solutions), rather than a map from vectors to vectors.

## P

**PEFT (Parameter-Efficient Fine-Tuning)** — methods that fine-tune a small subset of parameters while keeping the pretrained model mostly frozen. Includes LoRA, adapters, prompt tuning.

**Policy gradient** — a class of RL algorithms that directly optimize the policy parameters by estimating the gradient of expected return. REINFORCE, PPO, and GRPO are policy gradient methods.

**PPO (Proximal Policy Optimization)** — a policy gradient algorithm with a clipped surrogate objective that prevents excessively large policy updates. The most widely used RL algorithm; used in RLHF.

## Q

**Quantization** — reducing numerical precision of weights/activations (FP32 → FP16/INT8/INT4) to reduce memory and increase inference throughput. GPTQ, AWQ, and GGUF are quantization methods for LLMs.

## R

**RLHF** — see [[rlhf]].

**RoPE (Rotary Position Embedding)** — encodes position via rotation in the query/key space; enables relative position computation naturally in dot-product attention; default in Llama, Mistral, DeepSeek.

**Roofline model** — a performance model showing whether a computation is memory-bandwidth-limited (memory-bound) or compute-limited (compute-bound) based on arithmetic intensity.

## S

**SCM (Structural Causal Model)** — a triple (V, U, F) of endogenous variables, exogenous noise, and structural equations; Pearl's framework for representing causal relationships.

**Score function** — ∇_x log p(x): the gradient of the log-density with respect to x. Learned by score matching; the basis of score-based and diffusion generative models.

**Self-attention** — attention where queries, keys, and values all come from the same sequence. Every token attends to every other token; the core mechanism of transformers.

**SFT (Supervised Fine-Tuning)** — fine-tuning a pretrained model on labeled demonstration data. Stage 1 of RLHF; also the dominant fine-tuning approach for instruction-following.

**SSM (State Space Model)** — a model of sequential data via a hidden state: h_t = Ah_{t-1} + Bx_t, y_t = Ch_t. Mamba extends SSMs with input-selective (data-dependent) parameters.

## T

**Tensor parallelism** — distributing model weight matrices across GPUs along one dimension; requires all-reduce operations for matrix multiplications. Used in Megatron-LM.

**TF-IDF** — Term Frequency-Inverse Document Frequency: a classical weighting scheme for document retrieval, superseded by dense retrieval (BERT embeddings + FAISS) for most applications.

**Tokenization** — the process of converting text to discrete tokens. BPE (Byte-Pair Encoding) and SentencePiece are dominant methods; typical LLMs use 30k-100k token vocabularies.

## V

**VAE (Variational Autoencoder)** — a generative model combining an encoder q_φ(z|x) and decoder p_θ(x|z) trained via the ELBO. First deep latent variable model to enable smooth interpolation.

**Vector field** — in flow matching: v_θ(x, t) is a function that assigns a velocity vector to every point (x, t); integrating it traces a path from noise to data.

## Z

**ZeRO (Zero Redundancy Optimizer)** — a distributed training technique that partitions optimizer states, gradients, and model parameters across GPUs instead of replicating them. Stage 3 enables trillion-parameter training.

---

> **Agents** expand this glossary automatically when generating new topic pages: any `[[term]]` reference that doesn't exist in this file triggers an agent task to add the definition.
