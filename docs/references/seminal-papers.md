---
title: Seminal Papers
tags: [papers, seminal, test-of-time, reading-list]
updated: 2026-05-25
---

# Seminal Papers
> A curated list of papers that defined their area — filtered for lasting impact, not recency. Signal over noise.

Every paper here answers: *"If someone asks me to read one paper to understand this topic, which one?"*

---

## Foundations

### Optimization & Training
| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [Adam: A Method for Stochastic Optimization](https://arxiv.org/abs/1412.6980) | 2014 | Kingma & Ba | The default optimizer for neural networks |
| [Batch Normalization](https://arxiv.org/abs/1502.03167) | 2015 | Ioffe & Szegedy | Normalized activations → faster, more stable training |
| [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385) | 2015 | He et al. | Skip connections solved vanishing gradients for very deep nets |
| [Dropout](https://jmlr.org/papers/v15/srivastava14a.html) | 2014 | Srivastava et al. | Regularization by randomly dropping activations |

### Scaling
| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) | 2020 | Kaplan et al. | Performance scales predictably with compute, data, and model size |
| [Training Compute-Optimal Large Language Models (Chinchilla)](https://arxiv.org/abs/2203.15556) | 2022 | Hoffmann et al. | Data scales equally with model size; Kaplan was wrong about the ratio |

---

## Architectures

### Transformers
| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [Attention Is All You Need](https://arxiv.org/abs/1706.03762) | 2017 | Vaswani et al. | The transformer — ended the RNN era |
| [BERT](https://arxiv.org/abs/1810.04805) | 2018 | Devlin et al. | Bidirectional pretraining; fine-tuning paradigm |
| [Language Models are Few-Shot Learners (GPT-3)](https://arxiv.org/abs/2005.14165) | 2020 | Brown et al. | Scaling + in-context learning; few-shot without gradient updates |
| [An Image is Worth 16x16 Words (ViT)](https://arxiv.org/abs/2010.11929) | 2020 | Dosovitskiy et al. | Transformer for vision without convolutions |
| [FlashAttention](https://arxiv.org/abs/2205.14135) | 2022 | Dao et al. | IO-aware attention; 2-4× speedup; how attention is actually computed |
| [Mamba: Linear-Time Sequence Modeling](https://arxiv.org/abs/2312.00752) | 2023 | Gu & Dao | Selective state spaces; O(N) alternative to attention |

---

## Generative Models

| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) | 2013 | Kingma & Welling | VAEs — the foundation of deep generative modeling |
| [Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) | 2014 | Goodfellow et al. | GANs — adversarial training; sharp image synthesis |
| [Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) | 2020 | Ho et al. | DDPM — current dominant generative paradigm |
| [Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) | 2020 | Song et al. | Unified SDE framework; connects score matching and diffusion |
| [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) | 2022 | Lipman et al. | Simulation-free CNF training; the emerging standard |
| [Flow Straight and Fast: Rectified Flow](https://arxiv.org/abs/2209.03003) | 2022 | Liu et al. | Straight-line paths; FLUX and SD3 foundation |

---

## Language Models & Alignment

| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [InstructGPT](https://arxiv.org/abs/2203.02155) | 2022 | Ouyang et al. | RLHF applied to LLMs; ChatGPT's origin |
| [Constitutional AI](https://arxiv.org/abs/2212.08073) | 2022 | Bai et al. (Anthropic) | AI feedback replaces human feedback; scales alignment |
| [Direct Preference Optimization](https://arxiv.org/abs/2305.18290) | 2023 | Rafailov et al. | DPO — removed the reward model; the practical alignment standard |
| [Chain-of-Thought Prompting](https://arxiv.org/abs/2201.11903) | 2022 | Wei et al. | Step-by-step reasoning emerges from prompting |
| [DeepSeek-R1](https://arxiv.org/abs/2501.12948) | 2025 | DeepSeek AI | Pure RL produces reasoning comparable to o1 |

---

## Reinforcement Learning

| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [Playing Atari with Deep RL (DQN)](https://arxiv.org/abs/1312.5602) | 2013 | Mnih et al. | Deep RL at scale; experience replay + target networks |
| [Proximal Policy Optimization (PPO)](https://arxiv.org/abs/1707.06347) | 2017 | Schulman et al. | Clipped surrogate objective; the most-used policy gradient algorithm |
| [Mastering the Game of Go with Deep Neural Networks (AlphaGo)](https://www.nature.com/articles/nature16961) | 2016 | Silver et al. | Deep RL + MCTS beats world champion; proved the paradigm |
| [Mastering Chess and Shogi by Self-Play (AlphaZero)](https://arxiv.org/abs/1712.01815) | 2017 | Silver et al. | Tabula rasa RL via self-play; no human knowledge needed |

---

## Causal AI

| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [Causality (book Ch. 1-3)](https://bayes.cs.ucla.edu/BOOK-2K/) | 2009 | Pearl | SCMs, do-calculus, the ladder of causation |
| [Deep RL from Human Preferences](https://arxiv.org/abs/1706.03741) | 2017 | Christiano et al. | Original RLHF; reward learning from pairwise preferences |
| [Double/Debiased Machine Learning](https://arxiv.org/abs/1608.00060) | 2016 | Chernozhukov et al. | Causal inference at scale via cross-fitting |
| [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107) | 2021 | Schölkopf et al. | Bridge between causal theory and deep learning |
| [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893) | 2019 | Arjovsky et al. | Learn features invariant across environments |

---

## Scientific AI

| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561) | 2017 | Raissi et al. | PDEs as loss terms; mesh-free PDE solvers |
| [Fourier Neural Operator](https://arxiv.org/abs/2010.08895) | 2020 | Li et al. | Resolution-invariant operator learning; turbulence simulation |
| [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) | 2021 | Jumper et al. | Solved protein folding; Nobel Prize 2024 |
| [Accurate structure prediction with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) | 2024 | Abramson et al. | Diffusion-based joint structure prediction for all biomolecule types |
| [Learning skillful medium-range weather forecasting (GraphCast)](https://arxiv.org/abs/2212.12794) | 2022 | Lam et al. | GNN-based global weather forecasting in <60 seconds |

---

## Systems

| Paper | Year | Authors | Why it matters |
|---|---|---|---|
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | 2019 | Rajbhandari et al. | Shard optimizer/gradient/parameter across GPUs |
| [Megatron-LM: Training Multi-Billion Parameter Language Models](https://arxiv.org/abs/1909.08053) | 2019 | Shoeybi et al. | Tensor parallelism for LLMs |
| [Efficient Memory Management for LLM Serving with PagedAttention](https://arxiv.org/abs/2309.06180) | 2023 | Kwon et al. | KV cache as virtual memory; continuous batching |

---

> **Maintained by:** Editorial and pedagogical agents. Last updated: 2026-05-25.
> **Source policy:** arXiv, *.edu, official lab publications only.
