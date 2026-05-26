```yaml
---
title: State Space Models
track: 07-attention-memory-reasoning
tags: [state space models, SSM, sequence modeling, time series, Mamba]
depth: foundational
prereqs: [recurrent-neural-networks, transformers, time-series-analysis]
updated: 2024-10-26
has_mvb: false
---
# State Space Models
> **TL;DR:** State Space Models (SSMs) offer a computationally efficient alternative to Transformers for modeling sequential data, particularly excelling in tasks with long-range dependencies.

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [In production](#in-production) | Understand where SSMs are used |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

## What it is

Imagine you're trying to predict the stock market, or the next word in a sentence. Traditional methods struggle with the long-range dependencies inherent in these tasks. State space models offer a fresh approach, drawing on control theory to model sequential data more effectively. They represent a system's "state" and how it changes over time, allowing for more efficient processing of long sequences.

State space models (SSMs) are a class of models that describe the evolution of a system's internal state over time, based on inputs and previous states. Unlike recurrent neural networks (RNNs) or Transformers, SSMs often exhibit linear computational complexity with respect to sequence length, making them particularly appealing for handling long sequences. This efficiency stems from their ability to represent the entire history of a sequence in a compressed state vector, which is updated iteratively.

SSMs have seen a resurgence in deep learning, offering a compelling alternative to attention-based models for sequence modeling. They provide a structured way to capture temporal dependencies and have shown promising results in various applications, including language modeling, time series forecasting, and audio processing. Their ability to handle long sequences efficiently makes them a valuable tool in scenarios where Transformers become computationally prohibitive.

## Why it matters at the frontier

State space models are gaining traction at the frontier due to their potential to overcome the limitations of Transformers, especially in handling long-range dependencies and reducing computational costs. Research labs are actively exploring novel SSM architectures and training techniques to improve their performance and scalability. The ability to process long sequences efficiently opens up new possibilities for modeling complex systems and making accurate predictions in various domains.

One of the key open problems in the field is how to design SSM architectures that dynamically adapt their internal representations to the specific characteristics of different input sequences, optimizing both performance and computational efficiency across a wide range of tasks. Addressing this challenge could lead to more robust and versatile SSMs that can effectively handle diverse sequential data. Further research is also focused on improving the interpretability of SSMs, making them more transparent and understandable for practitioners.

## Core concepts

- **State Vector** — A representation of the system's internal state at a given time, encapsulating relevant information about its past and present.
- **State Transition Matrix** — A matrix that defines how the state vector evolves from one time step to the next, based on the current state and input.
- **Input Matrix** — A matrix that maps the input to the state vector, determining how the input influences the system's internal state.
- **Output Matrix** — A matrix that maps the state vector to the output, determining how the system's internal state is translated into observable outputs.
- **Feedforward Matrix** — A matrix that directly maps the input to the output, bypassing the state vector.
- **Impulse Response** — The system's output when presented with a brief input signal (an impulse), characterizing the system's dynamic behavior.
- **Linear Time-Invariance (LTI)** — A property of systems where the system's response to an input does not depend on when the input was applied.

## Mathematical foundations

State space models can be described by a set of equations that define the evolution of the system's state and output over time. In continuous time, the state-space representation is given by:

\[
\dot{x}(t) = A x(t) + B u(t)
\]

where \(\dot{x}(t)\) is the time derivative of the state vector \(x(t)\), \(A\) is the state matrix, \(B\) is the input matrix, and \(u(t)\) is the input. This equation describes how the internal state evolves over time.

\[
y(t) = C x(t) + D u(t)
\]

where \(y(t)\) is the output, \(C\) is the output matrix, \(x(t)\) is the state vector, \(D\) is the feedforward matrix, and \(u(t)\) is the input. This equation describes how the output is generated from the state and input.

The impulse response \(H\) of the system is given by:

\[
H = CA + D
\]

where \(H\) is the impulse response, \(C\) is the output matrix, \(A\) is the state matrix, and \(D\) is the feedforward matrix. The impulse response is a key property in understanding the system's behavior.

In discrete time, the state-space representation is given by:

\[
x_{k+1} = A x_k + B u_k
\]

where \(x_{k+1}\) is the state at the next time step, \(x_k\) is the state at the current time step, \(A\) is the state matrix, \(B\) is the input matrix, and \(u_k\) is the input at the current time step. This equation describes how the state evolves in discrete time steps.

\[
y_k = C x_k + D u_k
\]

where \(y_k\) is the output at the current time step, \(C\) is the output matrix, \(x_k\) is the state at the current time step, \(D\) is the feedforward matrix, and \(u_k\) is the input at the current time step. This equation describes how the output is generated in discrete time steps.

## Key algorithms / techniques

- **Mamba (2023)** — A selective state space model that dynamically adjusts its internal state based on the input, improving performance and efficiency for long sequences.
- **Structured State Space Sequence (S4) (2021)** — A class of SSMs designed to efficiently model long-range dependencies in sequential data by leveraging structured matrices.
- **HiPPO (2020)** — A family of continuous-time memory models that use high-order polynomial projections to approximate ideal memory dynamics.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Efficient Hybrid Language Model Compression through Group-Aware SSM Pruning | 2025 | Li et al. | Showcases state-of-the-art hybrid models and the benefits of compression. |
| SDE: A Simplified and Disentangled Dependency Encoding Framework for State Space Models in Time Series Forecasting | 2024 | Weng et al. | Highlights the application of SSMs in time series forecasting. |
| On Structured State-Space Duality | 2025 | Dao et al. | Formalized and generalized Structured State-Space Duality (SSD), showing an equivalence between SSMs and masked attention. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding | 2018 | Introduced BERT, a new language representation model that significantly advanced natural language understanding. |
| RoBERTa: A Robustly Optimized BERT Pretraining Approach | 2019 | Presented RoBERTa, an optimized BERT pretraining approach that improved performance by modifying training procedures. |

## Current SotA

Decoder-Hybrid-Decoder Architecture for Efficient Reasoning with Long Generation (SambaY) achieves state-of-the-art performance in long-range reasoning tasks (2025). MambaTS, with targeted improvements for LTSF, achieves strong results in long-term time series forecasting (2024). On Structured State-Space Duality (Dao et al. 2025) formalized and generalized Structured State-Space Duality (SSD), showing an equivalence between SSMs and masked attention.

## What's happening now

Research frontiers are focused on developing novel SSM architectures that can dynamically adapt to different input sequences, optimizing both performance and computational efficiency. Engineering and systems efforts are centered around integrating SSMs into existing deep learning frameworks and developing efficient implementations for large-scale deployment. A key open problem is how to develop state space models that effectively capture both long-range dependencies and complex, structured relationships within time series data, while maintaining computational efficiency and interpretability.

## In production

- NVIDIA — NeMo and Megatron-Core now officially enable end-to-end production-ready training and fine-tuning of state space models (SSMs), including SSDs like Mamba-2, and Griffin-based RG-LRU architectures. — Production-ready — [https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/](https://developer.nvidia.com/blog/nvidia-nemo-accelerates-llm-innovation-with-hybrid-state-space-model-support/)
- Meta AI — Multi-Head State Space Model (MH-SSM) as a drop-in replacement for multi-head attention in Transformer encoders, introduced for speech recognition. — Speech Recognition — [https://ai.meta.com/research/publications/multi-head-state-space-model-for-speech-recognition/](https://ai.meta.com/research/publications/multi-head-state-space-model-for-speech-recognition/)

## Minimum Valuable Build

For a hands-on build with this concept, see the MVB on [[transformers]].

## Code & implementations

- [Mamba](https://github.com/state-spaces/mamba) — Official implementation of the Mamba SSM.
- [Hugging Face SSM integration](https://huggingface.co/docs/transformers/model_doc/mamba) — Integration of SSMs within the Hugging Face Transformers library.

## What comes next

- [[transformers]] — State Space Models are emerging as a potential replacement for Transformers in certain applications, offering improved efficiency.
- [[recurrent-neural-networks]] — SSMs provide an alternative approach to modeling sequential data compared to traditional RNNs, addressing some of their limitations.

## Connected topics

- [Transformer Architecture](./transformer.md) — Transformers and state-space models are both sequence modeling architectures.
- [Diffusion Models](../02-generative-modeling/diffusion-models.md) — Diffusion models can be viewed and implemented using state space models.
- [KV Cache](../09-algorithms-systems-ai/kv-cache.md) — KV caches are used in attention mechanisms, which can be related to state-space models.
- [Agent Architectures](../01-ai/agent-architectures.md) — State-space models can be used within agent architectures for modeling.
- [Markov Decision Process](../06-reinforcement-learning/mdp.md) — State-space models are related to Markov Decision Processes in reinforcement learning.
- [Gaussian Processes](../05-statistical-probabilistic-ml/gaussian-processes.md) — Gaussian processes can be viewed as a type of state-space model.


## Further reading

- Dao et al. (2022) — "Long Range Arena: A Benchmark for Efficient Transformers" — [https://arxiv.org/abs/2111.03953] — Introduces a benchmark for evaluating the efficiency of Transformers and SSMs on long-range dependencies.
- Lilian Weng's survey on State Space Models (lil'log, 2023) — Provides an overview of the key concepts and recent advances in state space models.
- Tay et al. (2020) — "Efficient Transformers: A Survey" — [https://arxiv.org/abs/2009.06732] — Surveys techniques for improving the efficiency of Transformers, including connections to SSMs.
- Zhu et al. (2024) — "MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting" — [https://arxiv.org/abs/2405.16440v1] — Details targeted improvements for SSMs in long-term time series forecasting.