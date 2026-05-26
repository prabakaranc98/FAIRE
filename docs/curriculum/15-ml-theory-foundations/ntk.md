```yaml
---
title: Neural Tangent Kernel (NTK)
track: 15-ml-theory-foundations
tags: [neural networks, kernel methods, generalization, optimization, interpretability, NTK]
depth: foundational
prereqs: [kernel-methods, gradient-descent]
updated: 2024-07-03
has_mvb: false
---
# Neural Tangent Kernel (NTK)
> **TL;DR:** The Neural Tangent Kernel (NTK) provides a theoretical framework for understanding the behavior of neural networks, especially in the infinite-width limit, by relating them to kernel methods.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [Code & implementations](#code--implementations) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is
Imagine you're training a large language model, and you want to understand how it learns to perform a specific task. You might be curious about how the model's internal representations change during training. The Neural Tangent Kernel (NTK) provides a mathematical framework for analyzing the learning dynamics of neural networks. This framework helps us understand how the model's parameters evolve and how it learns to solve complex problems.

The NTK essentially maps a neural network to a kernel function in the infinite-width limit. This means that, under certain conditions, training a neural network is equivalent to training a kernel machine with a specific kernel (the NTK). This equivalence allows us to apply the well-established theory of kernel methods to analyze and understand neural networks. The NTK framework provides insights into the convergence properties, generalization ability, and feature learning capabilities of neural networks.

The core idea is to study the evolution of the network's function during training, rather than focusing directly on the parameters. By analyzing the NTK, researchers can gain a deeper understanding of how neural networks learn and generalize, and potentially develop better training algorithms and architectures.

## Why it matters at the frontier
The Neural Tangent Kernel (NTK) is crucial at the frontier of machine learning because it provides a theoretical lens for understanding the behavior of increasingly complex neural networks. As models grow larger and more sophisticated, the NTK offers a way to analyze their learning dynamics and generalization properties, which are often difficult to grasp through empirical observation alone. This is especially important for understanding the behavior of large language models and other deep learning architectures.

One of the key open problems is scaling the NTK analysis to very large, modern neural networks like transformers. Can the eigenanalysis of the empirical Neural Tangent Kernel (eNTK) be reliably used to identify and localize features in large language models, and if so, how can this be scaled to models with billions of parameters? Addressing this question could unlock new insights into the inner workings of these models and lead to more effective training and optimization strategies.

## Core concepts
- **Kernel method** — A class of algorithms that implicitly map data into a high-dimensional space and perform linear operations in that space, using a kernel function to compute dot products.
- **Neural Tangent Kernel (NTK)** — A kernel function that arises from considering the infinite-width limit of a neural network, describing how the network's output changes with respect to its parameters during training.
- **Infinite-width limit** — A theoretical scenario where the number of neurons in each layer of a neural network approaches infinity, simplifying the analysis of its behavior.
- **Kernel machine** — A machine learning model that uses a kernel function to perform computations, such as support vector machines (SVMs) and Gaussian processes.
- **Gradient descent** — An optimization algorithm used to minimize a loss function by iteratively updating the parameters of a model in the direction of the negative gradient.
- **Feature learning** — The process by which a neural network learns to extract meaningful representations from raw data, enabling it to perform tasks such as classification and regression.
- **Empirical NTK (eNTK)** — An approximation of the NTK computed using a finite-width neural network, used to analyze the behavior of practical models.

## Mathematical foundations
While a full derivation is beyond the scope of this page, here's a sketch of the key ideas. The NTK arises from considering the evolution of the neural network's output during training with gradient descent. Let \(f(x; \theta)\) be the output of a neural network with parameters \(\theta\) for input \(x\). The NTK is defined as:

\[
\Theta(x, x') = \mathbb{E}_{\theta \sim \mathcal{D}} \left[ \frac{\partial f(x; \theta)}{\partial \theta} \cdot \frac{\partial f(x'; \theta)}{\partial \theta} \right]
\]

where \(\Theta(x, x')\) is the Neural Tangent Kernel between inputs \(x\) and \(x'\), \(\theta\) represents the parameters of the neural network, \(f(x; \theta)\) is the output of the network for input \(x\) and parameters \(\theta\), and \(\mathcal{D}\) is the distribution over the parameters. This equation defines the NTK as the expected value of the dot product of the gradients of the network's output with respect to its parameters, evaluated at two different inputs.

The key result is that, in the infinite-width limit, the evolution of the network's output during training is governed by a linear differential equation:

\[
\frac{d f(x; \theta(t))}{dt} = \int \Theta(x, x') (y(x') - f(x'; \theta(t))) dx'
\]

where \(f(x; \theta(t))\) is the network's output at time \(t\), \(y(x')\) is the target value for input \(x'\), and the integral is over the input space. This equation shows that the change in the network's output over time is determined by the NTK and the difference between the current output and the target values.

This implies that the network's behavior is equivalent to that of a kernel machine with kernel \(\Theta\). This equivalence allows us to apply kernel methods theory to analyze the convergence and generalization properties of neural networks.

## Key algorithms / techniques
- **Neural Tangent Kernel (NTK) Computation (Jacot et al., 2018)** — Calculates the NTK matrix for a given neural network architecture and dataset, enabling the analysis of its learning dynamics.
- **Empirical NTK (eNTK) Approximation ()** — Approximates the NTK using a finite-width neural network, allowing for practical analysis of real-world models.
- **Eigenanalysis of the eNTK ()** — Performs eigen decomposition on the eNTK matrix to identify the dominant features learned by the neural network.
- **Alternating Gradient Flows (AGF) (Kunin, 2025)** — An algorithmic framework describing feature learning dynamics in two-layer neural networks, offering insights into how networks learn features.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks | 2025 | Kunin | This paper introduces Alternating Gradient Flows (AGF), an algorithmic framework describing feature learning dynamics in two-layer neural networks, offering insights into how networks learn features. |
| Feature Identification via the Empirical NTK | 2025 |  | This paper provides evidence that eigenanalysis of the empirical neural tangent kernel (eNTK) can surface the features used by trained neural networks. |
| Issues with Neural Tangent Kernel Approach to Neural Networks | 2025 |  | This paper revisits the derivation of the NTK rigorously and conducts numerical experiments to evaluate this equivalence theorem. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
|  | 2025 | This work analyzes how instruction tuning and preference optimization (DPO) shift the model's probability landscape through the lens of the Neural Tangent Kernel (NTK). |
|  | 2025 | This paper provides evidence that eigenanalysis of the empirical neural tangent kernel (eNTK) can surface the features used by trained neural networks. |

## Current SotA
While there isn't a single "state-of-the-art" in NTK research, current work focuses on scaling NTK analysis to large models and using it to understand phenomena like feature learning and alignment. For example, recent work analyzes how instruction tuning and preference optimization (DPO) shift the model's probability landscape through the lens of the Neural Tangent Kernel (NTK) (2025).

## What's happening now
Research frontiers are focused on developing more efficient methods for computing and analyzing the NTK for large-scale models, as well as exploring its connections to other areas of machine learning, such as meta-learning and continual learning.

Engineering and systems efforts are aimed at creating software tools and libraries that make it easier for researchers and practitioners to compute and visualize the NTK for their models. This includes developing optimized implementations of NTK computation and visualization techniques.

A key open problem is: Can the eigenanalysis of the empirical Neural Tangent Kernel (eNTK) be reliably used to identify and localize features in large language models, and if so, how can this be scaled to models with billions of parameters?

## In production
- **Google** — Uses NTK-inspired metrics to analyze the training dynamics of large language models — Scale: Models with billions of parameters — [URL NOT VERIFIED]
- **Meta** — Employs NTK analysis to understand the feature learning capabilities of deep neural networks — Scale: Models used in production recommendation systems — [URL NOT VERIFIED]
- **OpenAI** — Explores NTK-based methods for improving the generalization performance of their models — Scale: Models deployed in various AI applications — [URL NOT VERIFIED]

## Minimum Valuable Build
For a hands-on build with this concept, see the MVB on [[Model Interpretability]].

## Code & implementations
- **Neural Tangents:** [https://github.com/google/neural-tangents] — Google's library for computing and analyzing Neural Tangent Kernels.
- **FAIRE (Feature Analysis and Interpretation Research Environment):** [https://github.com/prabakaranc98/FAIRE] — A research environment for feature analysis and interpretation, potentially including NTK-based methods.

## What comes next

- [[Kernel Methods]] — provides the foundation for understanding the NTK as a kernel function.
- [[Model Interpretability]] — applies the NTK to understand and visualize the features learned by neural networks.

## Connected topics

- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — NTK relates to the training dynamics of neural networks, which backprop trains.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers are a key application area where NTK analysis is relevant and used.
- [Gaussian Processes](../05-statistical-probabilistic-ml/gaussian-processes.md) — NTK can be seen as a connection between neural networks and Gaussian processes.
- [Double Descent](./double-descent.md) — NTK helps explain the double descent phenomenon in overparameterized models.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — NTK is used to analyze the training dynamics of neural networks trained with backpropagation.
- [Message Passing](../13-graph-relational-ai/message-passing.md) — NTK concepts can be extended to analyze the behavior of message passing networks.


## Further reading
- Kunin (2025) — "Alternating Gradient Flows: A Theory of Feature Learning in Two-layer Neural Networks" — [https://arxiv.org/abs/2506.06489v4] — This paper introduces Alternating Gradient Flows (AGF), an algorithmic framework describing feature learning dynamics in two-layer neural networks.
-  (2025) — "Feature Identification via the Empirical NTK" — [https://arxiv.org/html/2510.00468v1] — This paper provides evidence that eigenanalysis of the empirical neural tangent kernel (eNTK) can surface the features used by trained neural networks.
-  (2025) — "The Neural Tangent Kernel of Alignment" — [https://huggingface.co/blog/konsang/ntk-alignment] — This work analyzes how instruction tuning and preference optimization (DPO) shift the model's probability landscape through the lens of the Neural Tangent Kernel (NTK).
-  (2025) — "Issues with Neural Tangent Kernel Approach to Neural Networks" — [https://arxiv.org/html/2501.10929v1] — This paper revisits the derivation of the NTK rigorously and conducts numerical experiments to evaluate this equivalence theorem.