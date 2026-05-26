```yaml
---
title: Fourier Neural Operator (FNO)
track: 12-physics-scientific-ai
tags: [neural operators, PDEs, Fourier transform, scientific computing, machine learning]
depth: foundational
prereqs: [deep-learning, fourier-analysis]
updated: 2024-10-26
has_mvb: false
---
# Fourier Neural Operator (FNO)
> **TL;DR:** Fourier Neural Operators (FNOs) are neural networks that learn mappings between function spaces by leveraging the Fourier transform, offering an efficient alternative for solving partial differential equations (PDEs).

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
Imagine trying to predict the weather a week from now, or simulating the flow of air around a new airplane design. These complex tasks require solving partial differential equations (PDEs), which can be computationally expensive and time-consuming. Fourier Neural Operators (FNOs) offer a promising alternative, using neural networks to learn the solutions to these equations directly from data, potentially speeding up simulations and improving accuracy. This approach has the potential to revolutionize fields from climate modeling to engineering design.

FNOs are a class of neural networks designed to learn mappings between infinite-dimensional function spaces. Unlike traditional numerical methods that discretize the domain and solve PDEs iteratively, FNOs learn the entire solution operator directly from data. This is achieved by representing functions in the Fourier domain, where differentiation becomes a simple multiplication, and using neural networks to learn the mapping between the Fourier coefficients of the input and output functions.

The key advantage of FNOs is their ability to generalize to different problem instances and resolutions without retraining. Once trained, an FNO can solve the same PDE with different initial conditions or boundary conditions, and even on finer grids, making them significantly more efficient than traditional solvers for many applications.

## Why it matters at the frontier
FNOs are gaining traction in frontier research due to their potential to accelerate scientific discovery and engineering design. Traditional numerical methods for solving PDEs often struggle with high-dimensional problems, complex geometries, and the need for real-time predictions. FNOs offer a data-driven alternative that can learn the underlying physics from simulations or experimental data, enabling faster and more accurate predictions.

The development of more efficient and versatile transformer-based architectures, like PDE-Transformer, is a key area of focus. The goal is to handle a wider range of physics simulations while maintaining high accuracy and scalability, particularly for complex, multi-physics problems. Furthermore, a critical open problem is: How can we develop FNO architectures that are robust to noisy or incomplete data, and can we quantify the uncertainty in their predictions, especially for chaotic systems where small errors can lead to significant divergence?

## Core concepts
- **Neural Operator** — A neural network that learns mappings between infinite-dimensional function spaces, as opposed to finite-dimensional spaces in traditional neural networks.
- **Fourier Transform** — A mathematical transformation that decomposes a function into its constituent frequencies, enabling efficient computation of derivatives in the frequency domain.
- **Kernel Integral Operator** — An integral operator with a learnable kernel function that maps an input function to an output function, forming the basis of the FNO architecture.
- **Function Space** — A set of functions that satisfy certain properties, such as continuity or differentiability, and are the domain and range of the learned mapping.
- **Operator Learning** — The process of training a neural network to approximate an operator that maps between function spaces, enabling the solution of PDEs and other functional equations.
- **Spectral Parameterization** — Representing functions in terms of their Fourier coefficients, allowing for efficient computation of derivatives and application of linear operators.
- **Generalization** — The ability of a trained FNO to accurately predict solutions for new problem instances or resolutions without retraining.

## Mathematical foundations
The core idea behind FNOs is to learn a mapping \(G\) between function spaces:
\[
u_t = G(u_0)
\]
where \(u_0\) is the initial condition, \(u_t\) is the solution at time \(t\), and \(G\) is the learned operator.

This can be expressed in the Fourier domain as:
\[
\hat{u}_{t+1}(\mathbf{k}) = R(\mathbf{k}) \hat{u}_t(\mathbf{k}) + \hat{W}(\mathbf{k})
\]
where \(\hat{u}_t(\mathbf{k})\) is the Fourier transform of the solution at time \(t\), \(\mathbf{k}\) is the wave number, \(R(\mathbf{k})\) is a learnable diagonal matrix representing the linear transformation in the Fourier space, and \(\hat{W}(\mathbf{k})\) is a learnable bias term. This equation says that the Fourier coefficients of the solution at the next time step are a linear combination of the Fourier coefficients at the current time step, plus a bias.

The FNO architecture typically involves lifting the input function to a higher-dimensional space, applying a linear transformation in the Fourier domain, and then projecting back to the original space:
\[
v(x) = P(u(x))
\]
where \(u(x)\) is the input function, \(P\) is a lifting operator, and \(v(x)\) is the lifted representation.

## Key algorithms / techniques
- **Fourier Neural Operator (FNO) (Li et al. 2020)** — Introduces the core FNO architecture, using the Fourier transform to efficiently learn mappings between function spaces for solving PDEs.
- **Adaptive Fourier Neural Operators (AFNO) (Guibas et al. 2021)** — Improves the efficiency of token mixing in Transformers by using adaptive Fourier transforms, enhancing performance on complex tasks.
- **Factorized Fourier Neural Operators (F-FNO) (Tran et al. 2023)** — Introduces a factorized approach to FNOs, further improving computational efficiency and scalability for large-scale simulations.
- **PDE-Transformer (Holzschuh et al. 2025)** — A transformer-based architecture specifically designed for surrogate modeling of physics simulations on regular grids, offering versatility and efficiency.

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs | 2021 | Kovachki et al. | Introduces the Neural Operator framework, which learns mappings between function spaces, with applications to solving partial differential equations. |
| Physics-Informed Neural Operator for Learning Partial Differential Equations | 2021 | Li et al. | Explores the use of physics-informed neural operators to learn partial differential equations. |
| Fourier Neural Operators for Non-Markovian Processes: Approximation Theorems and Experiments | 2025 | Lee et al. | Explores Fourier Neural Operators for Non-Markovian Processes, providing approximation theorems and experimental results. |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs | 2021 | Introduced the Neural Operator framework, learning mappings between function spaces for solving PDEs. |
| Factorized Fourier Neural Operators | 2023 | Introduced Factorized Fourier Neural Operators (F-FNO) as a learning-based approach for simulations. |

## Current SotA
PDE-Transformer achieves state-of-the-art performance on surrogate modeling of physics simulations on regular grids (Holzschuh et al. 2025). Adaptive Fourier Neural Operators (AFNO) improve efficiency in token mixing for Transformers (Guibas et al. 2021). Factorized Fourier Neural Operators (F-FNO) offer enhanced computational efficiency and scalability for large-scale simulations (Tran et al. 2023).

## What's happening now
Research frontiers are focused on developing more efficient and versatile transformer-based architectures, such as PDE-Transformer, to handle a wider range of physics simulations while maintaining high accuracy and scalability. A key area of investigation involves enhancing the ability of FNOs to handle complex geometries and boundary conditions, pushing beyond the limitations of traditional grid-based methods. Engineering and systems efforts are directed towards optimizing FNO implementations for deployment on high-performance computing platforms, enabling real-time predictions and large-scale simulations. This includes exploring techniques like quantization and distributed training to reduce computational costs. Open problems include developing FNO architectures that are robust to noisy or incomplete data and quantifying the uncertainty in their predictions, especially for chaotic systems where small errors can lead to significant divergence. Addressing these challenges is crucial for deploying FNOs in real-world applications where data quality and reliability can vary significantly.

## In production
While specific production deployments with verifiable sources are still emerging, the potential applications of FNOs are being actively explored in various industries:

*   **Climate Modeling:** Research groups are investigating FNOs for accelerating climate simulations, which require solving complex PDEs over long time scales. The ability of FNOs to generalize to different resolutions could significantly reduce the computational cost of these simulations.
*   **Engineering Design:** Companies in the aerospace and automotive industries are exploring FNOs for optimizing the design of aircraft and vehicles. FNOs could be used to quickly predict the performance of different designs, reducing the need for expensive physical experiments.
*   **Medical Imaging:** FNOs are being investigated for improving the accuracy and efficiency of medical image analysis. For example, FNOs could be used to reconstruct high-resolution images from limited data, enabling faster and more accurate diagnoses.

## Minimum Valuable Build
For a hands-on build with this concept, see the MVB on [[neural-operators]].

## Code & implementations
- [Neural Operators (Kovachki et al. 2021)](https://github.com/neuraloperator/neuraloperator) — Official implementation of the Neural Operator framework.
- [Adaptive Fourier Neural Operators (Guibas et al. 2021)](https://github.com/google-research/google-research/tree/master/afno) — Official implementation of Adaptive Fourier Neural Operators.
- [Factorized Fourier Neural Operators (Tran et al. 2023)](https://github.com/zongyi-li/fourier_neural_operator) — Official implementation of Factorized Fourier Neural Operators.

## What comes next

- [[neural-operators]] — provides the broader context of operator learning, of which FNO is a specific instance.
- [[pde-solving]] — explores traditional and machine learning-based methods for solving partial differential equations.

## Connected topics

- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — FNO uses neural networks, which are trained using backpropagation.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers and FNO are both neural network architectures for processing data.
- [Data Parallelism](../09-algorithms-systems-ai/data-parallelism.md) — Training FNO models often benefits from data parallelism for faster computation.
- [Diffusion Models](../02-generative-modeling/diffusion-models.md) — FNO can be used in similar contexts as diffusion models, such as image generation.
- [Bayesian Inference](../05-statistical-probabilistic-ml/bayesian-inference.md) — FNO can be used in conjunction with Bayesian methods for uncertainty quantification.
- [Contrastive Learning](../03-representation-learning/contrastive-learning.md) — FNO can be used in conjunction with contrastive learning for representation learning.


## Further reading
- Kovachki et al. (2021) — "Neural Operator: Learning Maps Between Function Spaces With Applications to PDEs" — [https://arxiv.org/abs/2108.08481v6] — Provides a comprehensive introduction to the Neural Operator framework and its applications.
- Li et al. (2021) — "Physics-Informed Neural Operator for Learning Partial Differential Equations" — [https://arxiv.org/abs/2111.03794] — Explores the use of physics-informed neural operators to improve the accuracy and efficiency of PDE solving.
- Guibas et al. (2021) — "ADAPTIVE FOURIER NEURAL OPERATORS: EFFICIENT TOKEN MIXERS FOR TRANSFORMERS" — [https://arxiv.org/pdf/2111.13587] — Introduces Adaptive Fourier Neural Operators, improving efficiency in token mixing for Transformers.
- Tran et al. (2023) — "FACTORIZED FOURIER NEURAL OPERATORS" — [https://export.arxiv.org/pdf/2111.13802v4.pdf] — Introduces Factorized Fourier Neural Operators (F-FNO) as a learning-based approach for simulations.
```