```yaml
---
title: Equivariant Networks
track: 12-physics-scientific-ai
tags: [equivariant networks, symmetry, group theory, neural networks, physics]
depth: foundational
prereqs: [convolutional-neural-networks, group-theory]
updated: 2024-05-03
has_mvb: false
---

# Equivariant Networks
> **TL;DR:** Equivariant networks are neural networks designed to respect symmetries in data, leading to more efficient learning and improved generalization when those symmetries are present.

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
Imagine you're designing a self-driving car. You want the car to recognize objects regardless of their orientation or position in the camera's view. Standard convolutional neural networks struggle with this, as a rotated image requires a completely new set of learned features. Equivariant networks offer a solution by building in the ability to recognize patterns that are preserved under transformations like rotation and translation.

Equivariance, in the context of neural networks, means that the network's output transforms in a predictable way when the input is transformed. For example, if you rotate the input image, the output feature map should also rotate by the same amount. This is in contrast to invariance, where the output remains the same regardless of the input transformation. By enforcing equivariance, these networks can learn more efficiently from less data and generalize better to unseen transformations.

Equivariant networks achieve this by incorporating mathematical structures that reflect the underlying symmetries of the data. This often involves using group theory to define how different transformations act on the input and intermediate layers of the network. The result is a neural network architecture that is inherently aware of the symmetries in the data, leading to improved performance in tasks where these symmetries are important.

## Why it matters at the frontier
Equivariant networks are increasingly important at the frontier of AI research, particularly in areas where data exhibits inherent symmetries. In scientific computing, for example, molecular dynamics simulations and fluid dynamics problems often possess rotational and translational symmetries. By incorporating these symmetries into the neural network architecture, researchers can develop more accurate and efficient models for predicting physical phenomena.

The development of equivariant networks addresses a key challenge in machine learning: how to effectively incorporate prior knowledge about the structure of the data into the learning process. The open problem is: How can we develop equivariant neural networks that achieve universality with practical computational complexity, particularly for high-dimensional data and complex symmetry groups, while maintaining efficiency and scalability for real-world applications? Addressing this question will enable the application of equivariant networks to a wider range of scientific and engineering problems, leading to more robust and generalizable AI systems.

## Core concepts
- **Symmetry** — A transformation that leaves an object or system unchanged.
- **Group** — A set of transformations with an operation that combines them, satisfying closure, associativity, identity, and invertibility.
- **Group Action** — The way a group transforms a space or a set of data.
- **Equivariance** — A property of a function where transforming the input results in a corresponding transformation of the output.
- **Invariance** — A property of a function where transforming the input leaves the output unchanged.
- **Representation Theory** — The study of how abstract algebraic structures can be represented as linear transformations of vector spaces.
- **Convolutional Neural Network (CNN)** — A type of neural network that uses convolutional layers to process data, particularly effective for image recognition.

## Mathematical foundations
While a full mathematical treatment requires delving into group representation theory, the core idea can be illustrated with a simple example. Consider a function \(f\) that maps an input \(x\) to an output \(y\). For equivariance with respect to a transformation \(g\) from a group \(G\), we require:
\[
f(g \cdot x) = g' \cdot f(x)
\]
where \(x\) is the input, \(y\) is the output, \(g\) is a transformation from the group \(G\), and \(g'\) is a corresponding transformation in the output space. This equation says that applying the transformation \(g\) to the input \(x\) and then applying the function \(f\) is equivalent to applying the function \(f\) to \(x\) first, and then applying the transformation \(g'\) to the output \(f(x)\).

In the context of convolutional networks, group convolutions generalize standard convolutions to incorporate group actions. The group convolution of a filter \(k\) with an input feature map \(f\) is defined as:
\[
(f \star_G k)(x) = \int_G f(g^{-1} \cdot x) k(g) dg
\]
where \(x\) is a point in the input space, \(g\) is an element of the group \(G\), and \(dg\) is the Haar measure on \(G\). This equation extends the standard convolution operation by integrating over all possible transformations in the group \(G\), ensuring that the resulting feature map is equivariant to the group action.

For a rotation group \(SO(2)\), the transformation can be represented by a rotation matrix \(R(\theta)\), where \(\theta\) is the rotation angle. The equivariance condition then becomes:
\[
f(R(\theta) \cdot x) = R(\theta) \cdot f(x)
\]
where \(R(\theta)\) is the rotation matrix, \(x\) is the input, and \(f(x)\) is the output. This equation ensures that if the input \(x\) is rotated by an angle \(\theta\), the output \(f(x)\) is also rotated by the same angle \(\theta\).

## Key algorithms / techniques
- **Group Convolutional Networks (G-CNNs)** (Cohen & Welling, 2016) — Generalize standard convolutions to be equivariant to a group of transformations.
- **Tensor Field Networks (TFNs)** (Thomas et al., 2018) — Designed to be rotation and translation equivariant for 3D point cloud data by using tensor fields to represent features.
- **Harmonic Networks** (Worrall et al., 2016) — Achieve translation and rotation equivariance by using circular harmonics as the basis functions for the convolutional filters.
- **cuEquivariance** (NVIDIA) — A CUDA-accelerated math library for equivariant neural networks, addressing both theory and performance hurdles in scientific AI.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Group Equivariant Convolutional Networks | 2016 | Cohen et al. | Foundational paper introducing group equivariant convolutional networks. |
| Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds | 2018 | Thomas et al. | Introduces Tensor Field Networks, designed for rotation and translation equivariance in 3D point clouds. |
| On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups | 2018 | Kondor & Trivedi | Explores the generalization of equivariance and convolution to compact groups. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Group Equivariant Convolutional Networks | 2016 | Introduced group equivariant convolutional networks, laying the groundwork for subsequent developments. |
| Harmonic Networks: Deep Translation and Rotation Equivariance | 2016 | Introduced Harmonic Networks, designed to be translation and rotation equivariant. |
| Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds | 2018 | Introduced Tensor Field Networks, which are designed to be rotation and translation equivariant for 3D point cloud data. |

## Current SotA
Pacini (2024) provides a characterization theorem for equivariant networks with point-wise activations, offering theoretical insights into their behavior. Benchmark evaluations for equivariant networks are not standardized as of 2024; the most widely cited comparison is qualitative description.

## What's happening now
Research is focused on developing more expressive and computationally efficient equivariant networks. This includes exploring new architectures that can handle complex symmetry groups and high-dimensional data. A key area is the development of universal equivariant networks that can approximate any equivariant function, similar to how standard neural networks can approximate any continuous function.

Engineering efforts are focused on developing software libraries and hardware accelerators that can efficiently implement equivariant networks. NVIDIA's cuEquivariance library is a notable example, providing CUDA-accelerated implementations of key equivariant operations. These tools are essential for deploying equivariant networks in real-world applications.

The open problem is: How can we develop equivariant neural networks that achieve universality with practical computational complexity, particularly for high-dimensional data and complex symmetry groups, while maintaining efficiency and scalability for real-world applications? Addressing this question will enable the application of equivariant networks to a wider range of scientific and engineering problems, leading to more robust and generalizable AI systems.

## In production
- NVIDIA — cuEquivariance, a CUDA-accelerated math library for equivariant neural networks — Addresses both theory and performance hurdles in scientific AI — [https://developer.nvidia.com/blog/accelerate-drug-and-material-discovery-with-new-math-library-nvidia-cuequivariance/]
- Salesforce — AI Model Serving using Amazon SageMaker AI — Deploys and manages large-scale, production-grade model deployments — [https://aws.amazon.com/blogs/machine-learning/how-salesforce-achieves-high-performance-model-deployment-with-amazon-sagemaker-ai/]

## Minimum Valuable Build
For a hands-on build with this concept, see the MVB on [[convolutional-neural-networks]].

## Code & implementations
- cuEquivariance: [https://developer.nvidia.com/blog/accelerate-drug-and-material-discovery-with-new-math-library-nvidia-cuequivariance/]

## What comes next

- [[group-theory]] — provides the mathematical foundation for understanding symmetries and transformations.
- [[convolutional-neural-networks]] — shows how to incorporate equivariance into standard CNN architectures.

## Connected topics

- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Equivariant networks often use backpropagation for training their parameters.
- [Contrastive Learning](../03-representation-learning/contrastive-learning.md) — Equivariant networks can be used in contrastive learning for representation learning.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers and equivariant networks both aim to improve model performance.
- [Data Parallelism](../09-algorithms-systems-ai/data-parallelism.md) — Data parallelism can be used to train equivariant networks on large datasets.
- [Bayesian Inference](../05-statistical-probabilistic-ml/bayesian-inference.md) — Bayesian methods can be used to analyze and improve equivariant networks.
- [Agent Architectures](../01-ai/agent-architectures.md) — Equivariant networks can be used in agent architectures for various tasks.


## Further reading
- Cohen et al. (2016) — "Group Equivariant Convolutional Networks" — [https://arxiv.org/pdf/1602.07576] — Introduces group equivariant convolutional networks, a method for building neural networks that are equivariant to group actions.
- Kondor & Trivedi (2018) — "On the Generalization of Equivariance and Convolution in Neural Networks to the Action of Compact Groups" — [https://ar5iv.labs.arxiv.org/html/1802.03690] — Explores the generalization of equivariance and convolution in neural networks to the action of compact groups, expanding the applicability of equivariant networks.
- Thomas et al. (2018) — "Tensor field networks: Rotation- and translation-equivariant neural networks for 3D point clouds" — [https://arxiv.org/pdf/1802.08219] — Introduces Tensor Field Networks, which are designed to be rotation and translation equivariant for 3D point cloud data.
- Pacini (2024) — "A Characterization Theorem for Equivariant Networks with Point-wise Activations" — [https://arxiv.org/html/2401.09235] — Provides a characterization theorem for equivariant networks with point-wise activations, offering theoretical insights into their behavior.