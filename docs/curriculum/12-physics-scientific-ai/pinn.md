```yaml
---
title: Physics-Informed Neural Networks (PINNs)
track: 12-physics-scientific-ai
tags: [neural networks, physics, PDEs, scientific computing, machine learning]
depth: foundational
prereqs: [deep-learning, partial-differential-equations]
updated: 2024-11-06
has_mvb: false
---
# Physics-Informed Neural Networks (PINNs)
> **TL;DR:** Physics-Informed Neural Networks (PINNs) are a type of neural network that incorporates physical laws, described by partial differential equations (PDEs), directly into the training process, enabling them to solve complex scientific problems with limited data.

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
Imagine trying to predict the weather a month from now, or simulating the complex flow of air around an airplane wing. These are examples of spatiotemporal dynamics problems, where understanding how things change over space and time is critical. Traditional methods struggle with the computational demands of these complex systems. This is where Physics-Informed Neural Networks (PINNs) come in.

PINNs are a class of neural networks designed to solve partial differential equations (PDEs) by incorporating the governing physical laws directly into the network's loss function. Instead of relying solely on data, PINNs use the PDE itself to guide the training process. This is particularly useful when data is scarce, expensive to obtain, or noisy, as is often the case in scientific and engineering applications. By minimizing the residual of the PDE, PINNs learn to approximate the solution while adhering to the underlying physics.

The core idea behind PINNs is to create a neural network that not only fits the available data but also satisfies the physical laws that govern the system. This is achieved by adding a physics-based loss term to the standard data-driven loss. The network is then trained to minimize both losses simultaneously, resulting in a solution that is both accurate and physically plausible.

## Why it matters at the frontier
PINNs address a critical need in scientific computing: the ability to model complex physical systems with limited data. In many real-world scenarios, obtaining sufficient data for purely data-driven models is either impossible or prohibitively expensive. For example, modeling turbulence or predicting the behavior of self-gravitating fluids requires vast amounts of high-resolution data, which can be challenging to acquire. PINNs offer a way to overcome these data constraints by leveraging known physics to guide the learning process.

The development of more robust and generalizable PINN architectures that can effectively handle complex, multi-scale physical phenomena without requiring extensive hyperparameter tuning or domain-specific knowledge is an active area of research. Overcoming these challenges would enable scientists and engineers to tackle a wider range of problems, from climate modeling to drug discovery, with greater accuracy and efficiency. The ability to accurately predict long-term turbulent flows in three dimensions, while also being computationally efficient enough for practical applications, remains a significant open problem.

## Core concepts
- **Partial Differential Equation (PDE)** — A mathematical equation that relates a function to its partial derivatives, describing how the function changes with respect to multiple variables.
- **Residual** — The error between the predicted solution of a PDE and the actual solution, used to quantify how well the PINN satisfies the governing equations.
- **Loss Function** — A function that quantifies the difference between the PINN's predictions and the true solution, guiding the training process to minimize this difference.
- **Physics-Informed Loss** — A component of the loss function that penalizes deviations from the governing physical laws, ensuring that the PINN's solution adheres to the underlying physics.
- **Automatic Differentiation** — A technique used to compute the derivatives of the neural network's output with respect to its inputs, enabling the calculation of the PDE residual.
- **Neural Network Architecture** — The structure of the neural network, including the number of layers, the number of neurons per layer, and the activation functions used, which can be tailored to the specific problem being solved.
- **Training Data** — The data used to train the PINN, which can include measurements of the system's state at various points in space and time, as well as boundary conditions and initial conditions.

## Mathematical foundations
PINNs aim to minimize a loss function that combines data-driven and physics-informed terms. A general form of the loss function can be written as:

\[
\mathcal{L} = \mathcal{L}_{data} + \lambda \mathcal{L}_{PDE}
\]

where \(\mathcal{L}_{data}\) is the data-driven loss, \(\mathcal{L}_{PDE}\) is the physics-informed loss, and \(\lambda\) is a weighting factor that balances the two terms. This equation says that the total loss is a weighted sum of the data-driven loss and the physics-informed loss, allowing the PINN to learn from both data and physical laws.

The data-driven loss \(\mathcal{L}_{data}\) measures the difference between the PINN's predictions and the available data:

\[
\mathcal{L}_{data} = \frac{1}{N_{data}} \sum_{i=1}^{N_{data}} ||u(x_i) - u_i||^2
\]

where \(N_{data}\) is the number of data points, \(u(x_i)\) is the PINN's prediction at location \(x_i\), and \(u_i\) is the corresponding data value. This equation says that the data-driven loss is the average squared difference between the PINN's predictions and the actual data values.

The physics-informed loss \(\mathcal{L}_{PDE}\) measures how well the PINN satisfies the governing PDE:

\[
\mathcal{L}_{PDE} = \frac{1}{N_{PDE}} \sum_{i=1}^{N_{PDE}} ||f(x_i)||^2
\]

where \(N_{PDE}\) is the number of collocation points, and \(f(x_i)\) is the residual of the PDE at location \(x_i\), computed using automatic differentiation. This equation says that the physics-informed loss is the average squared residual of the PDE, evaluated at a set of collocation points.

## Key algorithms / techniques
- **Automatic Differentiation (AD)** — A method for computing derivatives of functions implemented as computer programs. PINNs use AD to calculate the derivatives of the network's output with respect to its inputs, which are needed to evaluate the PDE residual.
- **Collocation Method** — A numerical method for solving PDEs by enforcing the equation at a set of discrete points (collocation points) within the domain. PINNs use collocation methods to minimize the PDE residual at these points.
- **Loss Weighting** — The process of assigning different weights to the data-driven and physics-informed loss terms. Proper loss weighting is crucial for balancing the influence of data and physics in the training process.
- **Adaptive Activation Functions** — Activation functions that adapt their shape during training, allowing the network to better capture the complex behavior of the solution. These can improve the accuracy and stability of PINNs.

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| LESnets (Large-Eddy Simulation nets): Physics-informed neural operator for large-eddy simulation of turbulence | 2024 | Zhao et al. | Introduces LESnets, a physics-informed neural operator for large-eddy simulation of turbulence. |
| Prediction of turbulent channel flow using Fourier neural operator-based machine-learning strategy | 2024 | Wang et al. | Explores the use of Fourier Neural Operators (FNOs) for predicting turbulent channel flow. |
| Modeling Turbulent and Self-Gravitating Fluids with Fourier Neural Operators | 2025 | Poletti et al. | Explores the application of Fourier Neural Operators to model turbulent and self-gravitating fluids. |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations | 2019 | Raissi et al. | Introduced the concept of PINNs and demonstrated their ability to solve a variety of forward and inverse problems involving PDEs.
| Deep Galerkin Method: Deep Learning for Solving Partial Differential Equations | 2018 | Sirignano and Spiliopoulos | Proposed a deep learning method for solving PDEs based on the Galerkin method, which is a precursor to PINNs.

## Current SotA
Jiang et al. (2025) introduced an Implicit Adaptive Fourier Neural Operator for long-term predictions of three-dimensional turbulence. Wang et al. (2024) explored the use of Fourier Neural Operators (FNOs) for predicting turbulent channel flow. Poletti et al. (2025) explored the application of Fourier Neural Operators to model turbulent and self-gravitating fluids.

## What's happening now
Research is focused on improving the robustness and generalizability of PINNs, particularly for complex, multi-scale problems. This includes developing new network architectures, activation functions, and training strategies that can better capture the behavior of the solution. Evolutionary algorithms are being explored to optimize PINN architectures and hyperparameters, addressing the challenges in PINN optimization and generalization.

Engineering efforts are focused on developing efficient and scalable implementations of PINNs that can be deployed on high-performance computing platforms. This includes optimizing the automatic differentiation process, reducing the memory footprint of the network, and parallelizing the training process. NVIDIA's LangGraph-based AI agent is an example of how these technologies are being scaled in production.

A key open problem is: How can we develop more robust and generalizable physics-informed neural operators that can accurately predict long-term turbulent flows in three dimensions, while also being computationally efficient enough for practical applications? Addressing this question would enable the application of PINNs to a wider range of scientific and engineering problems.

## In production
- NVIDIA — LangGraph-based AI agent — From a single user to 1000+ coworkers — [https://developer.nvidia.com/blog/how-to-scale-your-langgraph-agents-in-production-from-a-single-user-to-1000-coworkers/]
- Crexi — ML pipeline — Scalable model deployment on AWS — [https://aws.amazon.com/blogs/machine-learning/how-crexi-achieved-ml-models-deployment-on-aws-at-scale-and-boosted-efficiency/]
- Veriff — Model serving using Amazon SageMaker multi-model endpoints (MMEs) — Decreased deployment time by 80% — [https://aws.amazon.com/blogs/machine-learning/how-veriff-decreased-deployment-time-by-80-using-amazon-sagemaker-multi-model-endpoints/]

## Code & implementations
Official implementations are often released with the corresponding research papers. Frameworks like TensorFlow and PyTorch provide the necessary tools for building and training PINNs.

> *For a hands-on build with this concept, see the MVB on [[neural operator]].*

## What comes next

- [[neural-operator]] — generalizes PINNs by learning the mapping between function spaces directly from data, rather than relying on a fixed PDE.
- [[scientific-machine-learning]] — provides a broader overview of the field, including other techniques for integrating machine learning with scientific computing.

## Connected topics

- [Fourier Neural Operator (FNO)](./fno.md) — FNO is another neural operator used in scientific AI, similar to PINNs.
- [Equivariant Networks](./equivariant-networks.md) — PINNs can benefit from equivariance to handle symmetries in physical systems.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — PINNs rely on backpropagation to train the neural network components.
- [Optimization](../04-neural-networks-dl/optimization.md) — PINNs use optimization algorithms to minimize the loss function during training.
- [Gaussian Processes](../05-statistical-probabilistic-ml/gaussian-processes.md) — Gaussian processes can be used in conjunction with PINNs for uncertainty quantification.
- [Neural Tangent Kernel (NTK)](../15-ml-theory-foundations/ntk.md) — NTK analysis can be applied to understand the behavior of neural networks in PINNs.


## Further reading
- Raissi et al. (2019) — "Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations" — [URL NOT VERIFIED] — This paper provides a comprehensive introduction to PINNs and their applications.
- Lilian Weng's survey on Neural Operators (lil'log, 2021) — [URL NOT VERIFIED] — Offers an accessible overview of neural operators, including their relationship to PINNs.
- Zhao et al. (2024) — "LESnets (Large-Eddy Simulation nets): Physics-informed neural operator for large-eddy simulation of turbulence" — [https://arxiv.org/html/2411.04502] — This paper introduces LESnets, a physics-informed neural operator for large-eddy simulation of turbulence.
- Wang et al. (2024) — "Prediction of turbulent channel flow using Fourier neural operator-based machine-learning strategy" — [https://arxiv.org/html/2403.03051v4] — This paper explores the use of Fourier Neural Operators (FNOs) for predicting turbulent channel flow.