```yaml
---
title: Optimization
track: 04-neural-networks-dl
tags: [optimization, gradient descent, machine learning, deep learning, loss function]
depth: applied
prereqs: [linear-regression, backpropagation]
updated: 2024-10-26
has_mvb: true
---
# Optimization
> **TL;DR:** Optimization in machine learning is the process of finding the best set of parameters for a model to minimize a loss function, enabling accurate predictions and efficient model training.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is
Imagine you're training a cutting-edge AI model, and after weeks of work, it's finally ready. But when you deploy it, the model runs slower than expected, costing you money and frustrating users. Optimization is the key to unlocking the full potential of your AI, making it faster, more efficient, and more cost-effective.

In machine learning, optimization refers to the process of finding the best set of parameters for a model to minimize a loss function. The loss function quantifies the difference between the model's predictions and the actual values, so minimizing it means the model is making more accurate predictions. This is typically achieved through iterative algorithms that adjust the model's parameters based on the gradient of the loss function.

Optimization is crucial for training effective machine learning models, especially deep learning models with millions or even billions of parameters. Without efficient optimization techniques, training these models would be computationally infeasible, requiring vast amounts of time and resources.

## Why it matters at the frontier
Optimization is a critical bottleneck in pushing the boundaries of AI. As models grow larger and datasets become more complex, the need for efficient optimization techniques becomes even more pressing. Frontier labs are actively researching new optimization algorithms and strategies to enable the training of even larger and more sophisticated models.

One major open problem is developing a generalizable, automated method for selecting the optimal optimizer and its hyperparameters for a given model and dataset, without requiring extensive manual tuning or domain expertise. This would significantly reduce the time and resources required to train new models and accelerate the development of AI.

## Core concepts
- **Loss function** — A function that quantifies the difference between the model's predictions and the actual values, guiding the optimization process.
- **Gradient descent** — An iterative optimization algorithm that adjusts the model's parameters in the direction of the negative gradient of the loss function.
- **Learning rate** — A hyperparameter that controls the step size during gradient descent, determining how much the parameters are adjusted in each iteration.
- **Optimizer** — An algorithm that implements a specific optimization strategy, such as Adam, SGD, or RMSprop, to update the model's parameters.
- **Batch size** — The number of training examples used in each iteration of gradient descent, affecting the stability and speed of convergence.
- **Epoch** — One complete pass through the entire training dataset during the optimization process.
- **Regularization** — Techniques used to prevent overfitting by adding a penalty term to the loss function, encouraging simpler models.

## Mathematical foundations
The core goal of optimization is to minimize a loss function \(L(\theta)\) with respect to the model parameters \(\theta\). Gradient descent updates the parameters iteratively:
\[
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t)
\]
where \(\theta_t\) is the parameters at iteration \(t\), \(\eta\) is the learning rate, and \(\nabla L(\theta_t)\) is the gradient of the loss function with respect to the parameters at iteration \(t\). This equation says that the parameters are updated by moving in the opposite direction of the gradient, scaled by the learning rate.

In residual networks, the core building block is the residual block, which can be represented as:
\[
\mathbf{y} = \mathcal{F}(\mathbf{x}, \{W_i\}) + \mathbf{x}
\]
where \(\mathbf{x}\) is the input, \(\mathbf{y}\) is the output, \(\mathcal{F}\) represents the residual mapping to be learned, and \(\{W_i\}\) are the weights of the layers within the residual block. This equation represents the core structure of a residual block, where the input is added to the output of a series of layers.

Multi-objective Bayesian optimization aims to maximize the Expected Hypervolume Improvement (EHVI):
\[
\text{EHVI}(x) = \mathbb{E}[\text{HV}(f(x), R) - \text{HV}(R)]
\]
where \(\text{EHVI}(x)\) is the Expected Hypervolume Improvement at point \(x\), \(f(x)\) is the predicted objective values at point \(x\), \(\text{HV}\) is the hypervolume, and \(R\) is the reference point. This equation is central to multi-objective Bayesian optimization, where the goal is to find the best trade-off between multiple objectives.

## Key algorithms / techniques
- **Stochastic Gradient Descent (SGD)** — A basic optimization algorithm that updates parameters using the gradient computed on a single training example or a small batch.
- **Adam** — An adaptive optimization algorithm that combines the benefits of AdaGrad and RMSprop, using both momentum and adaptive learning rates for each parameter.
- **RMSprop** — An adaptive optimization algorithm that adjusts the learning rate for each parameter based on the historical magnitudes of its gradients.
- **Batch Normalization** — A technique that normalizes the activations of each layer within a batch, stabilizing training and accelerating convergence (Ioffe & Szegedy, 2015).
- **Residual Connections** — Architectural elements that add the input of a layer to its output, easing the training of very deep networks (He et al., 2015).

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Deep Residual Learning for Image Recognition | 2015 | He et al. | Introduced residual learning to ease the training of very deep neural networks. |
| Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift | 2015 | Ioffe and Szegedy | Introduced batch normalization to stabilize training and accelerate convergence. |
| Residual Alignment: Uncovering the Mechanisms of Residual Networks | 2024 | Li & Papyan | Investigated the mechanisms of residual networks, providing insights into how they function. |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| Deep Residual Learning for Image Recognition | 2015 | Introduced residual learning to ease the training of very deep neural networks, enabling the development of much deeper and more accurate models. |
| Identity Mappings in Deep Residual Networks | 2016 | Analyzed the behavior of residual networks, showing that the identity mappings are crucial for the performance of the network (He et al., 2016). |
| Wide Residual Networks | 2016 | Explored wider residual networks, demonstrating that increasing the width of residual blocks can improve performance (Zagoruyko & Komodakis, 2016). |

## Current SotA
Recent advancements focus on automating optimization strategies. NNGPT presents an open-source framework that turns a large language model (LLM) into a self-improving AutoML engine for neural network development (2025). AdaMuon proposes a novel optimizer that combines element-wise adaptivity with orthogonal updates for large-scale neural network training (2025).

## What's happening now
Research is actively exploring the use of large language models (LLMs) to automate the optimization process. LLMs are being used to generate and evaluate different optimization strategies, potentially leading to more efficient and robust training methods. LLMs are also being used to write efficient GPU kernels (2025).

Engineering efforts are focused on developing systems that can automatically tune the hyperparameters of deployed models. These systems use techniques such as Bayesian optimization and reinforcement learning to find the best configuration for a given model and environment.

Open problems include developing optimization algorithms that are less sensitive to the choice of hyperparameters, as well as methods for automatically selecting the best optimizer for a given task. Can we develop a generalizable, automated method for selecting the optimal optimizer and its hyperparameters for a given model and dataset, without requiring extensive manual tuning or domain expertise?

## In production
- Crexi — ML models deployment framework on AWS — Scalable, production-grade ML deployment — [https://aws.amazon.com/blogs/machine-learning/how-crexi-achieved-ml-models-deployment-on-aws-at-scale-and-boosted-efficiency/]
- Databricks — coSTAR framework for shipping AI agents — Scalable approach to shipping AI agents — [https://www.databricks.com/blog/costar-how-we-ship-ai-agents-databricks-fast-without-breaking-things]
- Google — OPPerTune — Optimizes deployed services with minimal disruption — [https://research.google/pubs/oppertune-post-deployment-configuration-tuning-of-services-made-easy/]

## Minimum Valuable Build
**What you're building:** A simple linear regression model trained using gradient descent, visualizing the loss function decreasing over time.
**Why this build:** Demonstrates the fundamental principles of optimization and how gradient descent can be used to find the optimal parameters for a model.
**Stack:** `scikit-learn==1.3.0`, `numpy==1.26.0`, `matplotlib==3.8.0`
**Estimated time:** 30 minutes

### The recipe

1. **Import necessary libraries:**
   ```python
   import numpy as np
   import matplotlib.pyplot as plt
   from sklearn.linear_model import LinearRegression
   from sklearn.model_selection import train_test_split
   from sklearn.metrics import mean_squared_error
   ```

2. **Generate synthetic data:**
   ```python
   X = 2 * np.random.rand(100, 1)
   y = 4 + 3 * X + np.random.randn(100, 1)

   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   ```

3. **Implement gradient descent:**
   ```python
   def gradient_descent(X, y, learning_rate=0.01, n_iterations=100):
       m = X.shape[0]
       theta = np.random.randn(2, 1)  # Initialize parameters
       X_b = np.c_[np.ones((m, 1)), X]  # Add bias term

       history = []
       for iteration in range(n_iterations):
           gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
           theta = theta - learning_rate * gradients
           history.append(mean_squared_error(y, X_b.dot(theta)))
       return theta, history
   ```

4. **Train the model using gradient descent:**
   ```python
   X_train_b = np.c_[np.ones((len(X_train), 1)), X_train]
   theta, history = gradient_descent(X_train, y_train, learning_rate=0.1, n_iterations=100)
   print("Theta found by gradient descent:", theta)
   ```

5. **Visualize the loss function:**
   ```python
   plt.plot(history)
   plt.xlabel("Iteration")
   plt.ylabel("Mean Squared Error")
   plt.title("Loss Function over Iterations")
   plt.show()
   ```

### Expected output
The code will output the learned parameters (theta) and display a plot showing the mean squared error decreasing over iterations, demonstrating the convergence of gradient descent. The plot should show a decreasing curve, indicating that the loss is decreasing as the model learns.

### Common failure modes
- **Loss increases instead of decreasing** → Reduce the learning rate. If the learning rate is too high, the algorithm might overshoot the minimum.
- **Slow convergence** → Increase the learning rate or the number of iterations.
- **NaN values in loss** → This usually indicates a very high learning rate causing the loss to explode. Reduce the learning rate drastically.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations
- scikit-learn: [https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html] — Official documentation for linear regression in scikit-learn.
- PyTorch Optimizers: [https://pytorch.org/docs/stable/optim.html] — Documentation for various optimization algorithms in PyTorch.

## What comes next
- [[Backpropagation]] — is the algorithm used to calculate the gradients of the loss function with respect to the model's parameters, enabling optimization.
- [[Regularization]] — are techniques used to prevent overfitting during optimization, improving the generalization performance of the model.

## Connected topics
- [Backpropagation](./backpropagation.md) — Backpropagation is a key algorithm for optimizing neural network parameters.
- [Markov Decision Process](../06-reinforcement-learning/mdp.md) — MDPs are often optimized using techniques like dynamic programming or policy gradients.
- [Reinforcement Learning from Human Feedback (RLHF)](../06-reinforcement-learning/rlhf.md) — RLHF uses optimization to align models with human preferences.
- [Contrastive Learning](../03-representation-learning/contrastive-learning.md) — Contrastive learning optimizes representations by contrasting similar and dissimilar examples.
- [Double Descent](../15-ml-theory-foundations/double-descent.md) — Double descent describes how model performance changes with optimization and model size.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers are optimized using techniques like gradient descent and attention mechanisms.


## Further reading
- He et al. (2015) — "Deep Residual Learning for Image Recognition" — [https://arxiv.org/abs/1512.03385] — Introduces residual connections, a key architectural innovation that significantly improved optimization in deep networks.
- Ioffe and Szegedy (2015) — "Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift" — [https://arxiv.org/abs/1502.03167] — Explains batch normalization, a technique that stabilizes training and accelerates convergence by normalizing activations within each batch.
- Lilian Weng's survey on Optimization (lil'log, 2017) — Provides a comprehensive overview of various optimization algorithms and techniques used in machine learning.
- Distill.pub interactive articles on optimization — Offers visual and interactive explanations of optimization concepts, providing deeper insights into the underlying mechanisms.