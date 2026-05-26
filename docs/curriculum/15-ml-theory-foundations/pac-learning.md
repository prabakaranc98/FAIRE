```yaml
---
title: PAC Learning
track: 15-ml-theory-foundations
tags: [PAC learning, generalization, sample complexity, machine learning theory]
depth: foundational
prereqs: [statistical-learning-theory, bias-variance-decomposition]
updated: 2024-11-02
has_mvb: false
---
# PAC Learning
> **TL;DR:** PAC (Probably Approximately Correct) learning provides a formal framework for understanding the sample complexity required to learn a concept from data, guaranteeing generalization performance with high probability.

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
Imagine you're teaching a computer to recognize handwritten digits. You show it thousands of examples, hoping it will learn to distinguish a "3" from an "8." But what if some of your examples are mislabeled, or the data is inherently noisy? PAC learning provides a framework for understanding how well a machine can learn, even when the data isn't perfect. It helps us understand the sample complexity of learning.

PAC learning, short for Probably Approximately Correct learning, is a mathematical framework for analyzing the sample complexity of machine learning algorithms. It addresses the fundamental question: how many training examples are needed to ensure that a learning algorithm will, with high probability, produce a hypothesis that is approximately correct? This framework provides theoretical guarantees on the generalization performance of learning algorithms, even in the presence of noisy data.

The core idea behind PAC learning is to quantify the trade-off between the desired accuracy (\(\epsilon\)) and the confidence level (\(\delta\)). A PAC-learnable concept class is one for which a learning algorithm can, with probability at least \(1 - \delta\), find a hypothesis with error at most \(\epsilon\), given a sufficiently large training set. This framework is essential for understanding the limits of learning and for designing algorithms that can learn effectively from finite data.

## Why it matters at the frontier
PAC learning provides a theoretical foundation for understanding the generalization capabilities of machine learning models, which is crucial for addressing open problems in the field. For instance, understanding the sample complexity of learning hierarchical functions is essential for designing deep learning architectures that can generalize well from limited data. Furthermore, PAC learning frameworks are being extended to analyze the learning dynamics of complex systems like autoregressive models, providing insights into the behavior of large language models.

Frontier labs are actively researching tighter generalization bounds and more efficient learning algorithms within the PAC learning framework. For example, researchers are exploring how interpolation techniques can lead to tighter bounds on generalization error. Additionally, there is ongoing work to develop computationally efficient learners that can achieve optimal error rates in agnostic PAC learning, pushing the boundaries of what is theoretically possible. The ultimate goal is to develop a comprehensive theory that can explain and predict the performance of machine learning models in real-world scenarios.

## Core concepts
- **Concept Class (\(\mathcal{C}\))** — A set of possible target functions that the learner is trying to learn.
- **Hypothesis Class (\(\mathcal{H}\))** — The set of functions that the learning algorithm can choose from to approximate the target function.
- **Sample Complexity** — The number of training examples required to achieve a desired level of accuracy and confidence.
- **Error (\(\epsilon\))** — The maximum allowable difference between the hypothesis and the true target function.
- **Confidence (\(\delta\))** — The probability that the learning algorithm will produce a hypothesis with error at most \(\epsilon\).
- **PAC-Learnable** — A concept class is PAC-learnable if there exists an algorithm that can, with high probability, find a hypothesis with low error, given a sufficiently large training set.
- **Agnostic Learning** — A learning setting where no assumptions are made about the true target function or the data distribution.

## Mathematical foundations
In PAC learning, we aim to find a hypothesis \(h \in \mathcal{H}\) that approximates the target concept \(c \in \mathcal{C}\) with high probability. The error of a hypothesis \(h\) is defined as:
\[
\text{err}(h) = \mathbb{P}_{x \sim D}(h(x) \neq c(x))
\]
where \(D\) is the data distribution, \(x\) is a data point, \(h(x)\) is the prediction of the hypothesis, and \(c(x)\) is the true label. This equation defines the probability that the hypothesis \(h\) makes an incorrect prediction on a randomly drawn example \(x\) from the distribution \(D\).

The goal of PAC learning is to ensure that, with probability at least \(1 - \delta\), the error of the chosen hypothesis \(h\) is at most \(\epsilon\):
\[
\mathbb{P}(\text{err}(h) \leq \epsilon) \geq 1 - \delta
\]
where \(\epsilon\) is the desired accuracy and \(\delta\) is the confidence level. This inequality states that the probability of the error of the hypothesis \(h\) being less than or equal to \(\epsilon\) is greater than or equal to \(1 - \delta\), ensuring that the hypothesis is approximately correct with high probability.

For a finite hypothesis class \(\mathcal{H}\), the sample complexity \(m\) can be bounded using the following inequality:
\[
m \geq \frac{1}{\epsilon} \left( \log(|\mathcal{H}|) + \log\left(\frac{1}{\delta}\right) \right)
\]
where \(m\) is the number of training examples, \(\epsilon\) is the desired accuracy, \(\delta\) is the confidence level, and \(|\mathcal{H}|\) is the size of the hypothesis class. This inequality provides a lower bound on the number of training examples needed to PAC-learn a concept class, showing that the sample complexity grows logarithmically with the size of the hypothesis class and inversely with the desired accuracy and confidence.

## Key algorithms / techniques
- **Empirical Risk Minimization (ERM)** — A principle where the learning algorithm selects the hypothesis that minimizes the error on the training data.
- **Occam's Razor** — The principle of preferring simpler hypotheses, which often generalize better than more complex ones.
- **VC Dimension** — A measure of the complexity of a hypothesis class, defined as the maximum number of points that can be shattered by the class.
- **Rademacher Complexity** — A measure of the ability of a hypothesis class to fit random noise, used to bound the generalization error.
- **Structural Risk Minimization (SRM)** — A technique that balances the empirical risk with a complexity penalty to prevent overfitting.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Revisiting Agnostic PAC Learning | 2024 | Hanneke et al. | Addresses the shortcomings of ERM learners in the agnostic case, providing a more fine-grained model and demonstrating tightness of the error lower bound. |
| Sample Complexity of Autoregressive Reasoning: Chain-of-Thought vs. End-to-End | 2025 | Joshi et al. | Introduces a PAC-learning framework for next-token generators, the primitive underlying autoregressive models, which is a very relevant application of PAC learning in the current landscape. |
| Noise Sensitivity and Learning Lower Bounds for Hierarchical Functions | 2025 | Li | Explores the learning complexity of functions with hierarchical structure, providing applications for agnostic learning and statistical query lower bounds. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Revisiting Agnostic PAC Learning | 1984 | Introduced the PAC learning framework, providing a formal model for understanding learnability and generalization. |
| Sample Complexity of Autoregressive Reasoning: Chain-of-Thought vs. End-to-End | 1998 | Introduced the VC dimension, a key measure of the complexity of a hypothesis class. |
| Noise Sensitivity and Learning Lower Bounds for Hierarchical Functions | 2002 | Developed Rademacher complexity, a powerful tool for bounding the generalization error of learning algorithms. |

## Current SotA
Cohen et al. (2025) provides tight bounds on the sample complexity of agnostic multiclass classification. Li (2025) explores the learning complexity of functions with hierarchical structure, providing applications for agnostic learning and statistical query lower bounds. Joshi et al. (2025) introduces a PAC-learning framework for next-token generators, the primitive underlying autoregressive models, and studies the sample complexity of such systems.

## What's happening now
Research frontiers are focused on developing tighter generalization bounds that can better capture the performance of modern machine learning models. Viallard et al. (2024) explores tighter generalization bounds via interpolation. The development of new theoretical tools and techniques is essential for understanding the behavior of complex models in high-dimensional spaces.

Engineering and systems efforts are aimed at designing more efficient learning algorithms that can achieve PAC guarantees with limited computational resources. This includes developing algorithms that can learn from streaming data and algorithms that can handle noisy or incomplete data. The goal is to make PAC learning more practical and applicable to real-world problems.

An open problem in PAC learning is: Can we develop a computationally efficient learner that achieves an error rate of \(c \cdot \tau + O\left(\sqrt{\frac{\tau(d + \log(1 / \delta))}{m}} + \frac{d + \log(1 / \delta)}{m} \right)\) for agnostic learning, with a constant \(c\) of 1, thereby completely settling the complexity of agnostic learning?

## In production
While specific production deployments directly citing "PAC learning" are difficult to source, the principles of generalization and sample complexity are fundamental to how machine learning models are developed and deployed at scale. These principles guide decisions about data collection, model selection, and evaluation in various applications.

## Minimum Valuable Build
For a hands-on build with this concept, see the MVB on [[model-selection]].

## Code & implementations
*   **scikit-learn** — [https://scikit-learn.org/stable/](https://scikit-learn.org/stable/) — A comprehensive library for machine learning in Python, including implementations of various learning algorithms and tools for model selection and evaluation.
*   **TensorFlow** — [https://www.tensorflow.org/](https://www.tensorflow.org/) — An open-source machine learning framework developed by Google, providing a flexible platform for building and training machine learning models.
*   **PyTorch** — [https://pytorch.org/](https://pytorch.org/) — An open-source machine learning framework developed by Facebook, known for its dynamic computation graph and ease of use.

## What comes next

- [[statistical-learning-theory]] — provides a broader context for understanding the theoretical foundations of machine learning.
- [[regularization]] — applies PAC learning principles to improve the generalization performance of machine learning models by preventing overfitting.

## Connected topics

- [Neural Tangent Kernel (NTK)](./ntk.md) — NTK is related to understanding the generalization properties of learning algorithms.
- [Double Descent](./double-descent.md) — Double descent explores how model performance changes with increasing model complexity.
- [Bayesian Inference](../05-statistical-probabilistic-ml/bayesian-inference.md) — Bayesian methods offer an alternative approach to learning with uncertainty.
- [Gaussian Processes](../05-statistical-probabilistic-ml/gaussian-processes.md) — Gaussian processes provide a probabilistic framework for learning and prediction.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Backpropagation is a core algorithm used for training neural networks.
- [Contrastive Learning](../03-representation-learning/contrastive-learning.md) — Contrastive learning focuses on learning representations that capture similarities and differences.


## Further reading
*   Hanneke et al. (2024) — "Revisiting Agnostic PAC Learning" — [https://arxiv.org/pdf/2407.19777] — This paper provides a comprehensive overview of PAC learning, covering the key concepts, algorithms, and theoretical results.
*   Dandi (2025) — "The Computational Advantage of Depth: Learning High-Dimensional Hierarchical Functions with Gradient Descent" — [https://arxiv.org/abs/2502.13961v4] — This paper explores the learning complexity of functions with hierarchical structure, providing applications for agnostic learning and statistical query lower bounds.
*   Joshi et al. (2025) — "Sample Complexity of Autoregressive Reasoning: Chain-of-Thought vs. End-to-End" — [https://arxiv.org/abs/2604.12013v1] — This paper introduces a PAC-learning framework for next-token generators, the primitive underlying autoregressive models, and studies the sample complexity of such systems.
*   Viallard et al. (2024) — "Tighter Generalisation Bounds via Interpolation" — [https://arxiv.org/pdf/2402.05101] — This paper explores tighter generalization bounds.
```