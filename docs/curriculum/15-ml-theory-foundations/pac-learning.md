---
title: PAC Learning
track: 15-ml-theory-foundations
tags: [pac-learning, sample-complexity, vc-dimension, generalization, statistical-learning]
depth: theoretical
prereqs: []
updated: 2026-05-25
---

# PAC Learning
> **TL;DR:** Probably Approximately Correct learning — a formal framework for asking when a learning algorithm can find a hypothesis that is approximately correct with high probability, from a bounded number of examples.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
PAC learning formalizes the question: how many examples do I need to learn a concept? A hypothesis class H is PAC-learnable if there exists an algorithm that, given ε (accuracy) and δ (confidence), outputs with probability ≥ 1-δ a hypothesis h with error ≤ ε, using a polynomial number of samples. The VC dimension characterizes the sample complexity of PAC learning for binary classification.

## Core concepts
- **PAC framework** — probably (≥1-δ) approximately (error ≤ε) correct
- **Sample complexity** — minimum examples needed for PAC learning
- **VC dimension** — the maximum number of points a hypothesis class can shatter
- **Shattering** — for any labeling of a set of points, some hypothesis in H achieves it
- **Fundamental theorem of statistical learning** — a class is PAC-learnable iff VC dimension is finite
- **Agnostic learning** — PAC learning when no hypothesis achieves zero training error

## Mathematical foundations
PAC sample complexity bound:
\[
m \geq \frac{1}{\epsilon}\left(\text{VC}(H)\ln\frac{1}{\epsilon} + \ln\frac{1}{\delta}\right)
\]

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Understanding Machine Learning: From Theory to Algorithms (Ch. 2-6) | 2014 | Shalev-Shwartz & Ben-David | Best textbook; arXiv version free |

## Current SotA
> *Updated: 2026-05-25*
Classical PAC theory doesn't directly explain deep learning (VC dimension of networks is enormous, yet they generalize). Modern extensions include algorithmic stability, PAC-Bayes, and the Decision-Estimation Coefficient (Foster et al. 2021) for interactive learning. Double descent has partially reconciled theory with practice.

## Connected topics
- Vc Dimension — the primary complexity measure in PAC theory
- Rademacher Complexity — data-dependent generalization bounds
- [Double Descent](./double-descent.md) — modern empirical phenomenon that challenges PAC predictions

## Further reading
- [Understanding Machine Learning](https://www.cs.huji.ac.il/~shais/UnderstandingMachineLearning/) — Shalev-Shwartz & Ben-David; free online
