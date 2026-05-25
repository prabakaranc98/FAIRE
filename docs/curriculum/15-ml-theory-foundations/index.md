---
title: ML Theory & Foundations
tags: [ml-theory, generalization, pac-learning, information-theory, optimization-theory, statistical-learning]
---

# Track 15 · ML Theory & Foundations

> The mathematical foundations of learning: PAC learning, generalization theory, optimization landscapes, information theory, and the theoretical questions behind modern deep learning.

Theory asks: why does learning work at all? What guarantees can we give? When does it fail? This track covers the mathematical infrastructure — from classical statistical learning theory to the open theoretical questions of deep learning.

---

## Topics

### Statistical Learning Theory
- [PAC Learning](pac-learning.md) — probably approximately correct framework, sample complexity
- [VC Dimension](vc-dimension.md) — Vapnik-Chervonenkis theory, shattering, growth function
- [Rademacher Complexity](rademacher-complexity.md) — data-dependent complexity measures, uniform convergence
- [Bias-Variance Tradeoff](bias-variance.md) — decomposition, interpolation regime, double descent

### Generalization in Deep Learning
- [Generalization in Deep Networks](generalization-deep.md) — overparameterization, implicit regularization, flat minima
- [Double Descent](double-descent.md) — the modern interpolation-generalization tradeoff
- [Implicit Bias of SGD](implicit-bias.md) — margin maximization, edge-of-stability, catapult phase

### Optimization Theory
- [Convex Optimization](convex-optimization.md) — gradient descent convergence, strong convexity, smoothness
- [Non-Convex Optimization](nonconvex-optimization.md) — saddle points, local minima, loss landscape geometry
- [Neural Tangent Kernel](ntk.md) — infinite-width limit, lazy training, kernel regime vs. feature learning

### Information Theory
- [Entropy & Mutual Information](entropy.md) — Shannon entropy, KL divergence, information bottleneck
- [Rate-Distortion Theory](rate-distortion.md) — compression, lossy coding, connections to representation learning
- [Minimum Description Length](mdl.md) — MDL principle, Kolmogorov complexity, compression as learning

### Probability Theory
- [Concentration Inequalities](concentration.md) — Hoeffding, Bernstein, McDiarmid, applications to ML
- [High-Dimensional Statistics](high-dimensional.md) — curse of dimensionality, Johnson-Lindenstrauss, random projections

---

## Connections to frontier research

- **Grokking** — delayed generalization as a window into learning dynamics
- **Mechanistic interpretability** — theory of what neural circuits compute
- **Scaling laws** — theoretical accounts of why compute × data = capability
- **Benign overfitting** — when overparameterized models generalize despite zero training loss

---

## Recommended entry points

Start with [PAC Learning](pac-learning.md) and [Bias-Variance Tradeoff](bias-variance.md) for classical theory. For modern relevance, [Double Descent](double-descent.md) and [Neural Tangent Kernel](ntk.md) connect theory to current practice.
