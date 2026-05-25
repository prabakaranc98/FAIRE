---
title: Gaussian Processes
track: 05-statistical-probabilistic-ml
tags: [gaussian-processes, kernel, nonparametric, regression, uncertainty]
depth: foundations
prereqs: [bayesian-inference]
updated: 2026-05-25
---

# Gaussian Processes
> **TL;DR:** A Bayesian nonparametric model that places a prior over functions — making predictions with principled uncertainty bounds, without specifying a fixed functional form.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **GP prior** — a distribution over functions; any finite collection of function values is jointly Gaussian
- **Kernel function** — k(x, x'); encodes similarity and smoothness assumptions
- **Posterior GP** — updated distribution after conditioning on observed data
- **Predictive mean and variance** — closed-form predictions with uncertainty
- **Marginal likelihood** — used for kernel hyperparameter optimization

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Gaussian Processes for Machine Learning](https://gaussianprocess.org/gpml/) | 2006 | Rasmussen & Williams | The canonical reference (open access) |

## Current SotA
> *Updated: 2026-05-25*
GPs remain essential for Bayesian optimization (hyperparameter tuning, active learning) and small-data scientific applications. Deep kernel learning and neural tangent kernels connect GPs to modern neural networks. Sparse GPs (inducing points) scale to larger datasets.

## Connected topics
- [[bayesian-inference]] — GPs are Bayesian nonparametric models
- [[ntk]] — infinite-width neural networks converge to GPs
- [[uncertainty-quantification]] — GPs provide calibrated uncertainty by design
