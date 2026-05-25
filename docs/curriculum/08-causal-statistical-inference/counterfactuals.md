---
title: Counterfactual Reasoning
track: 08-causal-statistical-inference
tags: [counterfactuals, potential-outcomes, rubin, pearl, causal-hierarchy]
depth: foundations
prereqs: [scm, do-calculus]
updated: 2026-05-25
---

# Counterfactual Reasoning
> **TL;DR:** Reasoning about what would have happened under different circumstances — "If X had been different, would Y have changed?" — the third and deepest rung of Pearl's causal hierarchy.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Counterfactual inference asks: given that we observed X=x and Y=y, what would Y have been if X had been x'? Unlike interventional reasoning (do-calculus), counterfactuals require knowing the exogenous noise values for a specific individual — they are person-level, not population-level. This is computed via the abduction-action-prediction procedure.

## Core concepts
- **Potential outcomes** — Y(x): the value Y would take under intervention do(X=x)
- **Fundamental problem of causal inference** — we can never observe both Y(1) and Y(0) for the same unit
- **Individual treatment effect** — ITE = Y(1) - Y(0); unobservable
- **Average treatment effect** — ATE = E[Y(1) - Y(0)]; estimable under assumptions
- **Abduction** — infer U from observed (X=x, Y=y) using the structural equations
- **Action** — modify the graph to reflect the counterfactual (e.g., set X=x')
- **Prediction** — forward-propagate through modified SCM to get Y_{X=x'}

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Causality](https://www.cambridge.org/core/books/causality/B0046844FAE10CBF274D4ACBDAEB5F5B) | 2009 | Pearl | Chapter 7 — counterfactual logic and potential outcomes |
| [Causal Inference: What If](https://www.hsph.harvard.edu/miguel-hernan/causal-inference-book/) | 2020 | Hernán & Robins | Potential outcomes framework; free textbook |

## Current SotA
> *Updated: 2026-05-25*
Counterfactual reasoning is a central challenge for AI interpretability: "what input would have changed this model's output?" Counterfactual explanations in ML use this framework. Causal representation learning aims to learn models where counterfactuals are well-defined.

## Connected topics
- [[scm]] — counterfactuals are computed using structural equations
- [[causal-discovery]] — discovering the graph that enables counterfactual queries
- [[disentanglement]] — learning representations where counterfactuals are meaningful

## Further reading
- [Counterfactual Explanations Without Opening the Black Box](https://arxiv.org/abs/1711.00399) — Wachter et al. 2017
