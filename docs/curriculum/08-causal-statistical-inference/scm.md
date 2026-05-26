---
title: Structural Causal Models
track: 08-causal-statistical-inference
tags: [scm, causal-graphs, dag, structural-equations, pearl]
depth: foundations
prereqs: []
updated: 2026-05-25
---

# Structural Causal Models
> **TL;DR:** A formal language for causation — directed acyclic graphs encoding causal relationships as structural equations, enabling reasoning about interventions and counterfactuals beyond what observational data alone allows.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
An SCM consists of: a set of variables V, a directed acyclic graph G where edges encode causal relationships, structural equations x_i = f_i(pa_i, u_i) for each variable, and exogenous noise variables u_i. The graph encodes the causal structure; the equations encode the functional form. This gives three levels of causal reasoning: observational (seeing), interventional (doing), counterfactual (imagining).

## Why it matters at the frontier
SCMs are foundational for causal AI, interpretability, and robust ML. They formalize what it means for an AI to understand causation rather than correlation — a prerequisite for systems that can reason about interventions, generalize under distribution shift, and answer "what if" questions.

## Core concepts
- **DAG** — directed acyclic graph; nodes = variables, edges = direct causes
- **Structural equation** — x_i = f_i(pa_i, u_i); defines how variable x_i is determined
- **Exogenous variables u_i** — noise/unobserved causes; independent across variables
- **d-separation** — graphical criterion for reading conditional independencies from the DAG
- **Markov condition** — each variable is independent of its non-descendants given its parents
- **Pearl's causal hierarchy** — three rungs: association (P(y|x)), intervention (P(y|do(x))), counterfactual (P(y_x|x'))

## Mathematical foundations
A simple SCM for Y ← X + noise:
- X = U_X (exogenous)
- Y = f(X, U_Y) = X + U_Y

Intervention (do(X=x)): remove the equation for X, replace with X = x, recompute Y.

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Causality (Chapters 1-3) | 2009 | Pearl | The foundational book; graphical models + do-calculus |
| [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107) | 2021 | Schölkopf et al. | Bridge between causal theory and ML |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [A General Identification Condition for Causal Effects](https://ftp.cs.ucla.edu/pub/stat_ser/R327.pdf) | 2006 | Shpitser & Pearl | Complete identification via do-calculus |

## Current SotA
> *Updated: 2026-05-25*
Causal representation learning — learning the SCM structure from data — is an active frontier. Current work connects disentanglement, ICA, and causal discovery. LLM-assisted causal discovery (using language models to propose causal structures) is emerging. Causal inference methods are being integrated into ML pipelines for robustness.

## Connected topics
- [The Do-Calculus](./do-calculus.md) — the algebra of interventions on SCMs
- [Counterfactual Reasoning](./counterfactuals.md) — the third rung of Pearl's causal hierarchy
- Disentanglement — learning causal factors of variation

## Further reading
- [Elements of Causal Inference](https://mitpress.mit.edu/9780262037310/elements-of-causal-inference/) — Peters, Janzing, Schölkopf (MIT Press open access)
