---
title: The Do-Calculus
track: 08-causal-statistical-inference
tags: [do-calculus, interventions, identification, pearl, causal-inference]
depth: foundations
prereqs: [scm]
updated: 2026-05-25
---

# The Do-Calculus
> **TL;DR:** Pearl's algebra for interventions — three rules that allow any causal query expressible in terms of observations, bypassing the need for randomized experiments when the causal graph is known.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
The do-calculus is a set of three inference rules for transforming expressions involving the do-operator P(Y|do(X)=x) into purely observational quantities P(Y|X). Given a causal graph G, these rules tell you when and how a causal effect can be identified from observational data. When identification is possible, you can estimate causal effects without running a randomized experiment.

## Core concepts
- **do-operator** — do(X=x): an intervention that sets X to x regardless of its causes
- **Identification** — a causal effect is identifiable if it can be expressed in observational terms
- **Backdoor criterion** — a sufficient condition for identification: control for a set Z that blocks all backdoor paths
- **Frontdoor criterion** — alternative identification when no valid backdoor set exists
- **Rule 1** — insertion/deletion of observations (d-separation)
- **Rule 2** — action/observation exchange (when do and conditioning are equivalent)
- **Rule 3** — insertion/deletion of actions (when intervention has no effect)

## Mathematical foundations
Backdoor adjustment formula:
$$P(Y | do(X=x)) = \sum_z P(Y | X=x, Z=z) P(Z=z)$$

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Causality, Chapter 3 | 2009 | Pearl | Definitive treatment of the do-calculus |
| [Causal Inference in Statistics: A Primer](http://bayes.cs.ucla.edu/PRIMER/) | 2016 | Pearl, Glymour, Jewell | Accessible introduction to do-calculus |

## Connected topics
- [[scm]] — do-calculus operates on structural causal models
- [[counterfactuals]] — the third rung uses do-calculus as its foundation
- [[observational-studies]] — do-calculus provides the theoretical basis for causal inference from observational data
