---
title: "Arc: Causal AI — From Correlation to Causation"
arc: causal-ai
super_domain: B-Modeling
tracks: [08-causal-statistical-inference, 05-statistical-probabilistic-ml, 03-representation-learning]
estimated_depth: "6-8 weeks, ~24 papers"
prereqs: [bayesian-inference, basic-probability, linear-algebra]
---

# Arc: Causal AI
> **What this arc builds:** A principled framework for reasoning about causes and effects — from structural causal models and the do-calculus to causal representation learning and its applications to robustness, interpretability, and biology.

## Why this arc exists

Statistics tells you what is associated. Causality tells you what would happen if you intervened. This gap — between correlation and causation — is one of the most important distinctions in science, and increasingly central to frontier AI.

The arc moves from the mathematical foundations (Judea Pearl's work) through modern causal inference methods (econometric tools, doubly-robust estimation) to the research frontier: learning causal structure from data, causal representations in deep learning, and applications in biology and AI safety.

## Prerequisites

Probability (conditional probability, Bayes' theorem), basic linear regression, graphical models (helpful but not required).

## The sequence

**Foundations**

1. **Potential outcomes framework** (foundational) — Neyman-Rubin; Y(1), Y(0); fundamental problem of causal inference.
2. **Randomized experiments** (applied) — RCTs; why randomization identifies causal effects; internal vs. external validity.
3. **Structural Causal Models** (foundational) — DAGs, structural equations, exogenous noise; Pearl's framework. [→](../../curriculum/08-causal-statistical-inference/scm.md)
4. **The Do-Calculus** (theoretical) — three rules for intervening; identification from observational data. [→](../../curriculum/08-causal-statistical-inference/do-calculus.md)
5. **Counterfactual reasoning** (theoretical) — abduction-action-prediction; the third rung of Pearl's hierarchy. [→](../../curriculum/08-causal-statistical-inference/counterfactuals.md)

**Causal Inference Methods**

6. **Confounding & backdoor adjustment** (applied) — controlling for confounders; when it works and when it fails.
7. **Propensity scores** (applied) — matching, IPW; balancing treatment and control groups.
8. **Instrumental variables** (applied) — when no valid backdoor set exists; IV identification.
9. **Regression discontinuity & DiD** (applied) — natural experiments; quasi-experimental designs.
10. **Double ML** (applied) — Chernozhukov et al.; debiased estimation via cross-fitting.
11. **Causal forests** (applied) — heterogeneous treatment effects; when effect varies across individuals.

**Causal Discovery**

12. **PC algorithm** (theoretical) — constraint-based causal discovery from independence tests.
13. **NOTEARS** (applied) — DAG learning as continuous optimization; differentiable causal discovery.
14. **Score-based discovery** (theoretical) — GES algorithm; asymptotic correctness guarantees.

**Causal Representation Learning**

15. **Invariant Risk Minimization** (frontier) — Arjovsky et al.; learn features invariant across environments. [→](https://arxiv.org/abs/1907.02893)
16. **Nonlinear ICA & identifiability** (theoretical) — recovering independent causal factors from mixed observations.
17. **Causal Representation Learning** (frontier) — Schölkopf et al. 2021; bridge between causal theory and deep learning. [→](https://arxiv.org/abs/2102.11107)
18. **Disentanglement** (applied) — learning separate dimensions for causal factors; β-VAE, TCVAE. [→](../../curriculum/08-causal-statistical-inference/disentanglement.md)
19. **Unifying Causal Rep. Learning** (frontier) — Yao et al. ICLR 2025; invariance principle perspective. [→](https://arxiv.org/abs/2409.02772)

**Applications**

20. **Causal AI for biology** (frontier) — perturbation modeling; GEARS; drug target discovery.
21. **Causal interpretability** (frontier) — causal abstractions; Geiger et al.; interpretability as causal intervention.

## Key figures

- **Judea Pearl** (UCLA) — SCMs, do-calculus, counterfactuals (Turing Award 2011)
- **Bernhard Schölkopf** (MPI-IS) — causal representation learning
- **Victor Chernozhukov** (MIT) — double ML, causal inference for econometrics
- **Martin Arjovsky** — Invariant Risk Minimization

## Essential reading sequence

1. *Causality* Ch. 1-3 — Pearl 2009 — SCMs and do-calculus
2. [Elements of Causal Inference](https://mitpress.mit.edu/9780262037310/) — Peters, Janzing, Schölkopf (open access) — rigorous modern treatment
3. [Estimating Causal Effects of Treatments in Randomized and Non-randomized Studies](https://psycnet.apa.org/record/1974-30770-001) — Rubin 1974 — potential outcomes
4. [Double/Debiased Machine Learning](https://arxiv.org/abs/1608.00060) — Chernozhukov et al. 2016
5. [Towards Causal Representation Learning](https://arxiv.org/abs/2102.11107) — Schölkopf et al. 2021
6. [Invariant Risk Minimization](https://arxiv.org/abs/1907.02893) — Arjovsky et al. 2019

## Current frontier anchors
> As of 2026-05-25

- **Causal representation learning** — learning SCM structure from high-dimensional data; active frontier
- **Causal abstractions in interpretability** — formalizing what a circuit "implements" using causal language (Geiger et al.)
- **Perturbation biology** — CRISPR screen modeling with causal models; GEARS, scPerturb

## What you'll know when done

1. Explain the difference between P(Y|X=x) and P(Y|do(X=x)) in concrete terms
2. Apply the backdoor criterion to determine when a causal effect is identifiable
3. Implement double ML for heterogeneous treatment effects on a tabular dataset
4. Explain what IRM is trying to do and why it doesn't fully solve out-of-distribution generalization
5. Describe what "causal representation learning" means and what the open identifiability problem is

## Branch points to other arcs

- **→ Generative Stack arc**: Disentanglement connects causal factors to latent variable models
- **→ RL arc**: Causal inference for off-policy evaluation; causal RL
- **→ Scientific AI arc**: Causal perturbation modeling in biology

## Where to go next

[Generative Stack arc →](../generative-stack/index.md) — Disentanglement and latent variable models

[Scientific AI arc →](../scientific-ai/index.md) — Causal perturbation modeling for biology
