---
title: Causal & Statistical Inference
tags: [causality, causal-inference, scm, do-calculus, counterfactuals]
---

# Track 08 · Causal & Statistical Inference

> Moving from correlation to causation: structural causal models, the do-calculus, counterfactual reasoning, and causal representation learning.

Statistical inference tells you what is associated. Causal inference tells you what would happen if you intervened. This distinction — correlation vs. causation — is one of the deepest ideas in science and increasingly central to frontier AI research.

---

## Topics

### Foundations of Causality
- [Structural Causal Models](scm.md) — DAGs, structural equations, exogenous noise, causal graphs
- [The Do-Calculus](do-calculus.md) — interventions, Pearl's do operator, identification
- [Counterfactual Reasoning](counterfactuals.md) — potential outcomes, counterfactual worlds, abduction-action-prediction

### Causal Inference Methods
- [Randomized Experiments](randomized-experiments.md) — RCTs, average treatment effects, internal vs. external validity
- [Observational Studies](observational-studies.md) — confounding, instrumental variables, regression discontinuity
- [Propensity Score Methods](propensity-scores.md) — matching, inverse probability weighting

### Statistical Inference
- [Hypothesis Testing](hypothesis-testing.md) — null/alternative, p-values, multiple comparisons, power
- [Estimation & Confidence Intervals](estimation.md) — MLE, method of moments, bootstrap
- [Information Theory](information-theory.md) — entropy, mutual information, KL divergence, capacity

### Causal Representation Learning
- [Disentanglement](disentanglement.md) — independent components, identifiability, causal factors
- [Causal Discovery](causal-discovery.md) — PC algorithm, FCI, score-based methods, LLM-assisted discovery
- [Out-of-Distribution Generalization](ood-generalization.md) — invariant risk minimization, domain generalization

---

## Connections to frontier research

- **Causal world models** — agents that understand interventions, not just correlations
- **Interpretability** — causal attribution for model behavior
- **Alignment** — causal framing of AI safety: what does the model actually optimize?
- **Scientific AI** — causal inference for biological systems, drug discovery

---

## Recommended entry points

Start with [Structural Causal Models](scm.md) and [The Do-Calculus](do-calculus.md). Then [Counterfactual Reasoning](counterfactuals.md) for the philosophical depth, and [Causal Representation Learning](disentanglement.md) for the frontier.
