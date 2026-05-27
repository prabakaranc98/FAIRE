---
title: Supervisor Report
description: Automated wiki health assessment and editorial priorities
---

# Supervisor Report

> Generated: **2026-05-27 07:49 UTC** by the Frontier Wiki supervising agent

## Wiki Health

| Metric | Value |
|---|---|
| Total pages (stubs + generated) | **71** |
| Stub pages (not yet generated) | **14** |
| Generated pages | **57** |
| Approved (conf ≥ 0.8) | **50** (87%) |
| Flagged (conf < 0.8) | **0** |
| Avg reviewer confidence | **0.77** |
| Tracks with content | **1** / 15 |
| Pages with MVB | **0** |
| Critical audit issues | **0** |
| Warnings | **9** |

## Editorial Analysis

We are in a strong structural position with over eighty percent of our target pages generated and a steady upward quality trend, but we must now address a persistent quality deficit of 0.08 to push our average reviewer confidence past the threshold. Our immediate bottleneck lies not in coverage, but in execution: a cluster of technically demanding pages—including actor-critic, do-calculus, and optimization—recently errored out during generation, while our self-supervised learning entry remains flagged due to low reviewer confidence. In the upcoming sprint, the team must prioritize targeted retries and revisions for these high-impact topics to stabilize our core content. Once these quality gaps are resolved, we should leverage our healthy twenty-one dollar budget to begin systematically converting foundational stubs like backpropagation and Bayesian inference. Fortunately, we face no critical audit issues or severe budget constraints, giving us the perfect operational runway to focus purely on mathematical rigor and prose refinement as we polish these complex algorithmic pages.

## Prioritised Work Queue

| # | Priority | Action | Topic | Track | Reason |
|---|---|---|---|---|---|
| 1 | 🔴 critical | improve | actor-critic | 06-reinforcement-learning | Previous attempt errored (conf=0.76) — retry with current prompts |
| 2 | 🔴 critical | improve | counterfactuals | 08-causal-statistical-inference | Previous attempt errored (conf=0.80) — retry with current prompts |
| 3 | 🔴 critical | improve | do-calculus | 08-causal-statistical-inference | Previous attempt errored (conf=0.78) — retry with current prompts |
| 4 | 🔴 critical | improve | optimization | 04-neural-networks-deep-learning | Previous attempt errored (conf=0.72) — retry with current prompts |
| 5 | 🔴 critical | improve | quantization | 09-algorithms-systems-for-ai | Previous attempt errored (conf=0.71) — retry with current prompts |
| 6 | 🔴 critical | improve | self-supervised-learning | 03-representation-learning | Reviewer flagged (conf=0.61) — needs targeted revision |
| 7 | 🔴 critical | improve | variational-inference | 05-statistical-probabilistic-ml | Previous attempt errored (conf=0.72) — retry with current prompts |
| 8 | 🟡 medium | generate | backpropagation | 04-neural-networks-deep-learning | Stub page — not yet generated |
| 9 | 🟡 medium | generate | bayesian-inference | 05-statistical-probabilistic-ml | Stub page — not yet generated |
| 10 | 🟡 medium | generate | bayesian-neural-networks | 05-statistical-probabilistic-ml | Stub page — not yet generated |
| 11 | 🟡 medium | generate | contrastive-learning | 03-representation-learning | Stub page — not yet generated |
| 12 | 🟡 medium | generate | data-parallelism | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 13 | 🟡 medium | generate | flash-attention | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 14 | 🟡 medium | generate | model-based-reinforcement-learning | 06-reinforcement-learning | Stub page — not yet generated |
| 15 | 🟡 medium | generate | pipeline-parallelism | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 16 | 🟡 medium | generate | policy-evaluation | 08-causal-statistical-inference | Stub page — not yet generated |
| 17 | 🟡 medium | generate | policy-gradients | 06-reinforcement-learning | Stub page — not yet generated |
| 18 | 🟡 medium | generate | quantization-aware-training | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 19 | 🟡 medium | generate | representation-learning | 03-representation-learning | Stub page — not yet generated |
| 20 | 🟡 medium | generate | reward-modeling | 01-ai | Stub page — not yet generated |

## Audit Issues

## Audit — 2026-05-27 07:49 UTC
> 71 pages scanned · 0 critical · 9 warnings

### Warning
- **curriculum/core/06-reinforcement-learning/concepts/ppo.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/06-reinforcement-learning/concepts/actor-critic.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/05-statistical-probabilistic-ml/concepts/mcmc.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/attention.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/02-generative-modeling/concepts/flow-matching.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/02-generative-modeling/concepts/latent-diffusion-models.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/03-representation-learning/concepts/self-supervised-learning.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/04-neural-networks-deep-learning/concepts/scaling-laws.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache.md** [no_arxiv]: No arXiv sources found — page needs academic citations

### Info
- **curriculum/core/06-reinforcement-learning/concepts/model-based-reinforcement-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/06-reinforcement-learning/concepts/mdp.md** [stale_sota]: Page updated 508 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/policy-gradients.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/06-reinforcement-learning/concepts/world-models.md** [stale_sota]: Page updated 232 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/ppo.md** [stale_sota]: Page updated 351 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/q-learning.md** [stale_sota]: Page updated 543 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/actor-critic.md** [stale_sota]: Page updated 568 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/policy-gradient.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/mcmc.md** [stale_sota]: Page updated 301 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/gaussian-processes.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-inference.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/uncertainty-quantification.md** [stale_sota]: Page updated 542 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/expectation-maximization.md** [stale_sota]: Page updated 561 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-neural-networks.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/em-algorithm.md** [stale_sota]: Page updated 602 days ago — SotA may be outdated
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/multi-head-attention.md** [stale_sota]: Page updated 360 days ago — SotA may be outdated
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/in-context-learning.md** [stale_sota]: Page updated 471 days ago — SotA may be outdated
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/positional-encoding.md** [stale_sota]: Page updated 238 days ago — SotA may be outdated
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/attention.md** [stale_sota]: Page updated 549 days ago — SotA may be outdated
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/long-context.md** [stale_sota]: Page updated 203 days ago — SotA may be outdated
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/retrieval-augmented-generation.md** [stale_sota]: Page updated 599 days ago — SotA may be outdated
- **curriculum/core/02-generative-modeling/concepts/diffusion-models.md** [stale_sota]: Page updated 599 days ago — SotA may be outdated
- **curriculum/core/02-generative-modeling/concepts/consistency-models.md** [stale_sota]: Page updated 542 days ago — SotA may be outdated
- **curriculum/core/02-generative-modeling/concepts/score-matching.md** [stale_sota]: Page updated 542 days ago — SotA may be outdated
- **curriculum/core/02-generative-modeling/concepts/flow-matching.md** [stale_sota]: Page updated 558 days ago — SotA may be outdated
- **curriculum/core/02-generative-modeling/concepts/generative-adversarial-networks.md** [stale_sota]: Page updated 234 days ago — SotA may be outdated
- **curriculum/core/02-generative-modeling/concepts/latent-diffusion-models.md** [stale_sota]: Page updated 330 days ago — SotA may be outdated
- **curriculum/core/02-generative-modeling/concepts/variational-autoencoders.md** [stale_sota]: Page updated 542 days ago — SotA may be outdated
- **curriculum/core/10-complexity-cognition-natural-intelligence/concepts/generalization.md** [stale_sota]: Page updated 188 days ago — SotA may be outdated
- **curriculum/core/10-complexity-cognition-natural-intelligence/concepts/compositionality.md** [stale_sota]: Page updated 574 days ago — SotA may be outdated
- **curriculum/core/10-complexity-cognition-natural-intelligence/concepts/emergence.md** [stale_sota]: Page updated 391 days ago — SotA may be outdated
- **curriculum/core/10-complexity-cognition-natural-intelligence/concepts/scaling-collapse.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/causal-representation-learning.md** [stale_sota]: Page updated 600 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/causal-discovery.md** [stale_sota]: Page updated 321 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/instrumental-variables.md** [stale_sota]: Page updated 207 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/mediation-analysis.md** [stale_sota]: Page updated 483 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/do-calculus.md** [stale_sota]: Page updated 563 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/counterfactuals.md** [stale_sota]: Page updated 412 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/policy-evaluation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/08-causal-statistical-inference/concepts/structural-causal-models.md** [stale_sota]: Page updated 559 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/jepa.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/simclr.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/self-supervised-learning.md** [stale_sota]: Page updated 597 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/contrastive-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/03-representation-learning/concepts/masked-autoencoders.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/representation-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/04-neural-networks-deep-learning/concepts/scaling-laws.md** [stale_sota]: Page updated 374 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/backpropagation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/04-neural-networks-deep-learning/concepts/residual-connections.md** [stale_sota]: Page updated 181 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/optimization.md** [stale_sota]: Page updated 545 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/batch-normalization.md** [stale_sota]: Page updated 592 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/distributed-training.md** [stale_sota]: Page updated 396 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/quantization-aware-training.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/model-parallelism.md** [stale_sota]: Page updated 238 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/tensor-parallelism.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache.md** [stale_sota]: Page updated 394 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/pipeline-parallelism.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/inference-optimization.md** [stale_sota]: Page updated 228 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/flash-attention.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/data-parallelism.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/01-ai/concepts/transformer.md** [stale_sota]: Page updated 511 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/reward-modeling.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/01-ai/concepts/alignment-safety.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/mechanistic-interpretability.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/mixture-of-experts.md** [stale_sota]: Page updated 572 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/rlhf.md** [stale_sota]: Page updated 417 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/chain-of-thought.md** [stale_sota]: Page updated 227 days ago — SotA may be outdated


---

*This report is regenerated automatically. The supervisor updates `agents/sprints/current.md` with the top work items.*