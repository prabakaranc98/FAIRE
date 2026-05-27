---
title: Supervisor Report
description: Automated wiki health assessment and editorial priorities
---

# Supervisor Report

> Generated: **2026-05-27 04:44 EDT** by the Frontier Wiki supervising agent

## Wiki Health

| Metric | Value |
|---|---|
| Total pages (stubs + generated) | **81** |
| Stub pages (not yet generated) | **10** |
| Generated pages | **71** |
| Approved (conf ≥ 0.8) | **64** (90%) |
| Flagged (conf < 0.8) | **4** |
| Avg reviewer confidence | **0.77** |
| Tracks with content | **10** / 10 |
| Pages with MVB | **0** |
| Critical audit issues | **0** |
| Warnings | **11** |

## Editorial Analysis

Wiki has 81 pages (10 stubs). 4 pages need revision. Focus: generate stubs and improve flagged pages.

## Prioritised Work Queue

| # | Priority | Action | Topic | Track | Reason |
|---|---|---|---|---|---|
| 1 | 🔴 critical | improve | bayesian-inference | 05-statistical-probabilistic-ml | Previous attempt errored (conf=0.81) — retry with current prompts |
| 2 | 🔴 critical | improve | bayesian-neural-networks | 05-statistical-probabilistic-ml | Previous attempt errored (conf=0.71) — retry with current prompts |
| 3 | 🔴 critical | improve | data-parallelism | 09-algorithms-systems-for-ai | Reviewer flagged (conf=0.61) — needs targeted revision |
| 4 | 🔴 critical | improve | do-calculus | 08-causal-statistical-inference | Reviewer flagged (conf=0.63) — needs targeted revision |
| 5 | 🔴 critical | improve | pipeline-parallelism | 09-algorithms-systems-for-ai | Previous attempt errored (conf=0.81) — retry with current prompts |
| 6 | 🔴 critical | improve | policy-evaluation | 08-causal-statistical-inference | Reviewer flagged (conf=0.60) — needs targeted revision |
| 7 | 🔴 critical | improve | variational-inference | 05-statistical-probabilistic-ml | Reviewer flagged (conf=0.66) — needs targeted revision |
| 8 | 🟡 medium | generate | attention-mechanisms | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 9 | 🟡 medium | generate | data-augmentation | 03-representation-learning | Stub page — not yet generated |
| 10 | 🟡 medium | generate | distributed-training-arc | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 11 | 🟡 medium | generate | gradient-descent | 04-neural-networks-deep-learning | Stub page — not yet generated |
| 12 | 🟡 medium | generate | llm-architecture-optimizations | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 13 | 🟡 medium | generate | mixed-precision-training | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 14 | 🟡 medium | generate | normalization | 04-neural-networks-deep-learning | Stub page — not yet generated |
| 15 | 🟡 medium | generate | post-training-quantization | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 16 | 🟡 medium | generate | potential-outcomes | 08-causal-statistical-inference | Stub page — not yet generated |
| 17 | 🟡 medium | generate | transformer-architecture | 04-neural-networks-deep-learning | Stub page — not yet generated |

## Audit Issues

## Audit — 2026-05-27 04:44 EDT
> 81 pages scanned · 0 critical · 11 warnings

### Warning
- **curriculum/core/06-reinforcement-learning/concepts/model-based-reinforcement-learning.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/06-reinforcement-learning/concepts/ppo.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/05-statistical-probabilistic-ml/concepts/mcmc.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/attention.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/02-generative-modeling/concepts/flow-matching.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/02-generative-modeling/concepts/latent-diffusion-models.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/08-causal-statistical-inference/concepts/do-calculus.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/08-causal-statistical-inference/concepts/policy-evaluation.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/03-representation-learning/concepts/self-supervised-learning.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/04-neural-networks-deep-learning/concepts/scaling-laws.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache.md** [no_arxiv]: No arXiv sources found — page needs academic citations

### Info
- **curriculum/core/06-reinforcement-learning/concepts/model-based-reinforcement-learning.md** [stale_sota]: Page updated 184 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/mdp.md** [stale_sota]: Page updated 508 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/policy-gradients.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/world-models.md** [stale_sota]: Page updated 232 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/ppo.md** [stale_sota]: Page updated 351 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/q-learning.md** [stale_sota]: Page updated 543 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/actor-critic.md** [stale_sota]: Page updated 568 days ago — SotA may be outdated
- **curriculum/core/06-reinforcement-learning/concepts/policy-gradient.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/mcmc.md** [stale_sota]: Page updated 301 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/gaussian-processes.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-inference.md** [stale_sota]: Page updated 572 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/uncertainty-quantification.md** [stale_sota]: Page updated 542 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/expectation-maximization.md** [stale_sota]: Page updated 561 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-neural-networks.md** [stale_sota]: Page updated 466 days ago — SotA may be outdated
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
- **curriculum/core/08-causal-statistical-inference/concepts/do-calculus.md** [stale_sota]: Page updated 550 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/potential-outcomes.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/08-causal-statistical-inference/concepts/counterfactuals.md** [stale_sota]: Page updated 412 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/policy-evaluation.md** [stale_sota]: Page updated 467 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/structural-causal-models.md** [stale_sota]: Page updated 559 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/jepa.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/simclr.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/self-supervised-learning.md** [stale_sota]: Page updated 464 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/contrastive-learning.md** [stale_sota]: Page updated 330 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/masked-autoencoders.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/representation-learning.md** [stale_sota]: Page updated 219 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/data-augmentation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/04-neural-networks-deep-learning/concepts/scaling-laws.md** [stale_sota]: Page updated 374 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/backpropagation.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/residual-connections.md** [stale_sota]: Page updated 181 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/transformer-architecture.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/04-neural-networks-deep-learning/concepts/normalization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/04-neural-networks-deep-learning/concepts/gradient-descent.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/04-neural-networks-deep-learning/concepts/optimization.md** [stale_sota]: Page updated 232 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/batch-normalization.md** [stale_sota]: Page updated 592 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/llm-architecture-optimizations.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/distributed-training.md** [stale_sota]: Page updated 396 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/quantization-aware-training.md** [stale_sota]: Page updated 410 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/mixed-precision-training.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/model-parallelism.md** [stale_sota]: Page updated 238 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/tensor-parallelism.md** [stale_sota]: Page updated 382 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/post-training-quantization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/distributed-training-arc.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache.md** [stale_sota]: Page updated 394 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/attention-mechanisms.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/pipeline-parallelism.md** [stale_sota]: Page updated 564 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/inference-optimization.md** [stale_sota]: Page updated 228 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/flash-attention.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/data-parallelism.md** [stale_sota]: Page updated 546 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/transformer.md** [stale_sota]: Page updated 511 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/reward-modeling.md** [stale_sota]: Page updated 388 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/alignment-safety.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/mechanistic-interpretability.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/mixture-of-experts.md** [stale_sota]: Page updated 572 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/rlhf.md** [stale_sota]: Page updated 417 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/chain-of-thought.md** [stale_sota]: Page updated 227 days ago — SotA may be outdated


---

*This report is regenerated automatically. The supervisor updates `agents/sprints/current.md` with the top work items.*