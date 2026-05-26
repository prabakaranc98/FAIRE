---
title: Supervisor Report
description: Automated wiki health assessment and editorial priorities
---

# Supervisor Report

> Generated: **2026-05-26 09:46 UTC** by the Frontier Wiki supervising agent

## Wiki Health

| Metric | Value |
|---|---|
| Total pages (stubs + generated) | **171** |
| Stub pages (not yet generated) | **170** |
| Generated pages | **1** |
| Approved (conf ≥ 0.8) | **1** (100%) |
| Flagged (conf < 0.8) | **10** |
| Avg reviewer confidence | **0.64** |
| Tracks with content | **1** / 15 |
| Pages with MVB | **0** |
| Critical audit issues | **0** |
| Warnings | **1** |

## Editorial Analysis

The Frontier Wiki is in a precarious state. While we have a decent number of pages, the vast majority are stubs, and our quality metrics are concerning. The average reviewer confidence is low at 0.64, and the quality trend over the last ten reviews is even worse, with an average confidence of 0.42 and no first-pass approvals. We have a significant coverage deficit, and a backlog of ten flagged pages that require immediate attention. The budget is also critically low. The most pressing issues are the flagged pages, particularly those related to agent architectures, AI hardware, and backpropagation, which need targeted revisions. The team should prioritize improving these flagged pages to raise confidence scores and address the quality deficit. Given the low budget, we need to be strategic in our revisions, focusing on high-impact improvements. The lack of first-pass approvals suggests a potential issue with the initial generation quality or a need to calibrate our reviewers.

## Prioritised Work Queue

| # | Priority | Action | Topic | Track | Reason |
|---|---|---|---|---|---|
| 1 | 🔴 critical | improve | agent-architectures | 01-ai | Reviewer flagged (conf=0.50) — needs targeted revision |
| 2 | 🔴 critical | improve | ai-hardware | 09-algorithms-systems-ai | Reviewer flagged (conf=0.35) — needs targeted revision |
| 3 | 🔴 critical | improve | backpropagation | 04-neural-networks-dl | Reviewer flagged (conf=0.45) — needs targeted revision |
| 4 | 🔴 critical | improve | bayesian-inference | 05-statistical-probabilistic-ml | Reviewer flagged (conf=0.35) — needs targeted revision |
| 5 | 🔴 critical | improve | bayesian-nn | 05-statistical-probabilistic-ml | Reviewer flagged (conf=0.35) — needs targeted revision |
| 6 | 🔴 critical | improve | bootstrapping-methods | 03-representation-learning | Reviewer flagged (conf=0.55) — needs targeted revision |
| 7 | 🔴 critical | improve | cell-simulation | 14-biology-life-sciences | Reviewer flagged (conf=0.25) — needs targeted revision |
| 8 | 🔴 critical | improve | chain-of-thought | 07-attention-memory-reasoning | Reviewer flagged (conf=0.65) — needs targeted revision |
| 9 | 🔴 critical | improve | circuit-complexity | 10-complexity-cognition | Reviewer flagged (conf=0.40) — needs targeted revision |
| 10 | 🔴 critical | improve | classical-algorithms | 09-algorithms-systems-ai | Reviewer flagged (conf=0.40) — needs targeted revision |
| 11 | 🟡 medium | generate | bias-variance | 15-ml-theory-foundations | Stub page — not yet generated |
| 12 | 🟡 medium | generate | classical-planning | 01-ai | Stub page — not yet generated |
| 13 | 🟡 medium | generate | climate-ai | 12-physics-scientific-ai | Stub page — not yet generated |
| 14 | 🟡 medium | generate | cnn | 04-neural-networks-dl | Stub page — not yet generated |
| 15 | 🟡 medium | generate | cognitive-architectures | 10-complexity-cognition | Stub page — not yet generated |
| 16 | 🟡 medium | generate | combinatorial-opt | 13-graph-relational-ai | Stub page — not yet generated |
| 17 | 🟡 medium | generate | complex-systems | 10-complexity-cognition | Stub page — not yet generated |
| 18 | 🟡 medium | generate | complexity-classes | 10-complexity-cognition | Stub page — not yet generated |
| 19 | 🟡 medium | generate | concentration | 15-ml-theory-foundations | Stub page — not yet generated |
| 20 | 🟡 medium | generate | consistency-models | 02-generative-modeling | Stub page — not yet generated |

## Audit Issues

## Audit — 2026-05-26 09:46 UTC
> 171 pages scanned · 0 critical · 1 warnings

### Warning
- **curriculum/07-attention-memory-reasoning/single-head-attention.md** [no_arxiv]: No arXiv sources found — page needs academic citations

### Info
- **curriculum/07-attention-memory-reasoning/efficient-attention.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/transformer.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/vision-language-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/single-head-attention.md** [stale_sota]: Page updated 377 days ago — SotA may be outdated
- **curriculum/07-attention-memory-reasoning/multimodal-generation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/tool-use.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/state-space-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/positional-encodings.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/scaling-laws-llm.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/instruction-tuning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/memory-architectures.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/multimodal-reasoning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/lm-pretraining.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/self-attention.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/07-attention-memory-reasoning/chain-of-thought.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/muzero.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/mdp.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/offline-rl.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/model-based-rl.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/distributional-rl.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/td-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/multi-agent-rl.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/ppo.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/dqn.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/actor-critic.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/dynamic-programming.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/policy-gradient.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/06-reinforcement-learning/rlhf.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/distribution-shift.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/directed-graphical-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/mcmc.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/gaussian-processes.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/bayesian-inference.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/uncertainty-quantification.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/bayesian-nn.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/variational-inference.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/hidden-markov-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/undirected-graphical-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/05-statistical-probabilistic-ml/em.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/scaling-laws.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/transfer-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/rnn.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/regularization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/backpropagation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/hyperparameter-tuning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/normalization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/residual-networks.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/optimization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/mlp.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/04-neural-networks-dl/cnn.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/admet.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/single-cell.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/dna-sequence-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/drug-target.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/protein-design.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/cell-simulation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/molecular-generation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/gene-networks.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/protein-structure.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/protein-lm.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/14-biology-life-sciences/epigenomics.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/simulation-acceleration.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/fno.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/molecular-simulation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/neural-odes.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/operator-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/equivariant-networks.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/geometric-dl.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/hamiltonian-networks.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/deeponet.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/climate-ai.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/12-physics-scientific-ai/pinn.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/02-generative-modeling/energy-based-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/02-generative-modeling/diffusion-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/02-generative-modeling/consistency-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/02-generative-modeling/score-matching.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/02-generative-modeling/flow-matching.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/02-generative-modeling/normalizing-flows.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/02-generative-modeling/generative-adversarial-networks.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/02-generative-modeling/variational-autoencoders.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/circuit-complexity.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/hardness-of-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/animal-intelligence.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/neuroscience-ai.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/memory-systems.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/cognitive-architectures.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/attention-cognition.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/emergent-capabilities.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/information-dynamics.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/complex-systems.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/10-complexity-cognition/complexity-classes.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/recommendation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/combinatorial-opt.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/message-passing.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/knowledge-graphs.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/gnn-expressivity.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/relational-biases.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/geometric-unification.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/scene-graphs.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/equivariant-gnn.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/molecular-property.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/graph-transformers.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/13-graph-relational-ai/spectral-graph.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/disentanglement.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/information-theory.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/propensity-scores.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/causal-discovery.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/hypothesis-testing.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/scm.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/observational-studies.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/do-calculus.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/estimation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/randomized-experiments.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/counterfactuals.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/08-causal-statistical-inference/ood-generalization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/mdl.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/ntk.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/double-descent.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/entropy.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/rademacher-complexity.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/rate-distortion.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/convex-optimization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/vc-dimension.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/high-dimensional.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/concentration.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/pac-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/bias-variance.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/generalization-deep.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/nonconvex-optimization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/15-ml-theory-foundations/implicit-bias.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/control-theory.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/slam.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/sim-to-real.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/visual-perception.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/tactile-sensing.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/locomotion.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/foundation-models-robotics.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/imitation-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/rl-robotics.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/kinematics.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/manipulation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/11-robotics-embodied-ai/world-models-robotics.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/03-representation-learning/geometry.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/03-representation-learning/probing.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/03-representation-learning/jepa.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/03-representation-learning/bootstrapping-methods.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/03-representation-learning/self-supervised-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/03-representation-learning/contrastive-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/03-representation-learning/masked-autoencoders.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/03-representation-learning/redundancy-reduction.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/01-ai/multi-agent-systems.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/01-ai/agent-architectures.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/01-ai/knowledge-representation.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/01-ai/constraint-satisfaction.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/01-ai/classical-planning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/01-ai/logic.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/01-ai/search-algorithms.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/gpu-architecture.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/gradient-checkpointing.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/model-parallelism.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/mixed-precision.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/classical-algorithms.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/kv-cache.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/speculative-decoding.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/ai-hardware.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/quantization.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/numerical-methods.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/inference-serving.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/custom-kernels.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/data-parallelism.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/09-algorithms-systems-ai/peft.md** [stub]: Page is still a stub — not yet agent-generated


---

*This report is regenerated automatically. The supervisor updates `agents/sprints/current.md` with the top work items.*