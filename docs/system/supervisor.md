---
title: Supervisor Report
description: Automated wiki health assessment and editorial priorities
---

# Supervisor Report

> Generated: **2026-05-27 14:54 EDT** by the Frontier Wiki supervising agent

## Wiki Health

| Metric | Value |
|---|---|
| Total pages (stubs + generated) | **111** |
| Stub pages (not yet generated) | **5** |
| Generated pages | **106** |
| Approved (conf ≥ 0.8) | **104** (98%) |
| Flagged (conf < 0.8) | **0** |
| Avg reviewer confidence | **0.78** |
| Tracks with content | **11** / 10 |
| Pages with MVB | **0** |
| Critical audit issues | **0** |
| Warnings | **124** |

## Editorial Analysis

The Frontier Wiki is in an exceptionally strong position as we near completion, with 111 total pages and 104 of our 106 generated articles successfully approved. Our coverage deficit sits at a perfect 0.0%, and a positive quality trend of +0.027 indicates our content standards are steadily rising. However, we must still address a minor quality deficit of 0.07, reflected in a conservative average reviewer confidence of 0.78 and a lingering backlog of 124 warnings. For the upcoming sprint, our immediate priority is to retry the errored `probabilistic-programming` page and systematically convert our remaining five stubs—focusing heavily on critical systems-level topics like `collective-communication`, `long-context-models`, and `rlhf-infrastructure-overview`. Systemically, our remaining $9.94 budget is more than sufficient to close these gaps in full-production mode. While our 70% first-pass approval rate shows healthy editorial rigor, we must guard against overly pedantic reviewer strictness to ensure these final, complex systems pages transition smoothly from stubs to approved assets without unnecessary friction.

## Prioritised Work Queue

| # | Priority | Action | Topic | Track | Reason |
|---|---|---|---|---|---|
| 1 | 🔴 critical | improve | probabilistic-programming | 05-statistical-probabilistic-ml | Previous attempt errored (conf=0.80) — retry with current prompts |
| 2 | 🟡 medium | generate | collective-communication | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 3 | 🟡 medium | generate | long-context-models | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 4 | 🟡 medium | generate | reinforcement-learning | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 5 | 🟡 medium | generate | rlhf-infrastructure-overview | 09-algorithms-systems-for-ai | Stub page — not yet generated |
| 6 | 🟡 medium | generate | what-can-you-build-next-template-04-neural-networks-deep-learning | 04-neural-networks-deep-learning | Stub page — not yet generated |

## Audit Issues

## Audit — 2026-05-27 14:54 EDT
> 111 pages scanned · 0 critical · 124 warnings

### Warning
- **curriculum/arcs.md** [missing_section]: Missing section '## The territory'
- **curriculum/arcs.md** [missing_section]: Missing section '## How it works'
- **curriculum/arcs.md** [missing_section]: Missing section '## Where the field is now'
- **curriculum/arcs.md** [missing_section]: Missing section '## What's still open'
- **curriculum/arcs.md** [missing_section]: Missing section '## Where to read next'
- **curriculum/arcs.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/06-reinforcement-learning/concepts/model-based-reinforcement-learning.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/06-reinforcement-learning/concepts/ppo.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/05-statistical-probabilistic-ml/concepts/mcmc.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/attention.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/02-generative-modeling/concepts/flow-matching.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/02-generative-modeling/concepts/latent-diffusion-models.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/08-causal-statistical-inference/concepts/do-calculus.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/03-representation-learning/concepts/self-supervised-learning.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/04-neural-networks-deep-learning/concepts/scaling-laws.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/04-neural-networks-deep-learning/concepts/layer-normalization.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/09-algorithms-systems-for-ai/concepts/reinforcement-learning-schedulers.md** [no_arxiv]: No arXiv sources found — page needs academic citations
- **curriculum/core/06-reinforcement-learning/concepts/model-based-reinforcement-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/06-reinforcement-learning/concepts/mdp.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/06-reinforcement-learning/concepts/policy-gradients.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/06-reinforcement-learning/concepts/world-models.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/06-reinforcement-learning/concepts/ppo.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/06-reinforcement-learning/concepts/q-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/06-reinforcement-learning/concepts/actor-critic.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/06-reinforcement-learning/concepts/policy-gradient.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/mcmc.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/gaussian-processes.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-inference.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/uncertainty-quantification.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-optimization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/expectation-maximization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/variational-inference.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/probabilistic-programming.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/05-statistical-probabilistic-ml/concepts/em-algorithm.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/multi-head-attention.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/in-context-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/positional-encoding.md** [persona_drift]: Persona(s) outside canonical set: ['curious-learner', 'cs-student', 'applied-engineer', 'theory-student', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/attention.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/long-context.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/07-attention-memory-reasoning-continual/concepts/retrieval-augmented-generation.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/02-generative-modeling/concepts/diffusion-models.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/02-generative-modeling/concepts/consistency-models.md** [persona_drift]: Persona(s) outside canonical set: ['curious-generalist', 'cs-student', 'applied-engineer', 'theory-student', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/02-generative-modeling/concepts/score-matching.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/02-generative-modeling/concepts/flow-matching.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/02-generative-modeling/concepts/generative-adversarial-networks.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher', 'curious-learner', 'theory-student', 'pm-decision-maker']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/02-generative-modeling/concepts/latent-diffusion-models.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/02-generative-modeling/concepts/variational-autoencoders.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/10-complexity-cognition-natural-intelligence/concepts/generalization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/10-complexity-cognition-natural-intelligence/concepts/compositionality.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/10-complexity-cognition-natural-intelligence/concepts/double-descent.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/10-complexity-cognition-natural-intelligence/concepts/scaling-collapse.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/08-causal-statistical-inference/concepts/causal-representation-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/08-causal-statistical-inference/concepts/causal-discovery.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/08-causal-statistical-inference/concepts/instrumental-variables.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/08-causal-statistical-inference/concepts/mediation-analysis.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/08-causal-statistical-inference/concepts/do-calculus.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/08-causal-statistical-inference/concepts/potential-outcomes.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/08-causal-statistical-inference/concepts/counterfactuals.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/08-causal-statistical-inference/concepts/structural-causal-models.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/03-representation-learning/concepts/jepa.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/03-representation-learning/concepts/simclr.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/03-representation-learning/concepts/self-supervised-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/03-representation-learning/concepts/contrastive-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/03-representation-learning/concepts/masked-autoencoders.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/03-representation-learning/concepts/representation-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/03-representation-learning/concepts/data-augmentation.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/scaling-laws.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher', 'curious-learner', 'theory-student']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/regularization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/backpropagation.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher', 'curious-learner', 'theory-student']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/residual-connections.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/transformer-architecture.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/normalization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/residual-networks.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/what-can-you-build-next-template-04-neural-networks-deep-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/transformers.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/gradient-descent.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/layer-normalization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/optimization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/adaptive-optimizers.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/04-neural-networks-deep-learning/concepts/batch-normalization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/llm-inference.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/connected-topics-template-09-algorithms-systems-for-ai.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/llm-architecture-optimizations.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/precision-scaling.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/distributed-training.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/compiler-optimizations-for-ml.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/communication-collectives.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/quantization-aware-training.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/policy-gradient-theory.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/reinforcement-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/mixed-precision-training.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/model-parallelism.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/tensor-parallelism.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'curious-learner', 'theory-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/post-training-quantization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/rlhf-infrastructure-overview.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/constrained-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher', 'curious-learner', 'pm-decision-maker']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/long-context-models.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/distributed-training-arc.md** [persona_drift]: Persona(s) outside canonical set: ['frontier-researcher', 'curious-generalist', 'theory-student']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/convex-optimization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/tensor-cores.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/collective-communication.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/curriculum-resampling.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/automatic-differentiation.md** [persona_drift]: Persona(s) outside canonical set: ['frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/attention-mechanisms.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'frontier-researcher', 'systems-architect']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/gradient-bucketing.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/reinforcement-learning-schedulers.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache-management.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/pipeline-parallelism.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/inference-optimization.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/flash-attention.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/curriculum-learning.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/quantization.md** [persona_drift]: Persona(s) outside canonical set: ['curious-learner', 'cs-student', 'applied-engineer', 'theory-student', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/model-deployment.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/data-parallelism.md** [persona_drift]: Persona(s) outside canonical set: ['applied-ai-ml-engineer']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/09-algorithms-systems-for-ai/concepts/quantization-basics.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/01-ai/concepts/transformer.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/01-ai/concepts/reward-modeling.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/01-ai/concepts/alignment-safety.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/01-ai/concepts/mechanistic-interpretability.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/01-ai/concepts/mixture-of-experts.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/01-ai/concepts/rlhf.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']
- **curriculum/core/01-ai/concepts/chain-of-thought.md** [persona_drift]: Persona(s) outside canonical set: ['cs-student', 'applied-engineer', 'frontier-researcher', 'curious-generalist', 'theory-student']. Allowed: ['applied-ai-engineer', 'applied-researcher', 'research-engineer']

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
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-inference.md** [stale_sota]: Page updated 558 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/uncertainty-quantification.md** [stale_sota]: Page updated 542 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-optimization.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/expectation-maximization.md** [stale_sota]: Page updated 561 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/bayesian-neural-networks.md** [stale_sota]: Page updated 546 days ago — SotA may be outdated
- **curriculum/core/05-statistical-probabilistic-ml/concepts/probabilistic-programming.md** [stale_sota]: Page updated 372 days ago — SotA may be outdated
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
- **curriculum/core/08-causal-statistical-inference/concepts/potential-outcomes.md** [stale_sota]: Page updated 479 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/counterfactuals.md** [stale_sota]: Page updated 412 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/policy-evaluation.md** [stale_sota]: Page updated 188 days ago — SotA may be outdated
- **curriculum/core/08-causal-statistical-inference/concepts/structural-causal-models.md** [stale_sota]: Page updated 559 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/jepa.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/simclr.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/self-supervised-learning.md** [stale_sota]: Page updated 464 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/contrastive-learning.md** [stale_sota]: Page updated 330 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/masked-autoencoders.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/representation-learning.md** [stale_sota]: Page updated 219 days ago — SotA may be outdated
- **curriculum/core/03-representation-learning/concepts/data-augmentation.md** [stale_sota]: Page updated 466 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/scaling-laws.md** [stale_sota]: Page updated 374 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/backpropagation.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/residual-connections.md** [stale_sota]: Page updated 181 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/normalization.md** [stale_sota]: Page updated 542 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/residual-networks.md** [stale_sota]: Page updated 188 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/what-can-you-build-next-template-04-neural-networks-deep-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/04-neural-networks-deep-learning/concepts/transformers.md** [stale_sota]: Page updated 346 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/gradient-descent.md** [stale_sota]: Page updated 214 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/layer-normalization.md** [stale_sota]: Page updated 603 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/optimization.md** [stale_sota]: Page updated 232 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/adaptive-optimizers.md** [stale_sota]: Page updated 234 days ago — SotA may be outdated
- **curriculum/core/04-neural-networks-deep-learning/concepts/batch-normalization.md** [stale_sota]: Page updated 592 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/llm-inference.md** [stale_sota]: Page updated 502 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/connected-topics-template-09-algorithms-systems-for-ai.md** [stale_sota]: Page updated 549 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/differentiable-optimization.md** [stale_sota]: Page updated 198 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/llm-architecture-optimizations.md** [stale_sota]: Page updated 459 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/precision-scaling.md** [stale_sota]: Page updated 558 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/distributed-training.md** [stale_sota]: Page updated 396 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/compiler-optimizations-for-ml.md** [stale_sota]: Page updated 207 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/communication-collectives.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/quantization-aware-training.md** [stale_sota]: Page updated 410 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/policy-gradient-theory.md** [stale_sota]: Page updated 377 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/reinforcement-learning.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/mixed-precision-training.md** [stale_sota]: Page updated 537 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/model-parallelism.md** [stale_sota]: Page updated 238 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/tensor-parallelism.md** [stale_sota]: Page updated 382 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/post-training-quantization.md** [stale_sota]: Page updated 410 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/rlhf-infrastructure-overview.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/constrained-learning.md** [stale_sota]: Page updated 378 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/long-context-models.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/convex-optimization.md** [stale_sota]: Page updated 377 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/tensor-cores.md** [stale_sota]: Page updated 391 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/collective-communication.md** [stub]: Page is still a stub — not yet agent-generated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/curriculum-resampling.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/automatic-differentiation.md** [stale_sota]: Page updated 378 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache.md** [stale_sota]: Page updated 394 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/attention-mechanisms.md** [stale_sota]: Page updated 423 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/gradient-bucketing.md** [stale_sota]: Page updated 183 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/reinforcement-learning-schedulers.md** [stale_sota]: Page updated 306 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/kv-cache-management.md** [stale_sota]: Page updated 238 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/pipeline-parallelism.md** [stale_sota]: Page updated 562 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/inference-optimization.md** [stale_sota]: Page updated 228 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/flash-attention.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/curriculum-learning.md** [stale_sota]: Page updated 378 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/model-deployment.md** [stale_sota]: Page updated 314 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/data-parallelism.md** [stale_sota]: Page updated 502 days ago — SotA may be outdated
- **curriculum/core/09-algorithms-systems-for-ai/concepts/quantization-basics.md** [stale_sota]: Page updated 198 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/transformer.md** [stale_sota]: Page updated 511 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/reward-modeling.md** [stale_sota]: Page updated 388 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/alignment-safety.md** [stale_sota]: Page updated 497 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/mechanistic-interpretability.md** [stale_sota]: Page updated 421 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/mixture-of-experts.md** [stale_sota]: Page updated 572 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/rlhf.md** [stale_sota]: Page updated 417 days ago — SotA may be outdated
- **curriculum/core/01-ai/concepts/chain-of-thought.md** [stale_sota]: Page updated 227 days ago — SotA may be outdated


---

*This report is regenerated automatically. The supervisor updates `agents/sprints/current.md` with the top work items.*