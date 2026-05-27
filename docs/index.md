---
title: Frontier Wiki
template: home.html
---

<!-- cache-bust 2026-05-27T0510EDT: force Fastly CDN to re-fetch after rapid deploys -->


## The 10 canonical tracks

FAIRE is organized around the 10 parallel learning tracks from [pracha.me/curriculum](https://pracha.me/curriculum). Each subject has its own arcs, concepts, key authors, and builds — slot in at any depth.

<div class="grid cards" markdown>

- **A · Foundations & Theory**

    [AI](curriculum/core/01-ai/index.md) · [NN & Deep Learning](curriculum/core/04-neural-networks-deep-learning/index.md) · [Statistical & Probabilistic ML](curriculum/core/05-statistical-probabilistic-ml/index.md)

- **B · Modeling**

    [Generative Modeling](curriculum/core/02-generative-modeling/index.md) · [Representation Learning](curriculum/core/03-representation-learning/index.md) · [Causal & Statistical Inference](curriculum/core/08-causal-statistical-inference/index.md)

- **C · Decision & Reasoning**

    [Reinforcement Learning](curriculum/core/06-reinforcement-learning/index.md) · [Attention, Memory, Reasoning, Continual](curriculum/core/07-attention-memory-reasoning-continual/index.md)

- **D · Systems & Cognition**

    [Algorithms & Systems for AI](curriculum/core/09-algorithms-systems-for-ai/index.md) · [Complexity, Cognition & Natural Intelligence](curriculum/core/10-complexity-cognition-natural-intelligence/index.md)

</div>

---

## Every page is one of four artifact types

Each subject converges on the same shape — **concepts**, **authors**, **arcs**, and **builds** — designed to feed into each other rather than sit as a bookmark pile.

- **Concepts** are encyclopedic, self-contained walk-throughs (Olah/Distill grade), not bullet-point summaries.
- **Authors** anchor the field to the people whose work shaped it.
- **Arcs** are roadmaps.sh-style learning paths through the concepts.
- **Builds** are Minimum Valuable Build recipes — runnable, persona-tagged, real artifact at the end.

---

## Built for three personas

Every page that carries an MVB targets these three. The schema enforces it — any other persona tag is a writing error.

| Persona | Comes to do | Time | Their MVB shape |
|---|---|---|---|
| **Applied AI/ML engineer** (forward-deployed) | Ship into production by Friday | Half a day – 1 working day | Fine-tune a real model and serve it with a measured latency target |
| **Research engineer** | Reproduce a paper's number on commodity hardware | 1–3 working days | A reproduced table or figure within ±5% of the published number |
| **Applied researcher** | Test one hypothesis with one falsifier | 2 days – 1 week | A 2–3 condition ablation with a plot and a falsification criterion |

Every MVB clears the **5-gate quality bar**: a real ship-able artifact · a concrete time-to-ship · real HuggingFace model + dataset IDs · a specific success metric · *hardness in the middle* (fine-tune OR reproduce OR ablate OR deploy — never just `pip install` + `pipeline()`).

---

## Source discipline

Every link traces back to a primary source:

- `arxiv.org` — papers and preprints
- `*.edu` — university lecture notes, course pages
- `huggingface.co` — model cards, datasets
- Official library docs (PyTorch, JAX, Diffusers...)
- *"In production" sections only:* official engineering blogs from frontier labs

No Medium. No Towards Data Science. No Wikipedia as a citation.
