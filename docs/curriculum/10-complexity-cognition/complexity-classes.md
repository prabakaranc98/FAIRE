---
title: Computational Complexity for AI
track: 10-complexity-cognition
tags: [complexity, p-np, seth, fine-grained-complexity, transformer-expressivity]
depth: theoretical
prereqs: []
updated: 2026-05-25
---

# Computational Complexity for AI
> **TL;DR:** The study of what computers can and cannot do efficiently — and how these limits apply to neural networks, attention mechanisms, and the hardness of learning tasks.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **P** — problems solvable in polynomial time
- **NP** — problems verifiable in polynomial time; P ≠ NP is the central open problem
- **SETH** — Strong Exponential Time Hypothesis; implies quadratic lower bounds for many problems
- **Fine-grained complexity** — quadratic vs. linear tradeoffs; not just polynomial vs. exponential
- **TC⁰ circuits** — the complexity class characterizing transformer computation
- **Attention hardness** — O(N²) attention cannot be computed in O(N^{2-ε}) time under SETH

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [On the Computational Power of Transformers and Its Implications in Sequence Modeling](https://arxiv.org/abs/2006.09286) | 2020 | Bhattamishra et al. | Transformer expressiveness vs. formal languages |
| [Lower Bounds for Attention by Keles et al.](https://arxiv.org/abs/2302.13214) | 2023 | Keles et al. | SETH-based lower bounds for exact attention |

## Connected topics
- [State Space Models](../07-attention-memory-reasoning/state-space-models.md) — SSMs sit below SETH attention lower bound; O(N) not O(N²)
- Ml Theory — computational hardness of learning connects to statistical learning theory
- [Emergent Capabilities in Large Models](./emergent-capabilities.md) — complexity theory tools for understanding phase transitions

## Further reading
- [Introduction to the Theory of Computation](http://math.mit.edu/~sipser/book.html) — Sipser; MIT open access chapters
