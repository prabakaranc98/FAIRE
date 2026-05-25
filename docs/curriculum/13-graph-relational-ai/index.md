---
title: Graph & Relational AI
tags: [graph-neural-networks, gnn, geometric-deep-learning, relational-reasoning, knowledge-graphs]
---

# Track 13 · Graph & Relational AI

> Learning over structured data: graph neural networks, geometric deep learning, relational reasoning, and knowledge graphs.

Graphs are the natural representation for relational structure — molecules, social networks, knowledge bases, physical simulations. Graph neural networks extend deep learning to this domain, and geometric deep learning provides the unifying mathematical framework.

---

## Topics

### Graph Neural Networks
- [Message Passing](message-passing.md) — aggregate-combine, GCN, GraphSAGE, GAT
- [Expressive Power of GNNs](gnn-expressivity.md) — Weisfeiler-Leman hierarchy, 1-WL, higher-order GNNs
- [Graph Transformers](graph-transformers.md) — attention over nodes and edges, positional encodings for graphs

### Geometric Deep Learning
- [Geometric Unification](geometric-unification.md) — the geometric deep learning blueprint: groups, symmetries, domains
- [Equivariant GNNs](equivariant-gnn.md) — E(3)/SE(3) equivariant networks for molecular data
- [Spectral Graph Methods](spectral-graph.md) — graph Laplacian, spectral convolution, ChebNet

### Relational Reasoning
- [Knowledge Graphs](knowledge-graphs.md) — entity-relation representations, KGE, TransE, RotatE
- [Relational Inductive Biases](relational-biases.md) — object-centric representations, relation networks
- [Scene Graphs](scene-graphs.md) — visual relation detection, structured scene understanding

### Applications
- [Molecular Property Prediction](molecular-property.md) — MPNN, DimeNet, SchNet, drug-target interaction
- [Recommendation Systems](recommendation.md) — collaborative filtering as graph learning
- [Combinatorial Optimization](combinatorial-opt.md) — GNNs for NP-hard problems, pointer networks

---

## Connections to frontier research

- **AlphaFold 3** — equivariant graph networks for protein-ligand structure prediction
- **Physics simulation** — particle-based simulation, graph networks as physics engines
- **Geometric foundation models** — large pretrained models over molecular graphs
- **Heterogeneous graphs** — multi-type nodes and edges in real-world knowledge systems

---

## Recommended entry points

Start with [Message Passing](message-passing.md) for how GNNs compute, and [Geometric Unification](geometric-unification.md) for the conceptual framework. [Equivariant GNNs](equivariant-gnn.md) is the frontier entry point for scientific applications.
