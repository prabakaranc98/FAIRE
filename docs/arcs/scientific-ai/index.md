---
title: "Arc: Scientific AI — From PDEs to Foundation Models for Science"
arc: scientific-ai
super_domain: E-Science
tracks: [12-physics-scientific-ai, 14-biology-life-sciences, 03-representation-learning, 07-attention-memory-reasoning]
estimated_depth: "6-8 weeks, ~25 papers"
prereqs: [partial-differential-equations, neural-networks, basic-chemistry-optional]
---

# Arc: Scientific AI
> **What this arc builds:** The ability to read, understand, and work at the frontier of AI applied to science — from the physics-informed networks that solve PDEs to AlphaFold 3 and the foundation models now transforming biology, chemistry, and Earth system science.

## Why this arc exists

Science has always been constrained by compute. Simulating a protein fold, solving the Navier-Stokes equations, or forecasting weather at 10km resolution required supercomputers running for days. Machine learning changes this by learning the structure of physical systems rather than simulating them from first principles — and doing so with a quality guarantee that keeps improving as data accumulates.

This arc traces that transformation. Physics-Informed Neural Networks (PINNs) showed that physical laws could be encoded as loss terms; Neural Operators (DeepONet, FNO) showed that you could learn maps between function spaces — not just point-to-point predictions; equivariant networks showed how to bake in symmetry; AlphaFold showed what happens when you get all of it right. The result is a new discipline: AI for scientific discovery, where models don't just fit data but respect physical law and generalize across scales.

## Prerequisites

Familiarity with neural networks and backpropagation. Basic understanding of differential equations (what a PDE is, what boundary conditions are) — you don't need to solve them analytically. Exposure to graphs and graph neural networks is helpful for the molecular sections. Basic chemistry concepts (atoms, bonds, force fields) help but are not required.

## The sequence

**Physics-Constrained Learning**

1. **Physics-Informed Neural Networks (PINNs)** (foundational) — encode PDE residuals as loss terms; mesh-free; solves forward and inverse problems simultaneously. [→](https://arxiv.org/abs/1711.10561)
2. **Automatic differentiation for PDEs** (theoretical) — computing spatial/temporal derivatives via autodiff; why neural networks as PDE solvers are differentiable end-to-end.
3. **Collocation methods & adaptive sampling** (applied) — where to place training points; failure modes of PINNs on stiff PDEs.
4. **Neural ODEs** (theoretical) — continuous-depth networks as ODEs; adjoint method for memory-efficient backpropagation. [→](https://arxiv.org/abs/1806.07366)

**Neural Operators**

5. **DeepONet** (foundational) — Universal Approximation Theorem for operators; branch and trunk networks; learning maps between function spaces. [→](https://arxiv.org/abs/1910.03193)
6. **Fourier Neural Operator (FNO)** (frontier) — learn in Fourier space; O(N log N) complexity; resolution-invariant; turbulence simulation. [→](https://arxiv.org/abs/2010.08895)
7. **Geo-FNO / NeRF for science** (applied) — handling irregular geometries; extending FNO beyond regular grids.
8. **AFNO / Transformer-based operators** (frontier) — Adaptive Fourier Neural Operator; foundation for weather forecasting models.

**Equivariant & Geometric Deep Learning**

9. **Symmetry and equivariance** (theoretical) — why SE(3) equivariance matters for molecules; equivariant maps vs. invariant features.
10. **Message-passing GNNs for molecules** (applied) — MPNN; atoms as nodes, bonds as edges; molecular property prediction.
11. **E(n)-equivariant networks (EGNN)** (applied) — position + velocity as node features; equivariant message passing; fast and practical. [→](https://arxiv.org/abs/2102.09844)
12. **SE(3)-Transformers / NequIP** (frontier) — full equivariant attention for molecular dynamics; neural force fields.

**Weather & Earth Systems**

13. **GraphCast** (frontier) — GNN-based global weather model; 10-day forecasts in <60s; Science 2023. [→](https://arxiv.org/abs/2212.12794)
14. **Pangu-Weather** (frontier) — 3D transformer for NWP; 24h forecasts outperforming ECMWF. [→](https://arxiv.org/abs/2211.02556)
15. **Aurora** (frontier) — Microsoft weather foundation model; 1.3B parameters; unified model across variables and resolutions.

**Structural Biology & Molecular AI**

16. **AlphaFold 2** (frontier) — evoformer + structure module; solved protein structure prediction; Nature 2021. [→](https://www.nature.com/articles/s41586-021-03819-2)
17. **AlphaFold 3** (frontier) — diffusion-based joint structure prediction for proteins, DNA, RNA, ligands; Nature 2024. [→](https://www.nature.com/articles/s41586-024-07487-w)
18. **ESMFold / ESM-2** (frontier) — protein language model; structure from sequence without MSA; Meta AI. [→](https://www.science.org/doi/10.1126/science.ade2574)
19. **GNoME** (frontier) — graph networks for materials discovery; 2.2M stable crystals; Nature 2023. [→](https://www.nature.com/articles/s41586-023-06735-9)

**Multimodal & Foundation Models for Science**

20. **Causal perturbation modeling** (applied) — GEARS; CRISPR screen modeling; predicting the effect of gene knockouts. [→](https://www.nature.com/articles/s41587-023-01905-6)
21. **Geneformer** (frontier) — transformer pretrained on single-cell transcriptomics; context-aware gene embeddings. [→](https://www.nature.com/articles/s41586-023-06139-9)
22. **scGPT** (frontier) — foundation model for single-cell biology; zero-shot cell type annotation, gene perturbation, multi-omic integration. [→](https://www.nature.com/articles/s41592-024-02201-0)
23. **V-JEPA 2 for robotic science** (frontier) — video-based joint embedding predictive architecture; world model enabling robot planning without action labels.

## Key figures

- **George Karniadakis** (Brown) — PINNs; physics-constrained learning
- **Lu Lu** (Yale) — DeepONet; neural operators
- **Zongyi Li** (Caltech → NVIDIA) — Fourier Neural Operator (FNO)
- **Demis Hassabis, John Jumper** (DeepMind) — AlphaFold 2/3 (Nobel Prize 2024)
- **Remi Lam** (DeepMind) — GraphCast
- **Alexander Rives** (Meta AI) — ESM protein language models
- **Yann LeCun** (Meta AI) — V-JEPA / JEPA framework

## Essential reading sequence

1. [Physics-Informed Neural Networks](https://arxiv.org/abs/1711.10561) — Raissi et al. 2017 — PDEs as loss terms
2. [Neural Operator: Learning Maps Between Function Spaces (DeepONet)](https://arxiv.org/abs/1910.03193) — Lu et al. 2019
3. [Fourier Neural Operator for Parametric Partial Differential Equations](https://arxiv.org/abs/2010.08895) — Li et al. 2020 — FNO
4. [Learning Equivariant Representations (EGNN)](https://arxiv.org/abs/2102.09844) — Satorras et al. 2021
5. [Highly accurate protein structure prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) — Jumper et al. 2021
6. [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) — Abramson et al. 2024
7. [Learning skillful medium-range global weather forecasting (GraphCast)](https://arxiv.org/abs/2212.12794) — Lam et al. 2022
8. [Evolutionary-scale prediction of atomic-level protein structure with a language model (ESMFold)](https://www.science.org/doi/10.1126/science.ade2574) — Lin et al. 2022

## Current frontier anchors
> As of 2026-05-25

- **AlphaFold 3** — diffusion-based joint structure prediction across all biomolecule types; Nature 2024
- **ESM3** — multimodal protein language model; sequence + structure + function; generates novel functional proteins
- **GNoME** — 2.2M stable crystal structures; accelerating materials discovery by 10×
- **Aurora (Microsoft)** — 1.3B parameter weather foundation model; unified forecasting across all variables and resolutions
- **scGPT** — single-cell foundation model for zero-shot biology; multi-omic integration
- **FNO + AFNO** — neural operators now outperforming numerical solvers on turbulence and climate emulation

## What you'll know when done

1. Explain what a neural operator is and why FNO is resolution-invariant
2. Describe the PINN training setup for a specific PDE (e.g., Navier-Stokes) — loss terms, collocation, boundary conditions
3. Explain why SE(3)-equivariance is necessary for molecular property prediction and how EGNN achieves it
4. Walk through AlphaFold 2's Evoformer architecture and what the structure module outputs
5. Identify which AI approach is appropriate for a given scientific problem: PINNs (known physics, inverse problem), FNO (operator learning, many instances), GNN (molecular property), foundation model (zero-shot biology)

## Branch points to other arcs

- **→ Generative Stack arc**: AlphaFold 3 uses diffusion; molecular generation; protein design
- **→ Causal AI arc**: Causal perturbation modeling in biology; GEARS connects causal graphs to gene networks
- **→ MLP → Transformer arc**: Evoformer (AlphaFold 2) is a transformer variant; geometric attention

## Where to go next

[Generative Stack arc →](../generative-stack/index.md) — Diffusion models for molecular generation and design

[Causal AI arc →](../causal-ai/index.md) — Causal reasoning for biology and perturbation modeling
