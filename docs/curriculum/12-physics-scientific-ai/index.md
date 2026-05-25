---
title: Physics & Scientific AI
tags: [physics-ai, neural-operators, pinn, scientific-ml, simulation, symmetry]
---

# Track 12 · Physics & Scientific AI

> AI for scientific discovery: physics-informed learning, neural operators, symmetry in networks, and the use of deep learning as a tool for science.

Scientific AI applies machine learning to physical simulation, scientific discovery, and understanding natural systems. It brings together physics constraints, geometric structure, and the expressiveness of neural networks to solve problems that classical numerical methods cannot.

---

## Topics

### Physics-Informed Methods
- [Physics-Informed Neural Networks](pinn.md) — PDE constraints, collocation points, boundary conditions
- [Neural ODEs & CDEs](neural-odes.md) — continuous-depth models, adjoint method, controlled differential equations
- [Hamiltonian & Lagrangian Networks](hamiltonian-networks.md) — conservation laws, symplectic integration, energy preservation

### Neural Operators
- [Operator Learning](operator-learning.md) — learning mappings between function spaces
- [Fourier Neural Operator](fno.md) — spectral convolutions, discretization invariance
- [DeepONet](deeponet.md) — branch-trunk architecture, approximation theory for operators

### Symmetry & Geometry
- [Equivariant Neural Networks](equivariant-networks.md) — E(3) equivariance, group theory, steerable CNNs
- [Geometric Deep Learning](geometric-dl.md) — the geometric unified framework, symmetry groups in DL

### Scientific Applications
- [Molecular Simulation](molecular-simulation.md) — force fields, molecular dynamics, energy prediction
- [Climate & Weather AI](climate-ai.md) — NeuralGCM, GraphCast, precipitation nowcasting
- [Accelerating Simulation](simulation-acceleration.md) — ML surrogates, latent dynamics, reduced-order models

---

## Connections to frontier research

- **AlphaFold & protein structure** — equivariant networks applied to molecular biology
- **Foundation models for science** — general-purpose scientific models trained on diverse data
- **ML for fusion** — magnetic confinement optimization, plasma control
- **AI for mathematics** — proof assistants, conjecture generation, symbolic regression

---

## Recommended entry points

Start with [Physics-Informed Neural Networks](pinn.md) for the constrained learning perspective, and [Fourier Neural Operator](fno.md) for the operator learning framework. [Equivariant Neural Networks](equivariant-networks.md) connects to the broader geometric deep learning arc.
