---
title: Protein Structure Prediction
track: 14-biology-life-sciences
tags: [protein-structure, alphafold, esmfold, structural-biology, folding]
depth: research
prereqs: []
updated: 2026-05-25
---

# Protein Structure Prediction
> **TL;DR:** Predicting a protein's 3D structure from its amino acid sequence — solved at near-experimental accuracy by AlphaFold 2, the single most impactful AI result in biology, enabling drug discovery and disease understanding at unprecedented scale.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Proteins fold into specific 3D shapes that determine their function. Predicting this structure from sequence alone was a 50-year grand challenge. AlphaFold 2 (2021) essentially solved it: it achieves accuracy competitive with experimental crystallography on most protein families. AlphaFold 3 (2024) extended this to protein-DNA/RNA-ligand complexes using a diffusion head.

## Why it matters at the frontier
AlphaFold 2 is the landmark demonstration that deep learning can solve fundamental problems in science. It has become infrastructure for drug discovery, enabled protein design (RFDiffusion, ProteinMPNN), and transformed structural biology. It's the reason biology is now one of the most active frontier AI research areas.

## Core concepts
- **Amino acid sequence** — the 1D string of amino acids that determines protein function
- **3D structure** — backbone + side chain atom positions in 3D space
- **Evoformer** — AlphaFold 2's core module; processes multiple sequence alignments (MSA) + pairwise representations
- **Structure module** — generates 3D coordinates equivariantly from Evoformer representations
- **MSA** — Multiple Sequence Alignment; evolutionary signal used by AF2
- **pLDDT score** — per-residue confidence; calibrated quality metric from AF2

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Highly Accurate Protein Structure Prediction with AlphaFold](https://www.nature.com/articles/s41586-021-03819-2) | 2021 | Jumper et al. (DeepMind) | AlphaFold 2 — landmark result |
| [Accurate structure prediction of biomolecular interactions with AlphaFold 3](https://www.nature.com/articles/s41586-024-07487-w) | 2024 | Abramson et al. (Google DeepMind) | AlphaFold 3 — extends to all biomolecular complexes |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [AlphaFold 2](https://www.nature.com/articles/s41586-021-03819-2) | 2021 | Near-solved the protein folding problem |
| [Language models of protein sequences at scale enable accurate structure prediction](https://www.biorxiv.org/content/10.1101/2022.07.20.500902v1) | 2022 | Lin et al. | ESMFold — single sequence without MSA |

## Current SotA
> *Updated: 2026-05-25*
AlphaFold 3 handles protein-ligand, protein-DNA, protein-RNA complexes with a diffusion head. Boltz-1/2 (MIT, 2024/25) are open-source AF3-class models. RoseTTAFold All-Atom (Baker lab) is another open-source alternative. The frontier is moving to protein dynamics (MD-level simulation) and conditional protein design.

## Connected topics
- [[protein-lm]] — protein language models as feature extractors
- [[equivariant-gnn]] — AF2's structure module is E(3)-equivariant
- [[protein-design]] — generating novel proteins with desired properties

## Further reading
- [A structural biology community assessment of AlphaFold2 applications](https://www.nature.com/articles/s41594-022-00849-w) — 2022
