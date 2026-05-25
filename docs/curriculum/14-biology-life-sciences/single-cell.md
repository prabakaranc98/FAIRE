---
title: Single-Cell Omics & Foundation Models
track: 14-biology-life-sciences
tags: [single-cell, scrna-seq, scgpt, scfoundation, cell-type, trajectory]
depth: research
prereqs: [self-supervised-learning, transformer]
updated: 2026-05-25
---

# Single-Cell Omics & Foundation Models
> **TL;DR:** Measuring gene expression in individual cells at scale — enabling cell type identification, developmental trajectory inference, and perturbation prediction — now with foundation models pretrained on tens of millions of single-cell profiles.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **scRNA-seq** — single-cell RNA sequencing; measures mRNA expression per cell
- **Cell type annotation** — classify each cell into a type based on its expression profile
- **Trajectory inference** — reconstruct differentiation paths between cell states over time
- **Perturbation prediction** — predict cell state after gene knockout/overexpression (CRISPR screens)
- **scGPT** — GPT-style foundation model pretrained on 33M single-cell profiles
- **Batch correction** — remove technical artifacts from different sequencing runs

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [scGPT: Toward Building a Foundation Model for Single-Cell Multi-omics Using Generative AI](https://www.nature.com/articles/s41592-024-02201-0) | 2024 | Cui et al. | scGPT — nature methods paper; 33M cell foundation model |
| [scFoundation: Large Scale Foundation Model on Single-Cell Transcriptomics](https://www.nature.com/articles/s41592-024-02305-7) | 2024 | Hao et al. | scFoundation — 50M cell pretraining |

## Current SotA
> *Updated: 2026-05-25*
scGPT and scFoundation represent the current SotA for multi-task single-cell analysis. Geneformer (Theodoris et al., Nature 2023) showed in silico perturbation for drug target discovery. The frontier: perturbation foundation models (CellOT, GEARS) and spatial transcriptomics models integrating spatial context.

## Connected topics
- [[protein-lm]] — analogous SSL pretraining paradigm for biological sequences
- [[causal-discovery]] — inferring gene regulatory networks from perturbation data
- [[molecular-generation]] — generating drug molecules that affect specific cell states

## Further reading
- [Best practices for single-cell analysis across modalities](https://www.nature.com/articles/s41576-023-00586-w) — Heumos et al. 2023
