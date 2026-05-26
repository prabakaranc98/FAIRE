---
title: Protein Language Models
track: 14-biology-life-sciences
tags: [protein-lm, esm, protrans, evolutionary-information, sequence-representation]
depth: research
prereqs: [self-supervised-learning, transformer]
updated: 2026-05-25
---

# Protein Language Models
> **TL;DR:** Transformer models pretrained on protein sequences learn rich structural and functional representations from evolutionary patterns — enabling structure prediction, functional annotation, and protein design without multiple sequence alignments.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## Core concepts
- **Amino acid vocabulary** — 20 standard amino acids as tokens (plus special tokens)
- **Masked language modeling** — BERT-style pretraining on protein sequences
- **ESM-2** — Meta's 650M–15B protein language model; trained on UniRef50
- **Contact prediction** — predicting 3D contacts from attention maps (ESM-1b)
- **Evolutionary information** — proteins with similar sequences share structure; this is the SSL signal
- **Inverse folding** — given structure, design sequence (ProteinMPNN, ESM-IF)

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Evolutionary-scale prediction of atomic-level protein structure with a language model (ESMFold)](https://www.science.org/doi/10.1126/science.ade2574) | 2022 | Lin et al. (Meta AI) | ESM-2 + structure prediction head |
| [Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences](https://www.pnas.org/doi/10.1073/pnas.2016239118) | 2021 | Rives et al. | ESM-1b — protein representations from SSL |

## Current SotA
> *Updated: 2026-05-25*
ESM-3 (Hayes et al., 2024) is a 98B multimodal protein model conditioning on sequence, structure, and function simultaneously. It can generate novel proteins and has demonstrated "reasoning" about protein biology. ProteinMPNN remains SotA for inverse folding.

## Connected topics
- [Protein Structure Prediction](./protein-structure.md) — PLMs provide features for structure prediction
- Protein Design — PLMs enable sequence generation for desired properties
- [Single-Cell Omics & Foundation Models](./single-cell.md) — similar SSL pretraining paradigm for cell state representations

## Further reading
- [Protein language models](https://www.nature.com/articles/s41587-024-02229-x) — Xu et al. 2024; review in Nature Biotechnology
