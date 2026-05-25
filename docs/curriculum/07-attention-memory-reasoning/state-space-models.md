---
title: State Space Models
track: 07-attention-memory-reasoning
tags: [ssm, mamba, s4, selective-ssm, sequence-models, recurrent]
depth: research
prereqs: [transformer]
updated: 2026-05-25
---

# State Space Models
> **TL;DR:** A family of sequence models based on linear dynamical systems — they process sequences recurrently in O(L) time while offering a convolutional view for efficient training; Mamba's selective SSMs match or exceed transformers on many tasks.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
State space models parameterize sequence-to-sequence maps as linear ODEs: dx/dt = Ax + Bu, y = Cx + Du. Discretized, these become recurrent operations. S4 (Gu et al.) showed that specific structured initializations of A produce excellent long-range dependency modeling. Mamba extended SSMs with input-dependent (selective) A, B, C matrices — enabling content-based reasoning that earlier SSMs lacked.

## Why it matters at the frontier
SSMs offer O(N) training (via convolution) and O(1) inference per token (via recurrence) — unlike transformers which are O(N²) in attention. For long-context modeling and hardware-constrained deployment, this matters enormously. Mamba-2 establishes a formal duality between SSMs and attention. Hybrid architectures (Jamba, Zamba) combine both.

## Core concepts
- **State space representation** — x'(t) = Ax(t) + Bu(t); y(t) = Cx(t) + Du(t)
- **Discretization** — converts continuous-time SSM to discrete recurrence via ZOH or bilinear
- **S4 initialization** — HiPPO matrix for A; enables learning of long-range structure
- **Selectivity** — Mamba's A, B, C depend on the input; enables content-based filtering
- **Hardware-aware scan** — parallel prefix scan for efficient recurrence on GPU
- **SSD** — State Space Duality; Mamba-2 shows SSM is a form of linear attention

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [Efficiently Modeling Long Sequences with Structured State Spaces (S4)](https://arxiv.org/abs/2111.00396) | 2021 | Gu et al. | S4 — first practical deep SSM |
| [Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) | 2023 | Gu & Dao | Mamba — input-selective SSM; COLM 2024 Oral |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| [S4](https://arxiv.org/abs/2111.00396) | 2021 | Long-range sequence modeling with structured SSMs |
| [Mamba](https://arxiv.org/abs/2312.00752) | 2023 | Selectivity — content-based reasoning in SSMs |

## Current SotA
> *Updated: 2026-05-25*
Mamba-2 (Dao & Gu, ICML 2024) establishes the SSD framework connecting SSMs and attention. Hybrid Mamba-Transformer models (Jamba, Samba, Zamba) combine the best of both. Mamba-3 (2026 preprint) scales the approach further. SSMs are also gaining traction for audio (SSAMBA), genomics (Caduceus), and long-context video.

## Connected topics
- [[transformer]] — SSMs as the recurrent alternative
- [[efficient-attention]] — both aim to reduce O(N²) attention cost
- [[titans]] — test-time learnable memory extends SSMs to >2M context

## Further reading
- [Structured State Spaces: Combining Continuous-Time Models with Deep Learning](https://arxiv.org/abs/2111.00396) — Gu et al.
