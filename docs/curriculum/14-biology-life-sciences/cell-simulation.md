---
title: Cell Simulation
track: 14-biology-life-sciences
tags: [single-cell, genomics, generative-models, systems-biology, drug-discovery]
depth: foundational
prereqs: [single-cell-genomics, generative-models, transformers]
updated: 2025-05-14
has_mvb: true
---

# Cell Simulation

> **TL;DR:** Cell simulation uses computational models to predict cellular behavior under perturbations, enabling virtual experimentation for drug discovery and systems biology.

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| Curious learner | [What it is](#what-it-is) | Build intuition |
| CS student | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Build something that works |
| Applied engineer | [In production](#in-production) → [MVB](#minimum-valuable-build) | Deploy a model |
| Theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Frontier researcher | [Current SotA](#current-sota) → [Open questions](#open-questions) | Identify open problems |

## What it is

Imagine you have a high-resolution map of a city, but you want to know how the traffic patterns would shift if you closed a major bridge. In biology, the "city" is a cell, and the "traffic" is the flow of gene expression—the process where DNA instructions are converted into functional proteins. Because cells are complex, non-linear systems, predicting how they react to a drug or a genetic change is difficult. Traditional wet-lab experiments are expensive and slow, often failing to capture the full dynamic range of cellular responses.

Cell simulation addresses this by creating "virtual cells"—computational models that represent the transcriptome, which is the complete set of RNA transcripts in a cell, or the cell's physical morphology. By training generative models on vast atlases of single-cell data, researchers learn the latent rules governing cellular transitions. This allows them to simulate perturbations *in silico* before moving to the bench, effectively turning the cell into a programmable system.

This shift changes how drug discovery is approached: instead of screening millions of compounds physically, researchers simulate the response of diverse cell types to these compounds. This reduces the search space for therapeutic candidates and provides a mechanistic understanding of why certain cells respond while others remain resistant, bridging the gap between descriptive genomics and predictive biology.

## Why it matters at the frontier

Cell simulation is the cornerstone of the next generation of drug discovery, as it enables the prediction of "in-the-wild" perturbation responses that are difficult to capture in controlled laboratory settings. By moving beyond static snapshots of gene expression, these models allow for the exploration of dynamic cellular trajectories, which is essential for understanding complex diseases like cancer or neurodegeneration where cell state transitions are key.

This field is currently blocking the transition from descriptive to predictive medicine. As models like Lingshu-Cell (Zhang et al., 2026, [https://arxiv.org/html/2603.25240](https://arxiv.org/html/2603.25240)) demonstrate, generative world models can now capture transcriptome dynamics with high fidelity. The ability to simulate these dynamics at scale is essential for developing personalized therapies, as it allows for the testing of drug efficacy on virtual representations of a patient's specific cellular profile.

## Core concepts

- **Virtual Cell** — A computational model that simulates the internal state and behavior of a cell in response to external stimuli.
- **Perturbation Response** — The change in cellular state (e.g., gene expression profile) following an intervention like drug treatment or gene knockdown.
- **Transcriptome Manifold** — The high-dimensional space where cellular states reside, constrained by regulatory networks.
- **Flow Matching** — A generative modeling technique that learns a vector field to transport data from a simple distribution to the complex manifold of cellular states.
- **Cellular World Model** — A generative model that learns the temporal and causal dynamics of cellular systems, allowing for multi-step simulation.

## Mathematical foundations

The objective in learning cellular dynamics is often to estimate the vector field \(v_t(x)\) that governs the transition of a cell state \(x\) over time \(t\):

\[ \mathcal{L}_{FM} = \mathbb{E}_{t, p_t(x)} [\| v_t(x) - u_t(x) \|^2] \]

where \(t \in [0, 1]\) is the time variable, \(p_t(x)\) is the probability path of the cellular state, \(v_t(x)\) is the learned vector field, and \(u_t(x)\) is the target velocity field. This term penalizes the difference between the model's predicted velocity and the ground truth velocity derived from the data distribution, directly formalizing the "Flow Matching" concept defined in the core concepts.

For transcriptome foundation models, we often use a transformer-based objective:

\[ \mathcal{L}_{LM} = - \sum_{i=1}^{N} \log P(g_i | g_{<i}, C) \]

where \(g_i\) is the expression level of gene \(i\), \(C\) is the cell-type context or ontology, and \(N\) is the total number of genes. This objective learns the joint distribution of gene expression by predicting the next gene in the sequence, conditioned on the cell's identity, which relates to the "Transcriptome Manifold" concept.

## Key algorithms / techniques

- **CellFlow** — Uses flow matching to simulate morphological changes by learning the vector field between cellular states, directly implementing the objective \(\mathcal{L}_{FM}\).
- **HarmonyCell** — Automates perturbation modeling by explicitly accounting for semantic and distribution shifts between different single-cell datasets.
- **Transcriptome Foundation Models** — Large-scale transformer models trained on cell atlases to learn universal representations of gene expression using the \(\mathcal{L}_{LM}\) objective.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [CellFlow](https://arxiv.org/abs/2502.09775v1) | 2025 | Li et al. | Introduces flow matching for cellular morphology. |
| [Benchmarking virtual cell models](https://arxiv.org/html/2604.27646v1) | 2026 | Wu et al. | Provides the standard benchmark for perturbation response. |
| [Lingshu-Cell](https://arxiv.org/html/2603.25240) | 2026 | Zhang et al. | Defines the generative world model approach for transcriptomes. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| [A single-cell gene expression language model](https://ar5iv.labs.arxiv.org/html/2210.14330) | 2022 | Connell & Khan | First application of LLM architectures to gene expression. |
| [Cell-ontology guided foundation model](https://arxiv.org/html/2408.12373) | 2024 | Yuan et al. | Incorporates biological priors into transcriptome models. |

## Current SotA

Lingshu-Cell (Zhang et al., 2026) achieves state-of-the-art performance in transcriptome generation and perturbation prediction. Benchmarking studies (Wu et al., 2026) show that flow-matching approaches like CellFlow (Li et al., 2025) outperform traditional VAE-based methods on the "in-the-wild" perturbation response task.

## What's happening now

Research is currently focused on "cellular world models" that can simulate long-term dynamics rather than just static state changes. Zhang et al. (2026) demonstrated that generative models can now capture the temporal evolution of transcriptomes, moving toward a true "digital twin" of the cell. This builds on the current SotA by moving from static generation to temporal trajectory simulation.

Engineering efforts are shifting toward automating the integration of heterogeneous single-cell datasets. Huang et al. (2026) introduced HarmonyCell to address the semantic and distribution shifts that occur when combining data from different sequencing technologies, which is a major bottleneck for scaling these models.

The open problem remains the integration of multi-modal data. While transcriptome models are mature, capturing the interplay between proteins, metabolites, and gene expression in a single unified model is still an active area of research.

## Open questions

> [!IMPORTANT]
> **Researcher:** How can we define a universal loss function that simultaneously optimizes for both transcriptome fidelity and morphological consistency in multi-modal cellular world models?

> [!IMPORTANT]
> **Engineer:** What are the optimal quantization and pruning strategies to deploy high-parameter cellular foundation models on edge devices for real-time laboratory analysis?

> [!IMPORTANT]
> **Open:** Can we develop a causal framework that distinguishes between correlative gene expression patterns and true regulatory drivers in perturbation response models?

## In production

- **Genentech** — Uses generative transcriptome models to prioritize drug targets in oncology, as detailed in their [research publications](https://www.gene.com/scientists/publications).
- **Insilico Medicine** — Deploys virtual cell simulations for small molecule discovery, documented in their [AI research portal](https://insilico.com/research).

## Minimum Valuable Build

### For the CS student (1 day · RTX 4070 / 12GB VRAM)
1. Install `scvi-tools` via `pip install scvi-tools`.
2. Load the PBMC dataset: `adata = scvi.data.pbmc3k()`.
3. Train the model: `model = scvi.model.SCVI(adata); model.train(max_epochs=20)`.
4. **Artifact:** A trained model checkpoint saved to `./model_checkpoints/`.
5. **Metric:** Achieve a Pearson correlation > 0.8 on held-out gene expression reconstruction using `model.get_normalized_expression()`.

### For the Curious Learner
Focus on visualizing the latent space. Use the trained model from the CS student build to generate a UMAP plot. Observe how different cell types cluster together, which demonstrates the model's ability to learn the "Transcriptome Manifold."

### For the Applied Engineer
Integrate the model into a pipeline that predicts the effect of a specific gene knockdown. Use `model.get_latent_representation()` to simulate the shift in cell state when a specific gene's expression is set to zero.

### For the Theory Student
Analyze the ELBO (Evidence Lower Bound) loss curve during training. Compare the reconstruction loss vs. the KL divergence term to understand the trade-off between data fidelity and latent space regularization.

### For the Frontier Researcher
Benchmark the model against the "in-the-wild" perturbation response task defined by Wu et al. (2026). Attempt to replace the standard VAE architecture with a Flow Matching objective to see if it improves the trajectory prediction accuracy.

---

## Code & implementations

- [scvi-tools](https://github.com/scverse/scvi-tools) — The standard library for probabilistic modeling of single-cell data.
- [CellFlow](https://github.com/li-lab/cellflow) — Official implementation of flow-matching for cellular morphology.

## This concept appears in

- [../../arcs/generative-biology/step-01-transcriptome-modeling.md](../../arcs/generative-biology/step-01-transcriptome-modeling.md) — This page serves as the foundational entry point for the generative biology arc.

## What comes next

Understanding the probabilistic foundations of cellular manifolds allows for the transition from static analysis to predictive simulation. Future work will focus on scaling these models to multi-modal data and integrating causal inference to move from correlation to intervention.

- [[/topics/single-cell-genomics]] — Provides the data manifolds that cell simulation models learn to navigate.
- [[/topics/generative-models]] — The underlying probabilistic framework used for learning cellular vector fields.
- [[/topics/systems-biology]] — The field that defines the regulatory networks these models aim to simulate.

## Connected topics

- [[/topics/single-cell-genomics]] — The primary data source for training virtual cell models.
- [[/topics/generative-models]] — The mathematical engine behind cellular world models.
- [[/topics/systems-biology]] — The biological context that validates simulation results.

## Further reading

- [Lilian Weng's survey on Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) — Provides the theoretical foundation for the generative techniques used in cell simulation.
- [Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — The seminal paper on the flow-matching technique used in modern cell simulation.