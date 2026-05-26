```yaml
---
title: Protein Structure
track: 14-biology-life-sciences
tags: [protein folding, structural biology, bioinformatics, drug discovery, AI, machine learning]
depth: foundational
prereqs: [amino-acids, protein-folding]
updated: 2024-10-26
has_mvb: false
---
# Protein Structure
> **TL;DR:** Protein structure refers to the three-dimensional arrangement of atoms in a protein, which is crucial for understanding its function and interactions, and is now being revolutionized by AI/ML-driven prediction and design.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [Code & implementations](#code--implementations) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters at the frontier](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is
Imagine you're designing a new drug to target a specific protein, but the protein's shape is constantly changing. Understanding these dynamic shifts, or protein conformational dynamics, is crucial for effective drug design. Recent advances in AI/ML, like AlphaFold2, have revolutionized our ability to predict protein structures from their amino acid sequences. Now, researchers are working to predict not just one structure, but the entire range of shapes a protein can adopt. This opens the door to designing drugs that can adapt to these dynamic changes.

Protein structure refers to the three-dimensional arrangement of atoms in a protein molecule. This structure is critical because it dictates the protein's function, specificity, and interactions with other molecules. Proteins fold into unique structures based on their amino acid sequence, and these structures can be organized into four levels: primary, secondary, tertiary, and quaternary.

The ability to accurately determine or predict protein structure is fundamental to understanding biological processes and developing new therapies. Experimental techniques like X-ray crystallography, NMR spectroscopy, and cryo-electron microscopy have traditionally been used to determine protein structures, but these methods are often time-consuming and challenging. The advent of AI/ML models has dramatically accelerated the process of structure prediction, enabling researchers to explore protein function and design novel proteins with greater efficiency.

## Why it matters at the frontier
Understanding protein structure is essential for deciphering the molecular mechanisms underlying biological processes and diseases. The vast majority of proteins' 3D structures remain unknown, which limits our ability to understand their roles in disease and to identify potential drug targets. AI/ML models, like AlphaFold, are now able to predict these structures with unprecedented accuracy, opening up new avenues for research and drug discovery.

At the frontier, researchers are pushing beyond static structure prediction to model protein dynamics and interactions. The open problem is: How can we develop AI/ML models that accurately predict the conformational dynamics of proteins directly from their amino acid sequences, enabling the design of drugs that effectively target these dynamic structures? Addressing this challenge could revolutionize drug discovery by enabling the design of drugs that target specific protein conformations and interactions, leading to more effective and precise therapies.

## Core concepts
- **Primary Structure** — The linear sequence of amino acids in a polypeptide chain, held together by peptide bonds.
- **Secondary Structure** — Localized folding patterns, such as alpha helices and beta sheets, stabilized by hydrogen bonds between amino acid residues.
- **Tertiary Structure** — The overall three-dimensional structure of a single polypeptide chain, determined by various interactions including hydrophobic interactions, hydrogen bonds, disulfide bridges, and salt bridges.
- **Quaternary Structure** — The arrangement of multiple polypeptide chains (subunits) in a multi-subunit protein complex.
- **Conformational Dynamics** — The range of different shapes or conformations that a protein can adopt over time, crucial for understanding its function and interactions.
- **Protein Folding** — The process by which a polypeptide chain acquires its native three-dimensional structure, driven by the sequence of amino acids and the surrounding environment.
- **Structure Prediction** — The computational process of predicting the three-dimensional structure of a protein from its amino acid sequence.
- **Homology Modeling** — A structure prediction technique that builds a protein model based on the known structure of a homologous protein.

## Mathematical foundations
While the prediction of protein structure using AI/ML models doesn't rely on a single, easily expressible equation, the underlying principles involve complex energy functions and optimization algorithms. One way to think about it is through the lens of energy minimization:

\[
E = E_{\text{bond}} + E_{\text{angle}} + E_{\text{dihedral}} + E_{\text{non-bonded}}
\]

where \(E\) is the total potential energy of the protein, \(E_{\text{bond}}\) is the energy associated with bond lengths, \(E_{\text{angle}}\) is the energy associated with bond angles, \(E_{\text{dihedral}}\) is the energy associated with dihedral angles (torsion angles), and \(E_{\text{non-bonded}}\) is the energy associated with non-bonded interactions (e.g., van der Waals forces, electrostatic interactions). This equation represents the total potential energy of the protein as a sum of different energy terms.

The goal is to find the structure that minimizes this energy, which corresponds to the most stable and likely conformation of the protein. AI/ML models, like AlphaFold, learn to approximate this energy landscape and efficiently search for the global minimum.

## Key algorithms / techniques
- **AlphaFold2 (DeepMind, 2020)** — A deep learning model that predicts protein structures with near-experimental accuracy by using attention mechanisms and a novel architecture to model the relationships between amino acids.
- **Rosetta (Baker Lab, University of Washington)** — A suite of computational tools for protein structure prediction, design, and analysis, employing a fragment-based approach to sample conformational space and optimize protein structures.
- **Molecular Dynamics (MD) Simulations** — A computational method that simulates the physical movements of atoms and molecules, allowing researchers to study the dynamic behavior of proteins over time.
- **Protein-SE(3) (Yu et al., 2025)** — A benchmark for SE(3)-based generative models used in protein structure design, facilitating comprehensive investigation and fair comparison of different methods.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Protein-SE(3): Benchmarking SE(3)-based Generative Models for Protein Structure Design | 2025 | Lang Yu et al. | Highlights the current state-of-the-art in generative models for protein design, offering a framework for comparing different methods. |
| From sequence to protein structure and conformational dynamics with AI/ML | 2025 | Alexander Ille | Discusses the use of AI/ML models to predict protein structure and conformational dynamics based on amino acid sequences. |
| Cell-ontology guided transcriptome foundation model | 2024 | Xinyu Yuan et al. | Introduces a cell-ontology guided transcriptome foundation model. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Highly accurate protein structure prediction with AlphaFold | 2021 | Demonstrates the groundbreaking accuracy of AI/ML in protein structure prediction, a feat previously considered impossible. |
| An automated approach to structure prediction of proteins associated with the SARS-CoV-2 virus | 2020 | Showcases the application of computational methods to rapidly predict the structures of viral proteins, aiding in drug discovery efforts. |

## Current SotA
AlphaFold2 achieves high accuracy in predicting protein structures from amino acid sequences (2021). Protein-SE(3) provides a benchmark for SE(3)-based generative models in protein design (2025). ConfRover (Bytedance-Seed, 2025) introduces a novel autoregressive model that simultaneously learns protein conformation and dynamics from MD trajectories, supporting both time-dependent and time-independent sampling.

## What's happening now
Research is focused on improving the accuracy and efficiency of protein structure prediction, particularly for challenging cases like membrane proteins and intrinsically disordered proteins. New AI/ML architectures are being developed to capture the complex relationships between amino acid sequence, structure, and function. Additionally, researchers are exploring methods to predict protein dynamics and interactions with other molecules.

Engineering and systems efforts are centered on deploying protein structure prediction models at scale, enabling researchers to rapidly generate structures for large numbers of proteins. Cloud-based platforms and high-performance computing resources are being leveraged to accelerate structure prediction and facilitate data sharing. Furthermore, there is a growing emphasis on developing user-friendly tools and interfaces that make protein structure prediction accessible to a wider range of researchers.

The open problem remains: How can we develop AI/ML models that accurately predict the conformational dynamics of proteins directly from their amino acid sequences, enabling the design of drugs that effectively target these dynamic structures? This includes addressing challenges such as modeling the effects of post-translational modifications, predicting protein-ligand interactions, and capturing the influence of the cellular environment on protein structure and dynamics.

## In production
- NVIDIA — Proteome-scale protein structure prediction using AlphaFold-Multimer — [https://developer.nvidia.com/blog/how-to-accelerate-protein-structure-prediction-at-proteome-scale/]
- AWS — Deploying and scaling OpenFold (a PyTorch-based protein folding model) — Achieves production-scale inference — [https://aws.amazon.com/blogs/machine-learning/run-inference-at-scale-for-openfold-a-pytorch-based-protein-folding-ml-model-using-amazon-eks/]
- AWS — Deploying the ESMFold protein structure prediction model — Production scale — [https://aws.amazon.com/blogs/machine-learning/accelerate-protein-structure-prediction-with-the-esmfold-language-model-on-amazon-sagemaker/]
- Metagenomi — Generating novel protein enzymes using the Progen2 model — Millions of novel enzymes, cost-optimized workflow — [https://aws.amazon.com/blogs/machine-learning/metagenomi-generates-millions-of-novel-enzymes-cost-effectively-using-aws-inferentia/]

## Minimum Valuable Build
For a hands-on build with this concept, see the MVB on [[protein-folding]].

## Code & implementations
- AlphaFold: [https://github.com/deepmind/alphafold](https://github.com/deepmind/alphafold)
- OpenFold: [https://github.com/aqlaboratory/openfold](https://github.com/aqlaboratory/openfold)
- Rosetta: Available through academic and commercial licenses.

## What comes next

- [[protein-folding]] — describes the process by which a protein acquires its three-dimensional structure.
- [[drug-discovery]] — explores how understanding protein structure facilitates the design of new therapeutics.

## Connected topics

- [Protein Language Models](./protein-lm.md) — Protein language models are directly related to understanding protein structure.
- [Message Passing](../13-graph-relational-ai/message-passing.md) — Message passing is used in graph neural networks for protein structure prediction.
- [GNN Expressivity](../13-graph-relational-ai/gnn-expressivity.md) — GNN expressivity is relevant to the ability of GNNs to model protein structures.
- [Equivariant GNN](../13-graph-relational-ai/equivariant-gnn.md) — Equivariant GNNs are used to model the symmetries in protein structures.
- [Fourier Neural Operator (FNO)](../12-physics-scientific-ai/fno.md) — FNOs can be used to model and predict protein structures based on their physical properties.
- [Equivariant Networks](../12-physics-scientific-ai/equivariant-networks.md) — Equivariant networks are useful for modeling the symmetries inherent in protein structures.


## Further reading
- Alexander Ille (2025) — "From sequence to protein structure and conformational dynamics with AI/ML" — [https://arxiv.org/abs/2504.14059v1] — Provides an overview of AI/ML methods for predicting protein structure and dynamics.
- Lang Yu et al. (2025) — "Protein-SE(3): Benchmarking SE(3)-based Generative Models for Protein Structure Design" — [https://arxiv.org/abs/2507.20243v1] — Introduces a benchmark for evaluating generative models in protein design.
- David Baker et al. — Rosetta@home — [https://www.rosettaathome.com/](https://www.rosettaathome.com/) — A distributed computing project for protein structure prediction and design.
- DeepMind — AlphaFold — [https://www.deepmind.com/research/highlighted-research/alphafold](https://www.deepmind.com/research/highlighted-research/alphafold) — Details the architecture and performance of the AlphaFold model.