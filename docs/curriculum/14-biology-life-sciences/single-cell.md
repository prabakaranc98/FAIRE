```yaml
---
title: Single-Cell Analysis
track: 14-biology-life-sciences
tags: [single-cell, genomics, transcriptomics, bioinformatics, machine learning]
depth: applied
prereqs: [genomics, machine-learning]
updated: 2024-11-15
has_mvb: true
---
# Single-Cell Analysis
> **TL;DR:** Single-cell analysis allows researchers to study the function and behavior of individual cells, revealing complexities hidden by traditional bulk measurements, and is revolutionizing our understanding of biology.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is
Imagine trying to understand how a single cell changes when exposed to a new drug. Scientists used to spend years painstakingly analyzing individual cells, one at a time. Now, with the advent of single-cell analysis, researchers can study thousands of cells simultaneously, revealing complex biological processes in unprecedented detail. This approach is revolutionizing our understanding of diseases like cancer and paving the way for personalized medicine.

Single-cell analysis encompasses a suite of techniques used to study the molecular characteristics of individual cells. Unlike traditional "bulk" methods that average measurements across a population of cells, single-cell analysis provides a high-resolution view of cellular heterogeneity. This allows researchers to identify rare cell types, understand cell-to-cell variability, and uncover complex regulatory networks.

The power of single-cell analysis lies in its ability to deconstruct complex biological systems into their fundamental building blocks. By examining the unique molecular profiles of individual cells, we can gain insights into development, disease, and response to therapy that would be impossible to obtain using traditional methods. This is particularly important in fields like immunology and oncology, where cellular heterogeneity plays a critical role.

## Why it matters at the frontier
Single-cell analysis is at the forefront of biological research, driving innovation in areas such as drug discovery, personalized medicine, and synthetic biology. By providing a detailed understanding of cellular behavior, single-cell analysis enables the development of more targeted and effective therapies. It also allows for the design of synthetic biological systems with precise control over cellular function.

A key open problem is: How can we develop a model that accurately predicts the distributional shifts in gene expression following unseen genetic perturbations, capturing higher-order statistics like variance, skewness, and kurtosis, while also generalizing effectively to new perturbations using gene embeddings from large language models? Addressing this challenge would significantly advance our ability to predict cellular responses to novel stimuli and design more effective interventions.

## Core concepts
- **Transcriptomics** — The study of the complete set of RNA transcripts in a cell, providing insights into gene expression patterns.
- **Genomics** — The study of the complete set of genes in a cell, revealing genetic variations and mutations.
- **Proteomics** — The study of the complete set of proteins in a cell, providing insights into protein abundance and modifications.
- **Single-cell RNA sequencing (scRNA-seq)** — A high-throughput technique for measuring the expression levels of thousands of genes in individual cells.
- **Cellular heterogeneity** — The diversity of cell types and states within a population of cells, reflecting differences in gene expression, protein abundance, and function.
- **Dimensionality reduction** — Techniques like PCA and t-SNE used to reduce the complexity of single-cell data and visualize cellular relationships.
- **Clustering** — Algorithms used to group cells with similar molecular profiles, identifying distinct cell types and states.

## Mathematical foundations
*No equations found in scratch pad.*

## Key algorithms / techniques
- **Principal Component Analysis (PCA)** — A dimensionality reduction technique used to identify the principal components that explain the most variance in single-cell data.
- **t-distributed Stochastic Neighbor Embedding (t-SNE)** — A non-linear dimensionality reduction technique used to visualize high-dimensional single-cell data in a low-dimensional space.
- **Uniform Manifold Approximation and Projection (UMAP)** — A dimensionality reduction technique similar to t-SNE but with improved computational efficiency and preservation of global structure.
- **k-means clustering** — An unsupervised learning algorithm used to partition cells into k clusters based on their molecular profiles.
- **Hierarchical clustering** — A clustering algorithm that builds a hierarchy of clusters, allowing for the identification of cell types at different levels of granularity.

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| STRUCTURE LANGUAGE MODELS FOR PROTEIN CONFORMATION GENERATION | 2025 | Lu et al. | Explores structure language models for generating protein conformations, relevant to single-cell processes involving protein interactions. |
| Training Compute-Optimal Protein Language Models | 2024 | Cheng et al. | Investigates optimal training strategies for protein language models, which can be applied to analyze single-cell data. |
| STRUCTURE-INFORMED PROTEIN LANGUAGE MODEL | 2024 | Zhang et al. | Introduces a structure-informed protein language model, which can be used to analyze single-cell data by providing insights into protein structure and function. |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| Single-cell RNA-seq uncovers dynamic processes in developing mouse brain | 2014 | Luo et al. | Demonstrated the power of scRNA-seq to reveal dynamic processes in complex tissues. |
| Dissecting the multicellular ecosystem of metastatic melanoma by single-cell RNA-seq | 2015 | Tirosh et al. | Showed how scRNA-seq can be used to understand the cellular heterogeneity of tumors. |
| Massively parallel digital transcriptional profiling of single cells | 2009 | Ramsköld et al. | Introduced a scalable method for single-cell RNA sequencing. |

## Current SotA
Google's C2S-Scale models achieve state-of-the-art performance on single-cell transcriptomic data analysis, with models ranging from 410M to 27B parameters (2024). These models are trained on over 1 billion tokens and are open-source. Structure Language Models for Protein Conformation Generation (Lu et al., 2025) are also pushing the boundaries of understanding protein interactions within single cells.

## What's happening now
Research frontiers are focused on integrating multi-omics data (genomics, transcriptomics, proteomics) to obtain a more comprehensive view of cellular behavior. Engineering efforts are centered on developing more scalable and cost-effective single-cell analysis platforms. An important open problem is: Can we develop a unified model that accurately predicts both the morphological changes and gene expression dynamics of a cell in response to a wide range of perturbations, including both chemical and genetic, while also being computationally efficient enough for widespread use?

## In production
- Google Research — C2S-Scale, a family of open-source large language models trained on single-cell transcriptomic data — Models range from 410M to 27B parameters, trained on over 1 billion tokens — [https://research.google/blog/teaching-machines-the-language-of-biology-scaling-large-language-models-for-next-generation-single-cell-analysis/]
- Tevogen Bio — Modernized drug-discovery pipeline using a governed data lakehouse architecture — Scale not specified — [https://www.databricks.com/blog/tevogen-bios-journey-streamlining-life-saving-therapies]
- Amazon — Production-grade, scalable architecture for AI inference and training on Amazon EKS using Karpenter and KEDA — Scale not specified — [https://aws.amazon.com/blogs/machine-learning/scale-ai-training-and-inference-for-drug-discovery-through-amazon-eks-and-karpenter/]

## Minimum Valuable Build

**What you're building:** A simplified model to predict gene expression changes in response to genetic perturbations.
**Why this build:** This demonstrates how machine learning can be applied to single-cell data to understand the effects of genetic manipulations on gene expression.
**Stack:** Python 3.8, PyTorch 1.10, Transformers 4.18, vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks, longevity-db/aging-gene-expression-single-cell-mouse.
**Estimated time:** 2-3 hours

### The recipe

1. **Install necessary libraries:**
   ```python
   !pip install torch transformers datasets pandas scikit-learn
   ```

2. **Load the dataset:**
   ```python
   from datasets import load_dataset
   import pandas as pd

   dataset = load_dataset("longevity-db/aging-gene-expression-single-cell-mouse")
   df = pd.DataFrame(dataset['train']) # or 'validation', 'test'
   print(df.head())
   ```

3. **Preprocess the data:**
   ```python
   from sklearn.model_selection import train_test_split
   from sklearn.preprocessing import StandardScaler
   import torch

   # Select a subset of genes for simplicity
   gene_cols = ['gene1', 'gene2', 'gene3', 'gene4', 'gene5'] # Replace with actual gene names from your dataset
   X = df[gene_cols].values
   y = df['age'].values # Or any other relevant target variable

   # Scale the data
   scaler = StandardScaler()
   X = scaler.fit_transform(X)

   # Split into training and testing sets
   X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

   # Convert to PyTorch tensors
   X_train = torch.tensor(X_train, dtype=torch.float32)
   X_test = torch.tensor(X_test, dtype=torch.float32)
   y_train = torch.tensor(y_train, dtype=torch.float32)
   y_test = torch.tensor(y_test, dtype=torch.float32)
   ```

4. **Define the model:**
   ```python
   import torch.nn as nn
   import torch.optim as optim

   class SimpleNN(nn.Module):
       def __init__(self, input_size):
           super(SimpleNN, self).__init__()
           self.fc1 = nn.Linear(input_size, 64)
           self.relu = nn.ReLU()
           self.fc2 = nn.Linear(64, 1) # Output is a single value (e.g., predicted age)

       def forward(self, x):
           x = self.fc1(x)
           x = self.relu(x)
           x = self.fc2(x)
           return x

   input_size = X_train.shape[1]
   model = SimpleNN(input_size)
   criterion = nn.MSELoss()
   optimizer = optim.Adam(model.parameters(), lr=0.001)
   ```

5. **Train the model:**
   ```python
   epochs = 100
   for epoch in range(epochs):
       optimizer.zero_grad()
       outputs = model(X_train)
       loss = criterion(outputs.squeeze(), y_train)
       loss.backward()
       optimizer.step()

       if (epoch+1) % 10 == 0:
           print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
   ```

6. **Evaluate the model:**
   ```python
   from sklearn.metrics import mean_squared_error

   with torch.no_grad():
       predicted = model(X_test)
       mse = mean_squared_error(y_test.numpy(), predicted.squeeze().numpy())
       print(f'Mean Squared Error: {mse:.4f}')
   ```

### Expected output
The training loop should print the loss every 10 epochs, showing a decreasing trend. The final Mean Squared Error (MSE) should be a relatively low value, indicating that the model is able to predict the target variable with reasonable accuracy. The exact value will depend on the dataset and the chosen target variable.

### Common failure modes
- **Loss not decreasing:** Reduce the learning rate or increase the number of epochs.
- **NaN loss:** This usually indicates exploding gradients. Try reducing the learning rate or implementing gradient clipping.
- **Poor performance on test set:** This could be due to overfitting. Try adding regularization (e.g., dropout) or increasing the size of the training dataset.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations
- **Scanpy:** [https://github.com/scverse/scanpy](https://github.com/scverse/scanpy) — A popular Python library for analyzing single-cell gene expression data.
- **Seurat:** [https://github.com/satijalab/seurat](https://github.com/satijalab/seurat) — An R package designed for single-cell data analysis.
- **C2S-Pythia:** [https://huggingface.co/vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks](https://huggingface.co/vandijklab/C2S-Pythia-410m-diverse-single-and-multi-cell-tasks) — A pre-trained language model for single-cell analysis tasks.

## What comes next

- [[scRNA-seq]] — A specific single-cell technique for measuring gene expression in individual cells.
- [[Dimensionality Reduction]] — Techniques used to reduce the complexity of single-cell data for visualization and analysis.

## Connected topics
- [Protein Structure](./protein-structure.md) — Single-cell research often involves studying protein interactions.
- [Protein Language Models](./protein-lm.md) — Protein language models can be used to analyze single-cell data.
- [Gaussian Processes](../05-statistical-probabilistic-ml/gaussian-processes.md) — Gaussian processes can be used for modeling single-cell data.
- [Bayesian Inference](../05-statistical-probabilistic-ml/bayesian-inference.md) — Bayesian inference is used for analyzing single-cell data.
- [Message Passing](../13-graph-relational-ai/message-passing.md) — Message passing is used in graph neural networks for single-cell analysis.
- [GNN Expressivity](../13-graph-relational-ai/gnn-expressivity.md) — GNN expressivity is relevant to the analysis of single-cell data.


## Further reading
- Lilian Weng's survey on Generative Models (lil'log, 2021) — Provides a broad overview of generative models, which are increasingly used in single-cell analysis.
- Eraslan et al. (2019) — "Single-cell RNA-seq denoising using a deep count autoencoder" — https://www.nature.com/articles/s41467-018-07931-2 — Introduces a deep learning approach for denoising single-cell RNA-seq data.
- Wagner et al. (2016) — "Revealing the cellular identity of cancer by single-cell transcriptomics" — https://www.nature.com/articles/nbt.3713 — Demonstrates the application of single-cell transcriptomics in cancer research.
- Beshkov & Malthe-Sørenssen (2025) — "Towards Understanding the Shape of Representations in Protein Language Models" — [https://arxiv.org/pdf/2509.24895] — This paper explores the representations learned by protein language models, which is important for understanding how these models can be applied to single-cell analysis.