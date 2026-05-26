```yaml
---
title: Protein Language Models
track: 14-biology-life-sciences
tags: [protein, language models, deep learning, bioinformatics, protein design]
depth: applied
prereqs: [deep-learning, natural-language-processing]
updated: 2024-11-01
has_mvb: false
---
# Protein Language Models
> **TL;DR:** Protein language models (pLMs) are deep learning models trained on vast protein sequence datasets, enabling the prediction of protein properties and the design of novel proteins with desired functions.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [In production](#in-production) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is

Imagine you're trying to design a new enzyme that can break down plastic waste, or a protein that can target and destroy cancer cells. The challenge? The sheer complexity of proteins, with their intricate 3D structures and vast sequence space. Protein language models are emerging as powerful tools to help scientists navigate this complexity, accelerating the discovery of novel proteins with desired functions.

Protein language models (pLMs) are deep learning models trained on massive datasets of protein sequences. These models learn the "language" of proteins, capturing the complex relationships between amino acids and their impact on protein structure and function. By learning from existing proteins, pLMs can predict various protein properties, such as stability, binding affinity, and even 3D structure.

The core idea is to treat protein sequences as sentences, where each amino acid is a word. Just as language models predict the next word in a sentence, pLMs predict the next amino acid in a protein sequence. This allows them to generate new protein sequences with desired characteristics, opening up exciting possibilities for protein design and drug discovery.

## Why it matters at the frontier

Protein language models are revolutionizing the field of protein engineering by enabling the design of novel proteins with desired functions. Traditional protein engineering methods are often slow, expensive, and require extensive experimental validation. pLMs offer a faster and more efficient way to explore the vast protein sequence space, identifying promising candidates for further development.

The ability to accurately predict protein properties and generate novel protein sequences has significant implications for various applications, including drug discovery, enzyme engineering, and materials science. However, challenges remain in developing pLMs that can effectively integrate structural information to improve the generation of protein conformations and accurately predict protein properties, while also being computationally efficient for large-scale applications. Addressing these challenges will pave the way for the widespread adoption of pLMs in protein engineering and accelerate the discovery of new and improved proteins.

## Core concepts

-   **Amino acid sequence** — The linear chain of amino acids that defines the primary structure of a protein.
-   **Protein structure** — The three-dimensional arrangement of atoms in a protein molecule, which is crucial for its function.
-   **Protein function** — The specific biological activity of a protein, such as catalyzing a chemical reaction or binding to a target molecule.
-   **Sequence space** — The vast set of all possible amino acid sequences for a given protein length.
-   **Embeddings** — Vector representations of amino acids or protein sequences that capture their semantic relationships.
-   **Attention mechanism** — A neural network technique that allows the model to focus on the most relevant parts of the input sequence when making predictions.
-   **Transfer learning** — A machine learning approach where a model trained on one task is fine-tuned for a different but related task.

## Mathematical foundations

While the specific equations vary depending on the architecture, the core idea is to model the probability of a protein sequence given its context. A common approach is to use a masked language modeling objective:

\[
P(\mathbf{x}) = \prod_{i=1}^{L} P(x_i \mid \mathbf{x}_{\setminus i})
\]

where \(\mathbf{x}\) is the protein sequence, \(x_i\) is the \(i\)-th amino acid, and \(\mathbf{x}_{\setminus i}\) is the sequence with the \(i\)-th amino acid masked.

This equation represents the probability of a protein sequence as the product of the probabilities of each amino acid given the rest of the sequence.

The probability \(P(x_i \mid \mathbf{x}_{\setminus i})\) is typically modeled using a neural network, such as a transformer:

\[
P(x_i \mid \mathbf{x}_{\setminus i}) = \text{softmax}(f(\mathbf{x}_{\setminus i})_i)
\]

where \(f\) is the neural network, and \(f(\mathbf{x}_{\setminus i})_i\) is the output of the network corresponding to the \(i\)-th amino acid.

This equation shows how the probability of an amino acid is determined by the softmax function applied to the output of the neural network, which takes into account the context provided by the rest of the sequence.

The model is trained to minimize the negative log-likelihood of the training data:

\[
\mathcal{L} = -\sum_{\mathbf{x} \in \mathcal{D}} \log P(\mathbf{x})
\]

where \(\mathcal{L}\) is the loss function, and \(\mathcal{D}\) is the training dataset.

This equation defines the loss function as the negative sum of the logarithms of the probabilities of the protein sequences in the training dataset, which the model aims to minimize during training.

## Key algorithms / techniques

-   **Transformer Networks** — (Vaswani et al., 2017) A neural network architecture that uses self-attention mechanisms to model long-range dependencies in sequences, enabling pLMs to capture complex relationships between amino acids.
-   **Masked Language Modeling (MLM)** — (Devlin et al., 2018) A training technique where the model is trained to predict masked amino acids in a protein sequence, forcing it to learn contextual representations of amino acids.
-   **Transfer Learning** — (Yosinski et al., 2014) A machine learning approach where a model trained on a large dataset is fine-tuned on a smaller, task-specific dataset, allowing pLMs to leverage knowledge learned from vast protein sequence data.
-   **Reinforcement Learning** — (Sutton & Barto, 2018) A machine learning paradigm where an agent learns to make decisions in an environment to maximize a reward signal, enabling the fine-tuning of pLMs for specific protein design goals.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Guiding Generative Protein Language Models with Reinforcement Learning | 2025 | Ferruz | This paper introduces a reinforcement learning framework (DPO_pLM) to optimize protein sequences using protein language models, enabling the design of functional proteins. |
| Training Compute-Optimal Protein Language Models | 2024 | Cheng et al. | This paper investigates optimal training strategies for protein language models. |
| STRUCTURE-INFORMED PROTEIN LANGUAGE MODEL | 2024 | Zhang et al. | This paper introduces a structure-informed protein language model. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Attention is All You Need | 2017 | Introduced the Transformer architecture, which is the foundation for many modern protein language models. |
| Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences | 2019 | Demonstrated the power of unsupervised learning on large protein sequence datasets for predicting protein structure and function. |
| Knowledge Distillation of a Protein Language Model Yields a Foundational Implicit Solvent Model | 2026 | This paper explores knowledge distillation techniques to create a foundational implicit solvent model from a protein language model. |

## Current SotA

Recent protein language models have achieved impressive results in predicting protein properties and generating novel protein sequences. For example, InstructPLM-MU achieves superior performance in protein mutation predictions compared to ESM3 after just one hour of fine-tuning (Xu et al., 2024). Structure Language Models have also shown promise for protein conformation generation (Lu et al., 2025).

## What's happening now

Research in protein language models is rapidly advancing, with a focus on improving the accuracy and efficiency of these models. One key area of research is the development of structure-informed pLMs, which incorporate structural information into the model to improve its ability to predict protein properties and generate novel protein sequences.

Engineering and systems efforts are focused on scaling up pLMs to handle even larger datasets and more complex protein design tasks. This includes developing efficient training algorithms and hardware architectures that can accelerate the training and inference of pLMs.

An open problem is: How can we develop protein language models that effectively integrate structural information to improve the generation of protein conformations and accurately predict protein properties, while also being computationally efficient for large-scale applications?

## In production

-   NVIDIA — BioNeMo Blueprint for Generative Protein Binder Design — Scalable, GPU-accelerated workflow for generative protein binder design. — [https://developer.nvidia.com/blog/accelerate-protein-engineering-with-the-nvidia-bionemo-blueprint-for-generative-protein-binder-design/]
-   Databricks — Accelerating Drug Discovery blueprint — Production-grade, end-to-end workflow for protein data from FASTA to AI-assisted insights. — [https://www.databricks.com/blog/accelerating-drug-discovery-fasta-files-genai-insights-databricks]
-   Amazon SageMaker — Fine-tuning and deploying a protein language model (ESM-2) — Production-friendly workflow for fine-tuning and deploying a protein language model (ESM-2) at scale. — [https://aws.amazon.com/blogs/machine-learning/efficiently-fine-tune-the-esm-2-protein-language-model-with-amazon-sagemaker/]

## Minimum Valuable Build

For a hands-on build with this concept, see the MVB on [[protein language model]].

## Code & implementations

-   Facebook Research's ESM: [https://github.com/facebookresearch/esm]
-   Hugging Face Transformers: [https://huggingface.co/docs/transformers/model_doc/esm]

## What comes next

-   [[Protein Design]] — Protein language models are a key component in modern protein design pipelines.
-   [[Drug Discovery]] — Protein language models can accelerate the identification of novel drug targets and the design of therapeutic proteins.

## Connected topics

- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Protein LMs often use the Transformer architecture for sequence modeling.
- [Message Passing](../13-graph-relational-ai/message-passing.md) — Some protein LMs use message passing on graph representations of protein structures.
- [GNN Expressivity](../13-graph-relational-ai/gnn-expressivity.md) — GNN expressivity is relevant when using graph neural networks in protein LMs.
- [Contrastive Learning](../03-representation-learning/contrastive-learning.md) — Contrastive learning is often used to train protein LMs on unlabeled data.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Protein LMs are trained using backpropagation to update the model's parameters.
- [KV Cache](../09-algorithms-systems-ai/kv-cache.md) — KV cache is relevant for efficient inference with large protein language models.


## Further reading

-   Lilian Weng's survey on Large Language Models (lil'log, 2023) — Provides a comprehensive overview of large language models, including their architecture, training techniques, and applications.
-   Rao et al. (2019) — "Biological structure and function emerge from scaling unsupervised learning to 250 million protein sequences" — [URL NOT VERIFIED] — Demonstrates the power of unsupervised learning on large protein sequence datasets for predicting protein structure and function.
-   Devlin et al. (2018) — "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding" — [URL NOT VERIFIED] — Introduces the BERT model, which is a foundational architecture for many modern protein language models.
-   Vaswani et al. (2017) — "Attention is All You Need" — [URL NOT VERIFIED] — Introduces the Transformer architecture, which is the foundation for many modern protein language models.