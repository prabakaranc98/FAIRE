```yaml
---
title: GNN Expressivity
track: 13-graph-relational-ai
tags: [GNN, graph neural networks, expressivity, Weisfeiler-Lehman, graph isomorphism]
depth: foundational
prereqs: [graph-theory, deep-learning, message-passing]
updated: 2024-07-02
has_mvb: false
---
# GNN Expressivity
> **TL;DR:** GNN expressivity refers to a GNN's ability to distinguish between different graph structures, which directly impacts its ability to learn and generalize across various graph-based tasks.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [Code & implementations](#code--implementations) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is
Imagine a social network where users are connected based on their interests and interactions. Now, picture a fraud detection system that analyzes financial transactions, represented as a graph where nodes are accounts and edges are transactions. These are just two examples of how graph data structures are used to model complex relationships. Graph Neural Networks (GNNs) are designed to learn from these graph-structured data.

GNN expressivity refers to the ability of a GNN to differentiate between different graph structures. A highly expressive GNN can map distinct graphs to distinct embeddings, allowing it to capture subtle structural differences. Conversely, a GNN with low expressivity may map different graphs to the same embedding, limiting its ability to learn and generalize effectively. This is crucial because many graph-based tasks, such as node classification, link prediction, and graph classification, rely on the GNN's ability to discern structural patterns.

The expressivity of a GNN is fundamentally tied to its architecture and the message-passing mechanism it employs. Simple GNNs, like Graph Convolutional Networks (GCNs), have limited expressivity, while more sophisticated architectures, such as Graph Transformers and higher-order GNNs, aim to overcome these limitations by incorporating more complex aggregation and update functions. Understanding GNN expressivity is essential for selecting the right architecture for a given task and for designing new GNNs that can capture increasingly complex graph structures.

## Why it matters at the frontier
GNN expressivity is a critical area of research because it directly impacts the performance of GNNs on a wide range of tasks, from drug discovery to social network analysis. As researchers push the boundaries of what GNNs can achieve, understanding and improving their expressivity becomes paramount. For example, in molecular property prediction, a GNN must be able to distinguish between molecules with subtle structural differences to accurately predict their properties.

The limitations in GNN expressivity also present a significant bottleneck in the development of graph foundation models. Current research focuses on designing GNN architectures that are provably more expressive than existing models while maintaining computational efficiency, particularly in the context of large and complex graphs. This includes exploring novel aggregation functions, incorporating higher-order structural information, and leveraging techniques from other areas of deep learning, such as Transformers, to enhance the expressive power of GNNs.

## Core concepts
- **Graph Isomorphism** — Determining whether two graphs are structurally identical, a fundamental problem in graph theory that serves as a benchmark for GNN expressivity.
- **Weisfeiler-Lehman (WL) Test** — A graph isomorphism test that iteratively aggregates neighbor information to refine node labels; GNN expressivity is often compared to the 1-WL test.
- **Message Passing** — The core mechanism in GNNs where nodes exchange information with their neighbors to update their representations.
- **Aggregation Function** — A function that combines the information received from neighboring nodes into a single representation, influencing the GNN's ability to distinguish different graph structures.
- **Node Features** — Attributes associated with each node in the graph, which provide initial information for the GNN to learn from.
- **Graph Representation Learning** — The process of learning vector representations (embeddings) of graphs that capture their structural and semantic properties.
- **Expressive Power** — The ability of a GNN to map different graph structures to distinct embeddings, indicating its capacity to capture subtle structural differences.

## Mathematical foundations
\[
h_i^{(l+1)} = \text{AGGREGATE}\left(h_i^{(l)}, \left\{h_j^{(l)}, \forall j \in \mathcal{N}(i)\right\}\right)
\]

where \(h_i^{(l)}\) is the hidden representation of node \(i\) at layer \(l\), and \(\mathcal{N}(i)\) is the set of neighbors of node \(i\). This represents the general form of message passing in GNNs.

This equation describes the aggregation of information from neighboring nodes, a key operation in GNNs, where each node updates its representation based on the aggregated information from its neighbors.

\[
\text{GCN}(X, A; \Theta) = \text{ReLU}\left(\tilde{D}^{-\frac{1}{2}}\tilde{A}\tilde{D}^{-\frac{1}{2}}X\Theta\right)
\]

where \(X\) is the input feature matrix, \(A\) is the adjacency matrix, \(\tilde{A} = A + I\) (with \(I\) being the identity matrix), \(\tilde{D}\) is the degree matrix of \(\tilde{A}\), and \(\Theta\) is the weight matrix. This is the equation for a Graph Convolutional Network (GCN) layer.

This equation defines the forward pass of a GCN layer, which performs a convolution-like operation on graph data by aggregating feature information from a node's neighbors, normalized by the degree matrix.

\[
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

where \(Q\) is the query matrix, \(K\) is the key matrix, \(V\) is the value matrix, and \(d_k\) is the dimension of the keys. This computes the attention weights and applies them to the values.

This is the fundamental equation for the self-attention mechanism, a core component of Transformer-based models used in GNNs, allowing the model to weigh the importance of different parts of the input when computing the output.

## Key algorithms / techniques
- **Graph Convolutional Networks (GCNs)** — Introduced by Kipf and Welling in 2016, GCNs perform convolution-like operations on graphs by aggregating information from neighboring nodes.
- **Graph Attention Networks (GATs)** — Proposed by Veličković et al. in 2018, GATs use attention mechanisms to weight the importance of different neighbors during aggregation.
- **Graph Transformers** — Leverage the Transformer architecture for graph learning, enabling the capture of long-range dependencies and complex relationships between nodes.
- **Higher-Order GNNs** — Extend the message-passing mechanism to consider higher-order neighborhoods, increasing the expressivity of the GNN.
- **Weisfeiler-Lehman (WL) Subtree Kernel** — A graph kernel based on the WL isomorphism test, used to measure the similarity between graphs and provide a benchmark for GNN expressivity.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| How Powerful are Graph Neural Networks? | 2018 | Xu et al. | Formally defines and analyzes the limitations of GNNs based on the Weisfeiler-Lehman (WL) test. |
| Weisfeiler and Lehman Go Neural: Higher-Order Graph Neural Networks | 2019 | Morris et al. | Extends the WL test to higher-order GNNs, showing how to increase expressivity. |
| Fast Graph Representation Learning with PyTorch Geometric | 2022 | Fey et al. | Provides a practical implementation of GNNs, allowing readers to experiment with different architectures. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Convolutional Neural Networks on Graphs with Fast Localized Spectral Filtering | 2016 | Introduced spectral graph convolutions, laying the foundation for many subsequent GNN architectures. |
| Semi-Supervised Classification with Graph Convolutional Networks | 2017 | Proposed the Graph Convolutional Network (GCN) model, a simple and effective GNN architecture. |
| Graph Attention Networks | 2018 | Introduced the Graph Attention Network (GAT), which uses attention mechanisms to weight the importance of different neighbors. |

## Current SotA
BiScale-GTR achieves state-of-the-art performance on molecular property prediction benchmarks (2024). Cardinality-Preserving Attention Channels for Graph Transformers show strong performance in molecular property prediction (2024). Chordless Structure-based Graph Neural Network (CSGNN) proves its expressiveness is strictly more powerful than the k-hop GNN (KPGNN) with polynomial complexity (2025).

## What's happening now
Research is actively exploring new GNN architectures that can overcome the limitations of existing models. This includes investigating novel aggregation functions, incorporating higher-order structural information, and leveraging techniques from other areas of deep learning, such as Transformers, to enhance the expressive power of GNNs. A key focus is on designing GNNs that are provably more expressive than the 1-WL test, while maintaining computational efficiency.

Engineering efforts are focused on developing scalable GNN systems that can handle large and complex graphs. This involves optimizing the message-passing mechanism, exploring distributed training strategies, and designing hardware-accelerated GNN implementations. Frameworks like TF-GNN and GraphStorm are continuously evolving to support the deployment of GNNs in real-world applications.

The open problem is: How can we design GNN architectures that are provably more expressive than existing models while maintaining computational efficiency, particularly in the context of large and complex graphs? This includes addressing challenges such as over-smoothing, scalability, and the development of robust evaluation metrics for GNN expressivity.

## In production
- Google — TF-GNN 1.0 — Large scale, with strong support for heterogeneous graphs. — [https://research.google/blog/graph-neural-networks-in-tensorflow/]
- Amazon — GraphStorm v0.5 — Enables real-time, enterprise-scale GNN inference for fraud prevention. — [https://aws.amazon.com/blogs/machine-learning/modernize-fraud-prevention-graphstorm-v0-5-for-real-time-inference/]
- Amazon — Training GNNs for protein structures — Training graph neural nets for millions of proteins. — [https://aws.amazon.com/blogs/machine-learning/train-graph-neural-nets-for-millions-of-proteins-on-amazon-sagemaker-and-amazon-documentdb-with-mongodb-compatibility/]

## Minimum Valuable Build
For a hands-on build with this concept, see the MVB on [[gnn-architectures-for-molecular-property-prediction]].

## Code & implementations
- **PyTorch Geometric (PyG)**: A library for deep learning on graphs and other irregular structures. [https://pytorch-geometric.readthedocs.io/en/latest/]
- **Deep Graph Library (DGL)**: A Python package designed for easy implementation of graph neural networks. [https://www.dgl.ai/]

## What comes next

- [[graph-attention-networks]] — explore how attention mechanisms can improve GNN expressivity by weighting the importance of different neighbors.
- [[graph-transformers]] — understand how Transformers can be adapted for graph learning, enabling the capture of long-range dependencies.

## Connected topics

- [Equivariant GNN](./equivariant-gnn.md) — Equivariant GNNs are related to GNN expressivity through their design.
- [Equivariant Networks](../12-physics-scientific-ai/equivariant-networks.md) — Equivariant networks share similar concepts with GNNs regarding expressivity.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Backpropagation is a fundamental concept used in training GNNs, impacting expressivity.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers and GNNs both aim to capture relationships, influencing expressivity.
- [Complexity Classes](../10-complexity-cognition/complexity-classes.md) — Complexity classes can be relevant when analyzing the computational limits of GNNs.


## Further reading
- Xu et al. (2018) — "How Powerful are Graph Neural Networks?" — [https://arxiv.org/abs/1810.00826] — This paper provides a theoretical analysis of the expressive power of GNNs based on the Weisfeiler-Lehman graph isomorphism test.
- Morris et al. (2019) — "Weisfeiler and Lehman Go Neural: Higher-Order Graph Neural Networks" — [https://arxiv.org/abs/1810.02244] — This paper extends the WL test to higher-order GNNs, showing how to increase expressivity by considering more complex graph structures.
- Kim et al. (2022) — "Pure Transformers are Powerful Graph Learners" — [https://arxiv.org/pdf/2207.02505] — This paper demonstrates that standard Transformers, without graph-specific modifications, can be effective graph learners.
- Chen et al. (2019) — "Path-Augmented Graph Transformer Network" — [https://ar5iv.labs.arxiv.org/html/1905.12712] — This paper introduces the Path-Augmented Graph Transformer Network, which builds upon Graph Convolutional Networks (GCNs) for molecular representation learning.
- Fey et al. (2022) — "Fast Graph Representation Learning with PyTorch Geometric" — [https://arxiv.org/abs/1903.02428] — This paper introduces PyTorch Geometric, a library for deep learning on graphs, and provides a practical guide to implementing various GNN architectures.
- Finkelshtein et al. (2025) — "Equivariance Everywhere All At Once: A Recipe for Graph Foundation Models" — [https://arxiv.org/abs/2506.14291v5] — This paper proposes a recipe for designing graph foundation models, emphasizing the importance of symmetries in graph machine learning.