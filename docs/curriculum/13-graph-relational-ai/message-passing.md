```yaml
---
title: Message Passing
track: 13-graph-relational-ai
tags: [graph neural networks, message passing, GNN, relational AI]
depth: foundational
prereqs: [graph-theory, neural-networks]
updated: 2024-11-04
has_mvb: true
---
# Message Passing
> **TL;DR:** Message passing is a fundamental computational paradigm for exchanging and aggregating information within graphs, enabling the creation of powerful node and graph representations.

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
Imagine you're trying to understand the relationships between people in a social network. Each person is a node, and their connections are edges. Now, imagine you want to predict which users are likely to become friends. Message passing is a technique that allows each user to gather information from their friends, and then use that information to make a prediction. This process is repeated across the network, allowing information to spread and influence the final outcome.

Message passing is a computational paradigm where nodes in a graph exchange information with their neighbors, updating their own state based on the received messages. This iterative process allows nodes to aggregate information from across the graph, enabling the learning of complex relationships and patterns. The core idea is to propagate information through the graph structure, allowing each node to "learn" from its surroundings.

Message passing is the foundation of Graph Neural Networks (GNNs). By iteratively updating node representations based on aggregated information from their neighbors, GNNs can learn powerful representations of nodes, edges, and entire graphs. This makes them suitable for a wide range of tasks, including node classification, link prediction, and graph classification.

## Why it matters at the frontier
Message passing is crucial at the frontier of graph-based machine learning because it provides a flexible and powerful framework for learning from structured data. It enables researchers to develop novel GNN architectures that can handle increasingly complex graph structures and data types. The ability to effectively propagate and aggregate information across graphs is essential for solving challenging problems in various domains, including drug discovery, social network analysis, and recommendation systems.

One key open problem is: How can we design message-passing mechanisms that effectively capture both the structural and contextual divergence in rich-text graphs, leading to improved representation learning in complex text datasets? Addressing this question would allow for better handling of complex data like text, going beyond simple graph structures. Furthermore, designing message-passing mechanisms that are provably robust to adversarial attacks on graph data, ensuring that the learned representations remain accurate and reliable even when the graph structure or node features are maliciously altered, is a critical area of research.

## Core concepts
- **Node:** A fundamental unit in a graph, representing an entity or object.
- **Edge:** A connection between two nodes, representing a relationship or interaction.
- **Message:** Information passed between nodes along edges during message passing.
- **Aggregation:** The process of combining messages received by a node from its neighbors.
- **Update function:** A function that updates a node's state based on its current state and aggregated messages.
- **Graph Neural Network (GNN):** A neural network architecture that uses message passing to learn representations of graphs.
- **Node Feature:** A vector of information associated with each node in the graph.

## Mathematical foundations
Message passing involves iteratively updating the feature vectors of nodes in a graph based on the features of their neighbors. The update rule can be expressed as:

\[
\mathbf{h}_i^{(l+1)} = \sigma\left(\mathbf{W}^{(l)} \cdot \text{AGGREGATE}\left(\{\mathbf{h}_j^{(l)} : j \in \mathcal{N}(i)\}\right)\right)
\]

where \(\mathbf{h}_i^{(l+1)}\) is the feature vector of node \(i\) at layer \(l+1\), \(\sigma\) is an activation function, \(\mathbf{W}^{(l)}\) is a weight matrix at layer \(l\), \(\text{AGGREGATE}\) is an aggregation function, and \(\mathcal{N}(i)\) is the set of neighbors of node \(i\). This equation says that the new representation of a node is a function of the aggregated representations of its neighbors.

The aggregation function can take various forms, such as mean, sum, or max. For example, using a mean aggregator:

\[
\text{AGGREGATE}\left(\{\mathbf{h}_j^{(l)} : j \in \mathcal{N}(i)\}\right) = \frac{1}{|\mathcal{N}(i)|} \sum_{j \in \mathcal{N}(i)} \mathbf{h}_j^{(l)}
\]

where \(|\mathcal{N}(i)|\) is the number of neighbors of node \(i\). This equation calculates the average feature vector of the neighbors of node \(i\).

The message passed from node \(j\) to node \(i\) can be defined as:

\[
\mathbf{m}_{ji}^{(l)} = f\left(\mathbf{h}_i^{(l)}, \mathbf{h}_j^{(l)}, \mathbf{e}_{ji}\right)
\]

where \(\mathbf{m}_{ji}^{(l)}\) is the message from node \(j\) to node \(i\) at layer \(l\), \(f\) is a message function, and \(\mathbf{e}_{ji}\) represents edge features between nodes \(j\) and \(i\). This equation defines the message as a function of the node features and edge features.

## Key algorithms / techniques
- **Graph Convolutional Networks (GCNs)** — Introduced by Kipf and Welling (2016), GCNs use a spectral graph convolution operation to aggregate information from neighbors.
- **Graph Attention Networks (GATs)** — Introduced by Veličković et al. (2018), GATs use an attention mechanism to weight the importance of different neighbors during aggregation.
- **Message Passing Neural Networks (MPNNs)** — Proposed by Gilmer et al. (2017), MPNNs provide a general framework for message passing, encompassing various GNN architectures.
- **Adaptive Depth Message Passing GNN (ADMP-GNN)** — Introduced by Abbahaddou et al. (2025), this method adaptively adjusts the depth of message passing in GNNs to improve performance.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| ADMP-GNN: Adaptive Depth Message Passing GNN | 2025 | Abbahaddou et al. | Introduces a method for adaptive depth message passing in GNNs. |
| Edge Directionality Improves Learning on Heterophilic Graphs | 2023 | Di Giovanni et al. | Investigates how edge directionality can improve learning on heterophilic graphs. |
| P2GNN: Two Prototype Sets to boost GNN Performance | 2026 | Jain et al. | Proposes a novel approach using two prototype sets to enhance the performance of Message Passing Graph Neural Networks (MP-GNNs). |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Semi-Supervised Classification with Graph Convolutional Networks | 2016 | Introduced Graph Convolutional Networks (GCNs), a foundational architecture for graph-based learning. |
| Graph Attention Networks | 2018 | Introduced Graph Attention Networks (GATs), incorporating attention mechanisms into message passing. |
| Neural Message Passing for Quantum Chemistry | 2017 | Proposed Message Passing Neural Networks (MPNNs), a general framework for message passing. |

## Current SotA
P2GNN achieves state-of-the-art performance by using two prototype sets to enhance the performance of Message Passing Graph Neural Networks (MP-GNNs) (Jain et al., 2026). Adaptive Depth Message Passing GNN (ADMP-GNN) adaptively adjusts the depth of message passing in GNNs to improve performance (Abbahaddou et al., 2025). Edge Directionality Improves Learning on Heterophilic Graphs (Di Giovanni et al., 2023).

## What's happening now
Research is focusing on developing more expressive and efficient message-passing mechanisms. This includes exploring novel aggregation functions, attention mechanisms, and ways to incorporate higher-order graph structures. The goal is to create GNNs that can capture more complex relationships and patterns in graphs.

Engineering efforts are focused on scaling GNNs to handle large-scale graphs and real-time applications. This involves developing efficient implementations of message-passing algorithms and leveraging distributed computing frameworks. The aim is to make GNNs practical for a wider range of applications.

A key open problem is: How can we design message-passing mechanisms that are provably robust to adversarial attacks on graph data, ensuring that the learned representations remain accurate and reliable even when the graph structure or node features are maliciously altered? Addressing this question is crucial for deploying GNNs in security-sensitive applications.

## In production
- LinkedIn — Kafka — Handles over 800 billion messages per day (≈175 TB) produced and 650 TB consumed daily, peaking at 13 million messages per second. — [https://engineering.linkedin.com/kafka/running-kafka-scale](https://engineering.linkedin.com/kafka/running-kafka-scale)
- LinkedIn — Apache Samza — Runs in production across multiple data centers. — [https://engineering.linkedin.com/samza/operating-apache-samza-scale](https://engineering.linkedin.com/samza/operating-apache-samza-scale)

## Minimum Valuable Build

**What you're building:** A simple Graph Neural Network (GNN) with a single message-passing layer to classify nodes in a citation network.
**Why this build:** This demonstrates the fundamental concept of message passing, where node features are updated based on information from neighboring nodes.
**Stack:** Python 3.8, PyTorch 1.10, PyTorch Geometric (PyG) 2.0.
**Estimated time:** 1-2 hours.

### The recipe

1. **Install PyTorch and PyTorch Geometric:**
   ```bash
   pip install torch==1.10.0+cu113 -f https://download.pytorch.org/whl/torch_stable.html
   pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-1.10.0+cu113.html
   ```

2. **Import necessary libraries:**
   ```python
   import torch
   import torch.nn as nn
   import torch.nn.functional as F
   from torch_geometric.datasets import Planetoid
   from torch_geometric.nn import MessagePassing
   from torch_geometric.utils import add_self_loops, degree
   ```

3. **Define a simple message-passing layer:**
   ```python
   class GCNConv(MessagePassing):
       def __init__(self, in_channels, out_channels):
           super(GCNConv, self).__init__(aggr='add')  # "Add" aggregation (Step 5).
           self.lin = nn.Linear(in_channels, out_channels)

       def forward(self, x, edge_index):
           # x has shape [N, in_channels]
           # edge_index has shape [2, E]

           # Step 1: Add self-loops
           edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

           # Step 2: Linearly transform node feature matrix.
           x = self.lin(x)

           # Step 3: Compute node degrees.
           row, col = edge_index
           deg = degree(col, x.size(0), dtype=x.dtype)
           deg_inv_sqrt = deg.pow(-0.5)
           deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
           norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

           # Step 4-5: Start propagating messages.
           return self.propagate(edge_index, x=x, norm=norm)

       def message(self, x_j, norm):
           # x_j has shape [E, out_channels]

           # Step 4: Normalize node features.
           return norm.view(-1, 1) * x_j
   ```

4. **Define a simple GNN model:**
   ```python
   class Net(torch.nn.Module):
       def __init__(self, in_channels, hidden_channels, out_channels):
           super(Net, self).__init__()
           self.conv1 = GCNConv(in_channels, hidden_channels)
           self.conv2 = GCNConv(hidden_channels, out_channels)

       def forward(self, x, edge_index):
           x = self.conv1(x, edge_index)
           x = F.relu(x)
           x = self.conv2(x, edge_index)
           return F.log_softmax(x, dim=1)
   ```

5. **Load the Cora dataset:**
   ```python
   dataset = Planetoid(root='/tmp/Cora', name='Cora')
   data = dataset[0]
   ```

6. **Train the model:**
   ```python
   device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   model = Net(dataset.num_node_features, 16, dataset.num_classes).to(device)
   data = data.to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

   model.train()
   for epoch in range(200):
       optimizer.zero_grad()
       out = model(data.x, data.edge_index)
       loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
       loss.backward()
       optimizer.step()

   model.eval()
   _, pred = model(data.x, data.edge_index).max(dim=1)
   correct = float (pred[data.test_mask].eq(data.y[data.test_mask]).sum().item())
   acc = correct / data.test_mask.sum().item()
   print('Accuracy: {:.4f}'.format(acc))
   ```

### Expected output
The code will print the accuracy of the trained model on the test set. Expect an accuracy of around 0.75-0.80.

### Common failure modes
- **CUDA out of memory:** Reduce the hidden_channels size in the Net class.
- **Incorrect PyTorch Geometric installation:** Ensure the PyTorch Geometric version is compatible with your PyTorch version.

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations
- **PyTorch Geometric (PyG):** [https://github.com/rusty1s/pytorch_geometric](https://github.com/rusty1s/pytorch_geometric)
- **Deep Graph Library (DGL):** [https://github.com/dmlc/dgl](https://github.com/dmlc/dgl)

## What comes next
- [[graph-convolutional-networks]] — implements a specific message-passing scheme using spectral convolutions.
- [[graph-attention-networks]] — introduces an attention mechanism to weight the importance of neighbors during message passing.

## Connected topics
- [GNN Expressivity](./gnn-expressivity.md) — Message passing is a core component of Graph Neural Networks (GNNs).
- [Equivariant GNN](./equivariant-gnn.md) — Equivariant GNNs utilize message passing to incorporate equivariance properties.
- [GNN Expressivity](./gnn-expressivity.md) — Message passing is a key operation in Graph Neural Networks (GNNs).
- [Equivariant Networks](../12-physics-scientific-ai/equivariant-networks.md) — Equivariant networks, like GNNs, can use message passing for symmetry handling.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Message passing can be seen as a form of backpropagation on graphs.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers use message passing-like mechanisms through attention.


## Further reading
- Kipf and Welling (2016) — "Semi-Supervised Classification with Graph Convolutional Networks" — [https://arxiv.org/abs/1609.02907] — This paper introduces Graph Convolutional Networks (GCNs), a foundational architecture for graph-based learning.
- Veličković et al. (2018) — "Graph Attention Networks" — [https://arxiv.org/abs/1710.10903] — This paper introduces Graph Attention Networks (GATs), incorporating attention mechanisms into message passing.
- Gilmer et al. (2017) — "Neural Message Passing for Quantum Chemistry" — [https://arxiv.org/abs/1704.01212] — This paper proposes Message Passing Neural Networks (MPNNs), a general framework for message passing.
- Lilian Weng's survey on Graph Neural Networks (lil'log, 2021) — Provides a comprehensive overview of various GNN architectures and their applications.