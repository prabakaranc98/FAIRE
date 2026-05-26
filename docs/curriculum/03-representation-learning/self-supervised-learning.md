```yaml
---
title: Self-Supervised Learning
track: 03-representation-learning
tags: [self-supervised learning, representation learning, contrastive learning, unsupervised learning]
depth: applied
prereqs: [representation-learning, unsupervised-learning]
updated: 2024-07-02
has_mvb: true
---
# Self-Supervised Learning
> **TL;DR:** Self-supervised learning (SSL) allows models to learn useful representations from unlabeled data by creating their own "pseudo-labels," mimicking the benefits of supervised learning without the need for extensive manual annotation.

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
Imagine a child learning to identify objects. The child repeatedly interacts with a toy, touching it, looking at it from different angles, and hearing its name. They learn to recognize the toy without explicit instruction. This mirrors how self-supervised learning (SSL) works, allowing models to learn from raw data by creating their own "tasks" or "challenges."

Self-supervised learning addresses the data bottleneck inherent in supervised learning. Instead of relying on manually labeled datasets, SSL leverages the inherent structure within the data itself to generate pseudo-labels. These pseudo-labels are then used to train the model, enabling it to learn meaningful representations without human intervention. This approach is particularly valuable when labeled data is scarce or expensive to obtain.

The core idea behind SSL is to create a pretext task that forces the model to learn useful features. For example, a model might be trained to predict missing parts of an image, or to recognize different views of the same object. By solving these pretext tasks, the model learns representations that capture essential aspects of the data, which can then be transferred to downstream tasks with minimal fine-tuning.

## Why it matters at the frontier
Self-supervised learning is crucial for pushing the boundaries of AI because it unlocks the potential of vast amounts of unlabeled data, which is far more abundant than labeled data. This is particularly important in areas like computer vision and natural language processing, where the availability of labeled data often limits the performance of supervised models.

At the frontier, researchers are actively exploring how to design more effective pretext tasks and training strategies that can lead to even better representations. A key open problem is how to design self-supervised learning methods that are robust to noisy or incomplete data, particularly in real-world scenarios where data quality is often a challenge. Overcoming this challenge will enable the development of more robust and generalizable AI systems.

## Core concepts
- **Pretext Task** — A task designed to learn useful representations from unlabeled data, where the task itself is not the ultimate goal but rather a means to learn features.
- **Pseudo-labels** — Labels automatically generated from the data itself, used to train the model in a self-supervised manner.
- **Contrastive Learning** — A technique where the model learns to group similar data points together while pushing dissimilar points apart in the embedding space.
- **Data Augmentation** — Applying transformations to the input data to create different views or versions of the same data point, used to improve the robustness and generalization of the model.
- **Embedding Space** — A high-dimensional space where data points are represented as vectors, such that similar data points are close together and dissimilar data points are far apart.
- **Transfer Learning** — The process of using representations learned from a self-supervised task to improve performance on a downstream task with limited labeled data.
- **Negative Sampling** — A technique used in contrastive learning to select dissimilar data points (negatives) to contrast with similar data points (positives).

## Mathematical foundations
The core idea behind contrastive learning is to learn an embedding space where similar data points are close together and dissimilar data points are far apart. This is typically achieved by minimizing a loss function such as the InfoNCE loss:
\[
L = -\mathbb{E}_{x \sim p_{data}} \left[ \log \frac{\exp(f(x) \cdot f(x^+)/\tau)}{\sum_{x' \in N} \exp(f(x) \cdot f(x')/\tau)} \right]
\]
where \(x\) is a data point, \(p_{data}\) is the data distribution, \(f(x)\) is the embedding function, \(x^+\) is a positive sample (similar to \(x\)), \(N\) is the set of negative samples (dissimilar to \(x\)), and \(\tau\) is a temperature parameter. This equation says that we want to maximize the similarity between \(x\) and \(x^+\) while minimizing the similarity between \(x\) and all \(x'\) in \(N\).

The embedding function \(f(x)\) is typically a neural network that maps the input data to a high-dimensional vector representation:
\[
f(x) = g(h(x))
\]
where \(h(x)\) is a feature extractor (e.g., a convolutional neural network) and \(g(x)\) is a projection head that maps the extracted features to the embedding space. The projection head \(g(x)\) is used to ensure that the embedding space is well-behaved and that the learned representations are useful for downstream tasks.

The temperature parameter \(\tau\) controls the sharpness of the contrastive loss:
\[
\tau > 0
\]
where a smaller \(\tau\) encourages the model to be more confident in its predictions, while a larger \(\tau\) makes the loss smoother. The choice of \(\tau\) can have a significant impact on the performance of the model.

## Key algorithms / techniques
- **SimCLR (Chen et al. 2020)** — A simple framework for contrastive learning of visual representations, using data augmentation and a contrastive loss to learn embeddings.
- **MoCo (He et al. 2020)** — A contrastive learning method that uses a memory bank to store representations of past data points, allowing for a larger number of negative samples.
- **BYOL (Grill et al. 2020)** — A self-supervised learning method that avoids negative samples by using two neural networks that learn from each other.
- **CLIP (Radford et al. 2021)** — A model that learns visual representations by contrasting images with their corresponding text descriptions.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| A Survey on Self-supervised Contrastive Learning for Multimodal Text-Image Analysis | 2025 | Khan et al. | Provides a comprehensive overview of contrastive learning in text-image models, categorizing approaches and discussing recent advancements and applications. |
| MetaCLIP 2: A Worldwide Scaling Recipe | 2025 | Li et al. | Presents a recipe for training CLIP from scratch on worldwide web-scale image-text pairs. |
| SimCLR: A Simple Framework for Contrastive Learning of Visual Representations | 2020 | Chen et al. | Introduces a simple yet effective contrastive learning framework, laying the groundwork for many subsequent advancements. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Learning Visual Features from Unlabeled Video | 2015 | Wang & Gupta | Introduced the idea of using motion cues in unlabeled video to learn visual features. |
| Representation Learning with Contrastive Predictive Coding | 2019 | van den Oord et al. | Proposed a contrastive learning approach based on predicting future representations from past ones. |
| Self-Supervised Representation Learning | 2020 | Jing & Tian | Provides a broad overview of self-supervised learning techniques and their applications. |

## Current SotA
MetaCLIP 2 achieves strong performance on image-text retrieval tasks, demonstrating the effectiveness of scaling CLIP to worldwide web-scale data (Li et al. 2025). Object Concepts Emerge from Motion achieves state-of-the-art results in unsupervised object discovery by leveraging motion cues (Zhang et al. 2025). MoSiC: Optimal-Transport Motion Trajectory for Dense Self-Supervised Learning achieves state-of-the-art results in spatiotemporally consistent representation learning (Zhang et al. 2025).

## What's happening now
Research in self-supervised learning is currently focused on developing more robust and efficient methods for learning representations from unlabeled data. This includes exploring new pretext tasks, contrastive learning techniques, and architectures that can better capture the underlying structure of the data.

Engineering efforts are focused on scaling self-supervised learning models to larger datasets and deploying them in real-world applications. This involves optimizing training pipelines, developing efficient inference methods, and addressing the challenges of data quality and noise.

Open problems in self-supervised learning include designing methods that are robust to noisy or incomplete data, developing techniques for learning representations that are transferable across different tasks and domains, and understanding the theoretical properties of self-supervised learning algorithms.

## In production
- **Google** — Uses self-supervised learning for pre-training language models like BERT, which are then fine-tuned for various NLP tasks — Scale: Millions of users — [research.google](research.google)
- **Meta** — Employs self-supervised learning for training large-scale image and video models, used in applications like content understanding and recommendation — Scale: Billions of users — [ai.meta.com/research](ai.meta.com/research)
- **Amazon** — Utilizes self-supervised learning for training models used in product search and recommendation systems — Scale: Hundreds of millions of users — [aws.amazon.com/blogs/machine-learning](aws.amazon.com/blogs/machine-learning)

## Minimum Valuable Build

**What you're building:** A simple contrastive learning model using a pre-trained vision transformer (ViT) on the CIFAR-10 dataset.
**Why this build:** Demonstrates how to train a model to group similar images together in the embedding space using self-supervised learning.
**Stack:** PyTorch 2.0, torchvision 0.15, transformers 4.30, CUDA (if available)
**Estimated time:** 1-2 hours

### The recipe

1. **Install dependencies:**
   ```bash
   pip install torch torchvision transformers
   ```

2. **Import necessary libraries:**
   ```python
   import torch
   import torchvision
   import torchvision.transforms as transforms
   from PIL import Image
   from transformers import ViTModel, ViTConfig
   import torch.nn as nn
   import torch.nn.functional as F
   ```

3. **Define data transformations:**
   ```python
   transform = transforms.Compose([
       transforms.Resize((224, 224)),
       transforms.RandomHorizontalFlip(),
       transforms.ToTensor(),
       transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
   ])
   ```

4. **Load CIFAR-10 dataset:**
   ```python
   trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                           download=True, transform=transform)
   trainloader = torch.utils.data.DataLoader(trainset, batch_size=32,
                                             shuffle=True, num_workers=2)

   testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                          download=True, transform=transform)
   testloader = torch.utils.data.DataLoader(testset, batch_size=32,
                                            shuffle=False, num_workers=2)
   ```

5. **Define the contrastive learning model:**
   ```python
   class ContrastiveModel(nn.Module):
       def __init__(self, vit_model_name="google/vit-base-patch16-224"):
           super(ContrastiveModel, self).__init__()
           self.vit = ViTModel.from_pretrained(vit_model_name)
           self.projection_head = nn.Sequential(
               nn.Linear(self.vit.config.hidden_size, 256),
               nn.ReLU(),
               nn.Linear(256, 128)
           )

       def forward(self, img1, img2):
           # Pass both images through ViT
           output1 = self.vit(img1).pooler_output
           output2 = self.vit(img2).pooler_output

           # Pass through projection head
           proj1 = self.projection_head(output1)
           proj2 = self.projection_head(output2)

           # Normalize
           proj1 = F.normalize(proj1, dim=1)
           proj2 = F.normalize(proj2, dim=1)

           return proj1, proj2
   ```

6. **Define the contrastive loss function:**
   ```python
   def contrastive_loss(proj1, proj2, temperature=0.1):
       # Calculate cosine similarity
       similarity_matrix = torch.matmul(proj1, proj2.T)

       # Mask out the diagonal (similarity with itself)
       mask = torch.eye(proj1.size(0), device=proj1.device).bool()
       similarity_matrix = similarity_matrix.masked_fill(mask, -1e9)

       # Create labels (positive pairs are the same index)
       labels = torch.arange(proj1.size(0), device=proj1.device)

       # Calculate loss
       loss = F.cross_entropy(similarity_matrix / temperature, labels)
       return loss
   ```

7. **Initialize the model, optimizer, and device:**
   ```python
   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   model = ContrastiveModel().to(device)
   optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
   ```

8. **Train the model:**
   ```python
   epochs = 10
   for epoch in range(epochs):
       for i, data in enumerate(trainloader, 0):
           images, _ = data
           # Create two versions of the same image using augmentations
           img1 = images.to(device)
           img2 = transform(Image.fromarray(images[0].cpu().numpy().transpose(1,2,0)* 0.5 + 0.5)).unsqueeze(0).to(device).repeat(images.size(0), 1, 1, 1) # Simple augmentation

           optimizer.zero_grad()
           proj1, proj2 = model(img1, img2)
           loss = contrastive_loss(proj1, proj2)
           loss.backward()
           optimizer.step()

           if i % 100 == 0:
               print(f"Epoch: {epoch}, Batch: {i}, Loss: {loss.item()}")
   ```

### Expected output
After training, the model should be able to generate embeddings where similar images are closer together in the embedding space. While a quantitative evaluation (e.g., clustering accuracy) requires additional steps, you should observe a decreasing loss during training, indicating that the model is learning to distinguish between similar and dissimilar images.

### Common failure modes
- **CUDA out of memory error:** Reduce the batch size in the data loaders.
- **Loss not decreasing:** Adjust the learning rate or temperature parameter in the contrastive loss.
- **Model overfitting:** Increase the strength of the data augmentations.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations
- **PyTorch Lightning Bolts:** [https://github.com/Lightning-AI/lightning-bolts](https://github.com/Lightning-AI/lightning-bolts) — Provides implementations of various self-supervised learning algorithms.
- **Hugging Face Transformers:** [https://huggingface.co/transformers](https://huggingface.co/transformers) — Offers pre-trained models and tools for self-supervised learning tasks.

## What comes next

Understanding self-supervised learning provides a foundation for exploring more advanced representation learning techniques.

- [[contrastive-learning]] — Contrastive learning is a key technique used in many self-supervised learning methods, focusing on learning representations by comparing similar and dissimilar data points.
- [[transfer-learning]] — Self-supervised learning is often used as a pre-training step for transfer learning, where the learned representations are fine-tuned on a downstream task.

## Connected topics
- [Contrastive Learning](./contrastive-learning.md) — Contrastive learning is a common technique used in self-supervised learning.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Backpropagation is used to train neural networks in self-supervised learning.
- [Optimization](../04-neural-networks-dl/optimization.md) — Optimization algorithms are crucial for training self-supervised learning models.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers are often used in self-supervised learning for tasks like language modeling.
- [Message Passing](../13-graph-relational-ai/message-passing.md) — Message passing is used in self-supervised learning on graph-structured data.
- [Diffusion Models](../02-generative-modeling/diffusion-models.md) — Diffusion models can be trained using self-supervised techniques.


## Further reading
- Mikulasch & Zenke (2026) — "Understanding Self-Supervised Learning via Latent Distribution Matching" — [https://arxiv.org/html/2605.03517] — Explores self-supervised learning by examining how it finds general-purpose latent representations from complex data.
- Zhang et al. (2026) — "MAXIMIZING INCREMENTAL INFORMATION ENTROPY FOR CONTRASTIVE LEARNING" — [https://arxiv.org/pdf/2603.12594] — Investigates contrastive learning by focusing on maximizing incremental information entropy.
- Cai et al. (2026) — "The Geometric Mechanics of Contrastive Representation Learning: Alignment Potentials, Entropic Dispersion, and Cross-Modal Divergence" — [https://arxiv.org/html/2601.19597v1] — Examines the geometric mechanics of contrastive representation learning, focusing on alignment potentials, entropic dispersion, and cross-modal divergence.
- Luthra et al. (2025) — "Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning" — [https://arxiv.org/html/2506.04411] — Explores the relationship between self-supervised and supervised contrastive learning.
```