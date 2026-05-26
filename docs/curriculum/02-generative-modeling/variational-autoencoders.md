```yaml
---
title: Variational Autoencoders
track: 02-generative-modeling
tags: [VAE, generative model, latent space, encoder-decoder, ELBO]
depth: foundational
prereqs: [autoencoders, probability, neural-networks]
updated: 2024-10-24
has_mvb: true
---
# Variational Autoencoders
> **TL;DR:** Variational Autoencoders (VAEs) are generative models that learn a latent representation of data, allowing for the generation of new, similar data points by sampling from that latent space.

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
Imagine you're trying to build a system that can generate realistic images of cats, but instead of just memorizing existing pictures, it can create entirely new ones. Or, consider a medical imaging system that can reconstruct 3D models from limited scan data. These tasks require a model that can understand the underlying structure of data and generate new instances. Variational Autoencoders (VAEs) are a key technology for this.

VAEs are a type of generative model that learn a compressed, latent representation of input data. Unlike standard autoencoders that learn a deterministic mapping, VAEs learn a probability distribution in the latent space. This probabilistic approach allows us to sample from the latent space and generate new data points that resemble the training data. The VAE architecture consists of two main components: an encoder, which maps the input data to a latent distribution, and a decoder, which maps samples from the latent distribution back to the original data space.

The key innovation in VAEs is the use of variational inference to train the model. Because the true posterior distribution over the latent variables is often intractable, VAEs approximate it using a simpler, tractable distribution, typically a Gaussian. The model is trained by maximizing a lower bound on the marginal likelihood of the data, known as the Evidence Lower Bound (ELBO). This ELBO balances the reconstruction accuracy of the decoder with the similarity between the approximate posterior and a prior distribution over the latent space, encouraging the latent space to be well-structured and continuous.

## Why it matters at the frontier
VAEs are crucial for addressing several open problems in generative modeling. One major challenge is generating high-fidelity, diverse samples from complex, multi-modal data. VAEs offer a framework for learning disentangled representations, where different latent dimensions capture independent factors of variation in the data. This disentanglement can enable more controllable generation, where specific attributes of the generated data can be manipulated by modifying the corresponding latent dimensions.

Frontier labs are actively exploring VAEs for various applications, including image and video generation, anomaly detection, and representation learning. Researchers are also investigating novel VAE architectures and training techniques to improve the quality and diversity of generated samples. For example, researchers are exploring ways to mitigate common artifacts in VAE training, such as blurry reconstructions and mode collapse (Li et al., 2025). The ability to generate realistic and diverse data is critical for advancing research in areas such as computer vision, natural language processing, and reinforcement learning.

## Core concepts
- **Encoder** — A neural network that maps input data to parameters of a latent distribution, typically the mean and variance of a Gaussian.
- **Decoder** — A neural network that maps samples from the latent distribution back to the original data space, generating a reconstruction of the input.
- **Latent Space** — A lower-dimensional space that captures the underlying structure and essential features of the data.
- **Variational Inference** — A technique used to approximate intractable posterior distributions by optimizing a tractable approximation.
- **Evidence Lower Bound (ELBO)** — A lower bound on the marginal likelihood of the data, used as the objective function for training VAEs.
- **Kullback-Leibler (KL) Divergence** — A measure of the difference between two probability distributions, used to regularize the latent space in VAEs.
- **Reparameterization Trick** — A technique that allows gradients to be backpropagated through stochastic nodes, enabling end-to-end training of VAEs.

## Mathematical foundations
\[\mathcal{L}_{VAE} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{KL}(q(z|x) || p(z))\]

where \(\mathcal{L}_{VAE}\) is the evidence lower bound (ELBO), \(x\) is the input data, \(z\) is the latent variable, \(q(z|x)\) is the approximate posterior, \(p(x|z)\) is the likelihood, \(p(z)\) is the prior, and \(D_{KL}\) is the Kullback-Leibler divergence.
This is the fundamental loss function for Variational Autoencoders, balancing reconstruction accuracy and regularization of the latent space.

\[D_{KL}(q(z|x) || p(z)) = \int q(z|x) \log \frac{q(z|x)}{p(z)} dz\]

where \(q(z|x)\) is the approximate posterior distribution of the latent variable \(z\) given input \(x\), and \(p(z)\) is the prior distribution of the latent variable \(z\).
This equation defines the Kullback-Leibler divergence, which measures the difference between the approximate posterior and the prior, encouraging the latent space to be regularized.

\[\text{ELBO} = \mathbb{E}_{q(z|x)}[\log p(x|z)] - \mathbb{E}_{q(z|x)}[\log q(z|x)] + \mathbb{E}_{q(z|x)}[\log p(z)]\]

where ELBO is the Evidence Lower Bound, \(x\) is the input data, \(z\) is the latent variable, \(q(z|x)\) is the approximate posterior, and \(p(x|z)\) is the likelihood, \(p(z)\) is the prior.
This is an alternative formulation of the ELBO, which is the objective function that VAEs aim to maximize during training.

## Key algorithms / techniques
- **Reparameterization Trick (Kingma & Welling, 2013)** — Enables gradient-based optimization of VAEs by expressing the latent variable as a deterministic function of the parameters of the approximate posterior and a noise variable.
- **Variational Inference (Jordan et al., 1999)** — Approximates the intractable posterior distribution over latent variables by optimizing a tractable variational distribution.
- **Adversarial Training (Goodfellow et al., 2014)** — Combines VAEs with Generative Adversarial Networks (GANs) to improve the quality and realism of generated samples.

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Auto-Encoding Variational Bayes | 2013 | Kingma & Welling | Introduces the core concept of VAEs, laying the foundation for all subsequent work. |
| Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders | 2024 | Rui Chen et al. | Demonstrates a state-of-the-art approach to improving VAE reconstruction quality, specifically for 3D shapes, introducing novel sampling strategies and attention mechanisms. |
| VIVAT: Virtuous Improving VAE Training through Artifact Mitigation | 2025 | Li et al. | Explores a novel approach to VAE training, specifically for mitigating common artifacts in KL-VAE training without requiring radical architectural changes. |

## Seminal papers & test-of-time
| Paper | Year | Key contribution |
|---|---|---|
| Auto-Encoding Variational Bayes | 2013 | Introduces the Variational Autoencoder (VAE) framework for generative modeling. |
| Stochastic Backpropagation and Approximate Inference in Deep Generative Models | 2014 | Introduces the reparameterization trick, enabling efficient training of VAEs. |

## Current SotA
Dora-VAE achieves state-of-the-art reconstruction quality for 3D shapes by using a sharp edge sampling strategy and a dual cross-attention mechanism (Rui Chen et al., 2024). VIVAT (Li et al., 2025) provides a systematic approach to mitigating common artifacts in KL-VAE training, enhancing the overall quality of generated samples. Jian et al. (2025) introduces FAE (Foundation Auto-Encoders), a foundation generative-AI model for anomaly detection in time-series data, based on Variational Auto-Encoders (VAEs).

## What's happening now
Research on VAEs is currently focused on improving the quality, diversity, and controllability of generated samples. New architectures, such as incorporating attention mechanisms and normalizing flows, are being explored to enhance the representational capacity of VAEs and reduce artifacts in generated data. Additionally, researchers are investigating methods for disentangling latent representations, allowing for more fine-grained control over the generated outputs.

Engineering efforts are focused on deploying VAEs in real-world applications, such as anomaly detection, image generation, and data compression. Companies are leveraging cloud platforms like Amazon SageMaker to scale VAE training and inference, enabling the processing of large datasets and the deployment of VAE models in production environments. Optimizations for inference speed and memory usage are also critical for deploying VAEs on resource-constrained devices.

A key open problem is developing VAE architectures that can effectively learn and generate high-fidelity, diverse, and controllable outputs from complex, multi-modal data sources, while also being computationally efficient and robust to noisy data. This requires addressing challenges such as mode collapse, blurry reconstructions, and the difficulty of learning disentangled representations. Further research is needed to develop novel training techniques and architectural innovations that can overcome these limitations.

## In production
- Amazon — Deploying variational autoencoders for anomaly detection with TensorFlow Serving on Amazon SageMaker — Not specified — [https://aws.amazon.com/blogs/machine-learning/deploying-variational-autoencoders-for-anomaly-detection-with-tensorflow-serving-on-amazon-sagemaker/](https://aws.amazon.com/blogs/machine-learning/deploying-variational-autoencoders-for-anomaly-detection-with-tensorflow-serving-on-amazon-sagemaker/)
- Zalando — Scalable, template-driven MLOps architecture for large-scale inference and production deployment on AWS SageMaker — Not specified — [https://aws.amazon.com/blogs/machine-learning/how-zalando-optimized-large-scale-inference-and-streamlined-ml-operations-on-amazon-sagemaker/](https://aws.amazon.com/blogs/machine-learning/how-zalando-optimized-large-scale-inference-and-streamlined-ml-operations-on-amazon-sagemaker/)
- Veriff — Scalable, cost-efficient production deployment of many computer-vision ML models on AWS SageMaker — Decreased deployment time by 80% — [https://aws.amazon.com/blogs/machine-learning/how-veriff-decreased-deployment-time-by-80-using-amazon-sagemaker-multi-model-endpoints/](https://aws.amazon.com/blogs/machine-learning/how-veriff-decreased-deployment-time-by-80-using-amazon-sagemaker-multi-model-endpoints/)
- Crexi — Scalable, production-grade ML deployment framework on AWS — Not specified — [https://aws.amazon.com/blogs/machine-learning/how-crexi-achieved-ml-models-deployment-on-aws-at-scale-and-boosted-efficiency/](https://aws.amazon.com/blogs/machine-learning/how-crexi-achieved-ml-models-deployment-on-aws-at-scale-and-boosted-efficiency/)

## Minimum Valuable Build
**What you're building:** A simple convolutional VAE that generates images of handwritten digits from the MNIST dataset.
**Why this build:** This build demonstrates the core principles of VAEs: encoding data into a latent space, and decoding latent vectors back into images.
**Stack:** PyTorch 2.0.0, torchvision 0.15.0
**Estimated time:** 2-3 hours

### The recipe

1. **Set up the environment:**
   ```python
   import torch
   import torch.nn as nn
   import torch.optim as optim
   from torchvision import datasets, transforms
   from torch.utils.data import DataLoader
   import matplotlib.pyplot as plt
   import numpy as np

   device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   print(f"Using device: {device}")
   ```

2. **Define the VAE architecture:**
   ```python
   class VAE(nn.Module):
       def __init__(self, latent_dim):
           super(VAE, self).__init__()

           # Encoder
           self.encoder = nn.Sequential(
               nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1),  # 28x28 -> 14x14
               nn.ReLU(),
               nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), # 14x14 -> 7x7
               nn.ReLU(),
               nn.Flatten(),
               nn.Linear(64 * 7 * 7, 256),
               nn.ReLU()
           )
           self.fc_mu = nn.Linear(256, latent_dim)
           self.fc_logvar = nn.Linear(256, latent_dim)

           # Decoder
           self.decoder = nn.Sequential(
               nn.Linear(latent_dim, 256),
               nn.ReLU(),
               nn.Linear(256, 64 * 7 * 7),
               nn.ReLU(),
               nn.Unflatten(dim=1, unflattened_size=(64, 7, 7)),
               nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1), # 7x7 -> 14x14
               nn.ReLU(),
               nn.ConvTranspose2d(32, 1, kernel_size=3, stride=2, padding=1, output_padding=1), # 14x14 -> 28x28
               nn.Sigmoid()
           )

       def encode(self, x):
           x = self.encoder(x)
           mu = self.fc_mu(x)
           logvar = self.fc_logvar(x)
           return mu, logvar

       def reparameterize(self, mu, logvar):
           std = torch.exp(0.5 * logvar)
           eps = torch.randn_like(std)
           return mu + eps * std

       def decode(self, z):
           x_hat = self.decoder(z)
           return x_hat

       def forward(self, x):
           mu, logvar = self.encode(x)
           z = self.reparameterize(mu, logvar)
           x_hat = self.decode(z)
           return x_hat, mu, logvar
   ```

3. **Define the loss function:**
   ```python
   def loss_function(x_hat, x, mu, logvar):
       reconstruction_loss = nn.functional.binary_cross_entropy(x_hat, x, reduction='sum')
       kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
       return reconstruction_loss + kl_divergence
   ```

4. **Load the MNIST dataset:**
   ```python
   transform = transforms.Compose([
       transforms.ToTensor()
   ])

   train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
   test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

   batch_size = 64
   train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
   test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
   ```

5. **Train the VAE:**
   ```python
   latent_dim = 20
   model = VAE(latent_dim).to(device)
   optimizer = optim.Adam(model.parameters(), lr=1e-3)

   epochs = 10
   for epoch in range(epochs):
       model.train()
       train_loss = 0
       for batch_idx, (data, _) in enumerate(train_loader):
           data = data.to(device)
           optimizer.zero_grad()
           x_hat, mu, logvar = model(data)
           loss = loss_function(x_hat, data, mu, logvar)
           loss.backward()
           train_loss += loss.item()
           optimizer.step()

           if batch_idx % 100 == 0:
               print(f"Epoch {epoch+1}/{epochs}, Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item() / len(data)}")

       print(f"Epoch {epoch+1}/{epochs}, Average Loss: {train_loss / len(train_loader.dataset)}")
   ```

6. **Generate new images:**
   ```python
   model.eval()
   with torch.no_grad():
       # Sample from the latent space
       num_samples = 16
       z = torch.randn(num_samples, latent_dim).to(device)
       generated_images = model.decode(z)

       # Display the generated images
       fig, axes = plt.subplots(4, 4, figsize=(8, 8))
       for i, ax in enumerate(axes.flatten()):
           img = generated_images[i].cpu().squeeze().numpy()
           ax.imshow(img, cmap='gray')
           ax.axis('off')
       plt.show()
   ```

### Expected output
After training, the code will display 16 generated images of handwritten digits. The images will not be perfect, but they should resemble digits from the MNIST dataset. You should also see the loss decreasing during training.

### Common failure modes
- **Loss not decreasing:** Reduce the learning rate or increase the number of epochs. → Try reducing the learning rate to 1e-4 or increasing the number of epochs to 20.
- **Generated images are blurry:** Increase the capacity of the encoder and decoder networks. → Add more convolutional layers or increase the number of channels in the existing layers.
- **CUDA out of memory error:** Reduce the batch size. → Try reducing the batch size to 32 or 16.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations
- [PyTorch VAE Example](https://github.com/pytorch/examples/tree/main/vae) — Official PyTorch example of a VAE implementation.

## What comes next
- [[Denoising Diffusion Probabilistic Models]] — VAEs learn a latent representation, while DDPMs learn to reverse a diffusion process to generate data.
- [[Generative Adversarial Networks]] — GANs offer an alternative approach to generative modeling, using a discriminator to guide the generator's learning process.

## Connected topics
- [Diffusion Models](./diffusion-models.md) — Diffusion models are another type of generative model, similar to VAEs.
- [Self-Supervised Learning](../03-representation-learning/self-supervised-learning.md) — VAEs are often used for self-supervised learning to learn useful data representations.
- [Score Matching](./score-matching.md) — Score matching is related to VAEs as a method for training generative models.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Backpropagation is used to train the neural networks within a VAE.
- [Optimization](../04-neural-networks-dl/optimization.md) — Optimization algorithms are used to train the parameters of a VAE.
- [Contrastive Learning](../03-representation-learning/contrastive-learning.md) — Contrastive learning is related to VAEs in learning useful data representations.


## Further reading
- Kingma & Welling (2013) — "Auto-Encoding Variational Bayes" — [URL NOT VERIFIED] — The original paper introducing VAEs and the reparameterization trick.
- Rui Chen et al. (2024) — "Dora: Sampling and Benchmarking for 3D Shape Variational Auto-Encoders" — [https://doi.org/10.48550/arXiv.2412.17808] — Presents Dora-VAE, a novel approach that enhances VAE reconstruction through a sharp edge sampling strategy and a dual cross-attention mechanism for 3D shape generation.
- Li et al. (2025) — "VIVAT: Virtuous Improving VAE Training through Artifact Mitigation" — [https://arxiv.org/html/2506.07863v1] — Introduces VIVAT, a systematic approach to mitigating common artifacts in KL-VAE training without requiring radical architectural changes.
- Jian et al. (2025) — "Towards Foundation Auto-Encoders for Time-Series Anomaly Detection" — [https://arxiv.org/abs/2507.01875v1] — Introduces FAE (Foundation Auto-Encoders), a foundation generative-AI model for anomaly detection in time-series data, based on Variational Auto-Encoders (VAEs).
```