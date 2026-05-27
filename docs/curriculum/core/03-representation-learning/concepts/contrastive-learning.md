---

## Contrastive Learning

Imagine you’re building a search engine that needs to understand images and text *together*. Contrastive learning lets you train a model to recognize that ‘dog’ and ‘golden retriever’ are similar, even if they look different, without needing to be labeled as such. This is a fundamental shift from traditional supervised learning, where you need to painstakingly label millions of images with "dog," "golden retriever," etc. Instead, contrastive learning teaches the model what makes something *itself* by comparing it to similar and dissimilar samples.

## The territory

Contrastive learning is a family of techniques that aim to learn representations by pulling similar samples closer together in embedding space and pushing dissimilar samples further apart. The core idea is to define a similarity metric and then train a model to maximize the similarity between positive pairs (samples belonging to the same class) and minimize the similarity between negative pairs (samples belonging to different classes). The field started with representation learning with contrastive predictive coding (Zhou et al., 2018), which introduced the core concept of contrasting similar and dissimilar samples.  Momentum Contrast (Herscovitch et al., 2019) then provided a simple yet effective contrastive loss function, dramatically improving unsupervised visual representation learning with minimal architectural changes.  More recently, Flow Matching (Lipman et al., 2024) generalized the continuous-time perspective, enabling the training of diffusion models with a single function evaluation, and pushing the boundaries of what’s possible with contrastive learning. The current frontier is scaling these techniques to handle massive datasets and multilingual models, as demonstrated by MetaCLIP 2 (Li et al., 2024).

## How it works

The core of contrastive learning is a loss function that encourages the model to learn representations where similar samples are close and dissimilar samples are far apart. Let’s break down the math:

The key idea is to rewrite the objective as \[ L(\theta) = \mathbb{E}_{x_0, t, \epsilon}\big[\|\epsilon - \epsilon_\theta(x_t, t)\|^2\big] \] where \(x_0\) is a clean sample, \(t\) is a timestep drawn uniformly from \(\{1, \dots, T\}\), \(\epsilon \sim \mathcal{N}(0, I)\) is the noise we added at training time, and \(\epsilon_\theta\) is the network we're learning. The term on the left captures the difference between the predicted noise and the actual noise, while the term on the right makes training tractable because it avoids computing the partition function.

Momentum Contrast (Herscovitch et al., 2019) introduces a temperature parameter, τ, to the loss function:

\[ L = - \sum_i \log( \exp(\frac{s_i}{τ}) / \sum_j \exp(\frac{s_j}{τ}) ) \]

where `s_i` is the similarity score between two samples. The temperature parameter τ controls the sensitivity of the loss to differences in similarity scores – a higher τ makes the loss less sensitive, encouraging the model to be more tolerant of dissimilar samples.

CLIP (Radford et al., 2021) uses a contrastive loss based on the probability of the correct label given an image:

\[ L = - \sum_i \log(P(y_i | x_i)) \]

where `x_i` is the image and `y_i` is the correct label. CLIP’s loss function is based on the probability of the correct label given an image, and it is used to train the model to align image and text embeddings.

## Where the field is now

State-of-the-art models are increasingly leveraging contrastive learning for a variety of tasks. Diffusion models, like Stable Diffusion (Rombach et al., 2022), use contrastive learning to train their denoising networks, achieving impressive image generation quality with minimal labeled data. MetaCLIP 2 (Li et al., 2024) demonstrated the practical application of contrastive learning at scale with a massive, diverse dataset of images and text from the web, addressing the “curse of multilinguality” – a common problem with multilingual models. The current frontier is focused on scaling these techniques to handle even larger datasets and improve their robustness to noise and adversarial attacks.

## What’s still open

Several open questions remain in the field of contrastive learning:

*   **The Curse of Multilinguality:** Why do multilingual models trained with contrastive learning often perform worse than their monolingual counterparts? How can we reliably prevent this degradation without sacrificing the benefits of leveraging diverse data?
*   **Noise Schedule Design:** The noise schedule is a critical component of diffusion models, and finding optimal schedules remains a challenge. How can we design noise schedules that are more robust to variations in data distribution?
*   **Contrastive Learning for Graphs:** How can we adapt contrastive learning to learn representations for graph data, which is increasingly important for tasks like node classification and link prediction?

## Where to read next

*   [DDPM (Ho et al., 2020)](https://arxiv.org/pdf/2006.11239) – implements the discrete training procedure that score matching enables.
*   [Flow Matching (Lipman et al., 2024)](https://arxiv.org/abs/2507.22062v1) – generalizes the continuous-time perspective using arbitrary paths.
*   [MetaCLIP 2 (Li et al., 2024)](https://arxiv.org/abs/2507.22062v1) – demonstrates the practical application of contrastive learning at scale with a massive, diverse dataset of images and text from the web.

## Build it

**What you’re building:** A simple image similarity task using a pre-trained CLIP model and a small dataset of images (e.g., a collection of different breeds of dogs).

**Artifact:** [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) – a CLIP-based diffusion model for generating high-quality images from text prompts.

**Success:** Achieve FID ≤ 20 on the Stanford Dogs dataset with a 4090 GPU in under 24 hours.

**Stack:**
- **Model:** [Stable Diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1)
- **Dataset:** [Stanford Dogs Dataset](https://vision.stanford.edu/ugrad/projects/dogs/)
- **Framework:** PyTorch 2.0.1
- **Compute:** RTX 3060 / M-series (Colab T4 possible)

---

This page is designed to be a starting point for readers to build their own image similarity systems using contrastive learning. The build is achievable with readily available tools and resources, and the success metric provides a clear goal for experimentation.

---