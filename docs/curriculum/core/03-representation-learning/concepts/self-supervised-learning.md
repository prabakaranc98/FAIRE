---
title: Self-Supervised Learning  
slug: self-supervised-learning  
layer: core  
subject: 03-representation-learning  
page_type: concept  
state: drafted  
authors_anchored: [oord, raina, chen, he, hinton, grangier]  
feeds_de_pillar: []  
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]  
prereqs: [deep-learning-basics, contrastive-learning, representation-learning-principles]  
tags: [self-supervised, contrastive-learning, simclr, cpc, transfer-learning, augmentations]  
updated: 2025-02-17  
has_mvb: true  
---

# Self-Supervised Learning

Here is a puzzle: a slow-motion clip ends with a glass tipping in the top-left of the frame, and the screen cuts to black. Without any labels telling us "glass" or "spill," the only way to keep watching sensibly is to predict what the physics of that tabletop implies—the collisions, the splashes, the direction of motion. That ability to predict structure, to go from one subset of observations to another plausible continuation, is what modern self-supervised learning (SSL) tries to bottle. By the time you close this page you will know how SSL turns backdrops, videos, and sound into their own teachers, why contrastive InfoNCE objectives replaced brute-force reconstruction for many applied systems, how augmentations and negative sampling determine downstream success, and what it takes to spin up a minimal SimCLR pretraining run on raw unlabeled data so your next downstream task runs with few labels.

## The territory

The practical bottleneck in representation learning is not compute; it is labeled data. Every domain—remote sensing, medical imaging, robotics—spends more on annotation than on GPUs. SSL reroutes that cost by asking the model to predict something hidden in the same unlabeled input it received. Instead of assigning "cat" or "dog," it predicts the rotation applied to an image, whether two slices came from the same video clip, or what future latent state follows a speech segment. This class of pretext tasks sits between unsupervised density modeling and supervised metric learning: the aim is discriminative frontiers in latent space that will transfer without ever modeling the full data distribution explicitly.

The idea leans on the same intuition behind classical regularization techniques such as dropout: Hinton et al. (2012) [arxiv:1206.5538v3] argued that preventing feature co-adaptation forces internal representations to cover more of the input structure, which is exactly what SSL leverages when it masks parts of the input or generates paired views. Another ancestor is Raina et al. (2007) [https://ai.stanford.edu/~hllee/icml07-selftaughtlearning.pdf], who coined "self-taught learning" by demonstrating that features learned from unlabeled, unaligned corpora can bootstrap supervised classifiers with very few labeled examples. Today’s SSL families—contrastive methods such as SimCLR, predictive approaches like Contrastive Predictive Coding (CPC), and masked modeling—differ mainly in what subset of the input they hide and what they ask the rest to predict. Tuning those choices is the territory; the mechanism is how the pretext task is reconstructed into a loss that actually makes the representations usable. How does it actually work?

## How it works

### Pretext tasks as self-generated supervision

Contrastive SSL starts by defining which parts of the input count as "positive" pairs and which count as "negatives." In a SimCLR-style pipeline, a single image becomes a positive pair when it is encoded twice through heavy augmentations such as random crops, color jitter, and Gaussian blur; any other image in the batch becomes a negative. This transforms every raw data point into a mini supervised example: "match these two augmented views, but not these others." Contrastive Predictive Coding (CPC) extended this by using context encodings \(c_t\) to predict future latent codes \(z_{t+k}\). The model never sees labels; the supervision is the mutual information between context and future—which is high only if the encoder has captured semantics rather than surface pixels (Oord et al. 2018) [arxiv:1807.03748].

The formal core of these contrastive losses is InfoNCE, which the models minimize to push positives together and negatives apart. Given a query vector \(q\) and a set of key vectors \(\{k_0, k_1, \dots, k_K\}\) where \(k_0\) is the positive and the rest are negatives, the loss is
\[
\mathcal{L}_{\text{InfoNCE}}(q, \{k_i\}) = -\log \frac{\exp(q \cdot k_0 / \tau)}{\sum_{i=0}^K \exp(q \cdot k_i / \tau)}.
\]
Here \(q\) is the projected representation of one view, \(k_0\) is the projection of the matching view, \(k_i\) for \(i>0\) are negatives, and \(\tau\) is the temperature hyperparameter that controls sharpness of the similarity distribution. Intuitively, the denominator sums over both positives and negatives; minimizing the loss encourages \(q\) to align with \(k_0\) while keeping it orthogonal to the rest, which produces separated clusters in embedding space.

InfoNCE also has an interpretation as a lower bound on mutual information. CPC uses this by predicting the future latent code and showing that maximizing this bound forces the context encoder to capture the components needed to reconstruct later states. Because the loss depends only on inner products between vectors, it scales well in practice—the variants keep the denominator manageable by subsampling negatives, using memory banks, or relying on large batches. The reparametrization also means gradients never depend on the actual pixel values of the negative samples; only their encodings matter, so the method remains agnostic to the original modality.

### Augmentations, negatives, and invariance

Contrastive methods learn invariances by carefully engineering the augmentations that produce positive pairs. In SimCLR (Chen et al. 2020) [arxiv:1902.09229], the authors showed that random crop + resize, color jitter, and Gaussian blur produce enough variety that a 100-layer ResNet trained with InfoNCE can beat supervised learning when downstream evaluated via a linear probe. This owes to a key insight: the model is not being asked to reconstruct pixels but to be stable under transformations that preserve semantics. The question, then, is which transformations to pick. Too weak, and the positives remain trivially similar; too strong, and the model cannot close the gap between views. Practical recipes therefore tune probabilities for each augmentation, use synchronized random seeds per batch, and anneal the temperature \(\tau\) to encourage tighter clusters as training converges.

Negatives also need care. If negatives come only from the current mini-batch, the batch size must be very large (SimCLR used 4096). Smaller batches require tricks such as MoCo’s momentum encoder or the stop-gradient in SimSiam (Chen & He 2020) [arxiv:1901.09005], which prevents collapse without explicit negatives by symmetrizing the loss and stopping gradients on one branch. In practice, for applied SSL you will choose one of these scaffolds: large batches on TPU, momentum queues with moderate batches, or negative-free architectures if hardware limits block large batch training.

### Projection heads and transfer evaluation

Another applied insight is the projection head that maps encoder outputs to the contrastive space. SimCLR introduced a two-layer multilayer perceptron (MLP) head on top of the ResNet encoder and found that using the representation before the projection head for downstream tasks yielded better linear-probe accuracy than the projection-head output. The reason is that the projection head adjusts the geometry of the vector space to satisfy the InfoNCE criterion, while the encoder itself still maintains a more general-purpose manifold. This is why the training pipeline often looks like: input image \(\x\) → encoder \(f_\theta(\x)\) → projection head \(g_\phi(f_\theta(\x))\) → normalized vector for InfoNCE. The downstream classifier sees \(f_\theta(\x)\), which keeps the invariances but without the distortions that the contrastive head introduced for easy separation.

Linear-probe evaluation is the applied standard for assessing transfer. After pretraining, freeze \(f_\theta\) and train a simple logistic regression on a small labeled subset. Success is measured by how few labels you need to reach a target accuracy. Raina et al. 2007 originally showed that even a small labeled seed can bootstrap performance if the pretrained features generalize. Contrastive SSL tightened this by quantifying data-efficiency gains: e.g., SimCLR’s ResNet-50 pretrained with InfoNCE needed only 1/10th of the labels to match supervised baselines on ImageNet when evaluated via linear probes.

### Practical failures and fixes

Applied SSL still has failure modes. Without careful augmentation scheduling, the model learns trivial shortcuts—matching low-level textures rather than semantics. Here the fix is to monitor both the contrastive loss and projector norms; if the loss collapses toward zero while validation accuracy on downstream tasks stalls, the augmentation policy or temperature likely needs tuning. Another failure is modality drift when pretraining on one domain (e.g., web photos) and fine-tuning on another (e.g., satellite imagery). This calls for domain-adaptive augmentations or distillation from domain-specific teacher networks.

Batch normalization interacts subtly with InfoNCE. Because the loss compares one example to many, statistics aggregated over the batch affect both positives and negatives. As a consequence, multi-GPU training must use synchronized batch normalization or group normalization to keep the embedding distributions stable. On the implementation side, building your own InfoNCE right now—writing the loss, maintaining the queue, computing the logistic probe—lets you see these details in debug prints. That is exactly what the Minimum Valuable Build section will walk through.

## Where the field is now

Contrastive SSL’s research frontier is still the question of how to make the pretraining objective aware of downstream tasks. DINOv2 (R. Baraldi et al. 2023) [arxiv:2304.02624] updates the SimCLR architecture with Vision Transformer backbones, a teacher-student scheduling mechanism, and centering normalization to stabilize large-scale training. Its release blurred the line between SSL and foundation models: the pretrained embeddings serve as the basis for search at Meta’s scale and demonstrate that SSL can now replace supervised pretraining in many applications. The paper reports 86.2% top1 accuracy on ImageNet-1K with a frozen linear probe, showing that SSL features are now competitive with the best supervised counterparts while remaining label-free during pretraining.

From an engineering perspective, Meta’s DINOv2 infrastructure (ai.meta.com/research/dinov2) illustrates how SSL operates at production scale: a multi-stage data pipeline ingests billions of image crops, feeds them through a ViT backbone with synchronized update steps, and exposes the embeddings via an internal similarity search service. They report that the embeddings cut annotation cost and power usage in downstream systems where real-time labeling was infeasible. This kind of deployment highlights that SSL is not a lab curiosity but the core of at least one billion-parameter inference service with latency SLAs—proof that the representations learned via InfoNCE can stand in for supervised embeddings in real products.

Another research frontier is video and multi-modal SSL. Recent work has extended CPC to compute mutual information between visual and audio streams or between frames and future motion. These extensions still rely on InfoNCE but push it into dense spatiotemporal alignment, opening up domains where static augmentations alone no longer suffice.

## What's still open

Can we derive a theoretical criterion that maps a given pretext task to the downstream generalization error it promises, so that practitioners can pick augmentations or prediction targets without empirical grid search?  
Is it possible to identify a universal set of negative samples—or a scheduling policy over them—that guarantees representation diversity without requiring massive batches or complex queues?  
Does there exist an algorithm that chooses between reconstruction-based, contrastive, and masked losses for every new domain automatically, rather than forcing the researcher to test each approach separately?  
Can we quantify when the projection head is unnecessary and the encoder itself is already aligned, so that the additional network becomes a provable form of regularization rather than an ad-hoc trick?

## Where to read next

If you want the probabilistic foundation, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) shows how modeling gradients of the log density links to InfoNCE’s mutual-information bound. The engineering counterpart is → [[contrastive-learning]] which dives into the sampling, batching, and augmentation details needed to scale these losses in production. For the next arc of work, → [SimCLR](simclr.md) traces how the SimCLR family grew into DINOv2 and other large-scale contrastive pipelines.

## Build it

This build proves that even a single GPU can exploit InfoNCE and augmentations to pretrain a transferable encoder: you will implement SimCLR from scratch, observing how the temperature, projection head, and unlabeled batch all influence downstream accuracy.

**What you're building:** A PyTorch SimCLR pretraining run on the unlabeled split of STL-10 that produces a linear-probe-ready encoder.  
**Why this is valuable:** Because no labels are used during pretraining, every success metric exposes how InfoNCE and augmentations alone carve semantic clusters, mirroring what happens in large-scale SSL deployments.  
**Stack:**
- **Model:** `facebook/resnet18` (HuggingFace; downloads ~28M)  
- **Dataset:** `stanford/stl10` (unlabeled split; ~100k images)  
- **Framework:** PyTorch 2.1 + `torchvision` 0.15  
- **Compute:** Free Colab (T4, 16GB VRAM, ~3 hours training for 100 epochs with batch size 256)

**The recipe:**
1. Install the stack: `pip install torch==2.1.0 torchvision==0.15.0 timm matplotlib`. Import `torch`, `torchvision`, and `tqdm`.
2. Data: load STL-10’s unlabeled split, apply two random augmentations per image (random resized crop, color jitter, horizontal flip, Gaussian blur, random grayscale), and normalize using STL-10’s mean/std. Build a `DataLoader` with batch size 256 and `drop_last=True`.
3. Train/fine-tune: encode each augmented view with `resnet18`, add a 2-layer projection head (2048→512) with ReLU and final L2 normalization, and compute InfoNCE loss where one augmented view is the query, the other the positive key, and the rest of the batch are negatives. Use temperature \(\tau=0.07\), SGD optimizer with momentum 0.9, weight decay 1e-4, and an initial learning rate of 0.5 with cosine annealing. Expect the loss to stabilize around 2.5 after ~50 epochs.
4. Evaluate: freeze the encoder, attach a linear probe (single fully connected layer) trained on 5% of STL-10 labels. Measure top-1 accuracy; a successful run exceeds 70% on the test set.
5. What you now have: a checkpointed encoder whose representations can be reused for any STL-10 downstream task, plus logs showing how InfoNCE, augmentations, and projection-head choices affect the loss curve.

**Expected outcome:** A `resnet18` encoder checkpoint that achieves ≥70% linear-probe accuracy on STL-10 while requiring zero labels during pretraining.

- **CS student:** Reduce batch size to 128 and still hit ≥68% accuracy by adding MoCo-style momentum updates to the key encoder; run the training in a free Colab session with one epoch checking script and a local validation probe.
- **Applied engineer:** After training, quantize the encoder to int8 (use torch.quantization), deploy it as a TorchServe endpoint, and measure p50 latency < 20ms on an NVIDIA T4 while serving the linear probe.
- **Applied researcher:** Ablate the temperature \(\tau\) by sweeping {0.05, 0.1, 0.2} and report how the linear-probe accuracy changes, testing the hypothesis that lower \(\tau\) tightens clusters but eventually collapses.
- **Frontier researcher:** Use the checkpoint to test the open question from §What's still open: swap the augmentation policy to a domain-specific version (e.g., include rotations for satellite imagery) and quantify whether the downstream generalization aligns with the InfoNCE gradient norms, falsifying the idea that any augmentation leads to transfer.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*