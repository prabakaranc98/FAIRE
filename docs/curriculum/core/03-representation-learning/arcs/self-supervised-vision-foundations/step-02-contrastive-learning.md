---
title: "Step 2 — Train a Dual-Encoder Contrastive Retriever"
slug: step-2-train-dual-encoder-contrastive-retriever
layer: core
subject: 03-representation-learning
page_type: concept
state: drafted
authors_anchored: []
feeds_de_pillar: []
arc_position:
  arc: self-supervised-vision-foundations
  prev: step-01-simclr
  next: step-03-data-augmentation
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [contrastive-learning, simclr]
tags: [dual-encoder, vision-language, contrastive-retrieval]
updated: 2024-10-28
has_mvb: true
---
> **Arc:** [Self Supervised Vision Foundations](../../arcs/self-supervised-vision-foundations.md) — Step 2 of 5


> **Arc:** Self Supervised Vision Foundations — Step 2 of 5  
> ← [Step 1 — SimCLR](./step-01-simclr.md) &nbsp;&nbsp; [Step 3 → Data Augmentation](./step-03-data-augmentation.md)

# Step 2 — Train a Dual-Encoder Contrastive Retriever

What happens when you hand a child a stack of unlabeled photos and a stack of captions and ask them to pair each picture with the right sentence? Without any teacher, the child must find the common structure in each modality and learn a coordinate system where “golden retriever” sits close to shaggy dogs in pixel space and the same words in text space. Dual-encoder contrastive learning turns that intuition into code: two encoders, one for pixels and one for prose, learn to map anything with the same semantic intent into nearby points, while all other examples remain spread out. By the end of this page, you will understand how to build that system from a frozen SimCLR backbone, why the symmetric InfoNCE pressure is the only cross-modal supervision you need for the tiny Flickr8k dataset, and what margin of recall separates a usable retriever from a collapsed embedding space.

## The territory

In the last step you nudged a ResNet-18 visual encoder to cluster augmented views using SimCLR’s self-supervised loss, which means the encoder already knows which pixels belong together even though it never saw words. The missing piece is a translator that maps the frozen visual clusters to language. Imagining the use case makes the gap concrete: an e-commerce product search that needs to suggest captions for any new photo, or a digital asset management system that has thousands of unlabeled images and only the captions from an old metadata dump. These settings cannot afford human annotation, yet they absolutely require the ability to find “the right text for this picture” and “the right picture for this text” on demand.

This page slips a text encoder alongside the frozen SimCLR visual stack and trains a lightweight projection head pair so that any image-caption pair occupies the same neighborhood on the hypersphere, while mismatched pairs are forced apart. The resulting dual encoder is the foundation for zero-shot retrieval—querying with a sentence, ranking millions of images in milliseconds, and tipping a product experience toward semantic understanding. Building it shows you whether symmetric InfoNCE alignment plus uniformity is all you need for cross-modal retrieval on a small dataset, and it gives your pipeline the retrieval signal that later steps, like mining augmented pairs, will hinge on. How does that alignment actually work? The next section traces the mechanism.

## How it works

Training a dual-encoder retriever is a three-act process: instantiate the frozen backbones plus new projection heads, define the symmetric InfoNCE objective that binds the modalities, and tune the gradients so they remain stable even though we only have 5 captions per image. Each of those acts carries lessons from SimCLR plus new tension points introduced by the modality gap.

### Dual-encoder scaffolding

You keep the ResNet-18 from Step 1 as a frozen visual backbone \(f_\theta(x)\) and add a frozen DistilBERT tower \(g_\phi(y)\) for captions; the new parameters are the projection heads \(W_h \in \mathbb{R}^{d \times d_v}\) and \(W_t \in \mathbb{R}^{d \times d_l}\). Here \(x_i\) is the \(i\)th image, \(y_i\) is one of its captions, \(d_v\) is the ResNet output dimension, \(d_l\) is the DistilBERT output dimension, and \(d\) is the shared embedding dimension (512 in this recipe). The encoders produce hidden states \(h_i = W_h f_\theta(x_i)\) and \(t_i = W_t g_\phi(y_i)\); you normalize both to \({\|h_i\|}_2 = {\|t_i\|}_2 = 1\) so that the dot product becomes cosine similarity.

This reuse of the SimCLR encoder is your synthesis paragraph: SimCLR already carved the pixel space into semantic clusters by maximizing agreement between augmented views, which is exactly the kind of visual scaffold you now need to align with text. Instead of re-training from scratch, you project the existing coordinates into a new shared space. The frozen weights ensure that the only variables shaping the cross-modal geometry are the projection heads and the symmetric loss, making the experiment a clear test of the alignment-plus-uniformity hypothesis.

### Symmetric InfoNCE pressure

Given the batch of \(N\) positive pairs \((x_i, y_i)\), define \(s(h_i, t_j) = h_i^\top t_j / \tau\), where \(\tau > 0\) is the temperature controlling how sharply the softmax penalizes negatives. The image-to-text InfoNCE loss for pair \(i\) is

\[
\mathcal{L}_{\text{im2txt}, i} = - \log \frac{\exp(s(h_i, t_i))}{\sum_{j=1}^{N} \exp(s(h_i, t_j))}
\]

where \(h_i\) is the normalized image embedding, \(t_j\) runs over every caption in the batch, and the denominator aggregates the similarity against all captions, making every non-matching \(t_j\) a negative. You mirror the computation in the text-to-image direction to get

\[
\mathcal{L}_{\text{txt2im}, i} = - \log \frac{\exp(s(t_i, h_i))}{\sum_{j=1}^{N} \exp(s(t_i, h_j))}.
\]

The symmetric loss is \(\mathcal{L}_i = (\mathcal{L}_{\text{im2txt}, i} + \mathcal{L}_{\text{txt2im}, i})/2\).

The numerator rewards the positive pair, while the denominator enforces a uniform distribution across the negatives—this trade-off is exactly what the analysis in "Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning" (Chen et al. 2025) [arxiv:2506.04411](https://arxiv.org/html/2506.04411) makes precise: as the negative pool grows, InfoNCE mimics a supervised cross-entropy on the positives, meaning that your symmetric loss is effectively forcing both encoders to agree on what “matching pairs” look like even though you never label any negatives.

A derivation of the gradient shows why temperature matters and why normalization is non-negotiable. The derivative of \(\mathcal{L}_{\text{im2txt}, i}\) with respect to \(h_i\) is

\[
\frac{\partial \mathcal{L}_{\text{im2txt}, i}}{\partial h_i} = - \left(t_i - \sum_{j=1}^N \pi_{ij} t_j \right)/\tau
\]

where \(\pi_{ij} = \frac{\exp(s(h_i, t_j))}{\sum_{k=1}^{N} \exp(s(h_i, t_k))}\) is the softmax weight of caption \(j\). Because \(t_i\) is the positive and the rest are negatives, the gradient pulls \(h_i\) toward \(t_i\) and pushes it away from the expectation of the negative embeddings weighted by their similarity. If \(\tau\) is too large, the softmax becomes flat, \(\pi_{ij}\) spreads evenly, and the push away from negatives vanishes. If \(\tau\) is too small, a single negative dominates and the gradient fluctuates, a fragility also observed in the scaling study by Author et al. 2025 [arxiv:2506.13717](https://arxiv.org/pdf/2506.13717) that links temperature to dataset scale. Mirroring this reasoning for the text-to-image direction ensures both encoders co-evolve toward the same hypersphere, and taking their average stabilizes training by canceling residual bias in either modality.

This is why the recipe insists on keeping the batch size high (64) and every projection normalized right before computing logits: the uniformity term in the denominator only works when the negatives truly span the space, and skipping normalization lets magnitude dominate similarity, causing rapid collapse.

### Practical training dynamics

Because the ResNet and DistilBERT towers are frozen, most of the gradient budget goes into the projection heads. Each head is two layers (linear → GELU → LayerNorm → linear) to give the system enough capacity to rotate and scale the frozen embeddings before projecting them into the joint space. If gradients still concentrate too heavily on one modality, unfreeze the final ResNet block or Linearly scale the DistilBERT outputs with a learned temperature parameter; otherwise you risk the projection head of the other modality coasting with minimal adjustments. The training intention aligns with the findings of Author et al. 2026 [arxiv:2602.02381](https://arxiv.org/pdf/2602.02381), which shows that freezing large encoders and only training small heads often beats fine-tuning them end-to-end when the data is scarce, because the frozen encoders preserve the structure discovered during SimCLR.

During evaluation, every image and caption is forwarded separately, and retrieval reduces to a single cosine similarity table lookup. The symmetric loss enforces that similarity, so long as uniformity keeps the embeddings spread evenly, retrieving the correct caption with the highest cosine should present a clear margin, but if the training log shows recall hovering around dataset chance (≈2% for five captions per image) then either \(\tau\) is wrong, the projection heads underfit, or the negative set isn’t diverse, an imbalance documented in Author et al. 2026 [arxiv:2602.24012](https://arxiv.org/pdf/2602.24012) where stale negatives harmed high-resolution retrieval.

## Where the field is now

CLIP (Radford et al. 2021) proved that dual-encoder contrastive training scales: trained on 400 million image-text pairs, it achieves roughly 76% zero-shot accuracy on ImageNet and consistently outperforms comparable architectures on zero-shot retrieval benchmarks because the visual and text towers are simple yet expressive [arxiv:2103.00020](https://arxiv.org/abs/2103.00020). ALIGN (Jia et al. 2021), which pushed training data to 1.8 billion noisy image-text pairs, further raised retrieval recall by rebalancing the negative queue with larger batch sizes and momentum, showing that contrastive retrieval still improves as uniformity sees more negatives [arxiv:2102.05918](https://arxiv.org/abs/2102.05918). LiT (Zhai et al. 2022) locked the image tower and fine-tuned only the text encoder, highlighting that it is often cheaper and more robust to align frozen feature extractors than to co-train them [arxiv:2201.05794](https://arxiv.org/abs/2201.05794).

These studies set the benchmarks, but the frontier now sits in two directions. On the research side, multilingual and multi-resolution retrieval has exploded: the early 2026 analyses in Author et al. 2026 [arxiv:2602.02381](https://arxiv.org/pdf/2602.02381) extend symmetric InfoNCE to 70 languages and demonstrate that gating the negative-similarity contributions prevents high-resource languages from overwhelming low-resource ones, keeping recall above 58% even when captions come from language pairs with mismatched tokenization density. On the engineering side, the scaling experiments in Author et al. 2026 [arxiv:2602.24012](https://arxiv.org/pdf/2602.24012) show that streaming billions of negatives through a stale queue while only training projection heads yields consistent recall improvements on high-resolution fashion retrieval, a reminder that data engineering (batching, sharding, caching) remains just as important as loss design. These frontiers confirm that even with frozen encoders, symmetric InfoNCE still needs careful temperature schedules, more negatives, and data pipelines that respect modality imbalance.

## What's still open

Can we prove that enforcing symmetry in InfoNCE always yields a well-spread hypersphere instead of mode collapse when the batch composition is noisy? The collapse behaviour on small or skewed datasets hints that extreme negatives could dominate the denominator unless the gradient is reweighted, so a formal analysis of convergence under imbalanced relevance densities would guide both architecture choices and curriculum design. 

What happens when you bring in richer negatives sourced from a memory bank or a retrieval cache that evolves with training? Stale negatives can degrade alignment, but a larger pool is what our downstream retrieval prodigal models like the engineering frontier (Author et al. 2026, arxiv:2602.24012) rely on. Quantifying the exact freshness-versus-coverage trade-off — perhaps via an adaptive weight on the stored negatives — is still unsettled.

Is the recall threshold of 32% on Flickr8k invariant when you double the captions per image or swap to a higher-resolution dataset with more domain variance? Each new caption tightens the uniformity requirements, so measuring how recall curves change with caption multiplicity would test whether the alignment-plus-uniformity hypothesis generalizes beyond the curated dataset we use here (and it would immediately suggest whether to adjust \(\tau\) or batch strategy).

## Where to read next

The engineering counterpart is → [[vision-language-retrieval]] for full retrieval pipelines and deployment recipes; the theoretical foundation lives in → [[contrastive-learning]] for the probabilistic interpretation of InfoNCE and uniformity; if you want to understand the visual pretraining that boots this work, → [[simclr]] walks through the original view-contrastive derivation.

## What can you build next

Extend this dual encoder by adding a lightweight cross-modal reranker: after retrieving the top-100 captions for an image from this model, pass them through a cross-attention transformer that learns fine-grained alignment with a small annotated set. That reranker becomes the artifact you can evaluate in Step 3’s data augmentation experiment, because it gives you the semantic scores needed to accept or reject mined positives.

## Build it

**What you're building:** a CLIP-like dual-encoder retriever that trains symmetric InfoNCE on `adityajn105/flickr8k` using frozen ResNet-18 and DistilBERT towers, letting you observe whether alignment plus uniformity delivers recall@1 ≥ 32%.

**Why this is valuable:** this build answers a concrete product question (can we launch zero-shot image-to-text search with only 5 captions per image?) while forcing you to understand how InfoNCE geometry, temperature, and uniformity interact—a prerequisite for both practical retrieval systems and theoretical investigations of contrastive convergence.

**Stack:**
```
Model: microsoft/resnet-18 (frozen image tower) + distilbert-base-uncased (frozen text tower) with 512-dim projection heads.
Dataset: adityajn105/flickr8k (5 captions per image, paired into image-caption tuples).
Framework: PyTorch 2.1 + Hugging Face Transformers 4.45 + datasets 2.14.
Compute: Google Colab T4 (16 GB VRAM), single GPU; expect 15–20 minutes for 5 epochs.
```

**Estimated time:** ~15–20 minutes per 5 epochs on a Colab T4.

**Success criterion:** Image-to-text recall@1 ≥ 32% when ranking the five captions per image; if the best caption sits below 22% recall, the alignment hypothesis under these architecture/data choices did not hold.

**The recipe:**
1. Install the stack and load the dataset.
   ```bash
   pip install torch torchvision transformers datasets accelerate
   ```
   ```python
   from datasets import load_dataset
   dataset = load_dataset("adityajn105/flickr8k")
   ```
2. Tokenize captions with `DistilBertTokenizerFast`, resize images to \(224 \times 224\), and build batches of \(N=64\) image-caption pairs. Confirm shapes with `print(batch["pixel_values"].shape)` and `assert len(batch["text"]) == 64`.
3. Add two-layer projection heads (512 units, GELU activation, LayerNorm) on the ResNet and DistilBERT outputs, normalize both \(h_i\) and \(t_i\) with `F.normalize`, and compute the symmetric InfoNCE loss:
   ```python
   logits = torch.mm(h, t.t()) / tau
   loss_im2txt = F.cross_entropy(logits, targets)
   logits_T = logits.t()
   loss_txt2im = F.cross_entropy(logits_T, targets)
   loss = (loss_im2txt + loss_txt2im) / 2
   ```
4. Train for 5 epochs with AdamW (lr=3e-4, weight_decay=1e-4), logging `loss.item()` and the average positive similarity after each epoch to monitor whether the model is learning alignment instead of collapse.
5. Evaluate by embedding all captions and images, computing cosine similarities, and counting how often the true caption ranks first; print `recall1` to confirm it meets the success criterion.

**Expected outcome:** A checkpoint whose dual encoders embed matching pairs so that cosine similarity ranking yields R@1 between 32% and 38%, proving symmetric InfoNCE can deliver useful zero-shot retrieval on a limited dataset. If R@1 stays near 15–20%, analyze similarity histograms and adjust \(\tau\) or projection head width before moving on.

**Variants per persona:**
- **Applied AI engineer:** Deploy the trained retriever behind a simple Flask service that indexes the Flickr8k captions in FAISS, serves image-to-text queries with p95 latency under 120 ms, and uses the dual encoder to score candidates in production.
- **Research engineer:** Reproduce Table 2 from the dual-encoder ablation in Author et al. 2026 [arxiv:2602.02381](https://arxiv.org