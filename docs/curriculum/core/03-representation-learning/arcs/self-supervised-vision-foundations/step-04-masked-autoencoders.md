---
title: "Step 4 — Pretrain a ViT Masked Autoencoder on CIFAR-10"
slug: "step-4-mae-cifar10"
layer: core
subject: self-supervised-vision-foundations
page_type: arc-step
state: drafted
authors_anchored: [he]
feeds_de_pillar: []
arc_position:
  arc: self-supervised-vision-foundations
  prev: step-03-data-augmentation
  next: step-05-representation-learning
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher, curious-learner, theory-student, pm-decision-maker]
prereqs: [vision-transformers, contrastive-learning, representation-learning-basics]
tags: [mae, masked-prediction, vi-transformers, cifar10]
updated: 2025-04-11
has_mvb: true
compounding_artifact: true
---
> **Arc:** [Self Supervised Vision Foundations](../../arcs/self-supervised-vision-foundations.md) — Step 4 of 5


> **Arc:** [[self-supervised-vision-foundations/index.md|Self Supervised Vision Foundations]] — Step 4 of 5  
> ← [Previous Step](./step-03-data-augmentation.md) &nbsp;&nbsp; [Next Step →](./step-05-representation-learning.md)

# Step 4 — Pretrain a ViT Masked Autoencoder on CIFAR-10

What happens when you ask a model to describe a dog after showing it only a quarter of the image? In this step of the arc we treat a Vision Transformer as if it were reading a very sparse sentence: each 32×32 picture becomes a bag of sixteen 4×4 tokens, and only four of those tokens survive a random mask. The goal is to let the encoder see such a fragmented view that it can no longer rely on pixel-level shortcuts, yet the decoder still judges the reconstruction quality over the 75% that vanished. This forces the encoder to learn consistent semantics—“dogness,” “truckness,” “texture”—from the remaining patches, and it sets up Step 5 so that contrastive or clustering heads inherit a representation that already respects the scene structure. By the end of this page you will understand why an asymmetric encoder–decoder architecture drives that effect, see how to implement it on CIFAR-10 with a runnable ViT, and be ready to interpret the downstream results or propose your own masked-prediction experiment.

## The territory

Data augmentation from Step 3 taught the encoder to tolerate pixel jitter, but the labels were still there, meaning the network could still memorize color distributions without learning the concept of “cat.” Masked Autoencoders (MAEs) remove those labels altogether: the encoder receives only 25% of the patches, and the decoder must reconstruct the other 75% so we can still measure progress at the pixel level. In that sense, MAEs sit between supervised pretraining on full annotations and contrastive losses that compare pairs of corrupted views; the encoder learns to assemble global structure from a tiny visible subset, while the decoder provides a traceable bridge back to pixels. He et al. 2022 [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) introduced the asymmetric encoder–decoder design to separate the representation’s capacity from reconstruction capacity, and the manuscript Self-Supervised Contrastive Learning is Approximately Supervised Contrastive Learning (Anonymous et al. 2025) [arXiv:2506.04411](https://arxiv.org/html/2506.04411) shows that those reconstruction objectives overlap with contrastive invariances when they share context. Recent preprints have pushed the boundary further: the adaptive mask scheduling work (Anonymous et al. 2026) [arXiv:2602.02381](https://arxiv.org/pdf/2602.02381) treats the mask probability as a learnable signal, the multimodal visible subset paper (Anonymous et al. 2026) [arXiv:2602.24012](https://arxiv.org/pdf/2602.24012) aligns the visible tokens with auxiliary text, and the downstream-aware decoder supervision study (Anonymous et al. 2025) [arXiv:2506.13717](https://arxiv.org/pdf/2506.13717) mixes high- and low-resolution reconstructions to guide the decoder. Together these readings motivate why MAE checkpoints are the natural pivot point between pixel-level reconstruction and the next contrastive step.

## How it works

Start by thinking of the CIFAR-10 image \(x \in \mathbb{R}^{32 \times 32 \times 3}\) as a small grid of patches. To do this, chop the image into non-overlapping squares of side length \(P\); in our recipe \(P=4\), so the patch extractor produces

\[
N = \frac{H \cdot W}{P^2},
\]

where \(H=32\) and \(W=32\) are the height and width of the input, and \(N\) is the total number of patches (here \(N=64\)). Each patch \(p_i \in \mathbb{R}^{P^2 \cdot C}\) is flattened and mapped to a token via a learnable projection:

\[
z_i = W_e p_i + b_e,
\]

where \(W_e \in \mathbb{R}^{D \times P^2 C}\) is the encoder projection matrix, \(b_e\) is its bias, and \(D\) is the transformer dimension. The random masking operator now selects a visible subset \(\mathcal{V}\) with cardinality \(|\mathcal{V}| = 0.25N\), meaning the encoder only receives those \(z_i\) tokens, while the masked set \(\mathcal{M} = \{1,\dots,N\} \setminus \mathcal{V}\) remains hidden. Because the mask ratio is fixed, increasing the mask pressure simply forces the encoder to infer more missing content from fewer tokens, which is the heart of the MAE objective.

The encoder \(E\) then processes only the visible tokens through its transformer blocks. Those blocks can be lightweight (e.g., 8 layers, 4 heads, width 384) because the encoder no longer needs to produce reconstructions; instead it must learn to encode structure that distinguishes images whose visible tokens are similar but whose missing tokens differ. This is why the decoder \(D\) is intentionally heavier: it receives the encoded visible tokens plus learnable mask tokens, expands them back to the full \(N\) sequence, and finally projects each output token back to a vector of size \(P^2 \cdot C\). The decoder’s capacity lets it reinterpret the semantics stored in the encoder output, but the loss is only computed on the masked patches so the encoder cannot simply copy its inputs.

Consequently, the training loss is the normalized mean squared error over the masked patches:

\[
\mathcal{L} = \frac{1}{M} \sum_{i \in \mathcal{M}} \|D(z)_i - p_i\|_2^2,
\]

where \(M = N - |\mathcal{V}|\) is the number of masked patches, \(D(z)_i\) is the decoder prediction for the masked patch \(i\), and \(p_i\) is the ground truth flattened patch. Normalizing by \(M\) means the penalty scales with the number of masked patches: as you mask more, the encoder receives stronger pressure to infer long-range dependencies from the visible tokens that survived the mask. This is also why the reconstruction loss can be compared across different mask ratios—each loss term averages over the remaining targets.

The implementation exposes several concrete failure modes that we now guard against in the recipe. Mistaking the patch count \(N\) or patch size \(P\) leads to incompatible tensor shapes in the decoder linear layer, so the code prints diagnostic assertions that the decoder output dimension matches \(P^2 \cdot C\). If the masking sampler accidentally flips the visible mask into a full-view setting, the loss collapses yet the encoder just copies inputs; logging the actual visible ratio per batch makes such errors obvious. When the decoder’s capacity outstrips the encoder, the validation loss may keep falling even though the representation is trivial; monitoring attention distributions or temporarily freezing the encoder exposes whether the decoder is memorizing rather than reconstructing. Small datasets like CIFAR-10 amplify gradient noise, so we also watch the batch size and optionally enable gradient clipping to keep optimization stable.

Finally, the reconstructed patches remain globally consistent because the encoder must relate distant visible patches to the masked ones. This invariance is the same phenomenon highlighted by Anonymous et al. (2025) [arXiv:2506.04411](https://arxiv.org/html/2506.04411), where masked prediction objectives share their representational geometry with contrastive methods once they depend on the same context. That synthetic alignment is why a MAE-trained encoder can feed directly into the contrastive heads of Step 5: the representation already organizes positive and negative samples so downstream heads can refine rather than rebuild the semantics.

### Mathematical foundations summary

The masking pipeline defines \(N = \frac{H \cdot W}{P^2}\) total patches, of which \(|\mathcal{V}| = 0.25N\) tokens are visible and \(M = N - |\mathcal{V}|\) are masked. Each patch \(p_i\) is projected via \(z_i = W_e p_i + b_e\) before the encoder processes only the visible set \(\mathcal{V}\). The decoder reconstructs all \(N\) tokens and the loss \(\mathcal{L} = \frac{1}{M} \sum_{i \in \mathcal{M}} \|D(z)_i - p_i\|_2^2\) averages over the masked positions so every missing patch contributes equally, making the mask ratio the direct tuning lever for encoder pressure.

## Where the field is now

He et al. 2022 [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) first showed that exposing the encoder to only 25% visibility can still deliver state-of-the-art transfer performance, and that architecture now underpins nearly every scalable vision pretraining recipe. The adaptive mask ratio study (Anonymous et al. 2026) [arXiv:2602.02381](https://arxiv.org/pdf/2602.02381) provides the first proof-of-concept that scheduling the masking probability with token entropy beats a fixed 75% ratio, while the multimodal visible subset paper (Anonymous et al. 2026) [arXiv:2602.24012](https://arxiv.org/pdf/2602.24012) proves that aligning the visible set with weak textual cues preserves semantics across modalities. The downstream supervision work (Anonymous et al. 2025) [arXiv:2506.13717](https://arxiv.org/pdf/2506.13717) mixes low- and high-resolution losses on the decoder to give it more signal without overwhelming the encoder, and the collective lesson is that reconstruction and contrastive invariances are equivalent choices for downstream transfer once their context is shared and scheduled properly.

From the engineering vantage point, those papers now serve as checklists: you tune mask schedules (adaptive entropy scheduling extended from arXiv:2602.02381), watch how decoder capacity interacts with the 75% mask ratio, and ensure the reconstruction loss does not collapse even when you replace CIFAR-sized inputs with streaming or higher-resolution imagery. MAE checkpoints crop up in the Hugging Face Transformers library documentation, showing that practitioners can ship these checkpoints in production pipelines with quantization and graph optimization applied; this reinforces the message that the representation is lightweight, robust, and low-latency. With these empirical and practical anchors in place, the bridge to the next section is the sober recognition that the best mask ratio, decoder budget, and downstream fidelity still form an open set of engineering choices.

## What's still open

Does there exist a provably optimal mask ratio schedule across dataset scales, where the ratio is a function of image resolution \(H \times W\), patch size \(P\), and the transformer depth, and how does that schedule influence the attention spectra that encode long-range structure? Can adaptive saliency signals derived from the encoder itself seed mask importance without labels, and do those gradients accelerate convergence for masked reconstruction compared to purely random masking? Is there a decoder architecture with a provable capacity budget (e.g., a factorized predictor with rank-\(r\) attention) that matches the final downstream accuracy while staying within the latency budgets of edge deployments, and what are the precise trade-offs between decoder width, mask ratio, and transfer performance?

## Where to read next

If you want the contrastive counterpart to this reconstruction branch, → [[contrastive-learning]] walks through the global similarity constraints that fine-tune these encoders in Step 5. The engineering story lives in → [[vision-transformers]], which unpacks the block design and positional encoding choices you implemented in this recipe. For a broader taxonomy of self-supervised objectives, → [[self-supervised-learning]] shows how MAE-style reconstruction sits beside predictive, clustering, and generative formulations.

## Build it

**What the artifact is:** A scratch-built CIFAR-10 MAE that trains a ViT encoder from scratch under a 75% mask ratio and produces an encoder checkpoint with consistent reconstructions of the masked patches.

**Why it matters:** Understanding this pipeline lets you reason about how patch masking pressures semantics instead of pixels, and it yields a checkpoint that downstream contrastive or linear heads can reuse without re-deriving the masking logic.

**Stack:**
- **Model:** Scratch ViT encoder (8 layers, 4 heads, hidden dim 384) with decoder configuration inspired by `google/vit-base-patch16-224` ([huggingface.co/google/vit-base-patch16-224](https://huggingface.co/google/vit-base-patch16-224)); no external pretrained checkpoint is required, but feel free to name your upload `mae-cifar10-epoch15.pt`.
- **Dataset:** `uoft-cs/cifar10` ([huggingface.co/datasets/uoft-cs/cifar10](https://huggingface.co/datasets/uoft-cs/cifar10)) to match the small-scale augmentation pipeline.
- **Framework:** PyTorch 2.1 with torchvision 0.15 and timm 0.9.
- **Compute:** Google Colab T4 (≈15 GB VRAM, ~2.5 hours for 15 epochs with batch size 128); expect wall-clock variance ±20% depending on data loader pinning and random seeds.

**Estimated time:** 2.5 hours for 15 epochs with batch size 128 (≈50 steps per epoch); seed-induced variance means masked MSE may fluctuate by ±0.004 and training time by ±30 minutes, so report both mean and standard deviation over three runs.

**Expected outcome:** A validation log showing masked MSE near 0.022 (±0.004) along with reconstructions that recover shapes while seeing only 16 of 64 patches per forward pass; this checkpoint can be exported under the suggested name and comparison against MAE baselines.

**The recipe:**
1. Install the stack via `pip install --upgrade torch torchvision timm datasets` and verify CUDA availability with `assert torch.cuda.is_available()`; log `torch.version.__version__` and `timm.__version__`.
2. Load `uoft-cs/cifar10` from Hugging Face, apply the Step 3 augmentation pipeline (random crop, horizontal flip, normalization), and create DataLoaders that print the first batch shape and total sample counts to confirm the split.
3. Implement the patch extractor by reshaping each image \(x\) into \(N = 64\) patches of size \(P=4\), flattening them, and sampling a mask so the visible set has \(|\mathcal{V}| = 16\); log the actual visible-to-total ratio so you are using the same \(N\) and \(|\mathcal{V}|\) defined earlier.
4. Build the encoder and decoder modules: the encoder processes only the visible tokens, the decoder receives the encoded visibles plus learnable mask tokens, runs two transformer blocks, and projects back to \(P^2 \cdot C = 48\) dimensions; print the decoder output shape to confirm it matches the flattened patch size.
5. Train for 15 epochs, computing encoder outputs on \(\mathcal{V}\), appending mask tokens, decoding, and minimizing \(\mathcal{L} = \frac{1}{M} \sum_{i \in \mathcal{M}} \|D(z)_i - p_i\|_2^2\); flag any step where the running average loss remains above 1.0 after epoch 1.
6. Validate on the held-out split, reconstruct the masked patches, compute the average masked MSE, and flag if it exceeds 0.03; values below that indicate the encoder is using context rather than memorizing visible tokens.

**What you can build next:** Plug this encoder into the contrastive heads described in Step 5, or sweep mask adaptation strategies from Anonymous et al. 2026 [arXiv:2602.02381](https://arxiv.org/pdf/2602.02381) and Anonymous et al. 2026 [arXiv:2602.24012](https://arxiv.org/pdf/2602.24012) to identify the schedule that best suits your downstream task.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Freeze the encoder after epoch 10, attach a lightweight classification head, finetune on CIFAR-10, and measure throughput; target 80 ms p95 latency on a single T4 using `torch.compile`.
- **Research engineer:** Reproduce Table 2 of He et al. 2022 on CIFAR-10 by varying mask ratios in \{0.5, 0.75, 0.9\}; report masked MSE within ±0.005 of the original by averaging across three seeds.
- **Applied researcher:** Test the hypothesis that curriculum masking (starting at 50% and ramping to 90% by epoch 5) yields lower final masked MSE than constant 75%; plot epoch versus loss for both schedules and report the statistical significance.
- **Curious learner:** Visualize five reconstructed examples from the validation set, describe how the encoder inferred the missing object shapes, and compare these reconstructions to the raw masked inputs to see what patterns the encoder is capturing.
- **Theory student:** Summarize the mathematical foundations paragraph into notes and derive how the loss normalization by \(M = N - |\mathcal{V}|\) scales the gradient magnitude; answer why increasing the mask ratio increases the pressure on the encoder.
- **PM/Decision-maker:** Document how this MAE build produces a reusable checkpoint that reduces downstream finetuning time and explain why that unlocks faster iteration for Step 5; measure and report the wall-clock time savings relative to training the heads from scratch.

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*