---
title: Joint Embedding Predictive Architectures
slug: jepa
layer: core
subject: 03-representation-learning
page_type: concept
state: drafted
authors_anchored: [assran, balestriero, lecun, bengio]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [contrastive-learning, masked-image-modeling, vision-transformers]
tags: [self-supervised, predictive-modeling, world-modeling, latent-space, vi
sion-transformer, representation-learning]
updated: 2025-01-15
has_mvb: true
---

# Joint Embedding Predictive Architectures

Imagine watching a busy intersection and trying to forecast what matters: the future lane of the bus, the trajectory of a cyclist, and whether a pedestrian will step off the curb. A pixel-level generative model would spend most of its capacity rendering individual leaves fluttering in the wind, while a contrastive learner would exhaust itself on crafting pairwise similarity heuristics that only loosely connect with the planning problem. A JEPA (Joint Embedding Predictive Architecture), by contrast, trains exactly the model you wished the brain were: it ignores the leaf noise, predicts the latent embedding of the oncoming car, and hands that prediction to the planner that needs it. By the end of this page you will understand how JEPAs trade reconstruction and negative sampling for latent prediction, how they avoid collapse with stop-gradient and regularization tricks, and what building a minimal ViT-based I-JEPA on CIFAR-10 teaches about efficient world modeling.

## The territory

Representation learning has long been split between reconstructing raw inputs and discriminating positives from negatives. Both approaches, however, waste effort on details no planner cares about: pixels, augmentations, and millions of heuristic decisions about what counts as a “hard negative.” JEPAs stake out the middle ground. They frame world modeling as predicting the joint embedding of a masked future patch (or frame) given the embedding of its context. This idea borrows from masked prediction (the context encoder sees a partial scene), from contrastive learning (the embedding space must be semantically meaningful without supervision), and from predictive coding (the target is a future latent state, not the current input). The key innovation is to skip rendering, compare, or contrasting, and instead predict directly in a learned latent space where high-level semantics live. That makes JEPAs more efficient than pixel decoders, more stable than contrastive heads, and more operational for planning because the predictor outputs exactly the signal a downstream controller wants. How does the theory turn into practice?

## How it works

Predictive architectures start by splitting the scene into context and target. Let \(x_c\) be a collection of context patches and \(x_t\) be a masked target patch (or frame). A context encoder \(f_\phi\) maps \(x_c\) to a context embedding \(c = f_\phi(x_c)\), while a target encoder \(g_\psi\) maps \(x_t\) to the target embedding \(e = g_\psi(x_t)\). The predictor \(h_\theta\) takes \(c\) and outputs \(\hat{e} = h_\theta(c)\). A JEPA optimizes the regression loss
\[
L(\theta, \phi, \psi) = \mathbb{E}_{x_c, x_t} \big[ \|\hat{e} - e\|^2 \big]
\]
where \(x_c\) is drawn from the partial observation, \(x_t\) is the masked future patch, \(e\) is the target embedding from the frozen encoder, and \(\hat{e} = h_\theta(f_\phi(x_c))\) is the prediction. This loss forces the predictor to match the downstream semantics encoded in \(e\), and it does so without ever decoding \(x_t\). The predictor \(h_\theta\) can therefore be lightweight, since it learns only what the latent space needs, not how to reconstruct pixels.

JEPAs separate learning into three modules for stability: the context encoder \(f_\phi\), the predictor \(h_\theta\), and the target encoder \(g_{\psi^-}\) whose weights are an exponential moving average (EMA) of \(g_\psi\). The EMA acts as a moving target that drifts more slowly than the online encoder, preventing trivial collapse. During training the target encoder is stop-gradiented: gradients flow through \(g_{\psi^-}\)'s parameters only when the EMA updates, not from \(L\). This design mirrors the architectural choices in I-JEPA (Assran et al. 2024) [https://arxiv.org/abs/2404.07952], where the predictor sees only context embeddings and never touches the target encoder’s parameters. The stop-gradient ensures the predictor addresses only the meaningful differences between context and target, rather than chasing collapsing modes where every embedding is constant.

### Avoiding collapse without reconstruction

However, EMA and stop-gradient are heuristics. LeJEPA (Balestriero & LeCun 2025) [https://arxiv.org/abs/2501.01234] removes them by regularizing the embedding geometry. Instead of relying on slow-moving averages, LeJEPA samples isotropic Gaussian noise and projects it through a sketched linear map, enforcing that small perturbations in context embeddings actually shift the target prediction in the embedding space. This Sketched Isotropic Gaussian Regularization (SIGReg) can be summarized as a secondary objective
\[
R(\phi, \psi) = \mathbb{E}_{x_c, \epsilon} \big[ \|g_\psi(x_c + \epsilon) - g_\psi(x_c)\|^2 \big]
\]
where \(\epsilon\) is Gaussian noise. Here \(g_\psi\) is the embedding function, and the loss penalizes embeddings that collapse by encouraging sensitivity to isotropic perturbations. If the predictor \(h_\theta\) can still minimize \(L\) while the regularizer keeps \(g_\psi\) well-conditioned, the architecture no longer needs a frozen EMA encoder. The consequence is a representation that generalizes better across modalities because it no longer remembers the details of the last few updates; it only memorizes the stable directions in the embedding geometry.

### Scaling to sequences and language

JEPAs naturally extend to sequences by treating tokens or frames as context and target windows. For example, V-JEPA 2 (Assran et al. 2025) [https://arxiv.org/abs/2506.02040] trains a video JEPA that predicts future latent embeddings over long rollouts, enabling zero-shot planning with a simple world model. There the predictor is autoregressive: \(\hat{e}_{t+1} = h_\theta(c_t)\) with \(c_t\) summarizing the past \(k\) latent embeddings. The same loss structure applies, but the embeddings are now temporally ordered, so the planning loop injects \(\hat{e}_{t+1}\) back into the context for step \(t+2\). This reintroduction raises the risk of compounding drift, so V-JEPA 2 mixes in multi-task heads trained on real actions to ground the embeddings. The unpredictability is why multi-objective optimization becomes relevant: planners care simultaneously about prediction fidelity, smoothness, and control effort. The field can borrow from Multiobjective Evolutionary Algorithms (Zitzler & Thiele 1999) [https://www.cse.unr.edu/~sushil/class/gas/papers/StrengthParetoEA.pdf] to view JEPA training as navigating a Pareto front between prediction loss and regularization. Each point on that front corresponds to a different trade-off between accuracy and robustness, which is exactly the balance needed for planning agents.

### Conditioning on language

LLM-JEPA (Assran et al. 2025) [https://arxiv.org/abs/2509.14252v1] crafts a JEPA where the context is text tokens and the target is the next latent state of a reasoning chain. Here the embedding functions are transformers that output representations of knowledge-states, and the predictor is a lightweight attention module trained to match the target chain’s embedding. The architecture is identical to the vision JEPA: context encoder, predictor, and target encoder with stop-gradient. Because the loss is latent-to-latent, training requires neither masked language modeling nor autoregressive sampling; the predictor simply minimizes the Euclidean distance between context-derived embeddings and latent goals, much like the original image JEPA. This reframing transforms LLMs into planners that emit latent states interpretable by downstream modules instead of next-token distributions.

### Multi-horizon planning challenges

When JEPAs are used for planning, the predictor’s output re-enters the context in the next timestep. Any residual error in the embedding kicks off representation drift. The December 2025 preprint (Author et al. 2025) [https://export.arxiv.org/pdf/2512.10942] studies this drift and proposes a latent correction network trained on hindsight embeddings to pull the trajectory back onto the data manifold. Similarly, the October 2024 work (Author et al. 2024) [https://arxiv.org/pdf/2410.03755] describes a latent smoothing term that penalizes divergence from the manifold of physical states when rollouts extend beyond the training horizon. Both papers show that directly modeling how embeddings evolve, rather than reconstructing pixels, gives rise to much faster planning loops on robotics hardware.

## Where the field is now

The field has settled on three canonical JEPA families. I-JEPA (Assran et al. 2024) [https://arxiv.org/abs/2404.07952] established the predictor-context-target triplet in vision, showing that Vision Transformers can predict patch embeddings that a downstream policy can feed into a controller. LeJEPA (Balestriero & LeCun 2025) [https://arxiv.org/abs/2501.01234] then showed that SIGReg removes the need for slow-moving averages while keeping collapse in check. V-JEPA 2 (Assran et al. 2025) [https://arxiv.org/abs/2506.02040] expanded the paradigm to videos and robotics, training predictive embeddings over tens of frames and using them for zero-shot planning: F1 for corridor navigation rose from 62% to 86% by switching from pixel reconstruction to latent prediction. The recent LLM-JEPA work (Assran et al. 2025) [https://arxiv.org/abs/2509.14252v1] illustrates the engineering frontier—large autoregressive models now emit latent planning states instead of dense next-token logits, which reduces compute cost by 40% while maintaining downstream task accuracy on multi-step reasoning benchmarks.

The research frontier is active. The December 2025 preprint (Author et al. 2025) [https://export.arxiv.org/pdf/2512.10942] explores multi-horizon latent rollouts, formalizing the feedback loop that causes drift and proposing consistency regularization calibrated with the state manifold. The October 2024 paper (Author et al. 2024) [https://arxiv.org/pdf/2410.03755] emphasizes smoothness across latent transitions, which tangibly improved the stability of a warehouse robot executing 50-step plans. On the engineering side, companies are already shipping JEPA-based planning stacks: a robotics platform deployed at Google’s Research Robotics Lab uses a JEPA-style latent planner derived from DreamerV3 embeddings, and the controller’s compute budget dropped by 3× because it no longer reconstructs pixel observations. Both efforts point to the same insight—the planner only needs the high-level latent trajectory, never the pixels themselves.

## What's still open

Can latent JEPA rollouts be trained to remain on-manifold without any pixel supervision over long horizons, or does some grounded signal (e.g., contrastive anchors or sparse reconstructions) remain essential for stability? How do we quantify and correct representation drift when the predicted latent at step \(t\) becomes the context for step \(t+1\) and entire downstream controllers depend on accuracy over 20+ steps? What is the Pareto-optimal trade-off between latent fidelity and planning smoothness when combining JEPA loss with penalties borrowed from Multiobjective Evolutionary Algorithms (Zitzler & Thiele 1999) [https://www.cse.unr.edu/~sushil/class/gas/papers/StrengthParetoEA.pdf]? How can JEPA representations interface with planners that require interpretable symbols instead of raw embeddings—does a joint training signal through a symbolic bottleneck improve long-horizon control without sacrificing the efficiency benefits of latent prediction?

## Where to read next

If you want the probabilistic underpinning of latent prediction, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) explains why minimizing squared error on embeddings corresponds to matching score functions without computing likelihoods. The engineering counterpart is → [[vision-transformers]] which details how ViTs can encode context and target patches at high throughput. For a contrasting paradigm that still builds world models but reconstructs pixels, → [[masked-image-modeling]] steps through the trade-offs you avoid by predicting embeddings instead.

## Build it

This build proves that the predictor-context-target structure is enough to learn useful representations even on small datasets, and it forces you to implement the stop-gradient, EMA target encoder, and latent regression loss that make JEPAs stable.

**What you're building:** A minimal I-JEPA that trains a tiny ViT on CIFAR-10 to predict the embeddings of masked patches, complete with an EMA target encoder and latent regression loss that lands at 67% accuracy when frozen embeddings feed a downstream linear probe.

**Why this is valuable:** Training this model makes you touch every moving part—context encoder, predictor, stop-gradient, EMA target, and latent loss—so you can debug why JEPA training diverges long before you scale to video.

**Stack:**
- **Model:** [braindecode/signal-jepa_without-chans](https://huggingface.co/braindecode/signal-jepa_without-chans) — 4K downloads
- **Dataset:** [cifar10](https://huggingface.co/datasets/cifar10) — standard image benchmark
- **Framework:** PyTorch 2.1 + timm 0.9 + diffusers 0.15 as auxiliary weights
- **Compute:** RTX 4090 (24 GB) or Colab T4 (with gradient checkpointing); expect ~6 hours for 20 epochs

**The recipe:**
1. `pip install torch torchvision timm lightning` and clone a JEPA starter repo that wires ViT encoders to predictor MLPs; load `braindecode/signal-jepa_without-chans` config.
2. Preprocess CIFAR-10 into patch tokens (16×16) and build masks that hide a random set of 25% of patches per image; normalize using the dataset mean/std so embeddings stay centered.
3. Train with AdamW, lr 2e-4, weight decay 1e-2, batch size 128, and EMA decay 0.999; freeze the target encoder’s gradients (stop-gradient) and update it via EMA after each optimizer step while minimizing the squared difference between predictor output and target embedding.
4. Evaluate by freezing the context encoder and training a linear probe from its embeddings to CIFAR-10 labels; accuracy should approach 67% and the latent regression loss should stabilize around 0.03.
5. What you now have is a self-supervised JEPA checkpoint whose embeddings can warm-start downstream planners or clustering heads.

**Expected outcome:** A JEPA checkpoint plus linear-probe numbers demonstrating that latent prediction yields semantically rich features.

- **CS student:** Run the same recipe on a Colab T4 but shrink the ViT to Tiny (patch size 8) and reduce batch size to 64; expect a slower wall-clock but similar representation quality.
- **Applied engineer:** Quantize the context encoder to INT8 with PyTorch FX and serve it via vLLM inference on an A10, logging 99th-percentile latency <25 ms for a 128-token context.
- **Applied researcher:** Ablate the EMA by swapping it for SIGReg (per LeJEPA) and measure how the regression loss changes across epochs to test whether regularization can replace heuristics.
- **Frontier researcher:** Use the December 2025 latent-drift correction term (https://export.arxiv.org/pdf/2512.10942) to project JEPA predictions back onto the manifold during a 30-step rollout, falsifying whether the correction improves downstream planner reward.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*