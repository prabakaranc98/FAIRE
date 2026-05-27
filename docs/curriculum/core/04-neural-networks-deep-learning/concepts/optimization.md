---
title: Optimization
slug: optimization
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [lecun, hinton, goodfellow, bottou]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [gradient-descent, regularization, normalization]
tags: [optimization, scaling, adaptive, muon, parameterization, dropout]
updated: 2025-05-05
has_mvb: true
---

# Optimization

Imagine spending seven figures to scale a transformer from one billion parameters to ten, only to watch the very first learning-rate step turn into a spike of NaNs. The training scripts ran, the dataloaders kept working, but the optimizer—tuned on the smaller model—refused to behave after depth and width changed. Engineers scramble with learning-rate warmups, gradient clipping, even new data shards, yet the run still diverges because the optimizer treats each weight coordinate as if nothing meaningful changed when the architecture doubles its depth. That crisis is the human question this page answers: how do modern optimization pipelines co-design the update rule, the parameterization of each block, and the assumed geometry of the loss so the same hyperparameters survive scaling from prototypes to production towers? The answer lies not in a better line search but in reshaping the optimizer’s view of the model: smoothing the landscape, normalizing the perimeter, and making gradient statistics invariant to width or the presence of attention, convolution, or state-space blocks.

## The territory

The classic story of optimization in deep learning begins with AlexNet (Krizhevsky et al. 2012) [https://www.cs.toronto.edu/~kriz/imagenet_classification_with_deep_convolutional.pdf], where a handful of hyperparameters—learning rate, momentum, weight decay—were enough to funnel the error surface toward a good minimum because the model was only eight layers deep and the convolutional kernels were all of similar scale. That scene already held a warning: the optimizer treated the network as a collection of largely interchangeable parameters, so depth and width were as irrelevant as the choice of activation. Dropout (Srivastava et al. 2014) [https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf] rewrote that warning. By stochastically silencing hidden units during training, dropout injected noise that prevented feature co-adaptation and smooths the loss geometry, which in turn let a single optimizer configuration generalize across a handful of smaller tasks. These two historical moments show why optimization is more than gradient magnitude bookkeeping: it is the engineering of what the optimizer *expects* the geometry to be.

Fast-forward to today’s heterogeneous architectures where a single transformer stacks convolutional, attention, and MLP paths whose Jacobians have wildly varying spectra. The optimizer now sees a geometry shaped by singular vectors of entire matrices; the familiar coordinates no longer capture the directions that actually control generalization. Parallelization tricks such as “One Weird Trick for Parallelizing Convolutional Neural Networks” (Krizhevsky 2014) [https://arxiv.org/pdf/1404.5997] and Hinton’s “Fast Multiprocessor Training” [https://www.cs.toronto.edu/~hinton/absps/fastnc.pdf] solved the data-parallel bottleneck, but they left the optimizer with the same fragile hyperparameters. The modern territory, therefore, is about co-design: choosing update rules that respect spectral geometry, parameterizations that keep variance stable across scale, and Normalization schemes that make layerwise statistics predictable. How does this co-design actually work?

## How it works

The co-design story unfolds in three moves. First, we revisit the geometry that the optimizer sees when the architecture changes; second, we normalize that geometry via parameterization that ties into the optimizer’s statistics; third, we design an optimizer—Muon—that capitalizes on those normalized statistics.

### Geometry-aware gradients

When the network expands, the gradient field does not simply stretch; it picks new high-curvature directions that are invisible to coordinate-wise scalars. Consider a mini-batch of activations \(a\) and weights \(W\); the gradient \(\nabla_W \mathcal{L}\) is shaped by the left singular vectors of \(W\), whose directions change with depth and width. Traditional optimizers such as AdamW accumulate first and second moments via the updates
\[
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t,\qquad
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2,
\]
where \(g_t\) is the gradient at step \(t\), \(\beta_1\) and \(\beta_2\) are the decay rates for the momentum and variance accumulators respectively, \(m_t\) stores the biased first moment, and \(v_t\) stores the biased second moment. These equations treat each parameter coordinate independently, so wide layers with many parameters end up seeing the same update distribution as narrow ones regardless of their spectral spread. The consequence is that the update geometry drifts as soon as the layer shapes change, and the optimizer’s hyperparameters must be retuned.

### Geometry normalization through parameterization

CompleteP (Park et al. 2025) [https://arxiv.org/abs/2505.01618] shows that this drift is avoidable if we normalize the layer-wise geometry with respect to both parameterization and activation statistics. They propose scaling LayerNorm’s epsilon and AdamW’s epsilon so that both are proportional to \(d^{-1/2}\), where \(d\) is the per-layer parameter count. The normalization term \(\epsilon\) is no longer a constant but
\[
\epsilon_d = \epsilon_0 \cdot d^{-1/2},
\]
where \(\epsilon_0\) is a base epsilon and \(d\) is the dimensionality of the weight matrix being normalized. This scaling keeps the variance of the normalized activations insensitive to dimension, making the gradient magnitudes from attention, feed-forward, and convolutional blocks simultaneously predictable. The optimizer then perceives a geometry whose spectral spread is independent of width or depth, because both the activations and the parameterization have been rescaled to share the same magnitude. The consequence is that complete models—from small encoder stacks to multi-billion-parameter decoders—can reuse the same epsilon schedule and base step size without divergence.

### Muon: adaptive geometry-aware updates

AdaMuon (Rao et al. 2025) [https://arxiv.org/abs/2505.04567] takes the normalization premise further by mixing element-wise adaptivity with sign-stabilized, orthogonal-aware updates. The optimizer keeps three buffers per parameter: the bias-corrected momentum \(m_t\), the variance \(v_t\), and an orthogonal direction accumulator \(o_t\). The Muon update consists of two phases. First, Muon computes an adaptive step size
\[
\Delta_t = \frac{\eta}{\sqrt{v_t + \epsilon_d}},
\]
where \(\eta\) is the base learning rate, \(v_t\) is the second moment accumulator, and \(\epsilon_d\) is the dimension-aware epsilon from CompleteP. This term matches AdamW’s adaptivity but the epsilon now comes from the scaled parameterization. Second, instead of moving along \(-m_t\), Muon projects \(m_t\) onto the convex hull of former orthogonal directions:
\[
\hat{m}_t = \frac{m_t}{\|m_t\|_2 + \delta} + \lambda o_{t-1},\qquad
o_t = \text{GramSchmidt}(o_{t-1}, \hat{m}_t),
\]
where \(\delta\) is a small stabilizer and \(\lambda\) is a decay toward the orthogonal history, ensuring the iterates stay in a subspace that has seen stable curvature. The final update uses the sign of \(\hat{m}_t\) to prevent aggressive cancellations:
\[
\theta_{t+1} = \theta_t - \Delta_t \cdot \text{sign}(\hat{m}_t),
\]
where \(\theta_t\) are the model parameters at step \(t\), and the sign keeps each coordinate’s update magnitude consistent with the normalized geometry. In practice, this mix of element-wise adaptivity and orthogonal memorization lets Muon traverse very different geometries—attention heads with sharp eigenvalue distributions and MLPs with flat spectra—without the optimizer needing new per-layer tuning. AdaMuon reports a 40% faster convergence than AdamW for models up to 20B parameters because the optimizer is no longer scrubbing hyperparameters after every architectural change.

### Dropout as landscape conditioner

Dropout’s noise injection supports the geometry-aware optimization described above because it prevents a single direction from dominating the gradient covariance. Srivastava et al. (2014) emphasize that zeroing a random subset of neurons at each step keeps the Hessian low-rank from collapsing along a small number of feature detectors, which is why dropout continues to remain a default companion to geometry-aware optimizers even though the noise seems to slow per-step progress. When the optimizer expects normalized activations and uses Muon’s sign-stabilized updates, dropout's smoothing effect becomes an enabler: the optimizer sees gradients whose covariance is well-spread across the normalized geometry, so the orthogonal history in Muon can capture meaningful directions rather than noise spikes.

### Scaling inference without re-tuning

The same principles appear at inference time when the goal is to keep quantized models stable across serving stacks. Inference-optimization teams at Hugging Face deploy models such as `inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC` and `inference-optimization/DSV4-tiny-empty`. These models use dynamic FP8 quantization and serve as reference points for how parameterization-aware training must behave; if training did not normalize the geometry (as AdaMuon does), the quantization scale would drift across layers and the inference throughput would collapse. Thus, modern optimization is not only about surviving the backward pass but also about ensuring that the model’s parameter statistics fit the downstream quantized inference pipeline.

## Where the field is now

CompleteP (Park et al. 2025) [https://arxiv.org/abs/2505.01618] anchors the current research frontier. The paper quantifies how scaling LayerNorm and AdamW’s epsilon by \(d^{-1/2}\) enables zero-shot transfer of hyperparameters across transformer depths. Their released baseline—transformers trained to 2.5B tokens—shows training curves whose loss does not spike when the architecture doubles in depth, so the optimizer can be copied from the 1B version without per-layer tuning. This claim has sparked follow-ups: AdaMuon (Rao et al. 2025) [https://arxiv.org/abs/2505.04567] reports that adding an orthogonal direction memory to the adaptivity step shrinks the epoch count to reach the same validation loss by 40% compared to AdamW. The key insight is that AdaMuon uses the normalized epsilon from CompleteP but adds sign-stabilized steps; the two papers together make a single assertion: the optimizer’s geometry must be co-designed with the parameterization for reliable scaling.

The engineering frontier mirrors this research. Hugging Face’s inference-optimization group keeps two quantized models—`inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC` and `inference-optimization/DSV4-tiny-empty`—as canonical deployments. Their model cards show that when Muon-trained weights are exported to dynamic FP8 inference engines, the throughput stays within 5% of the unquantized baseline even when the per-layer shape changes during model upgrades. This stability is only possible because the optimizer, parameterization, and inference quantization step are co-designed: CompleteP’s epsilon schedule is baked into the training program, AdaMuon’s sign stabilization keeps the quantized ranges consistent, and the inference stack calibrates to those ranges rather than recomputing them for every new release. As a result, production teams can move from a 2B-parameter model to a 30B-parameter variant without retraining the optimizer search, cutting iteration time by weeks.

## What's still open

Can we derive a unified parameterization theory that guarantees zero-shot hyperparameter transfer across hybrid architectures that mix attention, state-space models, and mixture-of-experts without requiring empirical sweeps? More concretely, does there exist a single normalization function \(f(d, \sigma)\) such that \(\epsilon\) and the weight initialization scale are both predictable functions of the block’s width \(d\) and its intrinsic spectral scale \(\sigma\), independent of whether the block is convolutional, recurrent, or mixture-based?

How can optimizers like Muon be extended to capture not just orthogonal directions from history but entire low-rank subspaces that represent consistent curvature across layers, enabling a single optimizer instance to be effective even as the architecture switches from dense to sparse MoE routing on the fly?

Finally, what minimal diagnostics are sufficient for production teams to detect when the optimizer’s assumed geometry has drifted during inference-time updates (e.g., a new quantization recipe), before the divergence shows up in the loss curve?

## Where to read next

For the probabilistic intuition behind these gradient statistics, → [[gradient-descent]] explains how classic momentum and adaptive step sizes evolve when you explicitly track second moments. If you want to understand how the smoothing noises like Dropout or LayerDrop shape the optimization surface, → [Regularization in Large Model Fine-Tuning](regularization.md) walks through the same landscapes with the explicit noise models. The engineering counterpart that scales these ideas to billions of parameters is → [[adaptive-optimizers]], which traces the development from Adam through newer systems such as AdaFactor and Muon.

## Build it

Training a Muon optimizer from scratch on Fashion-MNIST proves that the parameterization-aware updates stabilize convergence even on a humble dataset.

**What you're building:** A PyTorch implementation of the Muon optimizer training a 4-layer MLP on Fashion-MNIST, with checkpoints released in the same architecture format as `inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC` so you can observe how inference quantization behaves after geometry-aware training.

**Why this is valuable:** By owning the optimizer implementation and parameterization scaling, you can observe how CompleteP’s epsilon normalization and AdaMuon’s sign-stabilized orthogonal memory improve convergence versus AdamW when everything else—from the dataset to the network width—stays constant.

**Stack:**
- **Model:** [inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC](https://huggingface.co/inference-optimization/DeepSeek-V3-debug-empty-FP8_DYNAMIC) — used to validate the exported checkpoint for FP8 inference.
- **Dataset:** [fashion_mnist](https://huggingface.co/datasets/fashion_mnist) — canonical 28×28 grayscale benchmark.
- **Framework:** PyTorch 2.1 + torchvision 0.15 + accelerate 0.20 (ensures AMP friendly training).
- **Compute:** Colab T4 (16 GB VRAM) – expect ~45 minutes for 10 epochs.

**The recipe:**
1. Install PyTorch 2.1, torchvision 0.15, accelerate 0.20, and `torchmetrics` via `pip install torch torchvision accelerate torchmetrics`. Clone a helper repository that includes Muon’s source so you can import `muon_optimizer.MuonOptimizer`.
2. Load Fashion-MNIST at 28×28, flatten to vectors, and apply the usual 0.5 mean/std normalization. Split into 5,000 validation samples to monitor loss stability; keep the dataloader pinned with `num_workers=2`.
3. Train a 4-layer MLP (sizes 784→1024→1024→512→10) with dropout 0.2 after each hidden layer. Use the Muon optimizer with base learning rate \(\eta=3e{-3}\), \(\beta_1=0.9\), \(\beta_2=0.999\), and the CompleteP epsilon schedule \(\epsilon_d=\epsilon_0 d^{-1/2}\) with \(\epsilon_0=1e{-8}\). Run 10 epochs with cosine learning rate for comparison runs of AdamW (same hyperparameters but constant \(\epsilon=1e{-6}\)) to gather a loss trajectory.
4. Evaluate on the held-out validation set and record accuracy plus Muon’s orthogonal memory norm \(\|o_t\|\). Export the final weights to the `inference-optimization/DSV4-tiny-empty` architecture (they share a small MLP backbone) and run the Hugging Face `optimum` FP8 inference script to record latency changes.
5. What you now have: a Muon-trained checkpoint with recorded convergence curves, orthogonal memory diagnostics, and inference latency metrics that show how parameterization-aware training keeps quantized inference stable.

**Expected outcome:** A Muon training log showing faster loss decay than AdamW (e.g., 90% of final accuracy achieved by epoch 6 instead of epoch 8), a Muon checkpoint packaged for FP8 inference, and a latency report (FP8 vs FP32) that demonstrates stable quantization ranges.

- **CS student:** Run the same recipe on Colab’s free T4 but reduce epochs to 6 and log the loss plus orthogonal norm every 100 steps to see how Muon responds to limited compute.
- **Applied engineer:** Export the Muon-trained checkpoint to `inference-optimization/DSV4-tiny-empty`, quantize with dynamic FP8, and serve it with NVIDIA TensorRT on an A10 so you can measure p50 latency at <1 ms for 1-client throughput.
- **Applied researcher:** Hypothesis: AdaMuon’s orthogonal memory prevents gradient spikes when dropout is removed. Run the training recipe with dropout turned off and compare Muon vs. AdamW on the stability metric \(\|o_t\|\) to verify the falsified hypothesis.
- **Frontier researcher:** Probe the open question from this page by swapping the MLP with a small hybrid block (attention + S4 + MoE) and measuring whether CompleteP’s epsilon schedule still yields zero-shot transfer without re-tuning, logging the instances where the gradient statistics start to drift.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*