---
title: Mixture of experts
slug: mixture-of-experts
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [hinton, jordan, shazeer, bengio]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [mlp, conditional-computation, transformers, optimization-techniques]
tags: [sparsity, routing, conditional-computation, inference, dataset-routing, scale]
updated: 2024-10-01
has_mvb: true
---

# Mixture of experts

Imagine a hospital where every incoming patient, whether they need a Band-Aid or open-heart surgery, is made to walk through every department, see every specialist, and get every possible test before a single treatment starts. Dense neural networks operate the same way: each token or pixel is processed through every layer and every neuron, regardless of which computations would actually help predict the right label. Mixture-of-experts (MoE) throws up a triage desk at the network’s entrance. It asks “which specialists matter for this input?” and steers the work to two or three sub-networks (the experts) whose parameters will do the real job. The rest of the huge network stays asleep, so billions or trillions of parameters can exist without blowing up FLOPs or GPU memory. By the end of this page you will understand how MoE chooses specialists, how to train the gating device without collapsing to a single expert, and how to implement a Top-2 sparse MoE block that keeps inference cost close to a dense layer while still representing far richer functions.

## The territory

Language models, vision transformers, and multi-modal giants keep inventing ever-larger parameter counts because scale continues to buy capabilities. But pushing every input through every parameter is increasingly wasteful: most inputs only need a few forms of computation. Mixture-of-experts belongs to the conditional computation family, where different parts of the network are activated depending on the input. That idea was already present in Jacobs et al. (1991) “Making associative learning competitive” [https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts.pdf], where a gating network learned to softly assign responsibilities over separate predictor modules. Hierarchical mixtures in Hinton et al. (1993) [https://www.cs.toronto.edu/~hinton/absps/hme.pdf] extended the idea with an EM-style training loop and probabilistic layering, showing that specialization could be learned rather than engineered. MoE reuses that insight for transformers: the gating network routes each token to a sparse subset of expert MLPs, which means the model’s capacity is the sum total of all experts, but the compute is proportional only to the few experts that are active. The rest of this page explains how those gates are computed, how experts are kept balanced and trained, and how modern sparse routing keeps FLOPs low. How does the router decide which experts to wake? That mechanism is best understood by starting from the equations for Top-2 gating and load-balancing regularizers.

## How it works

### Input routing as sparse mixture

The simplest MoE block replaces a single dense feedforward block with a bank of \(E\) experts and a gating network that selects which of those experts process the token. Given an input vector \(x \in \mathbb{R}^d\), the gating network produces logits \(g(x) \in \mathbb{R}^E\) and a softmax gives normalized routing weights \(p(x)\):

\[
p_i(x) = \frac{\exp(g_i(x))}{\sum_{j=1}^{E}\exp(g_j(x))}
\]

where \(g_i(x)\) is the logit for expert \(i\) and \(p_i(x)\) is the probability that expert \(i\) will contribute to the output. To enforce sparsity, the gating network does not use all \(p_i\); instead, it selects the Top-\(k\) experts (typically \(k=2\)) with the highest \(p_i\) values and renormalizes them:

\[
\tilde{p}_i(x) = \begin{cases}
\frac{p_i(x)}{\sum_{j \in \text{Top-}k(x)} p_j(x)} & \text{if } i \in \text{Top-}k(x) \\
0 & \text{otherwise}
\end{cases}
\]

where \(\text{Top-}k(x)\) is the set of indices of the \(k\) largest logits. The mixture output is a weighted sum of the expert outputs \(h_i(x)\):

\[
\text{MoE}(x) = \sum_{i=1}^{E} \tilde{p}_i(x) h_i(x)
\]

where \(h_i(x) = \text{Expert}_i(W_i, x)\) is usually a small feedforward MLP parameterized by \(W_i\). Because only \(k\) experts contribute, the computational cost per token is \(O(k \cdot d^2)\) rather than \(O(E \cdot d^2)\), even though the total capacity—since each \(W_i\) is still stored—remains \(E\) times a dense block.

### Making gating stable

Sparse routing introduces a new training challenge: the gating network can collapse and send 100% of tokens to a single expert, wasting the rest of the bank. Early work in Hinton et al. (1993) and Jacobs et al. (1991) handled this with probabilistic EM-style responsibilities, but modern transformers train gating end-to-end with gradient descent. Switch Transformers (Shazeer et al. 2017) [https://arxiv.org/pdf/1701.06538.pdf] kept this idea alive with two new tricks: (1) a simple auxiliary loss that encourages a uniform distribution of tokens across experts, and (2) careful initialization so the gating logits start near zero. The auxiliary loss is:

\[
L_{\text{load}} = \lambda \cdot \sum_{i=1}^{E} \left(\frac{\text{Load}_i}{\text{Tokens}} - \frac{1}{E}\right)^2
\]

where \(\text{Load}_i\) counts how many tokens routed to expert \(i\) during the current batch, \(\text{Tokens}\) is the total number of tokens processed, and \(\lambda\) is a tunable coefficient. This term penalizes imbalance by pushing each expert’s fraction of tokens toward \(1/E\). Top-2 gating introduces two placements per token, which softens the load imbalance because tokens that would otherwise saturate one expert are spread over two weights.

Another regularizer is confidence-based entropy \(H(p)\): high entropy encourages the gate to use more experts, while low entropy yields sharper selection. Shazeer et al. (2017) experimented with both \(L_{\text{load}}\) and \(H(p)\) to find stable routing without sacrificing quality.

### Feedforward expert design

Each expert is typically a two-layer MLP of the form:

\[
h_i(x) = W_{i}^{(2)} \cdot \phi(W_{i}^{(1)} x + b_{i}^{(1)}) + b_{i}^{(2)}
\]

where \(W_{i}^{(1)} \in \mathbb{R}^{d_f \times d}\) maps the token to a higher-dimensional hidden layer of size \(d_f\), \(\phi\) is an activation (often GELU), and \(W_{i}^{(2)} \in \mathbb{R}^{d \times d_f}\) projects back to the transformer dimension. Because experts share the same overall structure, the overall block is plug-and-play: you can drop it in anywhere a dense FFN sits. The gating network is usually a lightweight linear layer with softmax, so the gating cost is negligible compared to the experts.

Mixture-of-experts interacts with the rest of the transformer as follows: after the attention module and before the residual add, insert the MoE block and add the resulting \(\text{MoE}(x)\) back into the residual path. In inference, because only two experts run per token, throughput stays high despite the enormous stored capacity.

### Sparse training dynamics and failure modes

The biggest failure mode arises when gating is triggered off-task. For example, if the gating logits are computed solely from the token embedding, they might align with superficial features (like punctuation) rather than the semantic work the experts are supposed to do. One mitigation is to condition gating on the token plus the current layer’s residual, so the router adapts to the evolving representation.

Shazeer et al. (2017) also introduced auxiliary metrics that track how “busy” each expert is. When one expert accumulates a lion’s share of the load—even with the auxiliary loss—it often signals that either the gating learning rate is too high or the initialization favored that expert. Layer normalization on the gating weights and gradient clipping limit sudden shifts in the routing distribution.

An orthogonal concern is expert redundancy: if every expert learns the same thing, the block degenerates into a dense layer. Conditional computation research from Bengio et al. (2013) [http://arxiv.org/pdf/1312.4314v3] argued that sparsity should be encouraged during training, either via dropout-like constraints or by penalizing overlapping activations. In practice, you can encourage diversity by adding noise to the gating logits or by applying a KL penalty between consecutive batches: the gate is rewarded for changing its pattern of selection rather than repeating the same experts.

### Scaling to very large models

MoE blocks allow the total number of parameters to grow linearly with \(E\). In practise, models like GShard (Shazeer et al. 2017) store hundreds of billions of parameters split across experts, but only the experts needed for a given batch are activated, so the active compute stays constant. During inference, a token’s route only touches \(k\) experts, so the FLOPs per token stay similar to a dense layer even as the model grows. This decoupling—train-time capacity vs. inference-time FLOPs—is the central insight that justifies trillion-parameter MoE models.

Modern training pipelines orchestrate this by sharding experts across devices. A token is sent over the network only to the hosts that own the selected experts, so the batch size and communication patterns dominate the wall-clock performance. Systems engineering is as important as the gating math: scheduling, batching, and expert placement determine whether MoE actually runs faster than a dense baseline.

## Where the field is now

Switch Transformers (Shazeer et al. 2017) demonstrated that sparse Top-1 gating could already match dense baselines while scaling parameters to 1.6T with the same number of FLOPs per token, but the load balancing terms were tuned manually. Since then, the research frontier has moved toward fine-grained routing. Recent work such as DeepSeekMoE (Dai et al. 2024) hardens that frontier by tracking per-token specialization and letting routing share features across similar tokens, which reduces redundant expert usage and lifts downstream perplexity on long-context tasks by about 5 points relative to Switch-style gating.

On the engineering frontier, Google’s Pathways and GShard systems pioneered device-aware MoE deployment. GShard (Shazeer et al. 2017 revisited in the engineering blog from research.google.com) shards experts across TPU cores, routes tokens through in-batch scheduling, and keeps all-to-all communication manageable. Vertex AI now exposes a MoE-backed “Pathways” service that can serve 600B-parameter models with sub-second latency by automatically batching tokens to the few experts they need and dropping the rest of the model’s compute.

These advances expose a clear research frontier: gating must be fast, balanced, and representation-rich all at once. New papers are exploring routing that conditions on extended context windows or that directly predicts the combination of experts rather than separate logits, but none have resolved the trade-offs between specialized representations and the load-balance penalties that hurt downstream fine-tuning.

## What's still open

How can we eliminate the trade-off between routing load balance and model representation quality without relying on hand-tuned auxiliary loss coefficients that degrade downstream performance? Every MoE today mixes a capacity-aware loss (to keep each expert busy) with the primary per-token loss, and the balance coefficient is tuned by hand—too much and the model focuses on uniform usage, too little and the routing collapses. A publishable investigation would ask: can a gating architecture be derived that inherently produces balanced routing while still optimizing the downstream objective, perhaps by deriving the gate’s loss from a single probabilistic model?

How can MoE routing be made more culturally aware of long-tail inputs so that rare tokens receive the specialized experts they need without copying the majority expert’s behavior? Current gates still struggle when they must represent both common and rare syntactic patterns, leading to bias. A solution could involve conditioning the gate on token frequency statistics or introducing a novelty detector that forces underutilized experts to stay alive.

Does dynamic expert pruning, where unused experts are retired at inference time, hurt model calibration compared to keeping the entire expert pool active? This question probes the long-term reliability of MoE deployment: if the gating network is allowed to permanently retire experts based on usage, we must understand whether the remaining experts still cover the problem space, especially under domain shift.

## Where to read next

If you want the probabilistic roots of routing, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) frames how gating mirrors the responsibilities in soft clustering; the engineering counterpart is → [[conditional-computation]] which details the system-level choices for sharding and device placement; for the next paradigm that shifts away from expert banks entirely, → [Flow matching](../../02-generative-modeling/concepts/flow-matching.md) generalizes the idea of sparse computation with continuous paths.

## Build it

This build proves that the MoE idea is not just a math trick but an engineering lever: you can swap a dense feedforward block in a classifier with a Top-2 sparse router and still train on MNIST with one minute of compute per epoch, all while watching the gating probabilities specialize to different digits.

**What you're building:** A PyTorch Top-2 MoE layer that replaces a dense MLP block inside a small classifier training on MNIST, visualizing how different experts focus on different digit strokes.

**Why this is valuable:** Training this block yourself forces you to implement the gating logits, the Top-2 selection, and the balancing loss; the artifact is a checkpoint + visualization that concretely shows how tokens are steered to two distinct expert MLPs, so you can explain how MoE decouples capacity from FLOPs.

**Stack:**
- **Model:** `hf-internal-testing/tiny-random-mlp` (1,000 downloads) — serves as the starting point whose FFN is replaced with our MoE block
- **Dataset:** `mnist` [https://huggingface.co/datasets/mnist] — normalized grayscale digits split into 60k train / 10k test
- **Framework:** PyTorch 2.1 + `torchvision==0.15` + `accelerate==0.28`
- **Compute:** Colab T4 (16GB) or equivalent (RTX 3060) — expect ~90 seconds per epoch for 60k samples with batch size 256

**The recipe:**
1. Install `pip install torch torchvision accelerate matplotlib numpy` and clone the repo that defines a simple classifier (use the HF tiny MLP config), then replace its feedforward block with your `SparseMoEBlock` module.
2. Data: load the HF `mnist` dataset, normalize images to \([0,1]\), flatten to 784-d vectors, and batch with shuffle; keep 5k samples aside for validation.
3. Train/fine-tune: in each forward pass, compute gating logits with a linear layer, select Top-2 experts per token using `torch.topk`, renormalize the probabilities, apply each expert MLP, and weight their outputs; include the load-balance loss \(L_{\text{load}} = \lambda \sum ( \text{Load}_i/\text{Tokens} - 1/E)^2\) with \(\lambda=0.01\); train for 10 epochs with learning rate \(1e^{-3}\), weight decay \(1e^{-5}\), and gradient clipping at 1.0—expect the training loss to drop below 0.05 within 6 epochs.
4. Evaluate: compute classification accuracy on the test set and log perplexity; additionally, plot the average gating probabilities per expert per digit class—expect each of the two experts to specialize on ~5 digits and the third gating weight to remain near zero (Top-2 uses only two experts, so one expert should dominate some digits with balanced contributions).
5. What you now have: a working MoE-enhanced MNIST classifier, a checkpoint containing the gating network + experts, and a set of visualizations showing how tokens route to experts.

**Expected outcome:** A checkpoint + visual report showing Top-2 gating specialization, demonstrating that the MoE block trains stably and that load-balancing regularization keeps both experts active.

- **CS student:** Run the same recipe on an RTX 4070 but reduce the expert count to 4 and the hidden dimension per expert to 128 so you can train a slightly larger model in under one hour and still see specialization in the gating plot.
- **Applied engineer:** After training, export the MoE block as a TorchScript module, quantize the experts to INT8 with `torch.quantization`, and serve the model through vLLM to hit a p50 latency under 45 ms on T4 inference instances by routing only the active experts per token.
- **Applied researcher:** Hypothesize that increasing the entropy penalty on the gate will force more expert diversity; vary \(\lambda\) from 0.001 to 0.05 and compare test accuracy + gating-entropy curves to determine whether the auxiliary loss still improves downstream quality without hurting loss convergence.
- **Frontier researcher:** Probe the open question by designing a gate whose loss is derived from a probabilistic mixture instead of a manually tuned load-balance term; instrument the experiment to falsify the claim that this probabilistic gate achieves uniform expert usage while matching the original accuracy within 0.2%.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*