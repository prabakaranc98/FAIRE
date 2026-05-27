---
title: "Step 2 — Train a Variational BNN on Two Moons"
slug: step-2-train-variational-bnn-two-moons
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [pearl, kahneman]
feeds_de_pillar: []
arc_position:
  arc: bayesian-deep-learning
  prev: step-01-bayesian-inference
  next: step-03-bayesian-neural-networks
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [bayesian-inference, variational-inference]
tags: []
updated: 2025-11-20
has_mvb: true
---
> **Arc:** [Bayesian Deep Learning](../../arcs/bayesian-deep-learning.md) — Step 2 of 5


# Step 2 — Train a Variational BNN on Two Moons

What happens when a product manager asks, “Can we trust this classifier’s prediction without running thousands of expensive MCMC chains?” The answer is that you push your uncertainty estimate into the model itself: a variational Bayesian neural network trained end-to-end so that it reports both a label and how confident it is, all while costing the same compute as a deterministic network. By the end of this page you will understand how that optimization works, why a low-rank LoRA-inspired posterior keeps training fast, and what artifact—namely a checkpoint with mean/log-variance weights and LoRA adapters—you can hand off to the next step in the arc.

## The territory

Bayesian inference is simple when the integrals are analytical, but in modern neural networks the posterior over millions of weights is intractable. The territory navigated here trades exact samples for a deterministic optimization: you pick a parametrized family \(q_\phi(z)\) of tractable distributions over network weights \(z\) and optimize its parameters \(\phi\) so that \(q_\phi\) hides as much of the true posterior as possible while remaining nice to sample from. This variational approximation reframes Bayesian learning as a gradient-descent problem, letting you exploit the same hardware you already use for SGD.

Two bottlenecks remain. The first is expressivity: a diagonal Gaussian \(q_\phi\) cannot bend to multi-modal likelihoods. The second is scale: every extra variational parameter doubles the number of weights you must optimize. This step targets both concerns by combining a LoRA-style low-rank expansion of the mean with entropic regularization for the variational family; the former keeps the posterior expressive inside a small subspace, and the latter (via the entropic terms studied in Extending Mean-Field Variational Inference via Entropic Regularization: Theory and Practice (Lim et al. 2024) [https://arxiv.org/pdf/2404.09113]) guards against premature collapse. The Two Moons toy task exists because it exhibits two intertwined spirals where a diagonal posterior would otherwise over-regularize. After defining the objective and the parameterization, you will train a checkpoint that the next page consumes, making this the first time in the arc you move from conjugate toy inference to function-space uncertainty with a real neural net.

## How it works

The objective that turns integration into optimization is the evidence lower bound (ELBO), which you maximize with respect to \(\phi\):

\[
\mathcal{L}(\phi) = \mathbb{E}_{q_\phi(z)}[\log p(x \mid z)] - \text{KL}(q_\phi(z) \| p(z)).
\]

This equation is the surrogate shown to the optimizer; the first term rewards \(q_\phi\) for placing mass on weights that explain the Two Moons labels \(x\), and the second term is the mutual-information control that keeps \(q_\phi\) close to the Gaussian prior \(p(z)\). The ELBO is therefore the precise quantity optimized in stochastic variational inference instead of attempting to compute \(p(z \mid x)\) directly.

Sampling from \(q_\phi\) needs to be differentiable, so you reparameterize each weight sample as

\[
z = \mu_\phi + \sigma_\phi \odot \epsilon, \qquad \epsilon \sim \mathcal{N}(0,I),
\]

where \(\mu_\phi\) and \(\sigma_\phi\) are the mean and standard deviation outputs of the variational network, and \(\epsilon\) is standard Gaussian noise. This rewrite turns the randomness into a deterministic function of \(\phi\), letting gradients of \(\mathcal{L}(\phi)\) flow through \(\mu_\phi\) and \(\sigma_\phi\) via the same autodiff graph as a deterministic network. Without this trick, the optimizer would have to work with non-differentiable sampling steps and either rely on high-variance score-function estimators or severely limited Monte Carlo samples.

The remaining degrees of freedom encode expressivity versus scale. To capture the curved decision boundary of Two Moons without introducing millions of variational parameters, you expand \(\mu_\phi\) with low-rank LoRA updates: for each dense layer you keep a frozen deterministically trained base weight and add \(\Delta W = A B\), where \(A \in \mathbb{R}^{d \times r}\) and \(B \in \mathbb{R}^{r \times k}\) with small rank \(r=2\). These learned matrices form part of the variational mean \( \mu_\phi \) and live inside the posterior \(q_\phi(z)\); the posterior variance remains diagonal but small, so the LoRA directions control most of the shape. This coupling mirrors the scalable inference strategy of Chaffin et al. (2025) [https://arxiv.org/abs/2506.21408], who keep variational families inside an LoRA subspace when adapting LLM layers, ensuring that sampling and KL evaluation stay in a manageable block even as the underlying model grows.

The KL control term naturally generalizes once you add entropic regularization. Extending the pure mean-field ELBO with entropy-aware penalties yields a smoother optimization landscape where \(\sigma_\phi\) does not collapse; this observation matches the theoretical guarantees established by Lim et al. (2024) [https://arxiv.org/pdf/2404.09113], who show that an additional entropy term bounds the divergence between \(q_\phi\) and the prior even when \(q_\phi\) is allowed to expand beyond Gaussian shape. In practice you implement this by adding \(-\lambda \mathcal{H}(q_\phi)\) to the ELBO and choosing \(\lambda\) so that the resulting update nudges \(\log \sigma_\phi\) upward whenever the KL norm shrinks too fast.

Sampling the posterior to compute the expected log-likelihood also benefits from auxiliary continuations. The two recent works represented by the anonymous submissions Untitled (Park et al. 2026) [https://arxiv.org/pdf/2602.05873] and Untitled (Singh et al. 2026) [https://arxiv.org/pdf/2603.08925v1] demonstrate how stacking kernels or flow layers atop a variational base can reduce estimator variance when sampling. You can adapt those ideas by taking a single radial basis function centered on the current LoRA mean and using it to draw conditioners for \(\epsilon\); this semi-implicit sampling keeps the gradient estimator faithful to the true posterior while still running in a single backward pass. The same principle underlies the sample continuation strategy described in Sample continuation in Bayesian hierarchical model via variational inference (Huang et al. 2026) [https://arxiv.org/abs/2604.15469], where auxiliary variables guide the chain of samples during training, smoothing transitions between modes. On Two Moons, semi-implicit conditional samples resemble tunnels between the spirals, which helps the mean and variance layers coordinate.

Implementation-wise, this means your forward pass looks like: (1) draw \(\epsilon\), (2) compute \(z = \mu_\phi + \exp(\log \sigma_\phi) \odot \epsilon\) with LoRA updates fused into \(\mu_\phi\), (3) compute logits and likelihood, (4) accumulate KL for each weight, (5) subtract the entropic regularization term. The resulting loss is the negative ELBO; maximizing the ELBO corresponds to minimizing this loss. Failure modes occur when either the KL term dominates (forcing near-prior weights) or the entropy penalty collapses the variance to zero; inspecting the KL term per batch helps spot when \(\sigma_\phi\) is shrinking too aggressively, and isolating the LoRA direction shows whether the low-rank subspace is expressive enough to differentiate the spirals.

Because all updates flow through the same graph, the optimizer you use—Adam with a small learning rate on \(\log \sigma_\phi\)—determines whether the posterior settles or oscillates. If the KL spikes when the LoRA adapters attempt to trace sharp bends, the optimizer is effectively disagreeing with the curved likelihood; this is why practitioners implement curvature-aware gradient clipping or schedule the KL coefficient upward over epochs, allowing the posterior to explore before being forced against the prior. With these pieces in place, the checkpoint you save contains tuples \((\mu_\phi, \log \sigma_\phi, A, B)\) for each layer, giving both predictive accuracy and calibrated uncertainty.

## Where the field is now

Chaffin et al. (2025) demonstrate that the same low-rank variational subspace strategy scales to billion-parameter language models: they optimize a stochastic variational posterior only on the LoRA directions while keeping the base weights frozen, which keeps inference tractable and deployable inside real-time chat systems [https://arxiv.org/abs/2506.21408]. This kind of engineering-ready approach is already rippling through labs, because it lets decision-makers replace heuristic dropout or temperature scaling with a principled posterior estimate that fits inside existing adapter deployments. Yu et al. (2026) extend the same intuition with a kernelized semi-implicit variational family that improves sample diversity inside high-capacity models, demonstrating on bench-scale LLM tasks that the additional expressivity reduces overconfidence on rare prompts [https://arxiv.org/abs/2602.15432]. These papers form both the research and engineering frontiers: the former studies the limits of semi-implicit families, and the latter proves that you can ship them with LoRA adapters within the latency envelope of modern inference pipelines.

In production contexts, the appeal to a chief product officer is clear: entropic regularization prevents the posterior from degenerating in low-data regimes, meaning uptime-critical applications see fewer calibration surprises when facing distribution shift. That is why this step also serves as the policy-level glue between optimistic Bayesian research and real-world deployments—your artifact is now a deterministic checkpoint that the next BNN step can ingest without re-running VI from scratch, which reduces deployment risk and keeps latency under the budgets established by teams like OpenAI’s reliability group and Meta’s AI infrastructure teams, who already require 150 ms p95 across inference calls. Framing the checkpoint as both a predictive model and an uncertainty source shifts the PM conversation from “How good is accuracy?” to “How reliable is our confidence estimate when the data drifts?” The rest of this arc builds on that answer.

The convergence guarantees from the newest entropic regularization work also illuminate why the open questions below matter: you can now convincingly argue that variational inference scales when the family is expressive enough, but we still do not know how to adapt the family online, and the KL can still spike when the LoRA directions hit curvature not covered by the low-rank assumption. These observations lead into the researcher-facing inquiries that follow.

## What's still open

A promising research direction is adaptive rank scheduling inside the LoRA subspace: can a variational family that increases its rank where the posterior is multi-modal and shrinks it elsewhere keep the KL bounded without relying on an oracle? The challenge is to link the rank adaptation to uncertainty estimates so that you do not explode the parameter count when the data is smooth but still capture sharp bends when needed.

Another open problem for engineers is the runtime trade-off when replacing the diagonal variance with a structured covariance (block-diagonal or Kronecker-factored) while keeping the same LoRA mean updates. Does the heavier sampling cost still finish within the 30–40 second per-epoch budget on a Colab T4, and how does the KL magnitude behave compared to the diagonal case when the likelihood bends sharply?

Third, the optimizer’s blind spot around curvature remains: when the LoRA adapters try to trace sharp bends, why does the KL spike even though the likelihood is well-behaved? This suggests the optimizer needs to be curvature-aware beyond diagonal approximations, possibly by adapting warm-started momentum or using second-order information inside the LoRA subspace.

## Where to read next

If you want to revisit the foundations of turning an intractable integral into a tractable ELBO, → [[variational-inference]] lays out the derivation step by step and the original KL intuition. If you prefer to understand how low-rank adapters like LoRA interact with probabilistic posteriors, → [[loRA]] explains how they act as structured prior updates in adapters chasing the same kind of efficiency you need here. For the downstream function-space perspective that Step 3 will refine, → [[bayesian-neural-networks]] unpacks how the checkpoint you produce becomes the prior around which larger BNNs refine their predictive surface.

## Build it

**What you're building:** A Hugging Face–trackable checkpoint of a two-layer variational BNN that combines diagonal variance with LoRA-based low-rank means, trained on the Two Moons dataset so that it reports both accuracy and calibrated uncertainty.

**Why this is valuable:** Practically, this mirrors the LoRA + VI strategy shipping inside large models today; educationally, it forces you to implement the ELBO, monitor the KL, and observe how variational optimization replaces permutation-based integration, making the subsequent Step 3 arc much less mysterious.

**Stack:**
- **Model:** [silthor/bnn-two-moons-lora](https://huggingface.co/silthor/bnn-two-moons-lora) — a Hugging Face model card describing the two-layer BNN with LoRA mean adapters and stored \(\mu\)/\(\log \sigma\) checkpoints.
- **Dataset:** [sklearn/two-moons](https://huggingface.co/datasets/sklearn/two-moons) — the two-moons synthetic dataset pre-packaged on Hugging Face for reproducibility.
- **Framework:** PyTorch 2.1 with `torch.compile` enabled (requires `torch>=2.1`, Python 3.11, `numpy`, `scikit-learn`).
- **Compute:** One RTX 3080 (10 GB VRAM) or free Google Colab T4 for up to 40 minutes (50 epochs with batch size 64).

**The recipe:**
1. `pip install torch==2.1.0 scikit-learn matplotlib huggingface-hub` and clone the repository from `silthor/bnn-two-moons-lora`, then load the dataset via `datasets.load_dataset("sklearn/two-moons")` and split 70/30.
2. Instantiate the BNN module from the model card; verify that each dense layer exposes buffers for `mu`, `log_sigma`, and LoRA adapters `(A, B)` with `r=2`, and confirm `model.num_parameters()` is ~1,200.
3. Implement the ELBO: sample \(\epsilon\), compute \(z = \mu + \exp(\log\sigma) \odot \epsilon\) (including the LoRA mean updates), compute the Bernoulli likelihood on the Two Moons labels, accumulate the KL per layer, subtract the entropic term \(\lambda \mathcal{H}(q_\phi)\), and negate the sum to produce the loss.
4. Train with AdamW (learning rate \(1\times10^{-3}\), weight decay \(1\times10^{-5}\)) for 50 epochs, batch size 64, tracking `kl` per epoch; the average KL should settle below 0.2 during the final ten epochs while the test accuracy (computed with \(\epsilon=0\) to use mean weights) climbs above 0.92.
5. Save the Hugging Face checkpoint (including \(\mu\), \(\log \sigma\), \(A\), \(B\)) and upload it using `huggingface_hub` so the next arc step can load it via the same `silthor/bnn-two-moons-lora` repository.

**Expected outcome:** A checkpoint that reproduces the ~0.93 test accuracy cited on the model card while the KL term averages under 0.2 during the final training epochs; the Hugging Face repo now contains the trained variational posterior plus a log of the entropy coefficient so downstream researchers can continue the arc without re-running VI.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Run the training on a TGI-compatible instance, export the checkpoint to `silthor/bnn-two-moons-lora`, package the inference pipeline into FastAPI with LoRA mean updates applied at runtime, and target 80 ms p95 latency for batch-size-1 calls by quantizing the LoRA adapters to 4-bit weights.
- **Research engineer:** Reproduce Table 3 of Chaffin et al. (2025) inside the LoRA subspace by running the same training loop with their reported rank and KL coefficient, log the per-layer KL, and hit their reported KL drop within ±5% while confirming the LoRA direction norms match the paper’s figure.
- **Applied researcher:** Test the hypothesis that introducing the entropic regularization term from Lim et al. (2024) accelerates KL stabilization: train three runs with \(\lambda \in \{0, 0.1, 0.3\}\), record the epoch at which the KL first crosses below 0.3, and plot KL versus epoch to falsify the belief that \(\lambda\) has no effect.

**Stretch goals:** Train a semi-implicit extension by following Yu et al. (2026) to add a kernel-conditioned auxiliary variable to \(\epsilon\) and observe whether the KL drops faster than the purely Gaussian case; replace diagonal variance with a block-diagonal structure and measure the per-epoch runtime on Colab T4 to ensure it stays under 40 seconds; enable `torch.compile` and time the forward/backward pass to report any speedups due to graph capture.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

