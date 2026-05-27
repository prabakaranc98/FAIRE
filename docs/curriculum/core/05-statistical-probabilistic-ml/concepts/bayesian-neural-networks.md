---
title: Bayesian neural networks
slug: bayesian-neural-networks
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [blundell, gal]
feeds_de_pillar: []
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [variational-inference, probabilistic-programming]
tags: []
updated: 2024-11-27
has_mvb: true
---

# Bayesian neural networks

Every autonomous system you build someday will face a data point whose sensors are inconsistent, whose context the training set never saw. The deterministic deep net that got you through the validation set will still make a confident prediction, because it is always asked to pick a single "best" set of weights. This is why, on a snowy evening, you want your perception stack to say "I do not know" before it commits to a risky maneuver. Bayesian neural networks (BNNs) are that upgrade: they treat every weight as a probability distribution rather than a constant, so the model can carry forward its uncertainty into every downstream decision. By the end of this page you will understand how that probability landscape is constructed, why it becomes tractable when we reparameterize each weight, where the BNN literature is pushing the frontier today, and how to train a small BNN yourself that can say "uncertain" when the iris species is ambiguous.

## The territory

In the spectrum from non-probabilistic deep learning up through full Bayesian hierarchical modeling, BNNs sit at the corner where neural nets meet statistical inference. The core question they answer is: instead of searching for a single weight vector \(w^*\) that minimizes training error, can we learn a posterior distribution \(p(w \mid D)\) over weights \(w\) given data \(D\) so that every prediction integrates over that uncertainty? This probabilistic view keeps the expressive capacity of deep networks while giving us calibrated probabilities, natural regularization, and principled safety checks. BNNs carve out their own arc inside probabilistic modeling, sitting just after [[variational-inference]] equips you with the tools to optimize expectations and before [[bayesian-optimization]] reuses that uncertainty to explore.

To do this without intractable integrals we rely on approximations. The classic trading of accuracy for tractability turns the posterior into a surrogate we can sample: instead of solving for the exact conditional, we pick a parametric \(q_\theta(w)\) and adjust its parameters \(\theta\) so that \(q_\theta\) is close to the true posterior. The territory of this page spans that choice of \(q_\theta\), the use of stochastic gradients for training, and how the same ideas extend from feed-forward nets to recurrent architectures and distillation pipelines. How does this posterior estimation actually work, and how does it make the model behave like it “knows what it does not know”? The next section walks through the inference machinery.

## How it works

### From deterministic weights to predictive distributions

Every neural network defines a conditional distribution \(p(y \mid x, w)\) for input \(x\) and output \(y\) once we fix weights \(w\). A BNN wraps that conditional with a prior and a posterior to describe the full data-generating process:

\[
p(y \mid x, D) = \int p(y \mid x, w)\,p(w \mid D)\,dw.
\]

Here \(p(y \mid x, D)\) is the predictive distribution we actually need, \(p(y \mid x, w)\) is the likelihood parameterized by weight instantiations \(w\), and \(p(w \mid D)\) is the posterior given dataset \(D\). Computing this integral exactly is impossible for deep nets, so we replace the posterior with an approximation \(q_\theta(w)\). Each prediction becomes

\[
p(y \mid x, D) \approx \mathbb{E}_{w \sim q_\theta(w)}[p(y \mid x, w)],
\]

which means the model propagates its own weight uncertainty into every downstream decision—uncertainty in \(q_\theta\) manifests as spread in the predictive probability vector.

### Learning the surrogate via Bayes by Backprop

Blundell et al. (2015) [arXiv:1505.05424] introduced Bayes by Backprop to turn this idea into tractable optimization. The algorithm posits a fully factorized Gaussian \(q_\theta(w)=\mathcal{N}(w \mid \mu, \sigma^2)\) for every weight and maximizes the evidence lower bound (ELBO)

\[
\mathcal{L}(\theta) = \mathbb{E}_{w\sim q_\theta(w)}[\log p(D \mid w)] - \mathrm{KL}(q_\theta(w) \,\|\, p(w)),
\]

where \(\log p(D\mid w)\) is the log-likelihood for data \(D\) under weights \(w\), and \(\mathrm{KL}(\cdot\|\cdot)\) penalizes deviation from the prior \(p(w)\). The first term encourages likelihood fit, and the second enforces the prior; when the prior is a zero-mean Gaussian with variance \(\sigma_p^2\), the KL term penalizes large means and variances in \(\mu\) and \(\sigma\) respectively.

Maximizing \(\mathcal{L}(\theta)\) with standard stochastic gradient descent is possible because Bayes by Backprop uses the reparameterization trick: draw \(\epsilon \sim \mathcal{N}(0, I)\) and set \(w = \mu + \sigma \odot \epsilon\), so gradients propagate through \(\mu\) and \(\sigma\). Each mini-batch update simulates a fresh weight sample while the same \(\mu\) and \(\sigma\) accumulate gradient information. The algorithm maintains a posterior belief across the entire weight space instead of converging to a Dirac delta. When the dataset contains ambiguous examples, the learned variance terms \(\sigma\) enlarge, signaling epistemic uncertainty back to the predictive distribution.

### Recurrent models with uncertainty

Temporal data adds another layer of complexity, because the same weight matrix is reused across many time steps. Fortunato et al. (2017) [arXiv:1704.02798] extended Bayes by Backprop to Bayesian recurrent neural networks (BRNNs). The BRNN parameterizes recurrent weights \(W_h\) and input weights \(W_x\) as Gaussians and applies local reparameterization updates inside the recurrence to keep variance computations manageable across long sequences. The optimization alternates between sampling a single set of weight draws for an entire sequence (to maintain consistent temporal dynamics) and using the reparameterized gradients to update \(\mu\) and \(\sigma\). In practice, this allows an RNN to signal when it has drifted into a new regime: the output variance grows when the hidden state receives inputs that were unseen during training, cueing subsequent planners to request human oversight or reinitialize.

### Bayesian dark knowledge and the student-teacher handshake

Another place BNNs shine is in distillation. Whereas standard distillation teaches a student to mimic a teacher’s point estimate, Bayesian Dark Knowledge (Hinton et al. 2015) [arXiv:1506.04416] trains the student on the teacher’s posterior predictive distribution instead of its logits. The teacher itself can be a BNN, so the student learns both the mean and the spread of the teacher’s beliefs. This has two consequences: first, the student inherits better-calibrated uncertainty than it would by regressing on hard labels, and second, the student can be deterministic (and cheaper to deploy) while still capturing essential probabilistic information. In the distillation phase the objective is again an expectation over \(q_\theta(w)\), but the target becomes the teacher’s predictive distribution, which may be computed as

\[
q_{\text{teacher}}(y \mid x) = \frac{1}{S} \sum_{s=1}^S p(y \mid x, w^{(s)}),\qquad w^{(s)}\sim q_{\text{teacher}}(w),
\]

where the \(w^{(s)}\) are samples from the teacher. The student minimizes the cross-entropy between its own predictive distribution and \(q_{\text{teacher}}\), so it learns to reproduce the shape of uncertainty, not just the most frequent class.

### Dropout as a scalable approximation

For larger models, directly storing \(\mu\) and \(\sigma\) for each weight becomes memory intensive, which is why researchers embraced dropout as a practical Bayesian approximation. Gal and Ghahramani (2016) [arXiv:1506.02142] showed that applying dropout at every layer during both training and inference is equivalent to sampling from a mixture of models, which approximates a Gaussian process prior on the output. Each dropout mask corresponds to a weight sample \(w\), and ensembles of forward passes through the dropout layer estimate posterior predictive statistics. This insight makes it possible to estimate epistemic uncertainty with standard architectures and small overhead: call the network \(T\) times with different dropout masks during inference, collect the logits \(f_t(x)\), and compute the predictive variance

\[
\text{Var}(f(x)) \approx \frac{1}{T} \sum_t f_t(x)^2 - \left(\frac{1}{T}\sum_t f_t(x)\right)^2.
\]

The same reasoning is used, for example, in [the 2017 arXiv preprint (Untitled, arXiv:1710.04759v1)](http://arxiv.org/pdf/1710.04759v1), which provides a collection of dropout-based baselines and open-source code to benchmark calibration on vision tasks. The dropout approximation trades some posterior fidelity for scalability, but it is often a better choice than point estimates when you need to quantify whether a prediction is trustworthy.

### Practical failure modes

Understanding where these approximations fail is critical. If the variational family \(q_\theta(w)\) is too narrow (say we fix \(\sigma\) to a small constant), the model collapses back to a deterministic network that cannot express uncertainty. If the prior \(p(w)\) is too weak (very large variance), the KL term offers no regularization and the approximate posterior becomes overly confident. Finally, BNNs trained on badly mis-specified likelihoods (e.g., using Gaussian likelihood for inherently multi-modal labels) may show misleading predictive variances; in those cases you must redesign the likelihood, for example using mixture outputs or normalizing flows to better capture \(p(y\mid x, w)\).

## Where the field is now

Two fronts show how BNNs are evolving. Research-wise, SWA-Gaussian (SWAG) by Maddox et al. (2019) [arXiv:1806.05594] continues to serve as a scalable hybrid: it fits a Gaussian approximation to the weight distribution around a stochastic weight-average point, sampling ensembles cheaply at inference time. SWAG matches or beats BNN baselines on ImageNet while maintaining the kind of calibration needed for risk-sensitive tasks, and it demonstrates that capturing posterior covariance is as important as tracking means.

The engineering frontier centers on making BNNs deployable. TensorFlow Probability’s tutorial and production docs now include a Bayesian neural network example with hierarchical priors, Flipout layers, and export pathways to TensorFlow Serving on Google Cloud [https://www.tensorflow.org/probability/examples/Bayesian_neural_network]. That documentation walks you through flipping the standard Dense layer into a Bayesian Dense layer that stores both \(\mu\) and \(\sigma\), and how to export the resulting SavedModel. The ability to run posterior predictive sampling directly in a serving graph means that real systems (medical imaging pipelines, auto-ML services) can request explicit confidence intervals without retraining a new model. Together these directions show research advancing the approximation quality while engineering efforts focus on production-friendly tooling that logs uncertainty alongside every inference.

## What's still open

1. How do we scale BNN inference to billion-parameter transformers without exploding the number of variational parameters? The open question is whether we can reuse low-rank subspaces or Kronecker-factored covariances so that \(\mu\) and \(\sigma\) occupy only a fraction of the original parameter budget while preserving meaningful uncertainty estimates.

2. Can we design priors that reflect structured inductive biases (e.g., equivariance, sparsity) so that the posterior covariance encodes not just magnitude uncertainty but also interpretable variations? Each proposed structure must come with tractable KL terms and stable gradients.

3. What is the best way to benchmark BNN calibration for out-of-distribution inputs in large pre-trained models? Any benchmark should control the data shift, capture failure modes, and compare deterministic ensembles versus explicit Bayesian inference so that we know when probabilistic modeling actually buys robustness.

## Where to read next

If you want the broader probabilistic inference view, → [[variational-inference]] recounts the ELBO derivation and the reparameterization trick that powers Bayes by Backprop. The engineering counterpart is → [[probabilistic-programming]] which explains how probabilistic languages (like Edward2) turn these ideas into composable modules with automatic inference. To see how uncertainty merges with decision-time planning, → [[bayesian-optimization]] shows how the same predictive distributions drive exploration and acquisition functions.

## Build it

**What you're building:** A Bayesian neural network classifier for the Iris dataset that reports prediction means and epistemic uncertainty and can be exported for lightweight inference.

**Why this is valuable:** You will gain hands-on experience turning a deterministic Keras model into a full BNN, see how uncertainty estimates evolve during training, and produce a reproducible artifact to argue for BNN deployment vs a baseline ensemble.

**Stack:**
- **Model:** `huggingface/tensorflow-probability/bayesian_neural_network` — publicly available BNN demo with 3.7k downloads and a model card showing Flipout layers for classification.
- **Dataset:** `uciml/iris` — a canonical multi-class dataset with four features, clean splits, and community standard preprocessing.
- **Framework:** TensorFlow Probability 0.21.0 + TensorFlow 2.16.0; Flipout layers and the SurrogatePosterior APIs implement Bayes by Backprop with reparameterization.
- **Compute:** Single RTX 4080 (16GB) or free Colab T4 (12GB) — training takes under 30 minutes.

**The recipe:**
1. Install the stack with `pip install --upgrade tensorflow==2.16.0 tensorflow-probability==0.21.0 datasets`. Confirm GPU access via `tf.config.list_physical_devices("GPU")`.
2. Load `uciml/iris`, shuffle with a fixed seed, and scale each feature to \([0, 1]\). Reserve 20% of the training split for validation so you can track uncertainty calibration.
3. Define a sequential BNN: `tfp.layers.DenseFlipout(64, activation="relu")`, `DenseFlipout(64)`, and `DenseFlipout(3)` with a `Categorical` likelihood. Use the `tfp.layers.default_mean_field_normal_fn` prior function and the `tfp.layers.default_mean_field_normal_fn` surrogate posterior.
4. Compile with `tf.keras.optimizers.Adam(learning_rate=1e-3)` and `tf.keras.losses.CategoricalCrossentropy(from_logits=True)`, then train for 150 epochs with an early stopping callback on validation negative log-likelihood. Log per-epoch values of the KL term and the predictive entropy.
5. Evaluate by Monte Carlo sampling: run 100 forward passes with different Flipout draws to compute predictive means and variances. Report accuracy (should be ≈97%) and average epistemic uncertainty on validation examples; verify that hard-to-separate pairs (Setosa vs Versicolor) have higher variance.

**Expected outcome:** A saved TensorFlow SavedModel that exposes a Bayesian inference endpoint (two outputs: mean logits and predictive variance). You also get a calibration plot showing predictive entropy vs accuracy and a CSV log of uncertainty across the validation set.

**Variants per persona:**
- **Applied AI/ML engineer:** Ship the model to TensorFlow Serving with a `PredictiveVariance` endpoint, instrument the inference graph to run 10 samples per request, and measure p95 latency on an A10 — target < 85 ms while maintaining accuracy above 95%.
- **Research engineer:** Reproduce Table 2 from Bayes by Backprop (Blundell et al. 2015) on UCI datasets using Flipout and hit mean test log-likelihood within ±0.05 nats; log your KL vs likelihood contributions to the ELBO.
- **Applied researcher:** Hypothesis: increasing the KL weight (temperature) will raise calibration without hurting accuracy; test three KL weights (0.25, 1.0, 4.0), plot calibration curves, and declare the null if reliability diagrams overlap within ±2% confidence intervals.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*