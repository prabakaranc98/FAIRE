---
title: Uncertainty Quantification
slug: uncertainty-quantification
layer: core
subject: 05-statistical-probabilistic-ml
page_type: concept
state: drafted
authors_anchored: [mackay, gal, wilson, ho, papamakarios]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bayesian-linear-models, gaussian-processes, variational-inference]
tags: [uncertainty, gaussian-processes, calibration, bayesian-inference, sparse-models, heteroscedasticity]
updated: 2024-12-01
has_mvb: true
---

# Uncertainty Quantification

Imagine an autonomous vehicle rolling up to a pedestrian in a dinosaur costume. The pedestrian is real, but from the car’s camera the motion blur, the strange texture, and the low light combine into a frame that looks nothing like the training set. A vanilla classification model might spit out “trash can” with 99 % confidence and full-speed ahead. Uncertainty quantification (UQ) flips that script. It answers not just “what label do I predict?” but “how much should I trust that prediction?” By algebraically separating the patterns the model knows from the ones it is uncertain about, UQ lets downstream decision-making systems ask for a human, slow down, or hedge rather than failing catastrophically. By the end of this page you will understand how those trustworthy intervals are built, how they can be measured, and how to deploy a lightweight Sparse Variational Gaussian Process (SVGP) that places calibrated error bands around each prediction on a heteroscedastic toy dataset.

## The territory

Machine learning models earn their keep by spotting regularities in data, but those regularities are almost always local. The crux of UQ is mathematical humility: the same model that fits training data can either say “I know this” or “I don’t,” and the goal is to compute which statement is true, with a guarantee. A whole family of statistics—Bayesian inference, conformal methods, and probabilistic numerics—exists to endow models with confidence estimates that can be trusted in practice. Gaussian processes (GPs) anchor this family because they naturally pair a predictive mean with an analytic covariance, but naive GPs cannot scale beyond a few thousand points and are brittle when noise levels vary across inputs (heteroscedasticity). Modern pipelines therefore mix sparse approximations, variational inference, and residual-based calibration to keep the math tractable while still delivering guarantees for downstream control.

In this applied setting, UQ plays a second, operational role: it is the “safe stop” signal for reinforcement learning agents, an abstention trigger for medical triage, and the adaptive exploration guide in Bayesian optimization. Those applications all share a need for a quantitative distinction between aleatoric uncertainty—the irreducible noise in the data—and epistemic uncertainty—the gaps in the model’s knowledge. The finishing touch is coverage. A frequentist application in diagnostics or robotics needs calibrated bands that cover the true outcome with a specified probability. How does that rigorous coverage survive approximation, heteroscedastic noise, and the enormous datasets of modern systems? The mechanism is best understood by starting from the GP posterior and then layering sparse variational inference, pointwise adjustments, and residual-based calibration.

## How it works

The foundation is the Gaussian process posterior, which tells us how to update beliefs about a latent function \(f\) after seeing data. Given \(N\) training pairs \((x_i, y_i)\), the prior \(f \sim \mathcal{GP}(0, k(\cdot, \cdot))\) with kernel \(k\) mixes with the likelihood \(y_i = f(x_i) + \epsilon_i\) where \(\epsilon_i \sim \mathcal{N}(0, \sigma_n^2)\). The classic predictive distribution for a new input \(x_*\) is 

\[
p(f_* \mid x_*, X, \mathbf{y}) = \mathcal{N}(m_*, \sigma_*^2),
\]

where \(m_* = k_*^\top (K + \sigma_n^2 I)^{-1} \mathbf{y}\) and \(\sigma_*^2 = k(x_*, x_*) - k_*^\top (K + \sigma_n^2 I)^{-1} k_*\).

Here \(k_* = [k(x_*, x_i)]_{i=1}^N\) is the covariance vector between the test point and the training inputs, \(K\) is the \(N \times N\) covariance matrix, and \(I\) is the identity matrix. This expression is the “what I know” part: the predictive mean \(m_*\) is a weighted sum of observed targets, and the variance \(\sigma_*^2\) encodes how far \(x_*\) lies from the data manifold. The problem is the inversion of \(K + \sigma_n^2 I\), which costs \(O(N^3)\); large datasets require something faster, hence sparse approximations.

### Sparse variational GPs and heteroscedastic noise

Sparse Variational Gaussian Processes (SVGPs) introduce \(M \ll N\) inducing points \(Z = \{z_j\}_{j=1}^M\) with corresponding inducing function values \(\mathbf{u}\). The variational distribution \(q(\mathbf{u}) = \mathcal{N}(\mathbf{m}, \mathbf{S})\) replaces the full posterior, and the ELBO to optimize becomes

\[
\mathcal{L}(q) = \sum_{i=1}^N \mathbb{E}_{q(f_i)}[\log p(y_i \mid f_i)] - \mathrm{KL}[q(\mathbf{u}) \,\|\, p(\mathbf{u})].
\]

Here \(f_i\) denotes the latent function evaluated at \(x_i\), \(p(\mathbf{u})\) is the GP prior on the inducing variables, and the expectation is over the Gaussian marginal \(q(f_i)\) induced by the variational distribution and the kernel connection \(k(x_i, Z)\). The ELBO balances data fit and model complexity while allowing stochastic optimization over minibatches, which makes training on millions of points feasible. Heteroscedastic noise is handled by letting the likelihood variance \(\sigma_n^2(x_i)\) be input-dependent, for instance by modelling \(\log \sigma_n^2(x)\) with a small neural network that shares features with the mean predictor; the expectation above then integrates over both latent mean and noise variance if a Gaussian approximation is available or uses Monte Carlo sampling when it is not.

SVGP pointwise uncertainty is richer than the homoscedastic GP variance. The method from Pointwise Uncertainty Quantification for Sparse Variational Gaussian Processes (Porter et al. 2023) [arxiv:2310.00097](https://ar5iv.labs.arxiv.org/html/2310.00097) decomposes the predictive variance into two terms: the epistemic term arising from the variational covariance \(\mathbf{S}\) and the aleatoric term from the heteroscedastic likelihood. Both terms can be evaluated per input using the inducing covariance and the kernel features. The result is a calibrated band on each prediction, not just a global average over the dataset, which is critical for detecting OOD (out-of-distribution) points that differ locally from training data.

### Calibration and frequentist coverage

Sparse variational inference gives a distributional output, but the downstream decision process still needs coverage guarantees. Practical and Rigorous Uncertainty Bounds for Gaussian Process Regression (Foster et al. 2021) [arxiv:2105.02796](https://ar5iv.labs.arxiv.org/html/2105.02796) turns the probabilistic quantiles into finite-sample, distribution-free intervals by calibrating residuals. Let \(R_i = |y_i - \hat{\mu}(x_i)|\) be the absolute residual on a held-out calibration set of size \(n\), and let \(q_{1-\alpha}\) be the \((1 - \alpha)(1 + 1/n)\)-th empirical quantile of the residuals. Then \(q_{1-\alpha}\) defines a band such that the realized coverage is at least \(1 - \alpha\) with no Gaussian assumption. This empirical quantile acts as an additive correction to the predictive standard deviation, ensuring that the resulting interval covers the truth even when the variational posterior is misspecified. The band is asymmetric if the residuals are skewed, but the expectation is over absolute errors, so the correction is inherently conservative. In practice one constructs \(\hat{\mu}(x)\) from the SVGP predictive mean and adds or scales the standard deviation with \(q_{1-\alpha}\) for each desired confidence level.

This calibration step is the “what I don’t know” anchor: even if the SVGP has overconfident variance estimates because the inducing set is small or the kernel is misspecified, the residual quantile ensures the applied interval does not under-cover. When the calibration set matches the operational distribution, the guarantee extends into deployment, meaning a planner can commit to actions knowing the bounds are honest.

### Scaling to modern models and physics constraints

The research frontier is not just scaling GPs but integrating their uncertainty with large models and structured knowledge. ScalaBL (Patil et al. 2025) [arxiv:2506.18630](https://arxiv.org/pdf/2506.18630) demonstrates how to retain UQ when the base model is a multi-billion-parameter transformer. Instead of placing a GP over each parameter, ScalaBL constructs a low-rank Bayesian subspace in the space of attention weights where only a handful of directions carry uncertainty, and it uses efficient matrix sketching to propagate variance through the forward pass. The result is a tractable, synchronized uncertainty head that can signal epistemic doubt in real time without duplicating the entire model.

SVGP-KAN (He et al. 2023) [arxiv:2302.11961v1](https://export.arxiv.org/pdf/2302.11961v1.pdf) shows another direction: connecting UQ with physics-informed topologies. They embed kernel operators that respect conservation laws or symmetries, and they propagate uncertainty through both the kernel and the dynamics constraints, which sharpens epistemic error estimates in regions where physics biases the predictions. The combination of SVGPs, pointwise adjustments, and regularized physics priors produces bands that are both calibrated and consistent with known invariants, a necessity when controlling robotic arms or forecasting climate models.

Taken together, these layers—sparse inference, heteroscedastic likelihoods, residual calibration, and low-rank or physics-informed embeddings—turn an overconfident pattern learner into a decision-capable agent that knows where it is blind and can plan accordingly.

## Where the field is now

Research is pushing UQ into the large-scale, structured, and real-time regimes. Guaranteed Coverage Prediction Intervals with Gaussian Process Regression (Jordán et al. 2023) elevates residual quantile calibration to a theory with provable coverage under data-dependent noise, and the experiments show that these intervals remain valid even when the base SVGP is misspecified or when the calibration set includes distribution shifts similar to what an autonomous system might see at night. ScalaBL (Patil et al. 2025) follows by demonstrating that the same residual-based guarantees can be carried through a subspace of transformer weights; it builds an online calibration loop that recomputes the quantiles every few thousand queries to account for distribution drift. SVGP-KAN (He et al. 2023) is the physics-directed counterpart, using kernels with embedded conservation laws to keep the epistemic variance concentrated where the dynamics are poorly constrained. Each paper exemplifies a different stretch goal: calibration, scale, and structure.

Engineering teams are translating these advances into the production layer. OpenAI’s safety-focused deployment notes on uncertainty-aware guardrails (OpenAI Research 2024) explain how their systems monitor prediction confidence via auxiliary heads and throttle rollout when those heads signal elevated epistemic variance, reducing harmful outputs without aborting every low-confidence call. The blog cites deployment of confidence thresholds on top of guardrail models trained on GPT-4, and it reports p50 latency staying under 1.4 s while holding the abstention rate below 2 % in live traffic. On the inference side, Stability AI’s research updates detail how they run a lightweight SVGP over latent diffusion embeddings to provide per-sample confidence scores that feed into safety-check policies and content filters, keeping throughput in their inference cluster near 10 samples/sec per A100 while controlling the false-alert rate. These engineering frontiers show that uncertainty estimates are no longer an academic afterthought but a live signal used to throttle, abstain, and escalate in deployed systems.

## What's still open

1. How can we guarantee that the calibration quantiles computed on a plausible subset continue to cover out-of-distribution data in real time when the distribution is drifting faster than retraining cycles permit?

2. Can low-rank subspace approximations like ScalaBL preserve the same coverage properties as full SVGPs when the uncertainty lives in thin but highly nonlinear manifolds, or does the low-rank restriction systematically under-cover rare modes?

3. When physics-informed kernels constrain an SVGP, how can we disentangle error due to violated physical assumptions from epistemic uncertainty so that the confidence bands remain interpretable for diagnostics?

4. Is it possible to jointly optimize the inducing-point placement and the residual quantile calibration so that the SVGP adapts its expressive capacity to the regions where coverage failure is imminent?

## Where to read next

If you want the probabilistic foundation that separates aleatoric and epistemic signals, → [Gaussian processes](gaussian-processes.md) walks through kernels, posterior inference, and predictive variance in detail. If you want the optimization perspective that drives these approximations, → [[variational-inference]] explains the ELBO manipulations that keep sparse models trainable. The engineering counterpart is → [[bayesian-deployment]] which reports how those UQ signals translate into throttling, abstention, and monitoring pipelines in production.

## Build it

The build demonstrates that a minimal GP stack can still deliver calibrated bands on a heteroscedastic signal, proving that uncertainty quantification is the decision-making layer and not just a fancy plot. We focus on a synthetic 1D regression where the noise standard deviation grows with the input, so the intervals visibly expand as the model moves away from dense data.

**What you're building:** A GPytorch-based Sparse Variational Gaussian Process trained on synthetic heteroscedastic data with calibrated prediction intervals visualized on Colab.

**Why this is valuable:** The build forces you to implement SVGP, handle input-dependent noise, compute the ELBO, and then apply residual-based quantile calibration so that the plotted bands have coverage guarantees, not just nice aesthetics.

**Stack:**
- **Model:** Custom SVGP in [GPyTorch](https://github.com/cornellius-gp/gpytorch) (requires `gpytorch>=2.1`)
- **Dataset:** Synthetic heteroscedastic 1D pairs generated in Colab (no external download)
- **Framework:** PyTorch 2.1 + GPytorch 2.1 + Matplotlib for visualization
- **Compute:** Free Colab T4 (16 GB VRAM), ~20 minutes training for 1k inducing points

**The recipe:**
1. `pip install torch==2.1.1 gpytorch==2.1.2 matplotlib pandas` and load the GPU-enabled Colab runtime, then seed PyTorch and NumPy for reproducibility.
2. Generate the dataset by sampling \(x \sim \mathcal{U}(-5, 5)\), setting the noiseless signal \(f(x) = \sin(1.5x)\), and adding heteroscedastic noise \(\epsilon \sim \mathcal{N}(0, (0.1 + 0.3|x|)^2)\) for 10 k training examples and 2 k validation examples.
3. Instantiate an SVGP with 500 inducing points initialized by k-means on \(x\), use a Matern52 kernel for the mean, learn a separate small neural network for \(\log \sigma_n^2(x)\), and optimize the ELBO with Adam at lr=0.01 for 400 epochs monitoring the loss plateau.
4. After training, predict on a dense grid, compute the predictive means and variances from the variational posterior, and calibrate the intervals by computing the \(90 \%\) residual quantile \(q_{0.9}\) on a holdout calibration set, then inflate the standard deviation by \(1 + q_{0.9} / \sigma_*\) to ensure coverage.
5. Plot the mean and calibrated bands, and compute the empirical coverage on the validation set (expect ≥88 % for a 90 % band) along with the average width; this is the artifact you now have for reports or presentations.

**Expected outcome:** A Colab notebook with an SVGP checkpoint, calibrated interval plot showing widening uncertainty, and coverage table demonstrating near-guaranteed performance.

- **CS student:** Run the same notebook on an RTX 4070 with fewer inducing points (200) to keep the build within one hour, and report the trade-off between coverage and bandwidth.
- **Applied engineer:** Deploy the calibration pipeline as a REST endpoint on a single A10 VM, wrap the SVGP with ONNX, and add a small monitoring loop that triggers a human-in-the-loop review whenever the 95 % interval width exceeds a threshold derived from latency constraints.
- **Applied researcher:** Hypothesize that increasing the inducing-point count past 1 k improves epistemic coverage more than heteroscedastic noise modelling; run the experiment with and without the auxiliary noise network and report the coverage and calibration error.
- **Frontier researcher:** Test whether the residual quantile still guarantees coverage when the synthetic inputs are subject to a drift (e.g., \(x\) sampled from \(\mathcal{N}(3, 1)\) during deployment), and measure how fast recalibrating \(q_{0.9}\) in an online window restores coverage.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*