---
title: Instrumental Variables
slug: instrumental-variables
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [pearl, heckman, imbens, rubin]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [structural-causal-models, potential-outcomes, selection-bias]
tags: [causal-inference, identification, two-stage-least-squares, randomized-control, causal-representation]
updated: 2025-11-01
has_mvb: true
---

# Instrumental Variables

Imagine trying to answer whether a new AI coding assistant lifts developer productivity by comparing the commits of teams who opt in versus those who do not. The opt-in teams are the most curious, the early adopters, the ones who already track their velocity obsessively. That hidden “motivation” confounder corrupts the comparison, making the tool look like a miracle even if it does nothing. Now imagine a rollout glitch that randomly exposes a subset of developers to the assistant—their access is no longer a self-selection story but a piece of variation created by the system. That glitch is an instrument: it nudges usage but, ideally, has no direct line to productivity beyond that nudge. Instrumental variables are the formal way of converting that kind of external variation into an unbiased causal estimate when standard regression cannot separate treatment from unobserved confounders. By the end of this page you will understand the algebra that reconstructs unbiased effects, how the assumptions fail in practice, and how to build a two-stage estimator in PyTorch so you can inspect every step yourself.

## The territory

Instrumental variables (IVs) sit at the junction between econometrics’ push for identification and modern causal representation learning. The typical problem is that the treatment \(d\) and outcome \(y\) are both correlated with some unobserved confounder \(u\). A naive regression \(y \sim d\) mixes the causal path \(d \rightarrow y\) with the spurious path \(u \rightarrow d\) and \(u \rightarrow y\), so the coefficient on \(d\) cannot be interpreted as a causal effect. Randomized controlled trials fix this by severing the \(u\) paths, but experiments are expensive, unethical, or impossible for many decisions. Instrumental variables offer a third way: find a measurable variable \(z\) (the instrument) that moves \(d\) but does not move \(y\) except through \(d\). The instrument acts like a proxy for randomization, allowing the analyst to recover the causal effect \( \tau \) even if \(u\) remains hidden. IVs are part of the broader strategy of using auxiliary variation—natural experiments, encouragement designs, policy changes, or glitches—to resurrect identification when observational data alone would mislead. Instrumental methods are also the engine behind recent causal representation work that exploits environmental shifts for identifiability. The mechanism is best understood by starting from the structural equation system that defines treatment, outcome, and instrument, and then plumbing the two-stage estimator that recovers \( \tau \) when the instrument satisfies its two core requirements.

## How it works

At its heart the IV strategy rewrites the causal link between \(d\) and \(y\) as a system of structural equations. Suppose the outcome is generated as

\[
y = \tau d + f(u) + \epsilon_y,
\]

where \( \tau \) is the causal effect we seek, \(f(u)\) aggregates the contribution of unobserved confounders \(u\), and \( \epsilon_y \) is idiosyncratic noise uncorrelated with everything else. The treatment itself is influenced by the instrument:

\[
d = \pi z + g(u) + \epsilon_d,
\]

with \( \pi \neq 0 \) capturing the relevance of the instrument \(z\), \(g(u)\) again pooling the unobserved confounders, and \( \epsilon_d \) noise orthogonal to \(z\). Without additional assumptions, the acoustic interference from \(u\) spoils a direct regression. The IV relies on two key conditions: relevance (\( \pi \neq 0 \)) and exclusion (the instrument influences \(y\) only through \(d\), so \(z\) is independent of \(y\) conditional on \(d\) and \(u\)). When these hold, the ratio of covariances isolates \( \tau \). Algebraically, if we regress \(d\) on \(z\) and then \(y\) on the predicted \(d\), the resulting coefficient confers the causal effect:

\[
\tau = \frac{\text{Cov}(z, y)}{\text{Cov}(z, d)},
\]

where covariances are evaluated under the distribution of observed data. The numerator captures how \(z\) nudges the outcome, while the denominator captures how \(z\) nudges the treatment. The noise from \(u\) cancels out because \(z\) is assumed independent of \(u\); the only variation passing through \(z\) reaches \(y\) via \(d\). This ratio is the population-level intuition behind the two-stage least squares (2SLS) estimator.

### Two-stage least squares

The practical estimator replaces population covariances with fitted regressions. Stage one regresses \(d\) on the instrument and any exogenous covariates \(x\):

\[
\hat{d} = \hat{\pi} z + \hat{\beta} x.
\]

Stage two regresses \(y\) on the fitted \(\hat{d}\) and the same covariates:

\[
y = \tau \hat{d} + \gamma x + \text{residual}.
\]

Because the first stage filters out the part of \(d\) correlated with \(z\) only, the second stage isolates the path that follows the instrument. When \(\tau\) is estimated by ordinary least squares in the second stage, the resulting coefficient is consistent even though the actual \(d\) may still correlate with \(u\). The algebra behind 2SLS can be understood as projecting both \(y\) and \(d\) onto the column space spanned by \(z\) and \(x\) before taking their ratio. This projection demarcates the variation coming from the instrument and leaves the confounder-induced variation orthogonal to the estimation.

### Diagnostics and failure modes

Relevance failure occurs when \(\text{Cov}(z, d)\) is close to zero. In finite samples the first-stage \(\hat{\pi}\) becomes noisy, and 2SLS suffers from weak instrument bias, pulling \(\tau\) towards the OLS estimate that mixes in confounding. A practical rule of thumb is to examine the F-statistic from the first-stage regression: values below 10 signal weakness. Exclusion failure happens when \(z\) has a direct path to \(y\); if \(z\) affects \(y\) outside of its influence on \(d\) (for example, if the random rollout glitch also changes developer morale independently of assistant usage), the estimated \(\tau\) attributes that direct effect to the treatment. Analysts test for this using overidentification tests when multiple instruments exist, or by searching for heterogeneity patterns inconsistent with the exclusion restriction.

Another failure arises when the instrument is correlated with measurement noise. In high-dimensional representation learning, the measured instrument might itself be entangled with latent confounders. Direct Effect Analysis (DEA) (Anonymous et al. 2025) [arxiv:2503.04358](https://arxiv.org/abs/2503.04358) provides a method for isolating the direct effect of candidate instruments by solving a generalized eigenvalue problem on learned representations, effectively disentangling the instrument from nuisance correlations. That framework shows how instruments can live not in raw features but in learned subspaces that remain invariant across noisy environments.

### Instrumental variables in causal representation learning

IVs have re-emerged in the representation learning community as a means of identifying latent causal variables across environments. Theoretically, Learning Causal Representations from General Environments (Anonymous et al. 2023) [arxiv:2311.12267](https://ar5iv.labs.arxiv.org/html/2311.12267) characterizes when latent variables are identifiable given changes in interventions. The identifiability results rely on having instruments or proxies that shift certain latent factors without touching others. The same paper shows how a structured generative model decomposed into instrumented subspaces can recover the causal graph up to permutation, as long as the instrument induces enough variation along one dimension while leaving others unchanged. This theoretical backing justifies modern practice: if you can train a representation extractor that produces features \(z\) whose variation matches an instrument’s target of action, you can apply IV estimation in the latent space even when the raw treatment is noisy.

Potential Outcome Rankings for Counterfactual Decision Making (Anonymous et al. 2025) [arxiv:2511.10776](https://arxiv.org/abs/2511.10776) takes the IV estimate \(\tau\) and turns it into rankings for individual-level decisions. After using instruments to get unbiased effect estimates, they order units by the predicted uplift and show that this ordering maximizes population utility under plausible risk aversion. This type of decision layer is what makes IVs actionable: once you disentangle the treatment’s causal effect, you can allocate resources to the units that benefit the most.

### Instruments beyond scalar features

In practice the instrument may not be a scalar variable but an entire experiment. Suppose randomization assigns 30% of developers to a feature flag that allows early access to the coding assistant. The assignment mechanism itself becomes \(z\); its randomness ensures relevance, while the exclusion restriction holds if the experiment affects productivity only through access. In complex systems where the "instrument" is an algorithmic change, engineers often treat the assignment policy as the instrument and verify exclusion by checking for balance on pre-treatment variables and by comparing multiple outcomes (e.g., productivity vs. error rate). When multiple instruments are available, one can stack them into an instrument matrix and estimate a vector of causal effects for multi-dimensional treatments, but every instrument still needs a domain argument for exclusion.

### Analytical interpretation of 2SLS

When the first-stage fitted values \(\hat{d}\) are used, the two-stage estimator solves

\[
\hat{\tau}_{2SLS} = \frac{(z^\top P_x y)}{(z^\top P_x d)},
\]

where \(P_x\) is the projection matrix onto the column space of covariates \(x\). This expression highlights that 2SLS is equivalent to replacing \(z\) with its residual after regressing out the covariates and then projecting \(y\) and \(d\) onto this space. The result is consistent as long as \(z\) remains uncorrelated with the error term in the structural equation for \(y\). The projection viewpoint also suggests alternative estimators such as limited information maximum likelihood and generalized method of moments when there are many instruments or heteroskedasticity, but the core idea remains: define a transformation of the data that removes confounder-induced variation while retaining the instrumented signal.

## Where the field is now

The research frontier is reimagining instruments in settings with representation learning and high-dimensional data. Instrumental Mechanism discovery leverages the idea that environment shifts provide the external variation needed for an instrument. Anonymous et al. (2024) [arxiv:2406.14302](https://arxiv.org/pdf/2406.14302) demonstrate an approach where agents learn to synthesize instruments from policy perturbations in reinforcement learning, allowing them to disentangle the effect of reward shaping from confounders during offline RLHF training. That work constructs a contrastive loss that enforces an exclusion restriction by penalizing representations that correlate with downstream rewards outside the instrumented pathway. The instrumentation generalizes to other domains, too: 2306.00542 (Anonymous et al. 2023) [arxiv:2306.00542](https://arxiv.org/pdf/2306.00542) extends the idea to time-series covariates, building a temporal instrument via lagged policy switches that maintain exclusion across long horizons. These papers set the theoretical groundwork for instruments that are themselves learned objects rather than hand-crafted features.

On the engineering side, production teams are deploying these ideas in attribution pipelines. The marketing measurement platform described by Anonymous et al. (2026) [arxiv:2603.25796](https://arxiv.org/pdf/2603.25796) formalizes how randomized offer assignments serve as instruments for ad exposure in large-scale systems. The paper outlines the instrumentation of user cohorts across millions of samples and shows how to guard against exclusion failure by cross-validating the instrument’s direct effect on alternative outcomes (such as app engagement). This engineering effort demonstrates that, at scale, the same identification principles used in econometrics can be embedded inside real-time pipelines, enabling companies to report unbiased uplift metrics rather than biased click-through correlations.

## What's still open

1. Can we algorithmically validate the exclusion restriction in high-dimensional learned representations without leaning on domain knowledge? In other words, is there a statistical test that distinguishes true exclusion from mere conditional independence in the latent space when the instrument itself is derived from neural features?

2. When instruments are themselves generated by models (learned encouragements), how can we certify that the relevance condition will hold out-of-distribution? Instruments derived from policy perturbations might work well in training but collapse when the environment shifts, so an open question is how to regularize the instrument’s influence to remain robust.

3. How does multi-instrument inference behave when instruments compete or interfere, such as when two different randomized campaigns simultaneously affect the same treatment? Understanding the geometry of the instrument matrix and its impact on finite-sample bias could unlock more reliable standard errors for complex deployments.

4. Can Potential Outcome Rankings be extended to settings where the causal effect is vector-valued (multi-task interventions) and instruments differ across dimensions, requiring a joint ranking strategy that respects the heterogeneity across treatments?

## Where to read next

If you want to see how the same identification argument is framed probabilistically, → [[potential-outcomes]] lays out the Neyman-Rubin causal model and how conditioning changes the estimand, while the engineering counterpart is → [[difference-in-differences]] explaining how instruments generalize the logic of policy rollouts. To connect to the broader arc of discovering latent causal graphs, → [[causal-representation-learning]] traces the environmental invariances that make identifiability possible.

## Build it

Instrumental variables are only useful when the analyst can see and manipulate the instrumented variation, and the best way to understand IVs is to implement the two-stage estimator yourself with datasets where you know the true effect. This notebook build exposes you to the entire pipeline: generate a synthetic marketing dataset, simulate a randomized email promotion as the instrument, fit the first and second stages in PyTorch, and inspect diagnostics such as the first-stage F-statistic and bias when the exclusion restriction fails. You will leave with intuition about each component of the IV assumptions and how fragile the estimate becomes when the instrument weakens.

**What you're building:** A PyTorch 2SLS estimator on a synthetic marketing dataset that uses a randomized email recommendation as an instrument to recover the true uplift of ad exposure on conversions.

**Why this is valuable:** Because IV estimation is about reconstruction, not prediction, implementing the estimator reveals how the first-stage filtering and second-stage projection interact, why relevance matters quantitatively, and where bias leaks in when exclusion fails.

**Stack:**
- **Model:** Custom 2-layer MLP first-stage predictor (PyTorch) — download count: built from scratch (no external checkpoint)
- **Dataset:** [ydata-synthetic/synthetic_data](https://huggingface.co/datasets/ydata-synthetic/synthetic_data) — well-known synthetic tabular benchmark you can transform into marketing covariates
- **Framework:** PyTorch 2.1 + scikit-learn 1.4 for metrics
- **Compute:** Free Colab TPU or T4 / ~30 minutes (sampling and fit on 10k rows)

**The recipe:**
1. `pip install torch==2.1 scikit-learn pandas matplotlib` and load the synthetic dataset from Hugging Face, retaining fields for baseline covariates \(x\), treatment proxy \(d_{\text{raw}}\), and a randomization seed column.
2. Preprocess by converting \(x\) to standardized tensors, define the instrument \(z\) as a Bernoulli draw seeded by the rollout column, and let the observed treatment \(d\) equal \(z + 0.7 d_{\text{raw}} + \mathcal{N}(0, 0.1)\) to mimic partial compliance; build the outcome \(y = 2 d + 1.5 x_1 + \mathcal{N}(0, 0.5)\) so you know the ground-truth effect.
3. Train the first-stage MLP \(d \sim z, x\) using mean squared error for 200 epochs with learning rate \(1e{-3}\); log the first-stage F-statistic to ensure the instrument is relevant.
4. Freeze the first-stage weights, compute fitted values \(\hat{d}\), and regress \(y\) on \(\hat{d}, x\) (second-stage OLS). Evaluate the bias by comparing \(\hat{\tau}\) to the known true effect (2.0) and plot the residual errors.
5. Introduce an exclusion violation (e.g., add \(0.5 z\) directly into \(y\)) and re-run the estimator to observe how \(\hat{\tau}\) drifts; log the difference in the second-stage coefficient to visualize failure modes.

**Expected outcome:** A runnable Colab notebook that outputs the 2SLS estimate, reports diagnostics (first-stage F-statistic, bias measure), and visualizes how the estimate shifts when the instrument loses exclusion.

- **CS student:** Force the build into a single Colab cell-by-cell narrative and replace the MLP with a linear layer so that it fits within an RTX 4070 budget; keep the same diagnostics to see how the simpler model behaves.
- **Applied engineer:** Deploy the trained 2SLS pipeline behind a Flask API, quantize the first-stage weights with `torch.quantization`, and serve predictions on an A10 with latency < 50 ms per inference while logging the instrument balances for auditing.
- **Applied researcher:** Hypothesize that stacking a second instrument derived from a different randomized campaign improves precision; add the new instrument into the first-stage regression using multi-variable 2SLS and compare variance reduction on held-out synthetic data.
- **Frontier researcher:** Probe the open question about exclusion testing by augmenting the representation learner: train a contrastive encoder that regularizes latent features to be orthogonal to \(z\)’s residuals, then assess whether the contrastive penalty stabilizes estimates under simulated exclusion violations while measuring the falsification criterion (the correlation between \(z\) and the second-stage residual).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*