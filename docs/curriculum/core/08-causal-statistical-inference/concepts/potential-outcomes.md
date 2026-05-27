---
title: Potential Outcomes
slug: potential-outcomes
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [little, rubin, imbens, hahn]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [counterfactuals, causal-graphs, treatment-effect-estimation]
tags: [causal-inference, potential-outcomes, missing-data, treatment-effects, propensity-score, instrumental-variables]
updated: 2024-11-15
has_mvb: true
---

# Potential Outcomes

Imagine a doctor watching a patient recover after taking a new antiviral. The lab tests look good, and the patient feels better—but the doctor will never see the alternate universe where that patient refused treatment. There is no second life to compare against the first, so any claim that the drug “caused” the recovery is built atop an absence: the counterfactual outcome. The potential outcomes framework was born exactly to make that absence visible, to treat causality as a missing-data problem and then ask, “what assumptions let us reconstruct the unseen value?” By the end of this page you will be able to write down the target estimand for an observational study, understand the assumptions that make inverse-probability weighting, instrumental variables, and doubly robust estimators consistent, and implement a minibatch Doubly Robust estimator on Lalonde’s dataset to see how the pieces interact in practice.

## The territory

The entire causal-inference stack sits on this one core realization: there are always two outcomes per unit—one for treatment, one for control—but only one is observed. Little and Rubin (2000) [https://ics.uci.edu/~sternh/courses/265/littlerubin_annrevepi2000.pdf] framed the problem in terms of missing data, explicitly modeling the vector of potential outcomes \((Y(1), Y(0))\) and observing a single slice \(Y = W\cdot Y(1) + (1-W)\cdot Y(0)\), where \(W\in\{0,1\}\) indicates treatment assignment. The observable world is a projection of the full potential-outcome table, and the projection is lossy because it mixes treated and untreated units. The goal is to recover features of the complete table (for example, the average treatment effect \(\mathbb{E}[Y(1)-Y(0)]\)) by plugging in reasonable assumptions about how units were assigned to treatment.

Causal graphs are the first way people try to justify those assumptions, but the potential outcomes language abstracts away from particular graph structures and starts with the estimand and the assumptions directly. This is why Rosenbaum and Rubin (1983) [https://www.stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf] emphasize the propensity score: by conditioning on a scalar function of covariates, we can mimic the balancing achieved by randomization without literally re-running the experiment. The potential outcomes framework is therefore the “missing data plus assumptions” lens that lets us speak rigorously about ignorability, propensity scores, instruments, and doubly robust estimators. How does that machinery actually work?

## How it works

### Notation and estimands

Let \(Y_i(1)\) and \(Y_i(0)\) denote the potential outcomes for unit \(i\) under treatment and control, respectively, and let \(X_i\) be the pre-treatment covariates. We observe only one of the potential outcomes: \(Y_i = W_i Y_i(1) + (1-W_i)Y_i(0)\), where the binary \(W_i\) is the treatment indicator. The causal effects of interest are functions of the missing second potential outcome; for example the average treatment effect (ATE) is
\[
\tau_{\text{ATE}} = \mathbb{E}[Y(1) - Y(0)],
\]
where the expectation is over the marginal distribution of units in the observational study. Each symbol carries meaning: \(Y(1)\) is the outcome if the unit had been treated, \(Y(0)\) if untreated, and the expectation averages over the full population. The fundamental problem of causal inference is clear: the difference inside the expectation cannot be measured for any single unit because we never see \(Y(1)\) and \(Y(0)\) simultaneously.

The usual way to proceed is to invoke assumptions that relate the observed data \((X,W,Y)\) to the unobserved potential outcomes. The Stable Unit Treatment Value Assumption (SUTVA) says the potential outcomes for unit \(i\) do not depend on the treatments assigned to other units, and consistency says that \(Y = Y(W)\) for the treatment actually received, ensuring the observed outcome is the corresponding potential outcome. These two assumptions, commonplace in experimental design, explicitly rule out interference and multiple versions of treatment, so that each unit’s single observation can be seen as the manifestation of the latent potential outcome under one of the two regimes.

### Ignorability, propensity scores, and balancing

To recover \(\tau_{\text{ATE}}\) from observational data we typically need ignorability: \(Y(1), Y(0) \perp W \mid X\). Conditionally on \(X\), treatment is as good as randomized, so the treated and untreated groups differ only by noise. Rosenbaum and Rubin (1983) [https://www.stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf] recognized that matching on high-dimensional \(X\) is impossible, but if we compute the propensity score \(e(X) = \mathbb{P}(W=1\mid X)\), then \(Y(1), Y(0) \perp W \mid e(X)\). This “balancing score” means we can adjust for a scalar instead of the full vector, turning the missing data problem into a weighted version of the law of large numbers.

A simple estimator uses inverse-probability-of-treatment weights (IPTW): the ATE is approximated by
\[
\hat{\tau}_{\text{IPW}} = \frac{1}{n}\sum_{i=1}^n \left[ \frac{W_i Y_i}{\hat{e}(X_i)} - \frac{(1-W_i) Y_i}{1-\hat{e}(X_i)} \right],
\]
where \(\hat{e}(X)\) is a logistic regression or machine-learned model for the propensity. Each component is annotated: \(W_i\) is the indicator, \(Y_i\) the observed outcome, and \(\hat{e}(X_i)\) the estimated propensity score. When the logistic model for \(e(X)\) is correct, the weights re-weight treated and control observations so that they approximate draws from the counterfactual mixture distributions. When the model is wrong but the outcome model is right, other estimators remain consistent, leading to doubly robust approaches.

### Instruments, LATE, and two-stage least squares

When ignorability fails because of unmeasured confounders, an instrumental variable \(Z\) that affects \(W\) but not \(Y\) except through \(W\) can rescue identification. Imbens and Angrist (1994) [https://scholar.harvard.edu/imbens/files/wo-stage_least_squares_estimation_of_average_causal_effects_in_models_with_variable_treatment_intensity.pdf] define the Local Average Treatment Effect (LATE) for compliers—units whose treatment status changes when \(Z\) changes. The standard linear two-stage least squares (2SLS) estimator operationalizes this: the first stage regresses treatment on the instrument and covariates,
\[
W_i = Z_i \pi + X_i \gamma + \nu_i,
\]
where \(Z_i\) is the instrument, \(X_i\) the controls, and \(\nu_i\) the residual. The predicted treatment \(\hat{W}_i = Z_i \hat{\pi} + X_i \hat{\gamma}\) replaces the true \(W_i\) in the second-stage regression
\[
Y_i = \hat{W}_i \beta + X_i \delta + \varepsilon_i,
\]
where \(\beta\) is the LATE and \(X_i\) again controls for remaining variation. Every symbol is annotated: \(Y_i\) is the observed outcome, \(\hat{W}_i\) is the fitted probability of treatment from stage one, \(X_i\) are the covariates, and \(\varepsilon_i\) is the remaining noise. When the instrument satisfies relevance (it predicts \(W\)) and exclusion (it affects \(Y\) only through \(W\)), the 2SLS estimator recovers the causal effect for compliers, a slice of the population that can be described entirely through the potential outcomes induced by \(Z\).

This instrument-based perspective is especially powerful when only part of the population can be nudged into treatment. The LATE gives a meaningful quantity even when the overall ATE is not identifiable, because it explicitly defines the subpopulation whose potential outcomes change with the instrument.

### Double robustness and semiparametric efficiency

Hahn (1998) [https://statweb.rutgers.edu/ztan/material/hahn98.pdf] studied the asymptotic behavior of semiparametric estimators in this framework and showed that combining a correctly specified propensity model with a misspecified outcome model (or vice versa) can still yield consistent estimates, hence “double robustness.” The Doubly Robust (DR) estimator blends a regression adjustment with propensity weighting:
\[
\hat{\tau}_{\text{DR}} = \frac{1}{n}\sum_{i=1}^n \left[ \hat{m}_1(X_i) - \hat{m}_0(X_i) + \frac{W_i(Y_i - \hat{m}_1(X_i))}{\hat{e}(X_i)} - \frac{(1-W_i)(Y_i - \hat{m}_0(X_i))}{1-\hat{e}(X_i)} \right].
\]
Here \(\hat{m}_1(X)\) and \(\hat{m}_0(X)\) are the predicted outcomes from separate regressions on treated and control units. The first term in the summand corrects the systematic difference in predicted outcomes, while the next two terms correct the residuals with inverse-propensity weights. Each part is annotated: \(\hat{m}_1\) and \(\hat{m}_0\) map covariates to outcomes, \(W_i\) toggles treatment, and \(\hat{e}(X_i)\) is the propensity score. If either \(\hat{m}\) or \(\hat{e}\) converges to the truth, the estimator is consistent. Estimating both simultaneously delivers asymptotic efficiency under standard conditions, because the influence function of the DR estimator cancels leading biases when both models are correctly specified, matching the semiparametric efficiency bound derived by Hahn.

Implementing a DR estimator therefore means choosing a parametric or machine-learned model for the outcome and another for the propensity, training them on the same data, and aggregating their predictions in the DR formula. The combination is the mechanism by which the missing potential outcomes are imputed and weighted, yielding an estimate that bridges regression and weighting.

## Where the field is now

The potential outcomes lens has been extended beyond tabular censuses to rich generative models that learn entire counterfactual distributions. Causal Diffusion Models (Zhang et al. 2024) [https://arxiv.org/abs/2403.09138] learn forward-noise processes conditioned on observed history and then run a reverse diffusion to sample plausible counterfactual trajectories, which is the first demonstration that diffusion-based generative models can encode time-varying confounding without explicitly modeling structural equations. These models output a distribution over \(Y(1)\) and \(Y(0)\) jointly, giving practitioners uncertainty quantification rather than point estimates, and they have begun to challenge sequential decision-making benchmarks where temporally correlated treatments would break standard ignorability-based estimators.

On the engineering front, Meta’s Counterfactual Reasoning and Learning Systems team uses a production-scale estimator that blends doubly robust weighting with real-time propensity updates while handling billions of ad impressions daily to optimize placement and bidding strategies [ai.meta.com/research/publications/counterfactual-reasoning-and-learning-systems-the-example-of-computational-advertising/](https://ai.meta.com/research/publications/counterfactual-reasoning-and-learning-systems-the-example-of-computational-advertising/). Because the system relies on adaptive weighting and touches every impression, it must recompute propensity scores with each policy change while still delivering low-latency predictions; the persistence of the DR structure at this scale shows how the theoretical guarantees carry through to real-world decision-making.

The motor of progress is now twofold: research pushes toward richer counterfactual distributions via generative models and treatments that evolve over time, while engineering tests the robustness of doubly robust architectures at latency and throughput scales dictated by ad auctions and recommendation feeds.

## What's still open

How can we compute tight, non-parametric bounds on counterfactual outcomes in Markov Decision Processes when multiple observationally equivalent causal models explain the same historical data? The usual potential outcome arguments collapse when the policy influences state transitions as well as rewards, and current diffusion-based proposals have not produced inequalities that hold uniformly over policy classes.

Can we design propensity-score estimators that remain efficient even when covariate distributions shift between training and deployment, without retraining the entire model? The current doubly robust estimators assume that the observed covariate distribution reflects future use, so distribution shifts can destroy the balancing property and bias the estimates.

Is there a principled way to integrate partial compliance with high-dimensional covariates under non-linear outcome models so that the LATE identified by 2SLS generalizes to the broader population? The complier-specific estimand is precise but narrow; whether machine learning and new identification conditions can widen the target while preserving interpretability is unresolved.

Could causal diffusion models be regularized to admit sensitivity analyses (e.g., bounded unobserved confounders) analogous to Rosenbaum’s sensitivity parameters, so that practitioners can trade off bias against generative flexibility in a transparent way?

## Where to read next

If you want the graphical story behind these assumptions, → [[causal-graphs]] shows how d-separation and confounding paths map to potential outcomes statements; if you want to dig into efficient semiparametric inference, → [[semiparametric-efficiency-inference]] makes explicit the influence-function calculations that underwrite the doubly robust estimator; for new modalities and structural assumptions, → [Instrumental Variables](instrumental-variables.md) describes how instruments and fuzzy compliance link to local estimands.

## Build it

Building a Doubly Robust estimator on the Lalonde dataset proves that the potential outcomes framework is practical: if at least one of the two nuisance models is correct, the system still recovers the treatment effect. The implementation forces the reader to fit a logistic regression propensity model, a linear outcome model, and then aggregate them in the DR formula, so they confront both the missing-data perspective and the weighting perspective at the same time.

**What you're building:** a minibatch Python implementation of a Doubly Robust estimator that produces the ATE for the Lalonde observational subset while reporting the standard error.

**Why this is valuable:** it makes the missing counterfactual visible by combining inverse-propensity weighting with regression error correction, directly enacting the mechanism that gives the estimator its robustness.

**Stack:**
- **Model:** logistic regression + ordinary least squares implemented with `scikit-learn 1.5.2`
- **Dataset:** [causaldata/lalonde](https://huggingface.co/datasets/causaldata/lalonde) — 1,700+ rows of treated and control units with covariates, earnings, and treatment indicators
- **Framework:** pandas 2.1.0 + scikit-learn 1.5.2 + statsmodels 0.14.0
- **Compute:** free Colab T4 (16 GB GPU, though the job runs entirely on CPU) or any laptop with 8 GB RAM; training completes in under 5 minutes

**The recipe:**
1. Run `pip install pandas scikit-learn statsmodels` in Colab and load the Lalonde dataset directly via `datasets.load_dataset("causaldata/lalonde")`.
2. Split the data into features `X` (age, education, race dummies, etc.), treatment `W`, and outcome `Y` (earnings at 1978), then standardize numeric columns and encode categorical ones to ensure the logistic model can learn stable coefficients.
3. Fit a logistic regression for \(W\) on \(X\) to get \(\hat{e}(X)\), and fit two linear regressions (\(\hat{m}_1\), \(\hat{m}_0\)) predicting \(Y\) from \(X\) on the treated and control groups separately; monitor the log-loss and RMSE curves to ensure they are within expected ranges (propensity log-loss ≲ 0.68, RMSE ≲ 2,500).
4. Compute the DR estimate using the formula in §How it works, aggregating the regression difference and the inverse-propensity weighted residuals, then bootstrap the data to derive standard errors for the treatment effect.
5. The artifact is a script that prints the point estimate, the 95% CI, and a comparison between DR, IPTW, and outcome-only regression, showing variance reduction when at least one model is well-specified.

**Expected outcome:** a runnable notebook that outputs the Lalonde ATE with bootstrap confidence intervals and reproduces the known DR estimate within ±0.5 of published values.

- **CS student:** Run the notebook on an RTX 4070 laptop, add a random forest for the propensity model using scikit-learn’s `RandomForestClassifier`, and report whether the DR estimate stays within ±1 of the logistic + linear baseline.
- **Applied engineer:** Package the notebook into a small FastAPI service that ingests new observations, runs the pretrained DR estimator, quantizes the logistic regression coefficients to float16 with ONNX Runtime, and targets a 50 ms p95 latency on an A10 instance.
- **Applied researcher:** Test the hypothesis that using a spline basis for the outcome regressions reduces bias when the treatment effect is heterogeneous; compare OBPS (outcome-based propensity scoring) to standard DR by varying the spline degrees and plotting bias vs. RMSE across folds.
- **Frontier researcher:** Probe the question from §What's still open about non-parametric bounds in Markov Decision Processes by extending the notebook to include a simple contextual bandit simulator, computing DR estimates under two different transition models, and checking whether the estimates stay within provable envelopes as the simulator switches regimes.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*