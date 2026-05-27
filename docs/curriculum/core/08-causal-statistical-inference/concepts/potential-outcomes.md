---
title: Potential outcomes
slug: potential-outcomes
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [rubin, rosenbaum, imbens, angrist, hahn]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [confounding, structural-causal-models, propensity-score, instrumental-variables]
tags: [causal-inference, potential-outcomes, doubly-robust, observational-studies, propensity-score, treatment-effect]
updated: 2025-02-06
has_mvb: true
---

# Potential outcomes

Imagine a physician who, on Monday, prescribes an antiviral pill and watches the patient’s fever break by Friday. Even with the full record of vitals, lab work, and compliance, there is a parallel universe that cannot be measured: what did that patient’s infection do on Monday in the version of the world where they only received a placebo? That counterfactual score is unobservable not because of noise or bad sensors but because the patient cannot simultaneously inhabit two histories. This “Time-Traveler’s Dilemma” is not metaphysical indulgence—it is the defining data problem of causal inference. By the end of this page you will see why the potential outcomes framework treats that missing counterfactual as a primitive, how the framework motivates doubly robust estimators and instrumental variables, and what practical recipes let you estimate average treatment effects even when treatment assignment was messy.

## The territory

Causal inference sits between randomized control trials, where we can measure every outcome under every treatment, and observational data, where each unit samples only one treatment path. The potential outcomes framework re-frames that middle ground as a structured missing-data problem: for each unit we posit two (or more) hypothetical outcomes, each representing what would have happened under a different intervention. In randomized experiments, randomization ensures that the observed outcomes are an unbiased sample of the potential outcomes for each treatment arm, so averaging them recovers the average treatment effect (ATE). In observational studies, however, treatment assignment is entangled with covariates, confounders, and compliance; the same outcome carries information about both the treatment you saw and the counterfactual you did not. This is why the propensity score, introduced in Rosenbaum and Rubin (1983) [https://www.stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf], becomes the balancing fulcrum: it summarizes assignment probabilities into a scalar that can be used to mimic randomization. The territory of potential outcomes therefore overlays two families: classical missing-data theory (prescribing how to impute the unobserved Y(1−T)) and selection bias correction (propensity-based reweighting, doubly robust estimation, instrumental variables). How does it actually work? We start by writing down the missing-data bookkeeping and then add assumptions that permit identification and estimation.

## How it works

Let \(T \in \{0,1\}\) denote the binary treatment assigned to a unit and let \(Y(1)\) and \(Y(0)\) be the potential outcomes under treatment and control, respectively. We only observe \(Y = T \cdot Y(1) + (1-T) \cdot Y(0)\) because each unit receives exactly one treatment. The individual treatment effect is \(Y(1) - Y(0)\), but it is unidentifiable without further assumptions because the data only provides either \(Y(1)\) or \(Y(0)\), never both for the same unit. The fundamental estimand is the average treatment effect
\[
\text{ATE} = \mathbb{E}[Y(1) - Y(0)],
\]
where the expectation is over the population of units under study. That is the quantity one would measure in a perfectly randomized trial. The potential outcomes framework turns the “missing counterfactual” into something we can reason about explicitly: everything we do is about recovering \(\mathbb{E}[Y(1)]\) and \(\mathbb{E}[Y(0)]\) from the observed mixture.

The first structural ingredient is the assignment mechanism. Let \(X\) denote pre-treatment covariates, and define the propensity score \(e(X) = \mathbb{P}(T=1 \mid X)\). Rosenbaum and Rubin (1983) proved that if treatment assignment is ignorable conditional on \(X\) (formally, \(T \perp (Y(0), Y(1)) \mid X\)) and every unit has a positive chance of receiving each treatment (the positivity condition \(0 < e(X) < 1\)), then \(Y(1)\) and \(Y(0)\) are independent of \(T\) after re-weighting by \(e(X)\). This justifies inverse probability weighting (IPW): the law of iterated expectations gives
\[
\mathbb{E}[Y(1)] = \mathbb{E}\left[\frac{T Y}{e(X)}\right], \qquad \mathbb{E}[Y(0)] = \mathbb{E}\left[\frac{(1-T)Y}{1-e(X)}\right],
\]
where each term takes the outcome observed under the assigned treatment, reweights it by the inverse probability of observing that treatment, and thus recovers the marginal expectation under a hypothetical randomized assignment. The propensity score collapses a high-dimensional balance problem into matching or weighting by a scalar—this is why the propensity score is called “the central role” of observational studies.

IPW, however, is notoriously sensitive to misspecification of the propensity model. Hahn (1998) [https://statweb.rutgers.edu/ztan/material/hahn98.pdf] analyzed the finite-sample bias of IPW when the model for \(e(X)\) is mis-specified, and he introduced the idea that combining outcome regression with selection models stabilizes the estimates. That leads directly to doubly robust estimation: if \(\mu_t(X)\) is a model for \(\mathbb{E}[Y \mid X, T=t]\), then the doubly robust estimator for \(\text{ATE}\) is
\[
\hat{\tau}_{\text{DR}} = \frac{1}{n} \sum_{i=1}^n \Big[ \mu_1(X_i) - \mu_0(X_i) + \frac{T_i (Y_i - \mu_1(X_i))}{\hat{e}(X_i)} - \frac{(1-T_i) (Y_i - \mu_0(X_i))}{1 - \hat{e}(X_i)} \Big].
\]
In this expression, \(\mu_t(X_i)\) adjusts the outcomes for observed covariates while the IPW residual terms correct for misspecification in the regression. The estimator is “doubly robust” because if either \(\mu_t(X)\) or \(\hat{e}(X)\) is consistent, the ATE estimate converges, significantly improving empirical stability when models are imperfect.

The potential outcomes framework also accommodates instrumental variables. Imbens and Angrist (1994) [https://scholar.harvard.edu/imbens/files/wo-stage_least_squares_estimation_of_average_causal_effects_in_models_with_variable_treatment_intensity.pdf] formalized the Local Average Treatment Effect (LATE), which identifies the treatment effect for compliers when compliance is imperfect. Let \(Z\) be an instrument that shifts treatment assignment but affects outcomes only through \(T\). With the monotonicity assumption \(T(1) \geq T(0)\), the LATE is
\[
\text{LATE} = \frac{\mathbb{E}[Y \mid Z=1] - \mathbb{E}[Y \mid Z=0]}{\mathbb{E}[T \mid Z=1] - \mathbb{E}[T \mid Z=0]},
\]
where the numerator measures the intention-to-treat contrast and the denominator rescues identification even when \((Y(1), Y(0))\) are unobserved for individuals who defy compliance. Conceptually, IV regression reverses the role of \(T\) in the structural equation and uses the instrument to project the treatment assignment onto the space orthogonal to confounders. This leads to the two-stage least squares algorithm, where the first stage predicts \(T\) from \(Z\) and covariates, and the second stage regresses \(Y\) on the fitted treatment from stage one. The potential outcomes formalism makes it clear that we are no longer estimating \(\text{ATE}\) for the whole population but the effect for the subgroup whose treatment would change if the instrument flipped—these are the compliers.

The missing-data perspective guides how we frame generalization. A key assumption is the Stable Unit Treatment Value Assumption (SUTVA), which states that each unit’s potential outcomes depend only on their own treatment and not on the treatments of others. Violations of SUTVA—interference, spillovers, network effects—turn the potential outcomes collection from a simple vector to a combinatorial explosion, so applications must justify its plausibility. When SUTVA holds, we can view the dataset as containing an “observed” indicator \(R_i = 1\) for the received treatment and “missing” indicators \(R_t = \mathbb{I}\{T=t\}\); potential outcomes for the unobserved treatments are missing data that we impute through modeling (propensity scores, regression, matching, or doubly robust methods).

The practical workflow is thus: (1) posit the potential outcomes \((Y(1), Y(0))\) and identify the estimand (ATE, ATT, LATE); (2) choose assumptions (ignorability, positivity, SUTVA, or valid instrument); (3) fit nuisance models for the propensity score \(e(X)\), the outcome models \(\mu_t(X)\), or the instrument \(Z\); (4) construct the estimator (IPW, regression adjustment, doubly robust, two-stage least squares); (5) assess why the estimator might fail (overlap violations, model misspecification, treatment effect heterogeneity). Littell and Rubin (2000) [https://ics.uci.edu/~sternh/courses/265/littlerubin_annrevepi2000.pdf] remind us that the framework is especially powerful in epidemiology because it focuses the researcher on articulating and justifying the assumptions in step (2) before harvesting estimates in steps (3)-(4). Without the potential outcomes bookkeeping, observational studies often hide these design choices.

In practice, the framework also underpins more ambitious generative counters. The recent “Robust Counterfactual Inference in Markov Decision Processes” paper (Zhang et al., 2025) [https://arxiv.org/abs/2502.13731] pushes the idea beyond static treatments by treating the sequence of potential outcomes along a trajectory as a diffusion process, allowing a generative model to sample entire counterfactual paths. In that setting, the variables \(Y_t(a)\) for each action \(a\) and timestep \(t\) are the potential transition distributions, and the estimand is a counterfactual transition probability matrix. The missing-data metaphor holds: we observe one path under one policy, but we want to infer what would have happened under a different policy without assuming a single structural model. This is the frontier that brings potential outcomes together with diffusion-based generative models: each sampled trajectory is a hypothesized complete set of potential outcomes over time, and we use the observed path plus model priors to bound the unobserved ones.

## Where the field is now

The last few years have blurred the boundary between prediction and potential outcomes estimation. The “Double Machine Learning” line of work (Chernozhukov et al., 2018) has matured into production-ready estimators that plug any black-box learner into the outcome and selection models, so the nuisance functions can be fitted with flexible forests or transformers while the treatment effect is still orthogonalized. The research frontier now is showing how generic generative models can govern the whole distribution of potential outcomes. Zhang et al. (2025) [https://arxiv.org/abs/2502.13731] take this direction in sequential decision-making, describing robust bounds on counterfactual transition probabilities when many structural causal models agree on the observational distribution. Their algorithm solves a bilevel optimization problem that leverages diffusion-based sampling to explore the space of counterfactual dynamics, providing one of the first non-parametric bounds on entire future trajectories even when the causal graph is underdetermined.

On the engineering front, Google’s measurement teams have woven potential outcomes into billions of dollars of ad products. The research.google blog entry on Causal Impact (Brodersen et al. 2015) still anchors their toolkit: experiments are modeled as potential outcomes with Bayesian structural time series for the control, and the difference between the posterior predictions and the observed outcomes yields the incremental effect. This infrastructure is deployed across YouTube and Display Ads, meaning the U.S. YouTube Ads platform computes per-campaign treatment effects daily while ingesting millions of impressions, clicks, and covariates. The engineering challenge there is stability under distribution shift—if the covariate support shrinks after an intervention, the propensity-based weights explode, so teams compensate with regularized models and truncation heuristics that are informed by Hahn’s sensitivity analyses.

The interplay between the statistical estimators and real systems is what keeps the field alive: the propensity score for balancing, IV for shifting compliance, doubly robust estimators for guarding misspecification, and generative counterfactual models for sequential decisions. The constructive tension between theory and practice is unavoidable because every production treatment effect must answer, “Which potential outcome hypothesis are we endorsing?” The field now is not just about cleaner averages but about accountable, deployable answers to that question.

## What's still open

Can we compute tight non-parametric bounds for counterfactual transitions in complex Markov Decision Processes when multiple incompatible structural causal models all perfectly fit the observed data, or does identifiability always force a combinatorial enumeration of models? That is the core question raised in Zhang et al. (2025) and it demands algorithms that balance sample efficiency with robustness to model misspecification.

Another open question is whether there exists a general theory that quantifies how much overlap (positivity) we lose when we match on high-dimensional representations learned by deep networks. The current practice is to regularize propensity-score models or to drop units with extreme scores, but no theoretical bound explains how these heuristics propagate into the bias of doubly robust estimators when the representation itself is fitted on the same data.

Finally, how do we extend the potential outcomes bookkeeping to interference settings where treatments of neighbors affect outcomes? The standard SUTVA assumption fails, yet many applications—vaccination campaigns, recommender systems, social networks—have interference baked in. The question is to identify designs or estimators that notice when SUTVA is violated (and quantify the spillover bias) rather than quietly average it away.

## Where to read next

If you want the structural story that led to the potential outcomes bookkeeping, → [Structural Causal Models](structural-causal-models.md) places directed graphs next to the same counterfactuals and explains how do-calculus produces the same estimands in simpler cases. If you prefer the historical starting point for propensity scores, → [[propensity-score]] walks through matching, weighting, and subclassification with a focus on interpretation. For the sequential decision extension, → [[reinforcement-learning-causal-inference]] explains how potential outcomes generalize to trajectories and where the current diffusions-based open problems begin.

## Build it

Estimating potential outcomes is only useful if the resulting treatment effect survives misspecification and sample variance. This build proves that with a realistic synthetic electronic health record dataset and a doubly robust estimator built from scratch in Python, you can recover both the ATE and plausible conditional potential outcomes while guarding against the usual selection biases.

**What you're building:** a doubly robust potential outcomes pipeline that estimates individual treatment effects for a synthetic EHR cohort and prints both ATE and calibrated counterfactual predictions.

**Why this is valuable:** it forces you to articulate the missing outcome for every patient, to model the noise through both propensity and regression learners, and to inspect what happens when one of these components fails.

**Stack:**
- **Model:** custom pipeline combining `LogisticRegression` for the propensity model and `GradientBoostingRegressor` for the outcome models (both from scikit-learn) with statsmodels-based inference on the doubly robust estimator.
- **Dataset:** synthetic EHR cohort generated in the notebook via `sklearn.datasets.make_classification`, augmented with treatment assignment based on risk scores (no external download required).
- **Framework:** Python 3.11 with `scikit-learn==1.3.2`, `statsmodels==0.14.0`, `pandas`, `numpy`, and `matplotlib`.
- **Compute:** free Colab T4 (16 GB RAM) — the training loop runs in under 30 minutes with 1e5 synthetic samples.

**The recipe:**
1. Install the stack with `pip install scikit-learn==1.3.2 statsmodels==0.14.0 pandas numpy matplotlib` and set a random seed for reproducibility.
2. Generate the synthetic EHR: sample 100k patients with covariates drawn from mixtures of Gaussians, define a logistic treatment mechanism (propensity) using a known coefficient vector, and sample outcomes whose noise variances differ by treatment.
3. Fit the nuisance models in cross-fitting folds: (a) train a logistic regression to estimate \(\hat{e}(X)\) for each fold, (b) train two gradient boosting regressors \(\hat{\mu}_0(X)\), \(\hat{\mu}_1(X)\) for control and treatment outcomes, and save out-of-fold predictions.
4. Construct the doubly robust estimator: plug the nuisance outputs into the DR formula, compute \(\hat{\tau}_{\text{DR}}\) as well as sample-level estimates \(\hat{Y}_i(1)\) and \(\hat{Y}_i(0)\) for each patient, and compute a standard error via the asymptotic variance from the influence function derived in statsmodels.
5. Evaluate: report the ATE estimate with 95% confidence interval, compute the mean squared error between \(\hat{Y}_i(t)\) and the simulated ground-truth potential outcomes (available from the synthetic generation), and plot the distribution of estimated individual treatment effects.

**Expected outcome:** a notebook-ready pipeline that prints an ATE within ±0.05 of the known truth, shows the distribution of individual conditional treatment effects, and saves a plot validating coverage.

- **CS student:** run the same pipeline on an RTX 4070 with cross-fitting expanded to 10 folds and measure how ATE error shrinks as you increase the number of trees in the gradient booster; report the MSE vs. tree depth.
- **Applied engineer:** wrap the trained nuisance models into a FastAPI endpoint that, given a batch of covariates, returns both propensity scores and doubly robust potential outcome predictions; deploy this on an A10 with quantized boosters (float16) and log latency to ensure p95 < 60 ms.
- **Applied researcher:** test the hypothesis that using gradient boosting for \(\mu_t(X)\) but linear regression for \(\hat{e}(X)\) is superior to the opposite configuration; document the hypothesis, record whether the doubly robust ATE changes by more than 0.02, and plot the residuals of each nuisance model.
- **Frontier researcher:** use the same synthetic cohort to probe the open problem from §What's still open: generate two causal graphs that agree on observational marginals but differ in their structural equations, and use diffusion-based counterfactual sampling (inspired by Zhang et al. 2025) to bound \(\mathbb{E}[Y(1)]\) under each graph, reporting whether the bounds overlap.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*