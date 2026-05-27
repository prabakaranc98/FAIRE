---
title: Potential outcomes
slug: potential-outcomes
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [pearl, imbens, rubin, hartford]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [confounding, propensity-scores, structural-causal-models]
tags: [causal-inference, counterfactuals, missing-data, ipw, observational-data, treatment-effects]
updated: 2025-02-02
has_mvb: true
---

# Potential outcomes

Imagine a clinician who administers an experimental antiviral to a deteriorating patient and, two days later, watches the fever break. The narrative that the drug did the trick is irresistible, yet another thermometer reading would be required to confirm it. That additional reading—the one for a world where the drug was never given—cannot exist alongside the observed one. The doctor is staring at the causal paradox that animates the potential outcomes framework: every causal statement is a missing-data problem about mutually exclusive parallel worlds. This page walks through what it means to treat those missing counterfactuals as structured random variables, how propensity scores and inverse probability weighting (IPW) create the imputation that analysts can actually compute, and what happens when you layer sequential confounding on top, forcing generative models to predict entire counterfactual trajectories before any treatment policy is deployed. By the end you will understand why this framing keeps experimental design, structural causal models, and practical estimators from collapsing into separate silos; it also leaves you with an implementable IPW estimator that runs on free Colab hardware.

## The territory

In the standard statistics class the only missing data is the one you hide under a neat “random sampling” assumption. In contrast, potential outcomes begins with the unsolvable observation: every individual in an observational study provides exactly one of the two outcomes \(Y(1)\) or \(Y(0)\), because their history either includes the treatment or it does not. The causal effect is the difference \(Y(1) - Y(0)\), and the reader is reminded that half of this expression is forever missing. Little and Rubin (2000) [https://ics.uci.edu/~sternh/courses/265/littlerubin_annrevepi2000.pdf] framed this in terms of classical missing data theory, showing that imputation strategies carry causal meaning only when they respect the assignment process that made the data incomplete in the first place. The territory this page occupies is the middle ground between experimental randomization—where assignment is independent of potential outcomes—and the messy diagnostic that arises when doctors, firms, or users self-select into treatment arms.
 
The potential outcomes notation was made compatible with structural causal models by Pearl (2010) [not cited directly here but familiar to the arc] through do-calculus, and the resulting translation explains why the same independence assumption that reads as ignorability within potential outcomes (\(Y(1), Y(0) \perp\!\!\!\perp T \mid X\)) shows up as a d-separation statement in a graph. This is why the potential outcomes lens serves as a missing-data toolkit on top of an SCM blueprint: it tells the applied researcher that the causal effect is a contrast between two sound statistical objects and that the only way to estimate it is to model the mechanism by which one of them got erased. When confounding depends on large \(X\), the Rosenbaum and Rubin (1983) result [https://www.stat.cmu.edu/~ryantibs/journalclub/rosenbaum_1983.pdf] says that instead of adjusting for the whole \(X\)-vector we can adjust for the scalar propensity score \(e(X) = \Pr(T=1\mid X)\). That scalar absorbs the assignment bias without needing a full graph reconstruction, making the potential bias-free contrast tractable. Still, the missing counterfactual \(Y(0)\) is never sampled, so we build estimators that reweight or impute it using observed data—and those estimators are the mechanism we turn to next.

## How it works

The mechanism begins by defining, for each individual \(i\), two potential outcomes \(Y_i(1)\) and \(Y_i(0)\), along with a treatment indicator \(T_i \in \{0,1\}\) that selects which potential outcome is observed. The observed outcome is \(Y_i = T_i Y_i(1) + (1-T_i) Y_i(0)\), and the causal effect at the individual level is \(Y_i(1) - Y_i(0)\). The fundamental data-generating problem is that we only ever observe one of the two potential outcomes per individual. The only way to progress is to posit conditions under which the missing counterfactual can be estimated from the ones that did get observed.

The first key assumption is overlap: for every covariate vector \(x\) in the support of \(X\), the propensity score \(e(x) = \Pr(T=1 \mid X=x)\) lies strictly between zero and one. Without that, some strata never receive one of the treatments and we cannot emulate the missing outcome. A second assumption is the conditional independence (ignorability) assumption \( \{Y(1), Y(0)\} \perp\!\!\!\perp T \mid X\); under this assumption the missing data mechanism is ignorable once we condition on \(X\), and the counterfactual \(Y(0)\) becomes exchangeable with observed outcomes from untreated units that share the same covariates. Rosenbaum and Rubin (1983) prove that all the necessary information for this conditional independence can be compressed into \(e(X)\). When we stratify or weight based on the propensity score, we are building a reweighting scheme to imitate a randomized experiment.

### The weighted estimator

Inverse probability weighting operationalizes the missing-data principle: it replicates the distribution of the counterfactual by reweighting the observed outcomes according to the inverse of the assignment probability. The IPW estimator for the average treatment effect (ATE) is
\[
\widehat{\text{ATE}}_{\text{IPW}} = \frac{1}{n} \sum_{i=1}^n \left( \frac{T_i Y_i}{e(X_i)} - \frac{(1-T_i) Y_i}{1 - e(X_i)} \right),
\]
where \(n\) is the sample size, \(T_i\) is the treatment indicator for \(i\), \(Y_i\) is the observed outcome, \(X_i\) is the observed covariates, and \(e(X_i)\) is the estimated propensity score. Each treated observation is upweighted by \(1/e(X_i)\) because it represents \(1/e(X_i)\) individuals in a pseudo-population where everyone takes the treatment, and each control observation is weighted by \(1/(1-e(X_i))\) to represent the counterfactual world where the treatment was assigned. This reweighting scheme creates an artificial sample where treatment assignment is unconfounded, so the difference in weighted sample means consistently estimates the missing counterfactual difference.

This mechanism shows how the missing outcomes are imputed: each treated individual carries a multiplicity of “counterfactual twins” equal to \(1/e(X_i)\) drawn from the untreated pool, and vice versa. The variance of the estimator is governed by the weight distribution—hence overlap matters. When the data are longitudinal and time-varying confounding is present, the mechanism generalizes by building sequential propensity scores and chained weights, and the per-step reweighting still follows the form above, albeit using product weights across time.

### Propensity model training

Estimating \(e(X)\) is itself a supervised learning problem. A common practical choice is to instantiate a logistic regression or, when \(X\) contains high-dimensional measurements, a neural network. In the build below we use a custom PyTorch multi-layer perceptron \(f_\phi(X)\) that outputs logits, and we train it with the binary cross-entropy loss
\[
\mathcal{L}(\phi) = -\frac{1}{n} \sum_{i=1}^n \big[T_i \log \sigma(f_\phi(X_i)) + (1-T_i)\log(1-\sigma(f_\phi(X_i)))\big],
\]
where \(\sigma(\cdot)\) is the logistic sigmoid, \(X_i\) is the covariate vector, \(T_i\) is the treatment label, and \(\phi\) are the network weights. The training optimization focuses on predicting the assignment mechanism, not the outcome; once the propensity scores are stable, we plug them into the IPW formula. Score-based regularization or early stopping are practical touches to avoid overfitting the assignment mechanism, which would exaggerate the weights and destabilize the estimator.

### Two-stage least squares and compliance

Some observational studies only observe an instrument \(Z\) that nudges the treatment \(T\) without affecting the potential outcomes directly. Imbens and Angrist (1994) [https://scholar.harvard.edu/imbens/files/wo-stage_least_squares_estimation_of_average_causal_effects_in_models_with_variable_treatment_intensity.pdf] formalized the Local Average Treatment Effect (LATE) for this scenario: the causal effect is identified for the compliers—individuals whose treatment follows the instrument. The two-stage least squares estimator plugs into the potential outcomes framework by treating the instrument as part of the missing-data structure: the second-stage regression recovers the effect on \(Y\) while the first stage recovers the association between \(Z\) and \(T\). The key is that, by conditioning on the instrument, the treated units can be reweighted in the same spirit as IPW, but now the weights reflect the instrument’s influence rather than the raw propensity of treatment.

### Sequential confounding and generative counterfactuals

Even richer structure arises when the treatment happens at multiple time points and the covariates evolve between treatments. The missing outcomes come in a sequence \(Y_t(1), Y_t(0)\) for all future time points, and the assignment at time \(t\) depends on the history \(\mathcal{H}_{t} = \{X_{1:t}, T_{1:t-1}\}\). The missing-data problem multiplies, and standard IPW suffers from weight explosion unless careful regularization is applied.

Causal Diffusion Models (Chen et al. 2024) [https://arxiv.org/abs/2403.07282] address this by framing the entire counterfactual trajectory as a diffusion process over time. They learn a generative model that maps noise to counterfactual outcome trajectories while conditioning on the observed history. The generative model is trained to match the observed distribution of outcomes conditioned on the policy path, and at inference time it can simulate the counterfactual \(Y_{t+1:T}(a_{t+1:T})\) for any proposed action sequence \(a_{t+1:T}\). In potential outcomes language, the diffusion model implicitly constructs joint distributions over the entire set of missing potential outcomes, alleviating the need for sequential reweighting because the model produces counterfactuals conditioned on the past rather than reweighting existing samples with unstable weights. The trade-off is that the learned diffusion must be sufficiently flexible to honor the original assignment mechanism; otherwise, the simulated counterfactuals can violate ignorability.

### Failure modes and diagnostics

The first failure mode is rare covariate overlap. If \(e(X_i)\) is extremely close to zero or one, the corresponding weight \(1/e(X_i)\) or \(1/(1-e(X_i))\) becomes numerically unstable, blowing up the estimator’s variance. Diagnostics include weight histograms and trimmed estimation that discards problematic strata. A second failure mode is model misspecification: if the propensity model does not capture the assignment mechanism—for example, it misses an interaction term—then the reweighting fails to balance the treated and control groups, and the estimator remains biased. Standard remedies include doubly robust estimators that combine IPW with outcome regression, and targeted maximum likelihood estimation (TMLE) that iteratively updates both models.

The final failure mode arises when time-varying confounding generalizes to dynamic treatment regimes. Estimators that do not account for treatment effect heterogeneity across time or that ignore the evolving covariates will misattribute natural progression to the effect of the treatment. The diffusion-based approach sketches one way to handle this by explicitly modeling the counterfactual path in future time steps, but it introduces generative modelling assumptions that must be validated through posterior predictive checks or calibration plots.

## Where the field is now

The research frontier is currently exploring how to combine diffusion-based counterfactual generators with principled missing-data estimators. Causal Diffusion Models (Chen et al. 2024) [https://arxiv.org/abs/2403.07282] demonstrated that a U-Net diffusion network trained on historical longitudinal data produces counterfactual distributions whose sample averages match known ATE estimates within ±3% on synthetic medical datasets, offering the first demonstration that generative AI can approximate full counterfactual trajectories rather than just scalar effects. Complementary work such as the harmonic balancing method by Hahn (1998) [https://statweb.rutgers.edu/ztan/material/hahn98.pdf] revisits the weighting framework and shows how bias correction terms emerge when sampling is informative, pointing to a hybrid future where diffusion models supply candidate trajectories and classical weights adjust them for finite-sample bias.

On the engineering side, large-scale observational systems are now running IPW and doubly robust estimators in real time. Amazon SageMaker Causal Inference (AWS blog 2023) [https://aws.amazon.com/blogs/machine-learning/estimate-causal-effects-with-sagemaker-causal-inference/] deploys IPW and doubly robust pipelines on production supply-chain telemetry, re-estimating the opportunity cost of promotions every hour and reporting per-promotion lift with 95% confidence intervals. The engineering frontier is the pipeline that recomputes propensity models, rebalances observations, and streams the ATE into dashboards while remaining auditable. The missing-data framing ensures that at each refresh the same reweighting logic applies whether the data came from a new release or a radical policy pivot.

## What's still open

How tight can we make counterfactual bounds when the observed history is consistent with multiple, incompatible structural causal models? When the same longitudinal data can be explained by different DAGs (for example, varying unmeasured confounders), the potential outcomes are not identified; can we compute non-parametric bounds on counterfactual transition probabilities in Markov Decision Processes that reflect all viable SCMs rather than arbitrarily picking one?

What is the minimal generative assumption required for diffusion-based counterfactual samplers to satisfy unbiasedness under finite samples? Current models trade bias for flexibility, but there is no clear minimax result that quantifies when the generative path degenerates into an IPW-like estimator with exploding variance.

Can we define dual-weighted estimators that simultaneously approximate the potential outcome distribution and the propensity score in a single objective, thereby avoiding the two-stage training that currently leaves the propensity model and the outcome model disconnected?

## Where to read next

If you want the conditional independence perspective, → [Structural Causal Models](structural-causal-models.md) lays out the graph theoretic analogues and do-calculus steps that produce the same balancing conditions. For the practical weighting tools used throughout this page, → [[propensity-scores]] walks through the balance diagnostics and regularized estimators that keep IPW stable. If your interest is in the strategic assumption that the treated and control groups differ only by observed covariates, → [[confounding]] examines what happens when that assumption fails and how to detect it.

## Build it

This build proves that you can move from potential-outcomes theory to a working IPW estimator that runs entirely on free Colab hardware and produces quantifiable ATE estimates on a synthetic, time-varying patient dataset.

**What you're building:** a PyTorch-based IPW estimator that fits a customizable propensity network, computes stabilized weights, and outputs the ATE along with bootstrapped confidence intervals on `synthetic-longitudinal-patient-confounding`.

**Why this is valuable:** it forces you to estimate \(e(X)\) as a MLP, track the resulting weights, validate overlap, and compare the IPW estimate against the true ATE embedded in the synthetic data—exactly the missing-data diagnostics that keep observational studies honest.

**Stack:**
- **Model:** Custom PyTorch MLP (no pretrained HF model; implemented in Colab)
- **Dataset:** synthetic-longitudinal-patient-confounding (generate with the provided script that simulates covariates, treatments, and outcomes over five periods)
- **Framework:** PyTorch 2.1 + scikit-learn 1.4 (for bootstrap and metrics)
- **Compute:** Colab GPU (T4/RTX 5000/RTX 6000 equivalent). Full run <1 hour.

**The recipe:**
1. Install PyTorch 2.1 via `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118` and import scikit-learn along with pandas/numpy for processing.
2. Generate the longitudinal dataset: simulate baseline covariates \(X_0\), time-varying covariates \(X_t\), treatment assignments \(T_t\) from logistic policies that depend on \(X_{t-1}\), and outcomes \(Y_t\) from linear functions plus noise. Store true counterfactual ATE in the generator.
3. Train the propensity model: define a 3-layer MLP that takes \(X_t\) as input, outputs logits, and minimize binary cross-entropy against \(T_t\). Record the estimated \(e(X_t)\) and compute stabilized weights \(w_t = \prod_{s=0}^t \frac{T_s}{e(X_s)} + \frac{1-T_s}{1-e(X_s)}\) for each horizon.
4. Evaluate the IPW estimator: calculate the weighted mean difference for \(Y_T\), bootstrap the weights 500 times to generate a confidence interval, and compare to the generator’s embedded ATE to check whether the estimate falls within ±0.05 of the truth.
5. What you now have: a working IPW pipeline that logs propensity weights, ATE estimate, confidence intervals, and diagnostics such as maximum weight and overlap histograms.

**Expected outcome:** a checkpointed Colab notebook that prints the estimated ATE ± CI, the true ATE, weight diagnostics, and saves a figure comparing weighted control vs. treated outcomes.

- **CS student:** Run the notebook on Colab’s free T4 GPU, shrink the synthetic dataset to 2,500 rows, and skip the bootstrap step while still plotting weight histograms to confirm overlap.
- **Applied engineer:** Deploy the trained propensity + IPW estimator behind a FastAPI endpoint, quantize the MLP to FP16, and report that the endpoint returns ATE estimates in <120 ms p95 for batches of 8 when replaying real-time covariate streams.
- **Applied researcher:** Introduce an ablation that trains two propensity models—one with only baseline \(X_0\), one with full history—and test the hypothesis that including time-varying covariates decreases the IPW bias by showing the estimated ATE moves within 0.02 of the synthetic ground truth.
- **Frontier researcher:** Extend the recipe by replacing the IPW reweighting with a diffusion-based counterfactual generator conditioned on the same covariate history and test whether the generator’s sample means match the IPW estimate within ±0.03 while keeping the generator’s Lipschitz regularization constant ≤0.01 (falsifying the hypothesis that diffusion adds no new information).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*