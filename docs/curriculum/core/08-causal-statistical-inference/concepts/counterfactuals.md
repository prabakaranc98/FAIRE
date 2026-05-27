---
title: Counterfactuals
slug: counterfactuals
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [pearl, halpern, johansson, pawlowski]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [structural-causal-models, do-calculus, potential-outcomes]
tags: [counterfactuals, structural-causal-models, abduction-action-prediction, deep-scm, causal-ads]
updated: 2025-04-10
has_mvb: true
---

# Counterfactuals

Here is the daily puzzle of online advertising: a person scrolls past your sponsored post, catches a second glimpse on the same page, clicks through, and buys the product. Your A/B test shows that exposure to the ad raises conversions by 2 percentage points, yet the business question the campaign owner cares about is quieter and harder: “Did this ad cause *that* purchase, or would this person have bought anyway?” Randomized experiments and observational averages give you the lift for the whole cohort but none of the story for that one user. Counterfactual reasoning inserts a missing dimension: an alternative reality for the same individual where the exposure was different, and it asks whether the outcome of interest changes. The rest of this page explains how structural causal models reconstruct those alternative realities, why deep representations and flows are now the engineering hooks, and how you can go from scratch data to a Deep SCM that executes Pearl’s Abduction-Action-Prediction loop on a free Colab notebook.

## The territory

Causal inference has long split into two territories. Randomized experiments and the do-calculus target population-level responses—“Does the treatment shift the mean outcome?”—and potential outcomes focus on contrasts between treated and control units. Counterfactuals live at their intersection but with a unique twist: they insist on a joint distribution over factual and hypothetical outcomes for the same individual. That insistence demands more than marginal statistics; it demands a structural story about how variables were generated and how the latent noise terms turned out for the observed case. Pearl (2009) [https://ftp.cs.ucla.edu/pub/stat_ser/r355-corrected-reprint.pdf] formalized this demand in the Abduction-Action-Prediction framework, showing that once you know the mechanisms and your priors over exogenous noise, the counterfactual answer is a probabilistic computation conditioned on the specific observational history.

Structural models give that machinery its bones. Each node is a deterministic function of its parents and a unique noise draw, so once you see a patient’s record you can infer the noise vector that must have produced it—this is the role of abduction. Halpern and Pearl (2005) [https://www.cs.cornell.edu/home/halpern/papers/actcaus.pdf] stress that the semantics of “cause” become crisp only in such fully specified models: each move away from the observed data is a contemplated change at the level of the structural equations. That is what “counterfactuals are semantic objects” means in Structural Counterfactuals: A Brief Introduction (???). The structural details give counterfactuals their leverage, and the rest of the page transitions to how those structures are specified, estimated, and used in practice.

## How it works

The mechansim behind counterfactuals is best understood as a synthesis of a structural causal graph, a factual history, and the three stages of Pearl’s counterfactual algorithm: abduction (learn the unobserved noises), action (replace the mechanism for the intervention), and prediction (roll the modified model forward). The deeper we dig, the more the math becomes an exercise in manipulating deterministic functions and Gaussian disturbances.

### Structural equations and the Abduction-Action-Prediction loop

Take a structural causal model \(M = (U, V, F)\), where \(V = \{V_1, \dots, V_n\}\) are the endogenous variables we observe, \(U = \{U_1, \dots, U_n\}\) are exogenous noise terms, and \(F = \{f_1, \dots, f_n\}\) are deterministic functions. The generative process is
\[
V_i := f_i(Pa_i, U_i)
\]
where \(Pa_i \subset V \setminus \{V_i\}\) denotes the parent set of \(V_i\) in the causal DAG, \(U_i\) is the exogenous noise for node \(i\), and each \(f_i\) is parameterized by structural parameters (which might come from domain knowledge or a learned neural net). The noises \(U_i\) are jointly independent under the SCM prior. With a fully specified model \(M\), the factual outcome for a unit is computed by sampling \(U \sim P(U)\) and computing \(V\) through the recursive structural equations; abduction, the first step of the counterfactual query, reverses that: given the observed factual data \(v^*\), we infer the posterior \(P(U \mid V {=} v^*)\). That posterior tightens the possible latent configurations to those consistent with the observation.

Abduction is usually intractable analytically, so modern practice replaces the exact posterior with amortized inference networks—a mapping \(g_\phi(v^*)\) that outputs a summary of the noise (e.g., mean and scale for each \(U_i\)). When the noises are Gaussian and the structural functions are differentiable, we can treat \(g_\phi\) as part of a variational encoder, and the loss includes a KL term back to the prior \(P(U)\). Once abduction gives us \(\hat{U}\), we proceed to action: say we want to counterfactually set a treatment node \(T\) to \(t'\). We replace the structural equation \(f_T\) with the constant \(T := t'\) and keep the abduced \(\hat{U}\) for every other node, generating a new outcome \(Y_{t'}\). Prediction computes the distribution of \(Y_{t'}\) while holding the other functions fixed, so the counterfactual depends on both the structural mechanisms and the inferred tail of the noise.

This entire calculus is validated in “Structural Counterfactuals: A Brief Introduction” [https://ftp.cs.ucla.edu/pub/stat_ser/r413-reprint.pdf], which walks through the semantics of how a unique exogenous noise assignment yields a unique counterfactual pair once action is applied.

### Building the structural model with deep components

If each \(f_i\) is a small tabular function, this framework stays transparent, but real-world counterfactuals involve high-dimensional covariates, complex heterogeneities, and unobserved confounders. Deep Structural Causal Models (Pawlowski et al. 2020) [https://arxiv.org/abs/2006.06826] bring neural nets into \(f_i\) and use normalizing flows to match the expressive posterior and prior over the noise terms. The key trick is to rewrite each structural equation as a latent-to-observed mapping \(V_i = f_i(Pa_i, U_i; \theta_i)\) where \(\theta_i\) parameterizes a neural network and \(U_i \sim \mathcal{N}(0, I)\). The flows now model \(P(U \mid V)\), enabling flexible abduction. For example, the structural model for a trial with treatment \(T\), covariates \(X\), and outcome \(Y\) becomes:
\[
\begin{aligned}
X &:= f_X(U_X), \\
T &:= f_T(X, U_T; \theta_T), \\
Y &:= f_Y(T, X, U_Y; \theta_Y).
\end{aligned}
\]
Here \(f_X\) might simply copy its noise if \(X\) is assumed exogenous; \(f_T\) is the policy network producing treatment probabilities; \(f_Y\) is the outcome network. Each noise \(U_i\) can be modeled by a flow \(h_{\psi_i}\) producing a density \(p_{\psi_i}(U_i)\) and trained by maximizing variational lower bounds. The entire model is optimized end-to-end: the abduction encoder takes an observation \((x, t, y)\), maps it to a latent \(u = g_\phi(x, t, y)\), and a reconstruction loss ensures \(f_Y(t, x, u_Y)\) matches \(y\).

The representation learning insight of Johansson et al. (2016) [https://arxiv.org/abs/1605.03661] augments this pipeline. Selection bias—differences in the covariate distribution between treated and control groups—damages abduction because the encoder sees different \(x\) distributions during training and inference. The solution is to learn a representation \(r = h_\eta(x)\) that is predictive of the outcome but indistinguishable between treatment groups via a domain-adversarial objective. The learned representation \(r\) then feeds into the structural equations: \(T := f_T(r, U_T)\), \(Y := f_Y(r, T, U_Y)\). This adversarial regularizer ensures the treatment assignment is “balanced” in representation space, allowing abduction to generalize to unseen combinations.

### Counterfactual query execution

With the structural equations and representation components in place, the sequence to evaluate a query “what is \(Y_{t'}\) given the observed history \(o = (x, t, y)\)?” is:

1. **Abduction:** Encode \(o\) via \(g_\phi(x, t, y)\) to obtain latent noise \(\hat{U} = (\hat{u}_X, \hat{u}_T, \hat{u}_Y)\). The variational loss encourages \(\hat{U}\) to match a prior \(p(U)\) and to reconstruct the observed \(y\).

2. **Action:** Replace the structural equation for the treatment with the hypothetical \(t'\), yielding \(T := t'\) while keeping \((\hat{u}_X, \hat{u}_Y)\) fixed and recomputing downstream nodes.

3. **Prediction:** Sample \(Y_{t'} = f_Y(t', x, \hat{u}_Y)\). Optionally average over multiple samples to approximate \(P(Y_{t'} \mid o)\).

If one also needs the individual treatment effect, compute \(\hat{Y}_1 - \hat{Y}_0\) using the two actions \(t' = 1\) and \(t' = 0\). The expectation and variance of this difference are the pointwise analogs of the average treatment effect.

### From theory to inference on health data

Counterfactual reasoning surfaces in practice when, for example, a physician wants to know whether a treatment caused remission for a patient. Observations might include patient history \(X\), whether the physician prescribed medication \(T\), and the recovered outcome \(Y\). Halpern and Pearl (2005) [https://www.cs.cornell.edu/home/halpern/papers/actcaus.pdf] emphasize that competing explanations correspond to different assignments of the exogenous noise vector \(\mathbf{U}\). The Deep SCM pipeline described above operationalizes this by (a) training inference networks on large observational cohorts to capture the posterior over \(\mathbf{U}\), (b) enabling intervention by rewiring \(f_T\), and (c) sampling the downstream outcome. In this way, a Deep SCM becomes a simulator of a counterfactual patient, letting practitioners test “if not for the drug…” scenarios in a fully probabilistic way.

### Failure modes and robustness

The key fragility of counterfactuals is identifiability. If the structural equations are under-specified or the latent space is not expressive enough, the abduction step might infer the wrong noise. Johansson et al. (2016) [https://arxiv.org/abs/1605.03661] argues that balancing representations mitigates bias but does not create identifiability when hidden confounders remain. Pawlowski et al. (2020) demonstrate that flows help but rely on architectural choices; the deeper the network for \(f_Y\), the more care is needed to avoid overfitting to the factual regime. Monitoring concordance between the predicted factual outcome and the observed \(y\) during training, along with counterfactual consistency checks (e.g., ensure \(Y_{t} \approx Y\) when \(t = T\)), is essential to detect posterior collapse or degeneracies.

## Where the field is now

Deep counterfactual inference currently balances two frontiers. The research frontier explores how to make counterfactual distributions identifiable in the presence of latent confounders and non-linear mechanisms. Pawlowski et al. (2020) [https://arxiv.org/abs/2006.06826] extend deep generative models by coupling conditional normalizing flows with the structural equations, training via joint likelihood maximization so that the resulting latent codes align with the true exogenous noise while still supporting high-dimensional observations such as images or electronic health records. A later line of research (e.g., CausalGPT 2023) introduces attention-based structural models that allow the mechanisms \(f_i\) to condition on subsets of variables dynamically, and ongoing benchmarks now compute policy value on synthetic twins data to compare identifiability across architectures.

The engineering frontier is about deploying counterfactual estimators under operational constraints. For example, large ad-tech teams measure lift via instrumentation that simulates counterfactual exposures; Google’s causal-impact library and its downstream productionization in Google Ads now routinely compute incremental revenue by fitting Bayesian structural time-series models that resemble the same abduction-action-prediction loop but with time-varying covariates and scalability constraints. These systems ingest billions of events per hour, isolate the latent noise for each ad impression via approximate Bayesian filtering, and recompute counterfactual revenue when budgets change. When ad platforms detect that a given impression chain aligns poorly with historical noise, they throttle the action path, which is essentially an engineering-grade rejection of counterfactual outcomes that would otherwise overpromise lift. Both research and production share a common measurement rubric: train structural models, enforce identifiability through architecture or instrumentation, and evaluate counterfactual samples against synthetic twins where the ground truth is known.

## What's still open

1. **How can deep latent variable SCMs guarantee identifiability when all confounders are latent and the mechanisms are highly non-linear?** Current flow-based approximations assume invertibility or Gaussian priors, but real-world mechanisms rarely satisfy those constraints, so a researcher could target identifiable families of flows or priors that admit closed-form inversion in the abduction step.

2. **Can counterfactual representation learning be extended to reinforcement learning settings where actions are not binary but high-dimensional and sequential?** The domain-adversarial objective in Johansson et al. (2016) relies on static treatment indicators; building analogous objectives for trajectory-level counterfactuals would enable policy evaluation with much richer action spaces.

3. **What statistical tests reliably detect when a counterfactual query leaves the support of the observational distribution?** Production systems often face “off-support” actions (e.g., showing an ad at a time never seen before), and we currently lack principled diagnostics linking support violations to posterior degradation without relying on held-out experiments.

4. **Is there a practical integration between Pearl’s structural frameworks and the potential-outcomes literature’s high-dimensional machine learning estimators that yields sharper uncertainty quantification for unit-level counterfactuals?** The combination would allow uncertainty estimates to be calibrated both by structural confidence (does the mechanism know this regime?) and by statistical error (does the estimator have low variance under the observed data?).

## Where to read next

If you want to drill into the causal diagrams that underpin all of this, → [Structural Causal Models](structural-causal-models.md) lays out the notation for DAGs, structural equations, and confounding; if you care about deriving interventional quantities without full counterfactuals, → [Do-calculus](do-calculus.md) explains the rules that let you rewrite \(P(Y \mid do(T))\); and if you prefer the Neyman–Rubin perspective on individual effects, → [[potential-outcomes]] contrasts that notation with Pearl’s and shows how the two schools of thought can be translated into the same algorithms.

## Build it

We now build a Deep Structural Causal Model from scratch so you can execute abduction, action, prediction on a synthetic medical cohort and inspect counterfactual samples. This recipe proves that the Abduction-Action-Prediction loop is not just algebra but code: you will infer noise posteriors with Pyro, reshape the treatment mechanism, and observe how the predicted outcome changes for the same patient under different actions.

**What you're building:** A Pyro-based Deep Structural Causal Model that simulates a medical treatment dataset, learns noise encodings, and answers counterfactual queries for a single patient.

**Why this is valuable:** Because it forces you to implement and debug every stage of Pearl’s framework—abduction via amortized inference, action as a do-operation, and prediction as sampling from the altered model—which is the conceptual leap from correlation to individualized attribution.

**Stack:**
- **Model:** Custom Deep SCM (Pyro + normalizing flows) — built in the notebook.
- **Dataset:** Synthesized medical cohort generated within the notebook using NumPy (mimics twins-style data and is governed by a known SCM).
- **Framework:** Pyro 2.0 + PyTorch 2.1 on Google Colab (with the `pyro` pip package).
- **Compute:** Free Colab T4 (16 GB RAM, 16 GB GPU memory); training finishes in ~15 minutes per run.

**The recipe:**
1. `pip install pyro-ppl==2.0.3 matplotlib torch==2.1.0 numpy pandas scikit-learn` and import Pyro, Torch, NumPy, Pandas, Matplotlib.
2. Generate the synthetic cohort by sampling noise \(U_X, U_T, U_Y \sim \mathcal{N}(0, I)\), computing covariates \(X = f_X(U_X)\), treatment logits \(p(T{=}1 \mid X) = \text{sigmoid}(w_X^T X + U_T)\), sampling \(T\), and outcomes \(Y = f_Y(X, T, U_Y)\); store the data in a PyTorch `TensorDataset`.
3. Define the Deep SCM with encoder `q(U | X, T, Y)` implemented as a normalizing flow that outputs mean/scale for each \(U_i\), and decoder functions \(f_T\) and \(f_Y\) parameterized by small MLPs; train with a loss combining reconstruction likelihood \(\log p(Y \mid f_Y(\cdot))\) and KL divergence between `q(U)` and the prior, monitoring that the observed \(Y\) is recreated for 95% of minibatches.
4. Evaluate by running the learned encoder on a held-out patient, performing action with \(T=1\) and \(T=0\), sampling \(Y_1\) and \(Y_0\), and comparing the difference \(\hat{Y}_1 - \hat{Y}_0\) to the known synthetic ground truth effect (expect RMSE < 0.1 due to the controlled generative process).
5. What you now have: an artifact that, given a factual patient history, produces counterfactual outcome distributions that you can interrogate, visualize, and report as evidence for or against treatment attribution.

**Expected outcome:** A Colab notebook that trains the Deep SCM and outputs counterfactual samples for a specific patient, together with plots of the posterior noise and the individual treatment effect.

- **CS student:** Reduce the hidden dimensions and use a single flow layer so the notebook runs in <5 minutes on an RTX 4070 while still showing the abduction-action-prediction plot.
- **Applied engineer:** Wrap the trained model in TorchScript, quantize the flow encoder, and demonstrate serving predictions at p50 latency < 40 ms on a local A10-like machine while logging counterfactual metrics.
- **Applied researcher:** Hypothesize that the adversarial representation in Johansson et al. (2016) reduces bias—add the domain discriminator, ablate it, and compare the bias (difference between \(\mathbb{E}[Y_1 - Y_0]\) estimated from treated vs. control) in the ablated vs. full model.
- **Frontier researcher:** Probe the open question of identifiability by replacing the Gaussian prior with a mixture of Gaussians for the noise variables and measure whether the counterfactual RMSE on synthetic twins decreases while keeping KL regularization stable.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*