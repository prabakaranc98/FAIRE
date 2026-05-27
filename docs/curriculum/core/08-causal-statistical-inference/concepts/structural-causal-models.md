---
title: Structural Causal Models
slug: structural-causal-models
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [pearl, shalit]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [probability, directed-graphs, statistics]
tags: [causal-inference, do-calculus, counterfactuals]
updated: 2024-11-14
has_mvb: true
---

# Structural Causal Models

When a new transit line opens in your city, you want to know how it changes commute times, not just whether it came with a marketing campaign. The core problem SCMs answer is exactly that: they let you trade the statistical correlation between the new line and commute time for a mechanistic story about how people choose trains, buses, or cars, and what would happen if you closed the line tomorrow. By the end of this page you will understand what ingredients go into an SCM, why they let you compute interventions and counterfactuals where plain observational models fail, how contemporary tools let you turn data into answers you can act on, and what concrete build you can ship today to demonstrate the machinery.

## The territory

Observational data is full of confounding: the people who ride the new line might also live near downtown, have better jobs, and would have short commutes regardless. Purely statistical models only see the joint distribution of features and outcomes, so they cannot reliably answer “what if we rerouted the line?” Structural causal models solve this by writing down the underlying mechanism that generates the data. Think of it as pairing a directed acyclic graph that encodes who influences whom with structural equations that spell out how each node responds to its parents and an independent noise source. These two pieces let you simulate “worlds where the line exists” and “worlds where we have data from a different city.” The formalization came from Judea Pearl’s early work on causal diagrams and structural models, which showed how graphical structure reveals when treatment effects are identifiable and how to isolate the effects of interventions [https://ftp.cs.ucla.edu/pub/stat_ser/r364.pdf](https://ftp.cs.ucla.edu/pub/stat_ser/r364.pdf).

SCMs sit at the heart of modern causal AI: they unify the goals of causal discovery, which tries to learn the graph from data, and causal effect estimation, which uses an assumed graph to derive quantities such as the average treatment effect (ATE). Later sections explain the mathematics, but first understand the big picture: the goal of SCMs is to turn mechanistic domain knowledge into equations you can plug into do-calculus and counterfactual reasoning, so that a scientist or engineer can ask “what happens under intervention?” and “what would have happened if…?” without needing randomized trials every time.

This is where the model-based approach pays off. When you sketch a DAG and check whether back-door paths are blocked, you replace guesses about confounders with visible criteria; when you write structural equations, you can sample counterfactual worlds and calculate policy effects. The rest of the page drills into how that exact logic works, how practitioners encode it in code, and how the field is pushing SCMs into high-dimensional, multi-agent, and production-grade settings.

## How it works

You build an SCM by specifying a DAG \(G\) and assigning to each observed node \(X_i\) (called endogenous) a structural equation driven by its parents \(PA_i\) within the graph and an independent noise term \(U_i\):

\[
X_i = f_i(PA_i, U_i)
\]

where \(f_i\) is the deterministic function describing how the parents cause \(X_i\), \(PA_i\) is the set of direct causes of \(X_i\) in the DAG, and \(U_i\) is a random variable representing background noise that is independent across equations according to the modeler’s assumptions. The pair \((G, \{f_i\})\) defines the structural causal model. The graph encodes conditional independencies; the equations encode the exact functional response.

### From observations to interventions

Causal questions usually ask about interventions, so we need a different probability than the usual observational distribution \(P(X)\). In Pearl’s notation, the intervention \(do(X=x)\) surgically sets the variable \(X\) to \(x\) and removes all arrows into \(X\), producing a modified graph \(G_{do(X=x)}\). The interventional distribution is then

\[
P(Y \mid do(X=x)) = \sum_{U} P(Y \mid x, U) P(U)
\]

where \(Y\) is the outcome we care about, \(U\) is the vector of exogenous noise variables, and the summation marginalizes over the unobserved factors. This formula is the basis for counterfactual reasoning: the somber statement “we cannot directly observe \(P(Y \mid do(X=x))\)” becomes actionable because the SCM specifies how to compute it using the structural equations and the graph’s independencies. In practice, we use rules such as the back-door criterion to decide which variables to condition on so that the interventional effect equals an observational conditional.

Back-door adjustment applies when there is a set \(Z\) that blocks all back-door paths from \(X\) to \(Y\). Written formally,

\[
P(Y \mid do(X=x)) = \sum_z P(Y \mid X=x, Z=z) P(Z=z)
\]

where \(Z\) is chosen so that conditioning removes confounding. The SCM graph guides that choice: if the graph shows an arrow \(Z \rightarrow X\) and \(Z \rightarrow Y\) with no other back-door paths, then the adjustment formula is valid. This is the practical procedure causal engineers follow when a simple regression would otherwise conflate several causal mechanisms.

### Counterfactuals and structural equations

SCMs go beyond interventions; they can answer “what if?” questions involving counterfactuals. A counterfactual query asks, “Given that I observed \(X=x\) and \(Y=y\) in the real world, what would \(Y\) have been if \(X\) had been \(x’\)?” This is encoded using twin networks or the abduction-action-prediction trio:

1. **Abduction**: infer the values of the exogenous noises \(U\) that explain the observed \(X=x\) and \(Y=y\). Because the structural equations are deterministic functions of parents and \(U\), you solve \(f_X(PA_X, U_X) = x\) and \(f_Y(PA_Y, U_Y) = y\) for the unobserved \(U\).

2. **Action**: replace the structural equation for \(X\) with \(X = x’\), effectively performing \(do(X=x’)\) on the modified SCM.

3. **Prediction**: run the modified equations forward with the same noise \(U\) derived in the abduction step.

This process produces counterfactual values \(Y_{X=x’}\). The key is that the structural equations retain the full causal story; without them, you cannot reason about hypothetical manipulations of the data.

### Encoding SCMs in code

DoWhy (Sharma et al. 2020) is a popular library that mirrors the SCM workflow: specify the graph and equations (via a causal graph string), choose estimation methods (e.g., propensity score matching), and validate assumptions with refuters and placebo tests [https://arxiv.org/abs/2011.03234](https://arxiv.org/abs/2011.03234). Under the hood, DoWhy uses data structures for the graph and leverages standard ML estimators to solve for \(P(Y \mid do(X=x))\). The pipeline demonstrates how close structural modeling sits to deployment: a developer writes a graph, an econometrician chooses back-door adjustments, and the system delivers causal estimates alongside diagnostics.

More expressive SCMs use latent-variable models such as CEVAE (Louizos et al. 2017) to learn hidden confounders from data [https://arxiv.org/abs/1610.06169](https://arxiv.org/abs/1610.06169). CEVAE assumes a latent variable \(Z\) influencing both \(X\) and \(Y\), introduces variational encoders and decoders, and optimizes a variational lower bound to recover \(P(Y \mid do(X=x))\) via learned representations. The architecture is a structural model because latent \(Z\) and observed nodes follow structural equations parameterized by neural networks, yet the training resembles a variational autoencoder with reconstructed confounders.

Recent work on counterfactual fairness (Kusner et al. 2017) uses SCMs to define fairness criteria grounded in realistic interventions [https://arxiv.org/abs/1703.06856](https://arxiv.org/abs/1703.06856). They propose that a decision \(Y\) is fair if it remains unchanged when the sensitive attribute \(A\) is counterfactually intervened on while holding other causes fixed. SCMs make this precise: you can perform the intervention \(do(A=a’)\) on the structural equations and recompute the outcome. If the outcome changes, the decision is unfair conditional on the SCM assumptions. This link between fairness and SCMs is an example of how the mechanistic representation gives actionable tests.

### Failure modes and domain knowledge

SCMs are only as good as the assumed graph and equations. If you omit a confounder or mis-specify a functional form, your do-calculus results will be wrong. The remedy is twofold: use domain knowledge (or experiments) to justify edges, and use refutation methods (placebo treatments, random data subsets) to detect instability. SCMs excel when you can enumerate the sources of influence; they struggle when the true mechanism involves latent, cyclic, or non-interpretable components. The rest of the field is working to scale these models to high-dimensional observation spaces while retaining the ability to interpret interventions.

## Where the field is now

Research and engineering splits of SCMs are alive and active. From a research perspective, the big push is to learn SCMs—both the graph and structural equations—from high-dimensional data. For instance, DAG-GNN (Yu et al. 2019) parameterizes the adjacency matrix of the causal graph with a graph neural network, optimizing a differentiable acyclicity constraint while simultaneously learning structural parameters [https://arxiv.org/abs/1901.10909](https://arxiv.org/abs/1901.10909). This allows SCM discovery on datasets with dozens of variables, and in benchmarks such as Sachs and protein signaling the learned graphs recover key biological edges that traditional constraint-based algorithms miss. CEVAE continues to be the standard for estimating treatment effects with latent confounders because it produces lower RMSE on the IHDP dataset than classical matching when the confounder is complex.

The engineering frontier is about production-grade inference. Databricks’ blog post “Why causal inference matters for data science teams” showcases how data engineers embed SCMs into production pipelines to answer questions like “What was the effect of our pricing change?” [https://databricks.com/blog/2023/08/09/why-causal-inference-is-critical-for-data-science-teams](https://databricks.com/blog/2023/08/09/why-causal-inference-is-critical-for-data-science-teams). The post describes a hybrid pipeline where upstream streaming data is stored in Delta Lake, an SCM is specified via a DAG in SQL, and treatment effects are recomputed nightly. AWS shares a concrete implementation in their SageMaker causality blog, showing how teams train causal ML models on SageMaker pipelines and monitor them for concept drift [https://aws.amazon.com/blogs/machine-learning/causal-inference-in-amazon-sagemaker/](https://aws.amazon.com/blogs/machine-learning/causal-inference-in-amazon-sagemaker/). Those deployments prove that SCM tools can run on cloud GPUs, connect to enterprise data, and provide explainable policy recommendations.

On the research side, factions are working on causal representation learning and counterfactual world models because SCMs as defined are limited to observed variables. Recent preprints explore disentangling independent causal mechanisms from sensor data so that the SCM equations themselves are learnable, even in video streams. That line of work aims to extend the reach of the Structural Causal Model to settings where the parent sets and functions are not obvious but can be recovered from invariances across environments, which opens the door to general AI systems that reason about interventions on raw pixels rather than tabular features.

## What's still open

1. **How can SCMs scale to thousands of variables and continuous actions while preserving identifiability?** Current differentiable SCM discovery methods face optimization challenges when the adjacency matrix becomes dense; new architectures or regularizers that exploit sparsity or modularity are needed.

2. **Can we learn valid SCMs in domains where some causal mechanisms are latent or inherently cyclic?** The assumption of acyclicity and full observability breaks in many control systems. Defining a theory of interventions and counterfactuals for dynamic systems with feedback remains unresolved.

3. **How do we verify SCM assumptions in high-stakes domains without randomized trials?** Placebo tests and refuters help, but automatic diagnostics that point to specific missing edges or mis-specified noise distributions would turn SCMs into trustworthy decision tools.

4. **What is the right way to combine causal representation learning with SCMs so that the learned latent space admits clear interventions?** Current methods either assume a reference SCM or learn representations without guaranteeing causal semantics; bridging this gap with provable identifiability would be a major advance.

## Where to read next

If you want the library and tooling perspective, → [[do-why]] walks through the schema–estimator–refuter pipeline that lets you iterate on an SCM quickly. The theoretical underpinnings of invariance and transportability live in → [[do-calculus]], while the practical signal for discovering graphs from data is developed in → [[causal-discovery]]. For those interested in counterfactual representations that build on SCMs, → [[counterfactual-representation-learning]] maps how to recover latent causes from raw inputs.

## Build it

**What you're building:** A causal effect estimation pipeline that specifies a small SCM for the IHDP dataset, adjusts for confounders, and validates the estimate with placebo tests from DoWhy.

**Why this is valuable:** You’ll leave with a runnable causal inference workflow that demonstrates structural equations, interventions, and counterfactual checks on a real dataset, which is the starting point for diagnostic dashboards or automated policy evaluation.

**Stack:**
- **Model:** DoWhy structural causal model with `sklearn` estimators (custom pipeline code, not a HuggingFace checkpoint).
- **Dataset:** [`causaldata/ihdp`](https://huggingface.co/datasets/causaldata/ihdp) (HuggingFace) — 747 samples with treatment indicator and outcomes for infant health.
- **Framework:** Python 3.10, DoWhy 0.7, scikit-learn 1.3, pandas 2.1, and matplotlib 3.8.
- **Compute:** Works on any CPU machine or free Colab (1–2 minutes runtime, <2 GB RAM, no GPU needed).

**The recipe:**
1. Install the stack with `pip install dowhy scikit-learn pandas matplotlib causaldata`. Load the IHDP dataset with `causaldata.load_dataset("ihdp")`, then convert to a pandas DataFrame, naming the treatment column “treatment,” outcome column “y,” and covariates as the rest.
2. Specify the SCM graph string:
   ```
   digraph {
     treatment -> y;
     mother_age -> treatment;
     birth_weight -> y;
     mother_age -> birth_weight;
     education -> treatment;
     education -> y;
   }
   ```
   This encodes the assumed confounding via `mother_age` and `education`. Structural equations are implicitly linear with logistic treatment and linear outcome; DoWhy will handle the estimation.
3. Create the DoWhy causal model, call `model.view_model()` to visualize, and choose the “backdoor” method with `propensity_score_matching` as the estimator. Fit the model, then call `model.estimate_effect(backdoor_estimator)` with `method_name="backdoor.propensity_score_matching"`. Expect an estimated ATE near the published 0.058 for IHDP.
4. Run DoWhy refuters: `placebo_treatment_refuter` and `data_subset_refuter` to verify that