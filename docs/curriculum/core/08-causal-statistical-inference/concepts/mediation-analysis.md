---
title: Mediation analysis
slug: mediation-analysis
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [pearl, schiefer, shalit, peters]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [structural-causal-models, counterfactuals, causal-effect-identification]
tags: [causal-inference, mediation, interpretability, representation-learning, llm, causal-effects]
updated: 2025-01-29
has_mvb: true
---

# Mediation analysis

Imagine prompting a large language model with “explain your steps,” then quietly shuffling or removing each chain-of-thought token before it reaches the user, yet the model still answers correctly. The performance boost from “think step-by-step” is real, and so is the paradox: if the reasoning tokens can be corrupted and the final answer still holds, are those tokens doing anything? Mediation analysis is the mathematical instrument that turns that observation into a testable theory. It treats each intermediate token or latent state as a candidate mediator and asks what portion of the total causal effect on the answer actually routes through that mediator. By the end of this page you will understand how to describe that routing rigorously, why the assumptions matter in practice, and how to build a toolchain that measures Natural Direct and Indirect Effects on synthetic LLM interventions using statsmodels and DoWhy.

## The territory

Mechanism is the question mediation analysis was invented to answer. Traditional causal inference tells us what happens when we flip a treatment—for example, whether presenting a chain-of-thought prompt increases LLM accuracy. Mediation analysis goes further by decomposing that improvement into pieces: the part that flows directly from the prompt to the answer, and the part that filters through the intermediate representation, such as generated reasoning tokens, retrieved facts, or an internal hidden layer. This decomposition is what makes causal inference explanatory rather than descriptive; it allows researchers to attribute outcomes not just to a treatment but to the pathways the treatment opens. The mathematical language that makes this precise is a structural causal model (SCM), where nodes represent random variables and edges represent deterministic structural equations plus noise. The mediator lives on one of these edges, and mediation analysis answers whether the edge is merely decorative or the actual conduit for the effect we observe. Positioning itself between the counterfactual machinery of SCMs and the practical estimators of econometrics, mediation analysis borrows from both: it uses counterfactual definitions to describe what “direct” and “indirect” mean, and it relies on regression or representation learning to estimate those quantities from data. How does it actually work? The mechanism is best understood by tracing the decomposition inside a SCM, writing the effects in terms of nested counterfactuals, and then checking whether those nested counterfactuals can be identified from observed data.

## How it works

Mediation analysis begins inside a structural causal model with three types of nodes: a treatment \(A\) (for example, whether we prompt the model with chain-of-thought), a mediator \(M\) (the tokens the model emits when reasoning), and an outcome \(Y\) (the final answer accuracy). Each node is a deterministic function of its parents and an independent noise term; \(M = f_M(A, U_M)\) and \(Y = f_Y(A, M, U_Y)\), where \(U_M\) and \(U_Y\) are exogenous noises. The treatment \(A\) is manipulated, so there exists \(Y_a\), the outcome we would observe if we forced \(A\) to take value \(a\), and \(M_a\) is the mediator under that intervention. The total causal effect (TE) of changing the prompt from \(a'\) to \(a\) is
\[
TE = \mathbb{E}[Y_a - Y_{a'}],
\]
where expectations are over all randomness in the SCM, including unobserved noise. This TE captures every path from \(A\) to \(Y\), but it does not tell us whether the effect transits through \(M\). To disentangle the pathways, Pearl (2014) defines two nested counterfactuals: the natural direct effect (NDE) modifies \(A\) while holding the mediator at the value it would have taken under \(a'\), and the natural indirect effect (NIE) changes the mediator while holding the treatment fixed at \(a\). Their definitions are
\[
NDE = \mathbb{E}[Y_{a, M_{a'}} - Y_{a', M_{a'}}], \quad NIE = \mathbb{E}[Y_{a, M_{a}} - Y_{a, M_{a'}}],
\]
where \(Y_{a,m}\) is the outcome when \(A\) is fixed to \(a\) and \(M\) is fixed to \(m\), and \(M_{a'}\) is the mediator under treatment \(a'\). These expressions are called “cross-world” because they compare outcomes across hypothetical worlds where \(A\) and \(M\) take values that could not coexist under a single intervention. Their sum recovers the total effect: \(TE = NDE + NIE\), which is true whenever the mediator is well defined.

### Identification via the causal mediation formula

To compute NDE and NIE from data, Pearl (2014) derives the causal mediation formula, which rewrites the cross-world expectation as an integral over observed variables:
\[
\mathbb{E}[Y_{a, M_{a'}}] = \sum_m \mathbb{E}[Y \mid A=a, M=m] \Pr(M=m \mid A=a').
\]
In this expression, \(A\) and \(M\) are the observed treatment and mediator; \(Y_{a,m}\) is replaced by the conditional expectation of \(Y\) given \(A=a\) and \(M=m\), while \(M_{a'}\) is replaced by the distribution of \(M\) under \(A=a'\). This substitution is valid when we assume sequential ignorability: conditional on pre-treatment covariates \(X\), there are no unobserved confounders of \(A\) and \(M\) nor of \(M\) and \(Y\), i.e., \(A \perp\!\!\!\perp M_{a'} \mid X\) and \(M \perp\!\!\!\perp Y_{a,m} \mid A,X\). Annotated, \(\mathbb{E}[Y \mid A=a, M=m]\) is a regression that predicts the outcome from treatment and mediator, and \(\Pr(M=m \mid A=a')\) is a mediator model that estimates what mediator value would have arisen without intervention. Sequential ignorability is strong, so we typically condition on a rich set of covariates \(X\)—for LLM reasoning this might include prompt difficulty, token frequency, or model temperature.

An alternative, advocated by the interventionist approach, avoids cross-world counterfactuals altogether by defining mediation through randomized manipulations of the mediator itself. “An Interventionist Approach to Mediation Analysis” (2020) reframes the problem as estimating the effect of \(A\) on \(Y\) when we intervene on \(A\) and then either allow or block the mediator from changing. That work introduces the idea of using auxiliary experiments—randomizing the mediator or its observed proxies—to isolate the direct effect. For example, to approximate \(Y_{a,M_{a'}}\), we can hold the mediator constant at its value under \(a'\) while applying treatment \(a\) by reusing cached reasoning tokens or by training a simulator that freezes the mediator embeddings. The interventionist formulation clarifies which assumptions are testable: if we can randomize \(M\) while keeping \(A\) fixed, we can estimate the same contrast as the cross-world definition without invoking untestable independence between unseen noise terms. That design also makes it easier to think about LLM interventions where we actively replace the mediator stream with synthetic tokens to simulate “no reasoning” conditions.

### Estimation and representation learning for mediators

In many applications the mediator is not a scalar but a high-dimensional latent representation—think of the continuous attention patterns or the hidden activations inside a transformer layer. The mediator models \(\mathbb{E}[Y \mid A=a, M=m]\) and \(\Pr(M=m \mid A=a')\) therefore must learn functions over these vectors. Learning representations for counterfactual inference (Shalit et al. 2016) [https://arxiv.org/pdf/1605.03661] provides a template: build a neural encoder that maps covariates and treatments to a balancing representation \(Z\), and then regress the outcome on \(Z\) with a regularizer that penalizes discrepancies between the treatment groups in \(Z\). Translating this idea to mediation analysis means encoding the mediator into \(Z\) so that the downstream regression is stable across hypothetical interventions. The representation learning step reduces variance in high-dimensional mediators, making it feasible to plug them into the mediation formula without hand-crafted features.

Disentangled Representation for Causal Mediation Analysis (2023) [https://arxiv.org/html/2302.09694] pushes this idea further: instead of encoding everything into a single embedding, it learns separate latent factors for the direct and indirect pathways via variational inference. The encoder outputs two distributions, one that controls the mediator’s effect on the outcome and another that models confounding variation. During training the architecture alternates between reconstructing the mediator and predicting the outcome, which approximately enforces the structural equation \(Y = f_Y(A, H_d, U_Y)\) where \(H_d\) captures the indirect pathway. Applying this to LLM chain-of-thought means structuring an encoder so that one latent factor responds to interventions on the reasoning tokens and another factor captures unrelated background variation like question difficulty. The disentanglement permits targeted interventions: to estimate the mediated effect we can clamp \(H_d\) to its distribution under \(A=a'\) while keeping the other latent factors free, aligning with the causal mediation formula but inside a learned representation space.

The upshot is that mediation analysis combines SCM-based definitions (Pearl 2014) with representation learning (Shalit et al. 2016, disentangled representations) and, when feasible, with randomized mediator interventions (An Interventionist Approach to Mediation Analysis 2020) to make the direct/indirect decomposition empirically tractable. This is why a practical toolkit for measured natural direct and indirect effects must handle three steps: (1) specify the influence diagram and collect covariates that satisfy ignorability or can be blocked via an intervention, (2) fit mediator and outcome models using classical regression or neural representations, and (3) combine those models via the mediation formula or its interventional analogue. The next section shows how the field is advancing beyond scalar mediators and into the modern demands of LLM interpretability.

## Where the field is now

The research frontier is currently shaped by efforts that both reinterpret mediation for high-dimensional mediators and apply it to large model reasoning. Disentangled Representation for Causal Mediation Analysis (Fong, Tran, and others 2023) [https://arxiv.org/html/2302.09694] demonstrates that representing direct and indirect pathways as separate latent codes lets the mediation formula operate in a learned manifold; using a dual variational objective regularized by the structural equations, that work recovers path-specific effects even when the mediator is a vector of hundreds of activations. On the LLM side, Untitled (2023) [https://arxiv.org/pdf/2305.15054], a study released alongside prompting benchmarks, perturbs chain-of-thought tokens by introducing random swap, deletion, or occlusion interventions and then applies mediation-style decompositions to quantify how much of the accuracy gain is genuinely carried by the reasoning tokens versus residual shortcuts. By combining causal mediation with synthetic mediator interventions, this line of work validates reasoning metrics that go beyond correlational attention scores.

The engineering frontier, meanwhile, is the adoption of these ideas in production inference pipelines. AWS’s “Causal Inference Made Easy with DoWhy” blog (2023) [https://aws.amazon.com/blogs/machine-learning/causal-inference-made-easy-with-dowhy/] describes how data scientists inside Amazon instrument policy changes through DoWhy’s API, deploying mediation-style estimators to attribute revenue changes to feature rollouts and enabling real-time dashboards of direct and mediated effects. DoWhy, which wraps statsmodels style regressions with structural model specification and refutation tests, is now part of the Amazon SageMaker causal inference offering, providing a production-grade path from a DAG to an estimated NDE/NIE pair. This industrialization of mediation analysis — grounded in the same SCM definitions from Pearl (2014) and the interventional learning advocated by 2020’s interventionist approach — shows that mediation analysis defines not just an interpretability experiment but a measurable production signal that can be monitored.

The consequence is that researchers pursuing LLM interpretability now have both a methodological toolbox and engineering scaffolding: the lens of cross-world decompositions, the practical estimator recipes in DoWhy and statsmodels, and libraries that can represent complex mediators as learned embeddings. Yet these tools are still being stress-tested for latent representations that are neither scalar nor easily perturbed, which leads us directly to the open problems below.

## What's still open

Can mediation analysis recover path-specific effects when the mediator is a high-dimensional, continuous latent representation that changes across multiple layers? The current disentangled encoders work on fixed-size vectors, but transformers produce sequences of contextual embeddings whose structure may not factor neatly into direct and indirect components.

How can we run interventional experiments on LLMs without access to the internal weights, especially when we only observe tokens? The interventionist approach proposes randomizing the mediator, but in practice researchers can only replace, mask, or simulate tokens; each of those manipulations introduces its own confounders.

Is there a principled way to regularize learned mediator representations so that they satisfy sequential ignorability, or an explicit test that tells us when the mediator and outcome models are misspecified? Without such a test, the mediation formula can produce misleading decompositions that look plausible but rest on invalid independence assumptions.

Finally, can we combine mediation analysis with fine-grained counterfactual representation learning (Shalit et al. 2016) to identify multiple mediators simultaneously while still providing uncertainty estimates, so that downstream explainability layers can say “this token contributed X% of the effect” with statistically sound confidence intervals?

## Where to read next

If you want the structural language that underpins these decompositions, → [Structural Causal Models](structural-causal-models.md) describes how DAGs encode interventions and counterfactuals, and the probabilistic backbone for nested counterfactuals lives in → [Counterfactuals](counterfactuals.md). The representation perspective continues in → [[causal-representation-learning]], which shows how learned embeddings balance treatment groups and support counterfactual queries, while the applied side of identifying transportable effects is documented in → [[causal-effect-identification]].

## Build it

This build proves that mediation analysis can be operationalized as an executable pipeline: a simple script on Colab will generate a synthetic dataset of prompt interventions, encode the mediators with HuggingFace embeddings, and estimate the Natural Direct and Indirect Effects using statsmodels regressions wrapped by DoWhy.

**What you're building:** A Colab notebook that simulates chain-of-thought prompts, encodes the mediator tokens with `sentence-transformers/all-MiniLM-L6-v2`, and reports the NDE and NIE of the reasoning tokens on final answer accuracy.

**Why this is valuable:** Estimating NDE and NIE forces you to execute every part of the mediation pipeline—generate intervention data, fit mediator and outcome models, apply the causal mediation formula, and inspect the robustness diagnostics—turning the abstract decomposition into a reproducible artifact.

**Stack:**
- **Model:** [https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) — 3.1M downloads, lightweight embedding model suitable for Colab T4.
- **Dataset:** Synthetic chain-of-thought dataset generated inside the notebook by sampling prompts, applying interventioned reasoning tokens, and recording binary accuracy outcomes.
- **Framework:** DoWhy 1.0.0 + statsmodels 0.14.2 + pandas 2.0 + scikit-learn 1.4 for preprocessing.
- **Compute:** Free Colab T4 (16GB RAM, 16GB VRAM) — data generation, embeddings, and regression complete in under 20 minutes.

**The recipe:**
1. Run `pip install dowhy==1.0.0 statsmodels==0.14.2 sentence-transformers==2.2.2 scikit-learn==1.4.0 wandb` and load DoWhy’s `CausalModel`, statsmodels `OLS`, and the HuggingFace embedding pipeline for `all-MiniLM-L6-v2`.
2. Generate the dataset: sample 1,000 prompts (e.g., arithmetic or word puzzles), simulate two treatments \(A \in \{0,1\}\) for “no reasoning/demoted chain-of-thought,” produce mediator vectors by embedding the generated tokens, and create \(Y\) as a noisy binary accuracy label influenced by both \(A\) and the mediator plus covariates like prompt-length.
3. Fit the mediator model \(\Pr(M \mid A, X)\) via scikit-learn’s `LinearRegression` or a shallow neural net on the embeddings, and fit the outcome model \(\mathbb{E}[Y \mid A, M, X]\) with statsmodels `OLS` (include interaction terms between \(A\) and the mediator representation).
4. Use DoWhy’s `CausalModel.identify_effect()` followed by `estimate_effect(method_name="backdoor.linear_regression")` to compute NDE and NIE via the causal mediation formula, inspect the refuters (placebo treatment, random common cause) to assess sequential ignorability, and compare the results to a naive difference-in-differences.
5. Visualize the estimated direct and indirect effects with 95% confidence intervals, plot the mediator distributions under \(A=0\) and \(A=1\), and archive the notebook on GitHub or Colab so the artifact documents the entire estimand path.

**Expected outcome:** A runnable Colab notebook that prints the Natural Direct and Indirect Effects, shows the mediator distributions, logs refuter diagnostics, and stores the script for reuse across other LLM interventions.

- **CS student:** Run the same notebook on a 200-example subset using only CPU (drop embedding batch size to 16); the smaller dataset keeps runtime under 30 minutes on an RTX 3060 while still producing stable NDE/NIE estimates.
- **Applied engineer:** Wrap the notebook in a FastAPI endpoint that accepts new prompts, applies the learnt mediator/outcome models, and responds with the estimated total, direct, and indirect effects; quantize the mediator regression with ONNX Runtime to hit p50 latency below 120ms on an NVIDIA A10.
- **Applied researcher:** Replace the pre-embedded tokens with a learned latent mediator via an autoencoder, re-run the beam of mediator/outcome models, and test the hypothesis that the latent mediator explains at least 60% of the treatment effect by comparing the NIE to the TE.
- **Frontier researcher:** Probe the open question of high-dimensional continuous mediators by training a disentangled VAE (as in Disentangled Representation for Causal Mediation Analysis) inside the notebook and falsify the assumption that sequential ignorability holds by constructing a simulated confounder; report when the estimated NIE deviates by more than 25% from the ground-truth path-specific effect.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*