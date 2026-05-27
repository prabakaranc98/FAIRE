---
title: Do-calculus
slug: do-calculus
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [pearl, shpitser, varici, tajer, acarturk]
feeds_de_pillar: []
arc_position:
  arc: [08-causal-statistical-inference]
  prev: [structural-causal-models]
  next: [causal-representation-learning]
mvb_personas: [curious-learner, cs-student, applied-engineer, applied-researcher, frontier-researcher, theory-student]
prereqs: [structural-causal-models, causal-diagrams, counterfactuals]
tags: [do-calculus, causal-inference, identifiability, interventions, structural-equations, policy-evaluation]
updated: 2024-11-10
has_mvb: true
---

# Do-calculus

What would you do if the only data you had came from the uncontrolled behavior of users, but you still needed to know what would happen if you forced a different action—say, changing a recommendation policy or setting a new medical dose? That is exactly the daily puzzle for policy analysts, social scientists, and health AI teams: they must predict the causal effect of an intervention that nobody has performed, using only observations of how the system already behaved. Do-calculus is the algebraic engine that makes that leap possible. By translating questions about imaginary interventions into statements about the real data you do see, it tells you whether the desired answer is identifiable and, if so, which combinations of conditional probabilities you must estimate. By the end of this page you will understand how the rules of do-calculus work inside a graphical model, why developers trust it to give rigorous identifiability guarantees, and how to build a lightweight pipeline that runs these rules on real datasets with ≤16 GB of GPU memory.

## The territory

Decision makers constantly conflate “What happened?” with “What would happen if I acted?” Standard statistics only speaks to the first: you can estimate how often an outcome followed a cause, but you cannot directly reason about the outcome of a hypothetical intervention. Structural causal models (SCMs) repair that gap by explicitly modeling how every variable is generated from its parents plus background noise, and by offering the language of interventions through the \(\textit{do}\)-operator. In this language, \(P(Y \mid do(X=x))\) represents the distribution of \(Y\) when the system is forced to set \(X\) to \(x\), as opposed to merely selecting datapoints where \(X=x\). The difficulty is that the interventional distribution is often not directly measurable; it depends on pathways that are opened or closed after we intervene.

Do-calculus sits at the heart of the causal inference territory between SCMs and policy evaluation. Once the SCM's causal graph is specified, do-calculus provides algebraic rules that allow you to rewrite queries that mention \(\textit{do}()\) interventions into expressions containing only ordinary conditional probabilities. If such a rewriting exists, the query is said to be “identifiable” from the observational data; otherwise, no amount of data will resolve the causal effect. Practically speaking, the territory this page covers is: given a causal graph, figure out if a target effect is identifiable and, if so, compute the observational expression that yields it. That identifiability result is the first half of any causal arc, and it connects directly to later work on learning representations where the causal structure itself is uncertain.

## How it works

The basic objects are the tuples of an SCM: \((\mathbf{U}, \mathbf{V}, \mathbf{F})\), where \(\mathbf{U}\) are exogenous noise variables, \(\mathbf{V}\) are observable variables, and \(\mathbf{F} = \{f_V\}\) is the set of structural equations \(V = f_V(pa(V), U_V)\). Every graph edge is read from cause to effect, so the graph records which parents \(pa(V)\subset \mathbf{V}\) appear in each structural equation. The observational distribution \(P(\mathbf{V})\) is defined by marginalizing over the exogenous noises, and because the SCM is recursive (acyclic), we can factorize it as \[P(\mathbf{V}) = \prod_{V \in \mathbf{V}} P(V \mid pa(V)).\] In this factorization, \(P(V \mid pa(V))\) is the distribution induced by \(f_V\) and the noise term \(U_V\).

When we intervene on \(X\) and set it to a value \(x\), we remove the structural equation for \(X\) and replace it with the constant \(x\); formally, we replace the factor \(P(X \mid pa(X))\) with a point mass \(\delta(X=x)\). The resulting interventional distribution is given by the truncated factorization \[P(\mathbf{Y} \mid do(\mathbf{X}=x)) = \sum_{\mathbf{Z}} \prod_{V \in \mathbf{Y} \cup \mathbf{Z}} P(V \mid pa(V)),\] where \(\mathbf{Z} = \mathbf{V} \setminus (\mathbf{Y} \cup \mathbf{X})\) collects the variables we marginalize over, and the product only includes the conditional factors for variables whose structural equations remain intact after the intervention. That expression exists in theory but not in practice because the conditional factors themselves might involve unobserved parents. Do-calculus decides whether we can eliminate the \(\textit{do}\) by combining three inference rules that are purely graph-theoretic.

The three rules of do-calculus manipulate statements of the form \(P(\mathbf{y} \mid do(\mathbf{x}), \mathbf{z})\) using the d-separation properties of the manipulation graph \(G_{\overline{X}}\) or \(G_{\underline{X}}\), where the notation \(\overline{X}\) means “remove incoming arrows to \(X\)” and \(\underline{X}\) means “remove outgoing arrows from \(X\).” The rules are:

1. **Insertion/deletion of observations**: If \(\mathbf{Y}\) is d-separated from \(\mathbf{Z}\) given \(\mathbf{X}\) in \(G_{\overline{\mathbf{X}}}\), then \(P(\mathbf{y} \mid do(\mathbf{x}), \mathbf{z}) = P(\mathbf{y} \mid do(\mathbf{x}))\). The rule tells you that an observed variable can be dropped if it carries no additional information about \(\mathbf{Y}\) once you fix the intervention.

2. **Action/observation exchange**: If \(\mathbf{Y}\) is d-separated from \(\mathbf{Z}\) given \(\mathbf{X}\) in \(G_{\overline{\mathbf{X}}, \underline{\mathbf{Z}}}\), then \(P(\mathbf{y} \mid do(\mathbf{x}), do(\mathbf{z}), \mathbf{w}) = P(\mathbf{y} \mid do(\mathbf{x}), \mathbf{z}, \mathbf{w})\). This rule allows you to replace a do-action with an ordinary observation whenever there are no backdoor paths from the intervened variable to \(\mathbf{Y}\).

3. **Insertion/deletion of actions**: If \(\mathbf{Y}\) is d-separated from \(\mathbf{Z}\) given \(\mathbf{X}\) in \(G_{\overline{\mathbf{X}}, \overline{\mathbf{Z}}}\), then \(P(\mathbf{y} \mid do(\mathbf{x}), do(\mathbf{z}), \mathbf{w}) = P(\mathbf{y} \mid do(\mathbf{x}), \mathbf{w})\). This rule says you can ignore an intervention when it no longer influences \(\mathbf{Y}\).

Do-calculus is complete for identification of \(P(\mathbf{Y} \mid do(\mathbf{X}))\) in recursive SCMs, meaning that if no sequence of the three rules can eliminate all \(\textit{do}\)-operators, then the query is not identifiable from \(P(\mathbf{V})\) (Shpitser & Pearl 2006) [https://arxiv.org/abs/1206.2086]. Because the rules only depend on d-separation, verifying their applicability is just a graph search issue, and the completeness result implies that once you have the causal graph, no statistical trick beyond do-calculus can salvage the desired interventional expression. The usual workflow launches from a query such as \(P(\text{Outcome} \mid do(\text{Treatment}))\): draw the graph, identify confounders, and test whether a back-door set \(\mathbf{Z}\) allows the adjustment formula
\[
P(Y \mid do(X)) = \sum_{z} P(Y \mid X, z) P(z),
\]
where \(z \in \mathbf{Z}\), \(X\) is the treatment, and \(Y\) is the outcome. Here, \(P(Y \mid X, z)\) and \(P(z)\) are observable conditional probabilities, and the sum occurs over values of \(\mathbf{Z}\). The adjustment formula is a direct consequence of Rule 1 plus the fact that \(\mathbf{Z}\) blocks all backdoor paths from \(X\) to \(Y\).

A more involved example occurs when you have unobserved confounders that leave no valid adjustment set. In such cases, do-calculus still allows you to factor out the intervention by sequentially applying Rule 2 to bring problematic variables inside ordinary conditioning and Rule 3 to remove irrelevant new interventions. When the rules succeed, you obtain a combination of observational distributions, often expressed as nested sums and products, that can be estimated from data. When they fail, the query is provably non-identifiable: only additional experiments or stronger assumptions can reveal the effect.

The operational engine that executes the three rules is the identification (ID) algorithm developed by Shpitser and Pearl. The algorithm partitions the graph into “c-components,” sets of variables connected by bidirected arcs (noise correlations). Within each c-component, it recursively applies do-calculus and marginalization to reduce the query to smaller subproblems; the recursion bottoms out either with a single causal factor \(P(V \mid pa(V))\) or with a failure when the intervention cannot be simulated by any sequence of graph edits. The algorithm’s correctness and completeness guarantee that the do-calculus manipulations you would perform by hand are automated and that the output expression is the simplest possible observational formula representing the causal effect.

Modern tooling takes this further by coupling do-calculus with graph search heuristics. For example, the ID algorithm is implemented in libraries such as DoWhy and Causalnex, where you explicitly supply the causal graph and the desired query, and the tool builds the do-calculus proof for you. These implementations are not merely convenience—they are critical when graphs have more than ten variables and manual reasoning becomes error-prone. The output is both the Boolean verdict about identifiability and the explicit mathematical expression that you then estimate from your dataset.

Do-calculus also underlies the theory of transportability and selection bias correction. By treating the distribution shift between domains as a special kind of intervention, do-calculus rules diagnose when you can combine experimental data from one environment with observational data from another to answer a query in a target population. The same three rules define whether the shift can be adjusted, making do-calculus the connective tissue between local experiments, global policy decisions, and representation learning when the underlying causal structure is partially known or even latent.

## Where the field is now

Researchers continue to push the limits of what can be identified when the graph itself is ambiguous or when the variables reside in learned representations. Varici, Acarturk, and Tajer (2024) generalize the classical identifiability theory to causal representation learning, proving that do-calculus still characterizes which latent distributions can be recovered from interventional experiments over learned features, and they provide achievability proofs that guide practical encoder designs [https://research.google/pubs/general-identifiability-and-achievability-for-causal-representation-learning/]. The same paper demonstrates that the completeness of do-calculus gives an upper bound on how much structure learning can hide within itself when only limited intervention data is available.

Bareinboim and Pearl (2012) show that do-calculus answers queries even when the data come from surrogate experiments that intervene on proxies rather than the treatment of interest; their z-identifiability results prove that you can combine experiments with observational data to “recover” the target intervention if certain graph-theoretic conditions hold [https://arxiv.org/abs/1202.4295]. This line of work remains the research frontier for deploying automated experimentation in the wild because it clarifies precisely which assumptions about surrogates and mediators are necessary for valid causal claims.

On the engineering front, Google Research’s 2024 effort “Learning to induce causal structure” uses neural architectures that internally encode do-calculus steps to iteratively propose and verify candidate DAGs from raw time-series data, treating the do-calculus transformations as differentiable constraints over predictions [https://research.google/pubs/learning-to-induce-causal-structure/]. The system shows promise for automating large-scale policy discovery in settings such as recommendation systems, where controlled experiments are expensive but causal reasoning is essential. Its deployment highlights that do-calculus is no longer just theoretical proof—it is embedded inside tooling that generates actionable policies for production teams.

## What's still open

Can do-calculus be extended to work with soft interventions where an action nudges but does not fix a variable, and if so, what graph surgeries replace the hard do-operator? How minimal can auxiliary proxy variables be when trying to identify a target effect through z-identifiability, and is there an efficient oracle that certifies the completeness of those proxies without enumerating all sets? When learning causal structure from high-dimensional data, can we quantify how do-calculus-guided regularization affects both identifiability and generalization under latent confounding? Finally, can the tractability of the ID algorithm be improved to handle thousand-variable graphs by leveraging sparsity or approximate d-separation while still providing guarantees about failure cases?

## Where this concept appears

Do-calculus is the workhorse of this arc: it is the logical next topic after [[structural-causal-models]] because it answers the fundamental “what happens if we intervene?” question left hanging by structural equations, and it is the bridge into [[causal-representation-learning]] because the same rules determine when latent features encode invariant causal information. Within broader causal inference stories, do-calculus also appears inside [[counterfactuals]] (where structural counterfactuals reduce to interventional queries) and inside [[policy-evaluation]] when evaluating new actions from logged data. The concept is referenced whenever a downstream arc needs to certify identifiability before deploying any learned policy.

## Where to read next

For the formal SCM setup and nested counterfactuals see [[structural-causal-models]]; for practical graph drawing and d-separation notation consult [[causal-diagrams]]; the engineering counterpart that shows how do-calculus feeds into representation learning is → [[causal-representation-learning]]; to close the loop with algorithms that enforce identifiability during policy optimization read → [[policy-evaluation]].

## Build it

**What you're building:** a do-calculus interpreter that computes \(P(\text{HeartDisease} \mid do(\text{Cholesterol}=c))\) under a specified DAG and backdoor-adjustment set, then estimates the identified expression from the `uciml/heart-disease` data.

**Why this is valuable:** it proves you can go from observational data and a credibly specified DAG to an explicit formula for the causal query you actually care about, and it produces a reproducible artifact (the identified expression plus numeric estimate) that you can present to stakeholders.

**Stack:**
- **Model:** none (the recipe builds symbolic expressions and derives estimators, so no pretrained model is required)
- **Dataset:** [uciml/heart-disease](https://huggingface.co/datasets/uciml/heart-disease) — 303 rows, 13 clinically meaningful columns
- **Framework:** Python 3.10 + `pandas`, `networkx`, `numpy`, `statsmodels`, `itertools`, `matplotlib`
- **Compute:** runs on CPU or Colab T4 (CPU-only, <16 GB RAM, seconds per run)

**The recipe:**
1. Install packages via `pip install pandas networkx numpy statsmodels matplotlib` and download the `uciml/heart-disease` dataset through the Hugging Face `datasets` library so every column is available as a pandas DataFrame.
2. Define the DAG by hand in code, for example: Age → Cholesterol, Exercise → Cholesterol, Cholesterol → HeartDisease, and let `Hypertension` act as a confounder for Cholesterol and HeartDisease. Use `networkx.DiGraph()` to store edges and `nx.descendants()` for traversal.
3. Ask the query \(P(\text{HeartDisease} \mid do(\text{Cholesterol}=c))\). Identify that the back-door set is `Hypertension` plus `Age`, then apply the adjustment formula \(P(\text{HeartDisease} \mid do(\text{Cholesterol}=c)) = \sum_{z} P(\text{HeartDisease} \mid \text{Cholesterol}=c, z) P(z)\) by grouping the data on \((\text{Cholesterol}, \text{Age}, \text{Hypertension})\), fitting a logistic regression to estimate \(P(\text{HeartDisease} \mid \text{Cholesterol}, z)\), and computing empirical frequencies for \(P(z)\).
4. Evaluate the identified expression by plugging in specific values of \(c\), plotting \(P(\text{HeartDisease} \mid do(\text{Cholesterol}=c))\) over the clinically observed range, and computing the average causal effect \(E[\text{HeartDisease} \mid do(\text{Cholesterol}=c_1)] - E[\text{HeartDisease} \mid do(\text{Cholesterol}=c_0)]\).
5. Export the derived expression, the estimated logistic coefficients, and the causal effect plot into a folder called `do_calculus_results` along with a markdown summary that records which rule(s) validated the adjustment set.

**Expected outcome:** a reproducible notebook or script that proves the query is identifiable, prints the algebraic adjustment formula, estimates the corresponding probabilities, and generates a plot or table that stakeholders can use to justify an action on cholesterol.

**Variants per persona (one line per active mvb_persona entry):**
- **Curious learner:** Visualize the DAG and highlight the back-door path so you can articulate each step to a non-technical audience.
- **CS student:** Add unit tests that assert the adjustment formula equals brute-force Monte Carlo estimates on a simulated SCM.
- **Applied engineer:** Wrap the notebook into an API that accepts new patient profiles and returns the causal effect estimate given specified interventions.
- **Applied researcher:** Ablate the back-door set by dropping one control variable at a time to measure the effect on bias and variance.
- **Frontier researcher:** Replace the hand-specified DAG with structure learned via bootstrapped invariances and rerun the identification proof to test robustness.
- **Theory student:** Derive the same adjustment set using the three do-calculus rules symbolically, and compare the result to the ID algorithm’s output.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

## Code & implementations

Production-ready libraries such as DoWhy encode the ID algorithm and do-calculus rules behind a simple `estimate_effect` call, and the official documentation on [https://docs.dowhy.org/en/stable/](https://docs.dowhy.org/en/stable/) walks through using the graphical interface plus automated identification. CausalNex and Ananke each expose graph utilities that check d-separation and enumerate adjustment sets, letting you plug in a DAG and immediately retrieve the observational estimand implied by do-calculus. Researchers also serialize do-calculus proofs in notebook form, attaching the sequence of rule applications to every identified effect so that the algebraic steps are auditable by domain experts.

## What comes next

After you can identify \(P(Y \mid do(X))\) in the observed graph, the natural follow-on is to learn the graph itself and then re-apply do-calculus to the learned structure, which is the goal of [[causal-representation-learning]]; simultaneously, you can apply do-calculus’ identification certificates to guide policy evaluation in [[policy-evaluation]] by restricting actions to those with verified estimands.

## Connected topics

The concept is tightly linked to [[structural-equations]] because the rules only operate once you know the functional form of each variable’s parents; it supports [[instrumental-variables]] when no back-door adjustment is possible; and it underlies [[transportability]] results that mix data from several environments while respecting the same graphical constraints.

## Further reading

Pearl’s original proof of completeness, the Shpitser & Pearl (2006) ID algorithm [https://arxiv.org/abs/1206.2086], Bareinboim & Pearl (2012) on z-identifiability [https://arxiv.org/abs/1202.4295], and the modern generalization to causal representations by Varici et al. (2024) [https://research.google/pubs/general-identifiability-and-achievability-for-causal-representation-learning/] are essential for the theory student, while the Google Research system “Learning to induce causal structure” [https://research.google/pubs/learning-to-induce-causal-structure/] offers a contemporary engineering case study.