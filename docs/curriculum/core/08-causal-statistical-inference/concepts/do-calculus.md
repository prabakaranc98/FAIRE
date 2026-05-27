---
title: Do-Calculus
slug: do-calculus
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [pearl, shpitser, bareinboim, heckman]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [structural-causal-models, causal-graphs]
tags: [causal-inference, do-calculus, structural-equations, front-door]
updated: 2024-11-23
has_mvb: true
---

# Do-Calculus

Imagine being in the position of the safety board for a new drug that is highly toxic in early trials. You know the drug could save lives, but you would be assassinated ethically if you forced a randomized control group to skip treatment or forced patients to ingest a lethal dose. The data you do have are passive records from doctors who chose treatments themselves, and yet the regulators still need a precise answer to “What if we had forced the drug on everyone?” Do-calculus is the mathematical device that resolves this ethical deadlock: instead of physically manipulating anything, it rewrites an intervention \(P(Y \mid do(X))\) into an expression built solely from what you observed, by treating the causal graph as a virtual laboratory. After reading this page you will see how the algebra aligns with the structural equations that underlie the graph, how each of the three rules corresponds to removing a “do” by appealing to a d‑separation, why the system is complete for identifiable effects, and how to practice the rewriting in code so that you can deliver counterfactual answers on top of purely observational logs.

## The territory

Do-calculus lies precisely between the structural causal models (SCMs) that describe mechanisms and the kinds of policy questions practitioners must answer without running new trials. An SCM gives you a graph where each node is a variable, each directed edge encodes a structural equation, and noise terms propagate through the graph when you sample from the joint distribution. In that world, a surgical intervention \(do(X = x)\) severs all incoming edges into \(X\) and forces \(X = x\); the resulting distribution \(P(Y \mid do(X = x))\) therefore lives in a different graphical model than the observational \(P(V)\). The territory problem is: can you express that interventional distribution using only \(P(V)\) and your knowledge of the graph, with no new data collection? The answer is yes, precisely when the intervention is identifiable, and do-calculus is the algebraic grammar that performs the translation. It borrows from the language of d‑separation so that every rewriting step is justified by a conditional independence that survives the surgical intervention. That is the shape of this answer: we cannot see the intervention, but we can compute it by rewriting the “do” away while preserving the invariances that the graph guarantees. How does this algebraic mechanism actually work?

## How it works

The key to do-calculus is that the “do” operator is not a probability measure but a change to the structural equations. Removing “do” therefore requires showing that the post-intervention distribution agrees with an observational sub-expression. There are exactly three rules that were distilled from the graphical language in the early work on SCMs; they appear in Pearl’s early exposition of causal diagrams (Pearl 1995) and are carefully unpacked in the “Introduction to Judea Pearl’s Do-Calculus” notes (Pearl 2013) [arxiv:1305.5506]. Shpitser and Pearl’s technical report “Identification of Joint Interventional Distributions” (ftp://ftp.cs.ucla.edu/pub/stat_ser/r402.pdf) proved that these rules are complete, so if an effect is identifiable, the rules will reach it; their later revisit (Shpitser & Pearl 2012) [arxiv:1210.4852v1] provides alternative characterizations that emphasize algorithmic implementation. The rules are best understood as manipulations of the truncated factorization that defines interventions.

When you intervene by setting \(X = x\), you create a mutilated graph \(G_{do(X)}\) where the edges into \(X\) are removed but all other edges remain. The truncated factorization of the interventional distribution is then

\[
P(V \setminus \{X\} \mid do(X = x)) = \prod_{Y \in V \setminus \{X\}} P(Y \mid \text{Pa}_G(Y))
\]

where \(\text{Pa}_G(Y)\) denotes the parents of \(Y\) in the original graph \(G\), and the product omits the factor for \(X\) because \(X\) is fixed by the intervention.
This expression shows that interventions amount to evaluating the observational conditional distributions but in a graph where the parents of \(X\) no longer influence it. Do-calculus then provides rules to swap that mutilated graph with the original observational graph whenever certain conditional independences hold.

### Rule 1: removing a do when the intervention can be ignored

Rule 1 allows you to drop the “do” when the variable you are intervening on is independent of the effect, given a conditioning set that d-separates them in the mutilated graph. Formally, if \(Y\) is d-separated from \(X\) by \(Z\) in \(G_{do(X)}\), then

\[
P(Y \mid do(X), Z) = P(Y \mid Z).
\]

Here \(Z\) does not contain descendants of \(X\) (because the mutilated graph removes the arrows into \(X\)). The intuition is that once the intervention disconnects \(X\) from \(Y\), the intervention no longer matters, and we can replace \(P(Y \mid do(X), Z)\) with the observational conditional \(P(Y \mid Z)\). This step is legal because the graphical d-separation encodes invariances that hold both before and after the intervention.

### Rule 2: exchanging do’s and observations

Rule 2 swaps an intervention on \(X\) with an observation on another variable \(Z\) when a d-separation in the graph with the intervention on \(X\) but without \(Z\)’s incoming edges holds. Specifically, if \(Y\) is d-separated from \(X\) by \(Z\) in \(G_{do(X, Z)}\) and \(Z\) is not a descendant of \(X\) in the original graph, then

\[
P(Y \mid do(X), do(Z), W) = P(Y \mid do(X), Z, W)
\]

for any additional set \(W\). Rule 2 is what lets us substitute real, observable values for future interventions—if intervening on \(Z\) does not change the effect conditionally, then we can simply observe \(Z\) instead.

### Rule 3: inserting or deleting observations with interventions

Rule 3 permits you to insert or remove observations from a “do” expression when a d-separation holds in the graph where both the intervention and the observation have been applied. If \(Y\) is d-separated from \(Z\) by \(X, W\) in \(G_{do(X)}\), then

\[
P(Y \mid do(X), Z, W) = P(Y \mid do(X), W).
\]

This means that once we have conditioned on \(W\) and applied the intervention on \(X\), observing \(Z\) does not provide extra information about \(Y\), so it can be dropped. The three rules collectively allow removal of a “do” by turning it into conditional probabilities that involve only observational data and previously justified substitutions.

### The front-door derivation as an illustration

The canonical front-door example shows the rewriting process in action. Suppose \(X\) causes \(Z\), \(Z\) causes \(Y\), and there is an unobserved confounder \(U\) between \(X\) and \(Y\). The confounder blocks direct identification of \(P(Y \mid do(X))\) because \(U\) introduces a spurious correlation. However, if (1) \(Z\) is fully mediated by \(X\), (2) \(Z\) blocks all paths between \(X\) and \(Y\) except through \(Z\), and (3) \(X\) and \(Z\) are not confounded, then the front-door formula gives

\[
P(Y \mid do(X)) = \sum_{z} P(z \mid X) \sum_{x'} P(Y \mid z, x') P(x').
\]

In this expression, \(P(z \mid X)\) and \(P(x')\) are observational probabilities, and \(P(Y \mid z, x')\) is the standard conditional distribution. The derivation is a sequential application of the three rules: use Rule 2 to replace \(do(X)\) with an observation on \(Z\), then Rule 1 to remove the do from the inner term, and finally Rule 3 to marginalize out \(X'\). The technical report (ftp://ftp.cs.ucla.edu/pub/stat_ser/r402.pdf) spells out the graphical conditions under which the sum over \(z\) is valid, ensuring that no unblocked back-door path re-enters the expression. This is the algebraic heart of do-calculus.

### Connecting the calculus to completeness and algorithms

Completeness means that if an effect is identifiable from \(P(V)\) and the graph, do-calculus will eventually rewrite it into an observable expression. Shpitser and Pearl (2006) formalized this through a recursive decomposition of identification problems and showed equivalence with the calculus-based derivation, while Shpitser and Pearl (2012) [arxiv:1210.4852v1] revisited the result with an eye toward implementation, emphasizing that the rules can be encoded in a search procedure. The more recent preprint Arxiv:2102.06626v1 extends this program to high-dimensional representation learning by showing how rule applications can be guided by invariance tests that “guess” partial graphs and then confirm them via do-calculus. Taken together, these works show that the three rules are not just heuristics but form a complete algorithmic system grounded in the truncated factorization of interventions.

### Failure modes

Do-calculus fails when the graph is misspecified or when the effect is non-identifiable: no sequence of rule applications will remove all “do” operators. It also fails if the required conditional independences cannot be established because latent confounders mask them. The completeness results tell us that such failures are not due to the insufficiency of the algebra; instead they reveal that either the causal effect truly cannot be estimated with the available data, or the graph is incomplete. Thus the calculus is both an inference engine and a diagnostic tool: failure to produce an observable expression is itself informative.

## Where the field is now

Research is currently pushing do-calculus into settings where the graph is partially unknown or the variables live in high-dimensional representations. Pearl’s introductory lecture notes (Pearl 2013) [arxiv:1305.5506] remain a touchstone for teaching the rules, while Shpitser and Pearl’s revisit (2012) [arxiv:1210.4852v1] provides the completeness proof that underlies modern identification algorithms. The 2021 preprint arXiv:2102.06626v1 builds on these foundations by proposing a causal representation learning pipeline that interleaves do-calculus rule application with neural invariance testing, allowing practitioners to hypothesize latent structures and then confirm them algebraically. On the engineering side, Databricks’ production causal AI platform (Databricks blog, 2023) uses causal graphs plus DoWhy-style engines to answer counterfactual business queries—by framing each question as a do-calculus derivation, the platform can automatically translate it into a sequence of SQL queries over logged exposures and outcomes, delivering answers that are auditable and reusable across feature stores. These developments show that the algebra of do-calculus now underpins both theoretical guarantees and real-world decision systems.

## What's still open

1. Can we guarantee identifiability when only partial graphs are available and the remaining relations are represented by neural latent variables, where the latent space itself is learned from scratch? 
2. How do we extend do-calculus to non-linear, high-dimensional systems so that the search for valid conditioning sets becomes tractable even when conditional independence tests rely on deep kernels?
3. Can we quantify the value of new measurements—that is, find the minimal set of additional variables whose observation would make an otherwise non-identifiable effect identifiable via do-calculus?
4. How can do-calculus be formalized when the causal graph evolves over time, such as in sequential decision-making, and the “do” operator must respect constraints on when and how interventions can be applied?

## Where to read next

If you want the structural backbone, → [Structural Causal Models](structural-causal-models.md) shows how SCMs define the graphs that the calculus manipulates. The engineering counterpart is → [Causal discovery](causal-discovery.md) which describes automated tools for learning those graphs from data, often using the same invariance principles that motivate rule applications. The theoretical foundation lives in → [[identification]] where the proofs from Shpitser & Pearl are unpacked, and the production arc is → [[causal-ml-pipeline]] which explains how to operationalize these estimators in a serving stack.

## Build it

Building the front-door estimator by hand forces you to trace every rule of do-calculus over your own synthetic graph and makes you accountable for each substitution instead of hiding the work behind a library. The recipe proves that the calculus is not an abstract proof but a sequence of verbal steps you can implement in Python on top of real data.

**What you're building:** a Python causal estimator that uses `networkx` to represent a graph with an unobserved confounder, and manually applies the three rules to compute \(P(Y \mid do(X))\) via the front-door formula from a synthetic dataset.

**Why this is valuable:** the build captures the hard part of applied do-calculus—the choice of conditioning sets and the algebraic substitutions—so running the code forces you to justify each rule before producing the final probability estimate.

**Stack:**
- **Model:** not applicable (graph + estimator)
- **Dataset:** synthetic `numpy` arrays of \(X, Z, Y\) constructed with known structural equations
- **Framework:** `networkx` for graphs, `numpy` for data + `matplotlib` for visualization
- **Compute:** free Colab T4 (12 GB VRAM) or any CPU notebook (a few seconds per sampling round)

**The recipe:**
1. `pip install networkx matplotlib numpy` and import the packages, then define a `networkx.DiGraph` with nodes \(X, Z, Y\) plus a latent confounder (store the confounder as metadata rather than a node).
2. Simulate data with \(X\) drawn from a Bernoulli, \(Z\) sampled deterministically from \(X\) plus Gaussian noise, and \(Y\) computed via a function of \(Z\), \(X\), and a latent confounder; store the joint distribution as a `numpy` array.
3. Write helper functions that compute conditional probabilities (e.g., \(P(Z \mid X)\), \(P(Y \mid Z, X')\), \(P(X')\)) from the joint by counting occurrences.
4. Apply the front-door formula: sum over \(z\) the term \(P(z \mid X)\) times the inner sum over \(x'\) of \(P(Y \mid z, x') P(x')\); use `matplotlib` to plot the resulting \(P(Y \mid do(X=0))\) and \(P(Y \mid do(X=1))\).
5. Document each substitution as a comment tied to a rule of do-calculus (Rule 1, 2, or 3) and verify that the observational expressions only involve the joint probabilities you computed.

**Expected outcome:** a runnable Colab notebook that demonstrates, step by step, how a counterfactual query with a latent confounder can be answered via the front-door formula and the three rules of do-calculus.

- **CS student:** Run the notebook on Colab with smaller synthetic datasets (e.g., 500 samples), add inline asserts that check the sum-to-one properties of each conditional probability, and plot the distributions with histograms to confirm the calculus derivation matches the simulation.
- **Applied engineer:** Extend the notebook into an API that loads logs from a CSV, computes the front-door estimator, and deploys the result via a Flask app with quantization of the coefficients to stay within 50 ms inference on an A10-backed container.
- **Applied researcher:** Hypothesize that the front-door estimator will drift when the latent confounder’s variance increases; vary the noise level in the simulator, recompute \(P(Y \mid do(X))\), and report the divergence between the estimator and the SCM ground truth to test if the algebraic steps remain valid.
- **Frontier researcher:** Use the notebook to probe the open question of partial graphs by removing one edge from the `networkx` representation, re-running the derivation, and checking whether any rule application fails—document the falsification criterion as “a rule cannot remove the remaining `do`” and collect the conditions under which identifiability collapses.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*