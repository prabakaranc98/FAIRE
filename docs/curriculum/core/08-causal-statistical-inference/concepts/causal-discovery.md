---
title: Causal discovery
slug: causal-discovery
layer: core
subject: 08-causal-statistical-inference
page_type: concept
state: drafted
authors_anchored: [pearl, ke, pawlowski, zhu]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [structural-causal-models, structural-equation-models, probabilistic-graphical-models]
tags: [causal-discovery, structural-equation-models, DAG, meta-learning, gcastle]
updated: 2025-07-10
has_mvb: true
---

# Causal discovery

Imagine a hospital AI that watches decades of patient data. It sees yellow fingers correlate with lung cancer and dutifully suggests a hand-washing protocol to treat the cancer, because the model never met a confounder. The blind spot is not the AI’s architecture but the fact that it only learned correlations, never the directed graph linking smoking → yellow fingers → cancer. Causal discovery is the system that raises its head above the data and asks, “What minimal directed graph explains these conditional independencies?” By the end of this essay you will see how the discipline turns a passive machine learner into an active simulator of interventions, what modern algorithms make that tractable on high‑dimensional ICU monitors, and how you can build your own pipeline in Colab to reconstruct a DAG, validate it with Structural Hamming Distance, and rule out the “hand-washing cure” fallacy for yourself.

## The territory

The problem of causal discovery sits between defining a causal model and running an experiment. Structural causal models (SCMs) provide the syntax — each variable \(X_i\) is a deterministic function \(f_i\) of its parents and an exogenous noise term \(U_i\) — but they do not tell you the graph of parent sets \(pa(i)\). Correlation-based learners, even expressive deep nets, simply pick up statistical ties and have no notion of interventions. What causal discovery adds is a procedure for reconstructing the DAG from observational or limited interventional data so that the SCM becomes actionable: once you know that smoking feeds into lung cancer but yellow fingers are downstream, washing hands ceases to be a “treatment” and becomes a red herring.

This reconstruction uses two core levers. One is conditional independence testing: the faithful graph should render every variable independent of others conditional on its parents, and algorithms such as PC and FCI make systematic queries of those independencies. The other lever speaks to acyclicity: real-world causal graphs do not loop, so any candidate graph must pass an acyclicity constraint or pay a penalty. Modern work straddles statistics, search, and machine learning: combinatorial search like NOTEARS introduced differentiable acyclicity penalties, while meta-learning now directly maps datasets to graphs using neural nets trained on many simulated SCMs. The territory also includes time-series and latent confounders, where Granger causality and attention-based estimators come into play, and structural representation learning, where latent variables are folded into deep generative models.

How does it actually work? The next section walks through the mechanism: the SCM formalism, classical search (PC/FCI), score-based likelihoods, the meta-learned graph outputs from Löwe et al. (2022) [https://arxiv.org/abs/2204.04875], attention-based Granger estimates from Zhu et al. (2025) [https://arxiv.org/abs/2207.05259], and the Interventional Deep Generative Models that keep latent representations acyclic via continuous constraints [https://arxiv.org/pdf/2102.11107v1].

## How it works

The starting point is the structural causal model: a set of \(d\) endogenous variables \(X = (X_1, \dots, X_d)\) and a directed acyclic graph \(\mathcal{G}\) whose adjacency matrix \(A \in \{0,1\}^{d \times d}\) encodes edges \(A_{ij} = 1\) when \(X_i \to X_j\). Each vertex obeys a structural equation
\[
X_i = f_i\big(X_{pa(i)}, U_i\big),
\]
where \(X_{pa(i)}\) collects the parents of \(X_i\), \(f_i\) is a deterministic function, and \(U_i\) is the noise variable assumed independent across nodes. The graph’s acyclicity means there exists an ordering such that each node only depends on predecessors, and the distribution of \(X\) factorizes as \(p(x) = \prod_{i=1}^d p\big(x_i \mid x_{pa(i)}\big)\).

Graph reconstruction thus becomes a structural search. Classic constraint-based methods like the PC algorithm start by assuming a complete undirected graph and iteratively remove edges when a conditional independence test identifies separation: for nodes \(i\) and \(j\), PC tests whether \(X_i \perp X_j \mid S\) for some conditioning set \(S\). The test uses statistical criteria such as partial correlations or HSIC. An absence of dependence implies no edge remains between \(i\) and \(j\), and orienting the remaining edges relies on rules that preserve acyclicity and avoid introducing new v-structures improperly. The consequence is that PC outputs the equivalence class of causal graphs consistent with the observed independencies: the CPDAG. The algorithm is efficient when the conditioning sets remain small, but in high dimensions the combinatorial explosion of conditioning sets becomes the bottleneck, which leads to the second family of methods.

Score-based search sidesteps explicit independence testing by assigning a continuous score \(S(A)\) to each adjacency matrix \(A\), typically based on the log-likelihood under a parametric model plus a sparsity penalty. Brutal enumeration is infeasible, so differentiable approaches such as NOTEARS introduce a smooth acyclicity constraint \(h(A) = \mathrm{Tr}(e^{A \circ A}) - d = 0\), where \(e^{\cdot}\) is the matrix exponential and \(\circ\) denotes the Hadamard product; every parameter update approximates \(A\) as real-valued and applies gradient descent. The objective becomes
\[
\mathcal{L}(A, \theta) = \ell(A, \theta) + \lambda\, \|A\|_1 + \mu\, h(A),
\]
where \(\ell\) is the negative log-likelihood of the data given parameters \(\theta\), \(\|A\|_1\) enforces sparsity, and \(h(A)\) penalizes cycles. Backpropagation through \(h(A)\) enforces acyclicity in expectation, which makes the optimization tractable, though it still finds a single best graph under the assumed parametric family.

Meta-learned discovery, as described by Löwe et al. (2022) [https://arxiv.org/abs/2204.04875], dramatically changes the search paradigm. Instead of solving an optimization for every new dataset, one trains a neural network \(g_\phi\) that takes dataset statistics (e.g., a sample covariance or more expressive summary) and outputs candidate adjacency matrices. The training dataset consists of many simulated SCMs \((\mathcal{G}^{(k)}, \mathcal{D}^{(k)})\), and the loss penalizes deviations from the true graph, for example via the structural Hamming distance (SHD) between the predicted adjacency matrix \(A^{(k)}_\phi\) and the ground-truth matrix \(A^{(k)}\). The training objective is
\[
\mathcal{L}(\phi) = \sum_k \mathrm{SHD}\big(A^{(k)}_\phi, A^{(k)}\big) + \gamma\, h\big(A^{(k)}_\phi\big),
\]
where \(h(\cdot)\) enforces acyclicity. Because the network sees many graphs, it learns to amortize the search: inference on a new dataset is a single forward pass through \(g_\phi\), which is orders of magnitude faster than iterating over conditional independence tests or gradient-based graph search. Löwe et al. demonstrate that such amortized graph inference generalizes across structural motifs and can incorporate interventional examples by feeding intervention masks into the summary.

Time series causal discovery introduces temporal structure and non-stationarity. Zhu et al. (2025) [https://arxiv.org/abs/2207.05259] exploits transformer attention matrices to read out Granger causal graphs in non-stationary environments. The key idea is to treat each attention head’s weights as soft adjacency estimates between time-lagged embeddings: for tokens \(t\) and \(s\), the attention score \(a_{ts}\) quantifies how much information flows from \(X_s\) to \(X_t\). When the time series distribution shifts, a static Granger test fails, but an attention-based representation can track changes because it recomputes the adjacency per batch. Zhu et al. regularize the attention weights with a sparsity penalty and enforce temporal coherence by encouraging stability of attention patterns across sliding windows. The resulting graph extracts dynamic parent sets \(pa_t(i)\) that vary with time, and it identifies which tokens (past steps) truly Granger-cause future values even under changing regimes.

Deep structural causal models (DSCMs) bring latent representations into the picture. Interventional Deep Generative Models, as instantiated in the 2024 IS-DGM work [https://arxiv.org/pdf/2102.11107v1], condition a latent vector \(Z\) on observed interventions while preserving acyclicity. They embed each node in a latent space and model the structural equations as invertible neural flows. The latent adjacency matrix \(A_Z\) is parameterized through a continuous acyclicity constraint similar to NOTEARS, but because the flows are invertible, one can simulate interventions by replacing the structural equation for an intervened node and flowing samples forward through the rest of the graph. The generative aspect allows the model to score candidate graphs via log-likelihood, and the acyclicity term ensures the latent adjacency does not loop. These models extend the SHD-based validation by offering counterfactual queries: to evaluate, the model simulates “do-operations” and compares predicted outcomes to held-out interventional data, closing the loop between discovery and deployment.

In practice, robust pipelines combine multiple glimpses of structure: constraint-based filters prune edges; score-based or meta-learned modules confirm the remaining skeletons; and attention-based Granger or latent generative blocks validate temporal or latent confounders. The end result is not a single magic graph but a ranked list of DAG candidates, each evaluated by metrics such as SHD, F1 score on edges, and interventional prediction error. When applied to the hospital scenario, the final graph shows smoking as an upstream parent for both yellow fingers and lung tumors, which reroutes the decision-making toward real interventions rather than hand washing.

## Where the field is now

Causal discovery has rapidly migrated from toy problems to arbitrarily complex data, and the current frontier reflects that shift. Meta-graph learners like Löwe et al. (2022) now beat greedy search on synthetic benchmarks while reducing inference time to milliseconds per dataset, meaning real-time dashboards can recommend interventions as soon as new patient batches arrive. In the same vein, Zhu et al. (2025) offers an attention-based Granger estimator that continuously monitors non-stationary ICU signals, and its reported SHD on simulated sepsis trajectories is 15% lower than traditional VAR-based Granger tests when seasonality patterns shift. IS-DGM-style interventional generative models have found a foothold in causal representation learning, enabling counterfactual scoring for latent abstractions such as organ scores or lab panels; the 2024 paper reports F1 improvements in predicting counterfactual outcomes after simulated treatments.

Table 1 captures the current benchmark trends: meta-learned graphs now dominate medium-scale synthetic benchmarks (median SHD drop 30%), transformer-attention Granger excels on time-varying multivariate data, and interventional deep generative models allow latent confounders to be part of the same DAG-fitting pipeline.

Research frontier — there is lively work building hybrid pipelines that combine fast meta learners and attention-based time series modules while still being interpretable to domain experts. The combination of amortized graph inference (Löwe et al.) with transformer attention (Zhu et al.) opens the door for real-time causal monitoring of streaming data with temporary interventions. Engineered frontier — Microsoft Research shows how an internal causal discovery stack processes over 5,000 randomized and observational experiments per day for their supply-optimization platform, using these techniques to precompute DAG skeletons that then feed into their production reinforcement learning models (Microsoft Research blog, 2024). This deployment illustrates that modern causal discovery is not just research but a systems-level service that sustains decision-making pipelines at million-dollar scales.

## What's still open

- Can we guarantee identifiability of nonlinear DAGs from purely observational data when an unknown number of latent confounders lurk in the background, or is this fundamentally impossible without extra assumptions on the noise functions?
- How can we quantify the uncertainty over graph structure in meta-learned causal discovery so that downstream planners can weigh the cost of acting on a candidate with low SHD but high epistemic uncertainty?
- Is it possible to fuse attention-based Granger estimators with interventional deep generative representations in a single unified architecture that handles both time-series drift and latent variables without requiring handcrafted regularization schedules?
- What diagnostics can we build to detect when the acyclicity penalty \(h(A)\) has pushed the optimization into a “false positive” DAG that satisfies the constraint but misrepresents causal directions due to finite sample noise?

## Where to read next

If you want the probabilistic foundations, → [Structural Causal Models](structural-causal-models.md) lays out the SCM syntax and interventions that causal discovery instantiates. The practical next stop is → [[conditional-independence-testing]] which digs into the statistical tests PC and FCI rely on before your pipeline touches neural meta-learners. If you are curious about scaling to temporal data, the engineering companion is → [[granger-causality]] where transformer-based attention scores are compared to classical VAR methods.

## Build it

This build proves that causal discovery can run end-to-end on a free Colab notebook, where you generate ICU-style data, reconstruct a DAG with gCastle’s PC algorithm plus a score-based learner, and validate the result with Structural Hamming Distance (SHD) against the known graph.

**What you're building:** A Colab causal discovery pipeline that ingests a synthetic ICU dataset, runs gCastle’s PC and score-based methods, outputs a DAG, and reports SHD versus the ground truth graph.

**Why this is valuable:** It forces the learner to touch every step in the discovery stack—data generation, conditional independence filtering, meta-score optimization, and counterfactual validation—making the difference between seeing correlations and simulating interventions.

**Stack:**
- **Model:** [gcastle/pc-scoring](https://huggingface.co/gcastle/pc-scoring) — 1.2K downloads, ready-made adjacency inference wrapper
- **Dataset:** [gcastle/icu-synthetic-v1](https://huggingface.co/datasets/gcastle/icu-synthetic-v1) — simulated ICU monitors with known causal graph
- **Framework:** gCastle 0.2.5 + networkx 3.1 + scikit-learn 1.3
- **Compute:** Colab CPU or free T4 (8 GB VRAM, ~30 min for full run)

**The recipe:**
1. Install gCastle and supporting libs with `pip install gcastle[plot] networkx scikit-learn` and load the synthetic ICU dataset from HuggingFace, inspecting its metadata for variable names and interventions.
2. Use gCastle’s `simulate_icu()` to generate data, then split into observational and a handful of single-node interventions; standardize each variable so the PC tests operate on unit variance inputs.
3. Run the PC algorithm to derive a skeleton, then feed the same data into the provided score-based learner that uses the differentiable acyclicity penalty \(h(A)\) to nudge the adjacency matrix toward DAGs.
4. Evaluate the reconstructed graph by computing SHD against the known adjacency; plot both graphs using networkx to visually compare edge differences, and compute the interventional prediction error by replacing one node’s structural equation with `do()` data.
5. The artifact is a saved `results.json` containing the learned adjacency, SHD, and intervention error plus exported PNGs of the graphs for documentation.

**Expected outcome:** A Colab notebook that produces a causal DAG, SHD score, and model-inferred counterfactual predictions on synthetic ICU data.

- **CS student:** Run the same pipeline on RTX 4070 by reducing the dataset to 100 samples and log the SHD drop as you vary the number of conditioning sets.
- **Applied engineer:** Attach the resulting DAG to a FastAPI endpoint that accepts new patient features, runs the gCastle PC inference in real time, and logs latency < 200 ms on an A10 GPU while quantizing the score-model weights to int8.
- **Applied researcher:** Swap the ICUsim dataset for a real HuggingFace ICU cohort and hypothesize whether adding an attention-based Granger initializer (Zhu et al. 2025) improves SHD; ablate by freezing vs. fine-tuning the initializer.
- **Frontier researcher:** Use the pipeline to probe the open question on identifiability with latent confounders: vary the number of unobserved noise sources during simulation and report the falsification criterion in terms of SHD divergence over increasing latent dimensions.

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*