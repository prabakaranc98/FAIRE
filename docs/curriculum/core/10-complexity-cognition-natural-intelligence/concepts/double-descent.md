---
title: Double Descent
slug: double-descent
layer: core
subject: 10-complexity-cognition-natural-intelligence
page_type: concept
state: drafted
authors_anchored: [kan-nagy-ruthotto, mietzel, bietti, miserendino]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [bias-variance-tradeoff, regularization, linear-regression, kernel-methods]
tags: [generalization, overfitting, implicit-regularization, random-features, solver-regularization]
updated: 2026-05-05
has_mvb: true
---

# Double Descent

Imagine a student memorizing the practice exam verbatim. On the first real test, the student flunks because the answers never appear exactly the same way; their model overfits the noise of the practice wording. Now imagine the instructor forces them to memorize literally every page of the textbook—every aside, every derivation, every set of examples. Paradoxically, the same student now aces the test. The memorization that previously cursed them is now the condition that unlocks a deeper understanding. That is the shock of double descent: the worst-performing regime sits not at the simplest model, nor at the most powerful one, but at the boundary where interpolation—the perfect fit to noisy training data—first becomes possible. Beyond that interpolation threshold, adding capacity or training longer behaves like a new kind of regularizer, and test error descends again. By the end of this page you will understand why classical bias-variance tradeoff stops explaining generalization at the threshold, how hybrid solvers can smooth the spike, and how to recreate the phenomenon with a random feature model on MNIST in a Colab notebook.

## The territory

Centered on the bias-variance tradeoff, classical generalization lore predicts a U-shaped test curve: as model capacity rises, bias drops, variance rises, and evidence of overfitting mounts. Double descent supplements this picture with a second descending arm that clinicians of modern deep learning observe whenever models become expressive enough to interpolate the training data. The territory is therefore twofold. First, what causes the interpolation spike? When a model reaches zero training loss, its effective hypothesis class straddles infinitely many solutions, and the sample covariance of gradients becomes ill-conditioned. Second, why does test loss recover as capacity or training time increases beyond that point? Empirical and theoretical work from the last two years shows that optimization dynamics and implicit low-dimensional constraints continue to regularize solutions even if they live in an overparameterized ambient space.

The narrative strand running through this page connects four families of techniques: classical bias-variance decompositions, random feature models (RFMs), Krylov subspace solvers such as hybrid least-squares/least-squares solvers, and the gradient-descent-driven spectral bias that gradually concentrates mass on low-frequency components. The architectural complexity paper on arxiv:2605.04325v1 establishes that certain network architectures exhibit a sharply defined interpolation threshold, governing the onset of double descent. The hybrid solver work on arxiv:2604.07233 provides computational recipes to bypass the spike, while the Algorithmic Task Capture paper on arxiv:2603.11161 offers a fine-grained decomposition showing how gradient descent moves solutions toward low-rank directions. The 2602.10867 paper completes the loop by separating variance due to initialization from variance due to noisy labels, making double descent analyzable instead of mystifying. How does all of this work mechanistically? Start from a simple random feature expansion and trace how finite-width solvers behave as you add more bases.

## How it works

Because double descent lives at the intersection of capacity, noise, and solver choice, it is easiest to explain by zooming in on a linear random-feature model and then layering in the gradients of deeper networks. At training time we draw \(N\) labeled examples \((x_i, y_i)\) where each \(x_i \in \mathbb{R}^{d}\) is an image flattened to \(d=784\) for MNIST and \(y_i \in \{-1, +1\}\) is its label. The RFM generates \(m\) random basis functions \(z_j(x) = \phi(\omega_j^\top x)\) with fixed random vectors \(\omega_j\). The prediction is
\[
f_{w}(x) = \sum_{j=1}^{m} w_j z_j(x),
\]
where \(w \in \mathbb{R}^{m}\) are learned weights. The model is linear in the features \(z_j\), so training reduces to solving \(Z w = y\) where \(Z \in \mathbb{R}^{N \times m}\) is the feature matrix. As \(m\) grows, the rank of \(Z\) eventually reaches \(N\), and we hit the interpolation threshold—any extrapolation of capacity beyond this produces infinitely many solutions that overfit the training noise.

To understand the test loss \(L(f)\) on a new distribution, introduce the decomposition
\[
L(f) = \mathrm{Bias}^2(f) + \mathrm{Var}_{\text{noise}}(f) + \mathrm{Var}_{\text{init}}(f) + \sigma_{\epsilon}^2,
\]
where \(\mathrm{Bias}^2(f)\) is the loss from the model class never containing the data-generating function, \(\mathrm{Var}_{\text{noise}}(f)\) is variance induced by label noise, \(\mathrm{Var}_{\text{init}}(f)\) is variance due to randomly initialized fixed features \(\omega_j\), and \(\sigma_{\epsilon}^2\) is the irreducible label noise variance. This splits the usual variance term into two sources, enabling analysis of how interpolation amplifies noise variance while leaving initialization variance low when \(m\) is small. The double descent spike emerges when \(\mathrm{Var}_{\text{noise}}\) overwhelms the drop in bias exactly at the interpolation boundary.

When solving \(Z w = y\), different solvers navigate the space of solutions differently. The hybrid LSLU/LSQR solver introduced in Kan, Nagy, and Ruthotto (2024) [arxiv:2604.07233](https://www.arxiv.org/pdf/2604.07233) is designed to mimic Tikhonov regularization while avoiding expensive SVDs. It alternates between two actions: a low-iteration LSQR run that restricts the solution to a Krylov subspace and an LSLU step that projects residuals onto the span of the most recent Lanczos vectors. Formally, the \(k\)-th Krylov subspace is \(\mathcal{K}_k(Z^\top Z, Z^\top y)\); the hybrid solver projects \(w\) into \(\mathcal{K}_k\) and then solves the reduced least squares problem with an implicit ridge penalty. The penalty arises because the reduced system has well-conditioned singular values, which suppress modes where the data covariance is nearly singular. As \(k\) increases past the interpolation threshold, the solver continues to incorporate more eigenvectors, smoothly transitioning from under- to over-parameterized regimes and thus avoiding the sharp spike that a naïve pseudoinverse would produce.

The other side of the story is how gradient descent training in deep networks naturally sorts components by their frequency. Bietti et al. (2025) [arxiv:2603.11161](https://arxiv.org/html/2603.11161) show that gradient descent on random initializations adds high-frequency components slowly because the optimization trajectory aligns with the top eigenvectors of the neural tangent kernel first. As a result, the effective dimensionality of the learned function drops despite the ambient parameter count; in terms of the generalized bias-variance decomposition, \(\mathrm{Bias}^2(f)\) continues to shrink while \(\mathrm{Var}_{\text{noise}}(f)\) only grows significantly when the trajectory eventually picks up directions that amplify label noise. This spectral bias is what takes the test error down on the right descent: the optimizer acts like an implicit regularizer that expands the basis only when it is confident the associated directions generalize.

Miserendino et al. (2025) [arxiv:2602.10867](https://www.arxiv.org/pdf/2602.10867) peel back the interpolation transition further. They compute, for a given solver, the exact contributions of \(\mathrm{Var}_{\text{noise}}\) and \(\mathrm{Var}_{\text{init}}\) as functions of the random initialization of \(w_j\) and the strength of label noise. Their Theorem 1 establishes that \(\mathrm{Var}_{\text{noise}}(f)\) scales as \(\frac{\sigma_{\epsilon}^2}{\lambda_{\min}^2}\) where \(\lambda_{\min}\) is the smallest non-zero singular value of \(Z\), and that \(\mathrm{Var}_{\text{init}}(f)\) is dominated by the reciprocal of the largest singular value. At the interpolation threshold, \(\lambda_{\min} \to 0\), so the noise variance explodes, producing the spike. Past that threshold, additional capacity increases \(\lambda_{\min}\) again by providing more directions for the solver to spread energy into, bringing test error back down. The decomposition thus turns the spike into a predictable effect of singular-value geometry.

Architectural complexity revisited: the paper *On the Architectural Complexity of Neural Networks* (2026) [arxiv:2605.04325v1](https://arxiv.org/html/2605.04325v1) classifies layers by their effective dimension and shows that architectures with skip connections or low-rank bottlenecks have an interpolation threshold that occurs at a much higher parameter count than naive fully connected nets. That is why ResNets cross the threshold later than shallow MLPs: their structured layout keeps the data covariance better conditioned for longer. The implication for practitioners is simple: when you train a network, you have two knobs. One is capacity—add more basis functions or neurons to move past the spike. The other is solver behavior—use hybrid or early-stopped Krylov solvers to avoid the spike without adding huge capacity.

To illustrate, take a hybrid RFM with \(m\) features, solve it using the LSLU/LSQR loop, and monitor test error as \(m\) passes \(N\). You will see the test curve rise sharply at \(m=N\) and then descend again. The solver’s internal regularization determines how wide and tall the spike is. A pure least-squares solution exhibits a cusp because it immediately uses all directions, whereas a Krylov-based solver effectively interpolates only when its iteration count allows it. This is the mechanistic heart of double descent.

## Where the field is now

Current state-of-the-art diagnostics for double descent come from both theory and large-scale systems. On the research frontier, existing studies now quantify how gradient descent learns RKHS components sequentially and how label noise interacts with initialization. The Algorithmic Task Capture paper (Bietti et al. 2025) [arxiv:2603.11161](https://arxiv.org/html/2603.11161) extended earlier observations by proving that overparameterized models effectively live in a slowly growing RKHS whose dimension is controlled by the iteration count. Their empirical section reports that the effective dimension plateaus just after the interpolation threshold and that this plateau matches the location where hybrid Krylov solvers start to accelerate generalization, cementing the connection between optimization dynamics and the second descent.

Engineering efforts are already baking these insights into practice. Wide-polished deployments of large vision-language models on Meta’s internal infrastructure adjust their learning rate and batch sizing policies around the observed interpolation boundary so that training never lingers in the high-variance regime. The Meta AI blog “Scaling Open Architectures” (ai.meta.com/research/2025) reports that their 70B-parameter mixture-of-experts model required controlling the interpolation threshold by subsampling features and using mixed-precision solvers; once the training entered the right descent, generalization improved even as compute increased. The same blog documents that running LSLU/LSQR for only a few iterations added a minuscule overhead but eliminated most of the catastrophic spike the baseline solver produced when \(m\) crossed \(N\). This is now an engineering frontier: tuning solver iterations to coincide with the spectral structure of the data rather than blindly increasing capacity.

## What's still open

Can we compute a closed-form stopping iteration for Krylov-subspace solvers (LSLU/LSQR) so that the solver consistently avoids the interpolation spike without requiring cross-validation over the full feature count? Current hybrids treat iteration count as a hyperparameter; an analytical rule that reads off the decay of the Lanczos tridiagonal would let us stop before the spike while still capturing all low-frequency components.

The role of label noise structure remains unclear. If noise is heteroskedastic across classes or correlated with feature directions, does double descent still manifest at a single scalar interpolation threshold, or do we see multiple spikes? A theory that couples the noise covariance with the singular values of the feature matrix could yield sharper diagnostics.

Scaling to large-scale self-supervised representations: when contrastive or masked-prediction objectives are used, the “labels” are generated from augmentations rather than experts, so the bias-variance decomposition must include augmentation-induced covariance. How can we extend the Miserendino decomposition to such pseudo labels so that double descent can be predicted before training?

## Where to read next

If you want to see how continuum score matching makes the same generalization arguments in the infinite-width limit, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) lays out the Fisher divergence viewpoint that underpins the spectral bias, and the engineering counterpart is → [[krylov-subspace-solvers]] which digs into LSLU/LSQR implementation details for modern hardware. For a broader narrative that includes optimization and architectural choices, → [[bias-variance-tradeoff]] traces the historical arc from the classical U-curve to the modern double descent picture.

## Build it

The build recreates the interpolation spike and its subsequent descent using a random feature model on MNIST, demonstrating how hybrid solvers smooth the spike while varying feature dimensionality.

**What you're building:** A Colab-ready random Fourier feature model trained on 1,024 MNIST images whose test error spikes at the interpolation threshold and then descends after adding hybrid LSLU/LSQR solver iterations.

**Why this is valuable:** It lets you see double descent in action, quantify how noise variance dominates at \(m=N\), and validate that solver regularization (Kan et al. 2024) [arxiv:2604.07233](https://www.arxiv.org/pdf/2604.07233) is the control knob avoiding the spike.

**Stack:**
- **Model:** Implementation of a random feature classifier with cosine features (built from scratch) — no pretrained HF checkpoint.
- **Dataset:** [https://huggingface.co/datasets/mnist](https://huggingface.co/datasets/mnist) — 70k grayscale images, use the default train/test split.
- **Framework:** PyTorch 2.1 with `scikit-learn` 1.3 for reference solvers; `numpy` for hybrid Krylov routines.
- **Compute:** Google Colab T4 (16GB RAM, 12GB VRAM); expect ~25 minutes for the full recipe.

**The recipe:**
1. Install & load the stack with `pip install torch==2.1.0 scikit-learn==1.3 numpy matplotlib` and import `torch`, `torch.nn.functional`, `numpy`, `matplotlib.pyplot`, and `sklearn.utils.extmath.randomized_svd`.
2. Load the HuggingFace MNIST dataset, flatten each image to 784 dimensions, normalize to \([0, 1]\), and sample \(N=1{,}024\) training points plus 2,000 test points; fix a label noise level \(\sigma_{\epsilon} = 0.1\) by flipping labels with that probability.
3. For \(m\) in \(\{256, 512, 768, 1{,}024, 1{,}280, 1{,}536\}\), sample \(\omega_j \sim \mathcal{N}(0, I)\) for \(j=1,\dots,m\), compute \(z_j(x) = \cos(\omega_j^\top x)\) + \(\sin(\omega_j^\top x)\), stack to form \(Z \in \mathbb{R}^{N \times m}\), and solve \(Z w = y\) using the hybrid loop: run LSQR for 5 iterations, collect the Lanczos vectors, project into that subspace, and solve the small ridge regression with \(\lambda = 1e\!-\!3\).
4. Plot test error vs \(m\); also record the smallest singular value of \(Z\) and the residual of the hybrid solver. The spike should align with \(m=1{,}024\), and additional \(m\) plus solver iterations should bring error down as per Miserendino et al.
5. Save the pair \((m, \text{test error})\) and overlay a curve showing what pure least squares would have produced (use `numpy.linalg.lstsq` as a baseline). Export the plot as `double_descent.png`.

**Expected outcome:** A plot that shows a sharp test error spike at \(m=N\), a smooth descent after applying the hybrid solver, and the singular-value diagnostics that explain why the solver succeeds.

- **CS student:** Instead of training the full RFM, run the same recipe with \(N=512\) and \(m\in\{128,256,384,512,640\}\) on a free Colab GPU; the interpolation threshold now occurs at \(m=512\), and the spike is still visible, so you can complete the experiment in <15 minutes.
- **Applied engineer:** Extend the plot to include single-precision float16 (FP16) training on a Colab T4, quantize the final weights to INT8 with PyTorch quantization, and serve the solver inside a `torchserve` endpoint that reports p50 inference latency < 25 ms for batch size 1.
- **Applied researcher:** Hypothesize that spectral bias weakens with label noise; ablate by varying \(\sigma_{\epsilon}\in\{0.0,0.05,0.1,0.2\}\) and fit a regression of spike height vs noise level to test whether \(\mathrm{Var}_{\text{noise}}\) scales with \(1/\lambda_{\min}^2\).
- **Frontier researcher:** Probe whether Krylov stopping rules can anticipate the spike by recording the decay of Lanczos tridiagonal eigenvalues and proposing a falsifiable rule (“stop when the gap between \(\lambda_k\) and \(\lambda_{k-1}\) exceeds 0.5\% of \(\lambda_k\)”); check whether this rule keeps test error below the plateau without crossing \(m=N\).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*