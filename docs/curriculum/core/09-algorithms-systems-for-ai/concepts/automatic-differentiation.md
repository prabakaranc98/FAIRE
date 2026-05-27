---
title: Automatic Differentiation
slug: automatic-differentiation
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [baydin, pearlmutter, werbos, biggs]
feeds_de_pillar: []
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher, frontier-researcher]
prereqs: [computational-graphs, backpropagation]
tags: [optimization, gradients, compilers, machine-learning]
updated: 2025-05-14
has_mvb: true
---

# Automatic Differentiation

When you need to optimize a complex physical simulation or a deep neural network, you require the gradient of your output with respect to millions of parameters. If you rely on finite differences—approximating the derivative by perturbing each input by a small step \(h\) and measuring the change—you encounter a fundamental dilemma: if \(h\) is too large, your truncation error dominates; if \(h\) is too small, catastrophic round-off cancellation destroys your precision. Alternatively, symbolic differentiation often causes "expression swell," where the mathematical expression tree grows exponentially, rendering your program unmanageable. Automatic differentiation (AD) solves this by treating the computer program itself as a composition of elementary operations, allowing for the exact, systematic execution of the chain rule over program traces. This enables gradient computation with the same temporal complexity as the original program, providing the bedrock for modern machine learning.

## The territory

Automatic differentiation sits at the intersection of compiler theory, numerical analysis, and machine learning. It is neither a numerical approximation nor symbolic manipulation; it is a transformation of the program itself. By decomposing any algorithm into a directed acyclic graph (DAG) of primitive steps—additions, multiplications, and transcendental functions—AD extracts sensitivity information from arbitrary code without manual derivation. The field is traditionally divided into two primary modes: forward and reverse. Forward-mode AD propagates derivatives alongside function values, making it efficient for functions with few inputs and many outputs. Reverse-mode AD, which powers modern deep learning, propagates derivatives backward from the output, making it the only tractable choice for functions with millions of parameters and a single scalar loss. This approach bridges the gap between high-level mathematical modeling and low-level hardware execution, forming the basis for how we train models today.

## How it works

The core idea of AD is to decompose a program into a sequence of elementary operations. Suppose we have a function \(y = f(x)\). We represent this as a series of intermediate variables \(v_i\), where \(v_i = \phi_i(v_j)_{j < i}\). Here, \(v_i\) is the result of an elementary operation \(\phi_i\) applied to previous variables \(v_j\). By the chain rule, the derivative of the output with respect to the input is the product of the local derivatives of each operation.

In forward-mode AD, we compute the derivative of each intermediate variable with respect to the input as we go. We define the "dual number" \(v_i + \dot{v}_i \epsilon\), where \(\epsilon^2 = 0\). When we perform operations on these dual numbers, the real part tracks the function value, and the dual part tracks the derivative. This is highly efficient for functions where the number of inputs is small, as described in the taxonomy by Baydin et al. (2018) [https://arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767). The complexity is \(O(N)\) where \(N\) is the number of inputs, but it remains independent of the number of outputs.

Reverse-mode AD, or backpropagation, reverses this flow. We first perform a forward pass to compute all intermediate values \(v_i\). Then, we perform a backward pass to compute the "adjoints" \(\bar{v}_i = \frac{\partial y}{\partial v_i}\), where \(\bar{v}_i\) represents the sensitivity of the final scalar output \(y\) with respect to the intermediate variable \(v_i\). The adjoint of an intermediate variable is computed by summing the contributions from all operations that use it:
\[ \bar{v}_j = \sum_{i: j \in \text{inputs}(\phi_i)} \bar{v}_i \frac{\partial \phi_i}{\partial v_j} \]
where \(\bar{v}_j\) is the adjoint of the variable \(v_j\), and \(\frac{\partial \phi_i}{\partial v_j}\) is the local partial derivative of the operation \(\phi_i\). This summation effectively aggregates the gradient flow from all downstream paths. Pearlmutter & Siskind (2008) [https://engineering.purdue.edu/~qobi/papers/toplas2008.pdf](https://engineering.purdue.edu/~qobi/papers/toplas2008.pdf) demonstrated that AD can be implemented in functional frameworks by treating the program as a higher-order function, avoiding "perturbation confusion" where the derivative of a derivative is incorrectly computed. Furthermore, Werbos (1990) [https://arxiv.org/abs/1812.01892](https://arxiv.org/abs/1812.01892) showed that when the computational graph contains cycles, the backward pass naturally unfolds the graph over time, performing backpropagation through time (BPTT).

## Where the field is now

The field has shifted from static graph construction to dynamic "define-by-run" approaches. PyTorch popularized this by building the graph on the fly during the forward pass, allowing for flexible control flow. The frontier is currently moving toward "JIT-compiled AD." Systems like JAX use XLA to compile the entire computational graph into fused GPU kernels, which solves the memory-bandwidth bottlenecks inherent in naive reverse-mode implementations by minimizing global memory access. Biggs (2000) [https://people.cs.vt.edu/~asandu/Public/Qual2011/Optim/Biggs_2000_AD.pdf](https://people.cs.vt.edu/~asandu/Public/Qual2011/Optim/Biggs_2000_AD.pdf) laid the groundwork for these compiler-based approaches. Modern systems now utilize source-to-source AD, where the compiler transforms the user's code into a new program that computes both the function and its gradient, as detailed in the implementation guide by Wang et al. (2024) [https://arxiv.org/html/2402.16020v2](https://arxiv.org/html/2402.16020v2). In production, frameworks like PyTorch and JAX are the industry standard, powering large-scale training at companies like Meta, Google, and OpenAI, where JIT-compilation is essential for scaling to thousands of GPUs.

## What's still open

Despite the maturity of AD, several frontiers remain. First, how can compilers automatically synthesize optimal, fused GPU kernels for reverse-mode AD of highly dynamic, control-flow-heavy code without incurring massive memory-bandwidth bottlenecks? Second, can we develop "checkpointing-aware" AD that automatically decides which intermediate activations to store and which to recompute to minimize memory usage in massive models? Finally, standard AD fails at non-differentiable points such as `step` functions or sorting; can we develop mathematically rigorous "surrogate gradients" that allow gradient-based optimization to proceed through these discontinuities without losing convergence guarantees?

## Where to read next

If you want the probabilistic foundation, → [[score-matching]] gives the likelihood-free training perspective that AD compiles down to. The engineering counterpart is → [[flash-attention]] explaining how the graph is made fast enough to be trained at this scale. For the next paradigm, → [[flow-matching]] generalizes the noising process to arbitrary continuous paths, requiring higher-order derivatives that test the limits of current AD engines.

## Build it

You will build a lightweight, reverse-mode AD engine ("micro-grad") from scratch in pure Python.

**What you're building:** A reverse-mode AD engine that trains a single-neuron classifier on a 2D Moons dataset.
**Why this is valuable:** It forces you to implement the backward pass manually, revealing the memory-management challenges of reverse-mode AD.
**Stack:**
- **Model:** Custom single-neuron MLP (Python)
- **Dataset:** `sklearn.datasets.make_moons`
- **Framework:** Pure Python + NumPy
- **Compute:** CPU (local)

**The recipe:**
1. Define a `Value` class that stores `data`, `grad`, and `_backward` function.
2. Implement operator overloading for `__add__`, `__mul__`, and `__relu__` to build the graph.
3. Implement the `backward()` method that performs a topological sort and calls `_backward` on each node.
4. Train the neuron using stochastic gradient descent on the Moons dataset (learning rate 0.01, 100 iterations, batch size 1).
5. Verify convergence by ensuring the loss drops below 0.1.

**Expected outcome:** A visual plot of the decision boundary and a loss curve showing convergence.

**Variants per persona:**
- **Applied AI/ML engineer:** Profile `torch.autograd` on a 7B model; reduce peak VRAM by 20% using `torch.utils.checkpoint` and verify latency.
- **Research engineer:** Reproduce the gradient calculation of a simple MLP in PyTorch and verify the result against your custom engine within 1e-6 tolerance.
- **Applied researcher:** Test the hypothesis that recomputing activations (gradient checkpointing) is faster than storing them for a 100-layer MLP; plot time vs. memory.
- **Frontier researcher:** Implement a "surrogate gradient" for a `step()` function and test if it allows convergence in a discrete-input model; report the final accuracy on the `make_moons` dataset.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*