---
title: Automatic Differentiation
slug: automatic-differentiation
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [baydin, pearlmutter, siskind, werbos]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [computational-graphs, chain-rule]
tags: [optimization, gradients, compilers, backpropagation]
updated: 2025-05-14
has_mvb: true
---

# Automatic Differentiation

When you need to optimize a complex system—whether it is a neural network with billions of parameters or a flight control algorithm for a spacecraft—you need gradients. If you rely on numerical finite-difference approximations, where you perturb inputs by a small amount to estimate the slope, you face catastrophic round-off errors and high computational costs. If you attempt to solve this using symbolic differentiation, the branching logic and iterative loops inherent in modern software trigger "expression explosion," where the symbolic representation grows exponentially in size, rendering it intractable. Automatic differentiation (AD) bypasses both traps by instrumenting the code itself. It treats the algorithm as a composition of elementary operations, applying the chain rule systematically to transform any executable program into its own exact derivative engine. By doing so, AD provides the precise gradients necessary for high-stakes optimization without the instability of numerical methods or the bloat of symbolic expansion.

## The territory

Automatic differentiation sits at the intersection of compiler theory and numerical analysis. It is the engine that powers modern deep learning, yet it is fundamentally distinct from the symbolic manipulation found in tools like Mathematica or the finite-difference approximations used in classical engineering. The core problem AD answers is how to compute the derivative of a function \(f: \mathbb{R}^n \to \mathbb{R}^m\) defined by a computer program, where the program may contain arbitrary control flow, loops, and recursion.

The shape of the answer is a transformation of the program's execution trace. Rather than approximating the slope of a function by perturbing inputs, AD decomposes the program into a sequence of elementary operations—additions, multiplications, and transcendental functions—whose derivatives are known. By applying the chain rule to this sequence, AD computes exact derivatives at a cost proportional to the original function evaluation. This family of techniques is broadly categorized into forward-mode and reverse-mode differentiation. Forward-mode AD propagates derivatives alongside the function values, making it efficient for functions with few inputs and many outputs. Reverse-mode AD, which is the backbone of backpropagation, propagates gradients from the output back to the inputs, making it the standard for functions with many inputs and a single scalar output, such as the loss functions in neural networks. This concept appears throughout the [[09-algorithms-systems-for-ai]] arc, serving as the foundational layer for training any differentiable model.

## Core concepts

*   **Computational Graph:** A directed acyclic graph representing the sequence of operations in a program, where nodes are operations and edges are data dependencies.
*   **Forward-Mode AD:** A mode of differentiation that computes the derivative of the output with respect to a single input by propagating tangents through the graph.
*   **Reverse-Mode AD:** A mode of differentiation that computes the gradient of a scalar output with respect to all inputs by propagating adjoints backward through the graph.
*   **Perturbation Confusion:** A common failure mode in nested differentiation where the derivative of a derivative is incorrectly computed due to variable shadowing.
*   **Define-by-Run:** An AD paradigm where the computational graph is constructed dynamically during the execution of the program (e.g., early PyTorch).
*   **Define-by-Transform:** An AD paradigm where the program is analyzed and transformed into a static graph or optimized IR before execution (e.g., JAX).

## How it works

The mechanism of automatic differentiation relies on the observation that any computer program can be decomposed into a sequence of primitive operations. If we represent this sequence as a computational graph, each node corresponds to an elementary operation, and each edge represents the flow of data.

Consider a function \(y = f(x)\) composed of a sequence of intermediate variables \(v_i\). We define the function as a series of assignments \(v_{i} = \phi_i(v_{i-1}, \dots, v_{i-k})\), where \(\phi_i\) is an elementary operation. In forward-mode AD, we compute the tangent \(\dot{v}_i = \frac{\partial v_i}{\partial x}\). By the chain rule, we have \(\dot{v}_i = \sum_{j < i} \frac{\partial \phi_i}{\partial v_j} \dot{v}_j\), where \(\frac{\partial \phi_i}{\partial v_j}\) is the partial derivative of the operation \(\phi_i\) with respect to its \(j\)-th argument. This approach is highly efficient when the number of inputs is small.

Reverse-mode AD proceeds in two phases: a forward pass to compute function values and a backward pass to compute gradients. We define the adjoint \(\bar{v}_i = \frac{\partial y}{\partial v_i}\), representing the sensitivity of the output \(y\) to the intermediate variable \(v_i\). During the backward pass, we compute \(\bar{v}_j = \sum_{i: j \in \text{parents}(i)} \bar{v}_i \frac{\partial \phi_i}{\partial v_j}\). This summation accumulates contributions of all downstream operations. As Baydin et al. (2018) [https://arxiv.org/abs/1502.05767](https://arxiv.org/abs/1502.05767) detail, the choice between these modes depends on the dimensionality of the input and output spaces.

To handle nested derivatives, Pearlmutter & Siskind (2008) [https://engineering.purdue.edu/~qobi/papers/toplas2008.pdf](https://engineering.purdue.edu/~qobi/papers/toplas2008.pdf) introduced a formal variable-tagging mechanism to prevent perturbation confusion. Furthermore, Werbos (1990) [https://axon.cs.byu.edu/~martinez/classes/678/Papers/Werbos_BPTT.pdf](https://axon.cs.byu.edu/~martinez/classes/678/Papers/Werbos_BPTT.pdf) established that backpropagation through time is a specific application of reverse-mode AD. For a comprehensive guide on implementation, see the step-by-step introduction by Wang et al. (2024) [https://arxiv.org/html/2402.16020v2](https://arxiv.org/html/2402.16020v2), and for the automation of nested matrix derivatives, refer to the work by Tesfatsion [https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/AutomationNestedMatrixDer.KTP.pdf](https://faculty.sites.iastate.edu/tesfatsi/archive/tesfatsi/AutomationNestedMatrixDer.KTP.pdf).

## Where the field is now

The state of the art has shifted from simple graph-based implementations to compiler-integrated systems. Modern frameworks like JAX [https://github.com/google/jax](https://github.com/google/jax) utilize a "define-by-transform" approach, operating on functional intermediate representations to enable operator fusion and XLA-based kernel compilation. This evolution represents a continuous trajectory from the early manual instrumentation of code to modern systems that treat differentiation as a first-class compiler transformation.

In production, AD is the backbone of large-scale training at companies like Meta, Google, and OpenAI, where it enables the training of models with trillions of parameters. The engineering frontier is currently dominated by the challenge of "just-in-time" differentiation, where the computational graph is generated and optimized on the fly to minimize memory footprint. As discussed in the survey by Biggs (2000) [https://people.cs.vt.edu/~asandu/Public/Qual2011/Optim/Biggs_2000_AD.pdf](https://people.cs.vt.edu/~asandu/Public/Qual2011/Optim/Biggs_2000_AD.pdf), the integration of AD into the compiler stack is essential for achieving performance that rivals hand-optimized C++ code.

## What's still open

Despite the maturity of AD, several fundamental questions remain. First, can compilers automatically generate and fuse optimal, hardware-specific GPU kernels for dynamic, control-flow-heavy reverse-mode AD without relying on manual engineering? Second, how can we formalize the debugging of AD engines? When a gradient is incorrect, it is often impossible to determine whether the error lies in the user's code, the AD transformation, or the compiler optimization. Finally, is there a universal representation for AD that can bridge the gap between high-level functional languages and low-level hardware accelerators?

## Where to read next

If you want the probabilistic foundation, → [[score-matching]] gives the likelihood-free training perspective that AD compiles down to. The engineering counterpart is → [[flash-attention]] explaining how the operations within the computational graph are made fast enough to be trained at scale. For the next paradigm in program transformation, → [[flow-matching]] generalizes the differentiation process to arbitrary continuous paths, moving beyond the discrete steps of traditional backpropagation.

## Build it

**What you're building:** A dual-number class that computes exact derivatives of arbitrary scalar functions.

**Why this is valuable:** It forces you to implement the chain rule manually, revealing how AD "instruments" code.

**Stack:**
- **Model:** Custom Python Class
- **Dataset:** Synthetic data (e.g., `numpy.linspace`)
- **Framework:** Pure Python + NumPy
- **Compute:** CPU (instant)

**The recipe:**
1. Define a `Dual` class with `__add__`, `__mul__`, and `__pow__` methods that handle the \(a + b\epsilon\) arithmetic, where \(\epsilon^2 = 0\).
2. Implement the chain rule for elementary functions (sin, cos, exp) within the `Dual` class.
3. Create a simple function: \(f(x) = x^2 + \sin(x)\).
4. Use your `Dual` class to compute the derivative at \(x=2.0\).
5. Verify your result against a finite-difference implementation using a small \(\Delta x\).

**Expected outcome:** A derivative value that matches within numerical precision for the tested functions.

- **CS student:** Implement the `Dual` class and use it to differentiate a polynomial; verify the derivative is exact.
- **Applied engineer:** Use `torch.autograd.grad` to profile the memory usage of a 7B parameter model's backward pass; identify the "activation checkpointing" threshold.
- **Applied researcher:** Test if forward-mode AD is faster than reverse-mode for a function with 100 inputs and 100 outputs; plot the crossover point.
- **Frontier researcher:** Implement a "second-order" AD engine by nesting your `Dual` class; test if it correctly computes the Hessian of a simple quadratic.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*