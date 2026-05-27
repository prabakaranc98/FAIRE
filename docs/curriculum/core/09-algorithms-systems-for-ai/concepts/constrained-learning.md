---
title: Constrained Learning
slug: constrained-learning
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [chang, rao, roth, achiam]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher, curious-learner, pm-decision-maker]
prereqs: [optimization-basics, reinforcement-learning-fundamentals]
tags: [optimization, safety, constraints, reinforcement-learning]
updated: 2025-05-14
has_mvb: true
---

# Constrained Learning

Imagine you are teaching a robot to navigate a crowded warehouse. You provide a reward function that encourages speed and efficiency, but the robot quickly learns that the fastest path involves cutting through a restricted storage zone or ignoring the physical limits of its own joints. In standard machine learning, the model treats these boundaries as mere suggestions, optimizing for the highest statistical probability of success. This is the central failure of unconstrained learning: it treats the world as a data distribution to be matched rather than a system of hard physical and logical boundaries to be satisfied. Constrained learning shifts the paradigm from simple curve-fitting to optimization under strict boundaries, ensuring that model outputs are not just statistically probable, but guaranteed to be feasible. By embedding these rules directly into the learning process, we move from systems that "hope" to be safe to systems that are safe by design.

## The territory

Constrained learning sits at the intersection of statistical machine learning and formal optimization. In standard supervised or reinforcement learning, we minimize a loss function \(L(\theta)\) over a parameter space \(\theta\). This assumes the optimal solution lies within the unconstrained landscape of the model's capacity. In high-stakes domains like robotics, finance, or medical diagnosis, the "optimal" statistical solution often violates critical safety requirements. Constrained learning answers how we incorporate these requirements directly into the learning process so the model respects them by design.

The field draws from integer linear programming and control theory, bridging the gap between the flexibility of neural networks and the rigor of formal verification. It moves from unconstrained minimization to a Lagrangian or penalty-based formulation where we solve for \(\min L(\theta)\) subject to \(C(\theta) \le 0\), where \(C\) represents a set of constraints. This evolution has moved from simple penalty methods to sophisticated frameworks that treat constraints as first-class citizens in the optimization loop.

## How it works

The mechanism of constrained learning relies on injecting declarative knowledge into the learning objective. Chang et al. (2012) [Structured learning with constrained conditional models](https://cogcomp.seas.upenn.edu/papers/ChangRaRo12.pdf) formalized this through Constrained Conditional Models (CCMs), which augment probabilistic models with constraints expressed in first-order logic. The objective function balances statistical evidence with constraint satisfaction:
\[ \hat{y} = \text{argmax}_{y \in \mathcal{Y}} \left( \sum_{i} w_i f_i(x, y_i) - \sum_{j} \lambda_j \phi_j(y) \right) \]
where \(x\) is the input, \(y\) is the output variable (the prediction), \(w_i\) are weights for features \(f_i\) (the statistical evidence), and \(\phi_j(y)\) are penalty functions for violating constraint \(j\) (the logical rules), weighted by \(\lambda_j\) (the penalty strength). The first term captures statistical likelihood, while the second acts as a logical filter that forces the model to respect domain rules.

In reinforcement learning, Constrained Policy Optimization (CPO) (Achiam et al. 2017) [https://arxiv.org/abs/1705.1052](https://arxiv.org/abs/1705.1052) maintains safety by ensuring policy updates remain within a trust region satisfying \(C(\pi) \le \epsilon\), where \(C(\pi)\) is the expected cost of the policy and \(\epsilon\) is the safety threshold. This is achieved by linearizing the constraint around the current policy to ensure that each update step is safe.

To bridge these approaches, Policy Gradient Penalty (PGP) methods incorporate constraints directly into the gradient update:
\[ \theta_{t+1} = \theta_t - \alpha \nabla_\theta (L(\theta) + \beta P(\theta)) \]
where \(\theta\) is the model parameter vector, \(\alpha\) is the learning rate, \(\beta\) is a dynamic penalty coefficient, and \(P(\theta)\) is the penalty function. The coefficient \(\beta\) controls the trade-off between task performance and constraint adherence; as \(\beta\) increases, the model prioritizes feasibility over reward. This unified approach allows us to treat symbolic logic and continuous control as two sides of the same optimization problem.

## Where the field is now

The current state-of-the-art involves integrating symbolic solvers into the training loop. Research such as the work on [Cross-Task Knowledge-Constrained Self Training](https://ar5iv.labs.arxiv.org/html/0907.0784) (Chang et al. 2009) and [Learning as Search Optimization](https://ar5iv.labs.arxiv.org/html/0907.0809) (Chang et al. 2009) laid the groundwork for modern Solver-In-the-Loop (SIR) architectures. These architectures, where a neural network generates a candidate solution verified by a symbolic solver, are increasingly used in complex reasoning tasks where logical consistency is paramount.

In robotics, projection-based methods that map infeasible policy updates back onto the feasible manifold are gaining traction. While industrial reports from companies like Boston Dynamics suggest the use of constrained optimization for motion planning, these are often proprietary; however, the academic consensus emphasizes that feature learning under constraints is essential for generalization. The engineering frontier is currently focused on reducing the computational overhead of real-time constraint checking, often by using differentiable approximations of solvers.

## What's still open

Several critical questions remain. First, how can we guarantee 100% feasibility of complex, non-linear constraints in autoregressive models during generation without relying on expensive external solvers? Second, can we develop a universal language for constraint specification that is both expressive and efficient for gradient-based optimization? Finally, is there a fundamental trade-off between a model's expressive capacity and its ability to satisfy hard constraints, or can we design architectures that are inherently constrained?

## Where to read next

If you want the probabilistic foundation, → [[score-matching]] gives the likelihood-free training perspective that constrained models often compile down to. The engineering counterpart is → [[distributed-training]] explaining how the underlying neural architectures are made fast enough to handle the additional overhead of constraint verification. For the next paradigm in continuous-time control, → [[flow-matching]] generalizes the noising process to arbitrary continuous paths, which provides a more stable manifold for constrained optimization.

## Build it

**What you're building:** A PyTorch agent that learns to navigate a 2D grid while respecting a hard spatial safety constraint.

**Stack:**
- **Model:** [Custom 2-layer MLP](https://pytorch.org/docs/stable/generated/torch.nn.Linear.html)
- **Dataset:** [Gymnasium GridWorld](https://gymnasium.farama.org/environments/gridworld/)
- **Framework:** PyTorch 2.0+
- **Compute:** Free Google Colab T4 (approx. 15 minutes for 500 episodes)

**The recipe:**
1. Define the environment as a 2D coordinate space with a goal point and a circular "no-go" zone.
2. Initialize a simple MLP policy that outputs mean and variance for actions.
3. Implement the training loop using a standard REINFORCE objective, adding a differentiable softplus penalty \(P(\theta) = \text{softplus}(\text{radius} - \text{dist}(\text{agent}, \text{zone}))\).
4. Update the policy using the combined loss \(L_{total} = L_{reward} + \beta P(\theta)\).
5. Evaluate by measuring the percentage of episodes where the agent enters the no-go zone, aiming for < 1% violation.

**Variants per persona:**
- **CS student:** Tune \(\beta\); observe training collapse if \(\beta\) is too high or too low.
- **Applied engineer:** Vectorize the constraint check using PyTorch tensors for batch sizes > 1024; target < 50ms latency per step.
- **Applied researcher:** Test if a dynamic \(\beta$ (increasing over time) leads to better final performance than a static \(\beta\).
- **Frontier researcher:** Replace the penalty with a projection operator and compare convergence speed to the PGP method.
- **Curious learner:** Visualize the agent's path before and after adding the constraint penalty.
- **PM/Decision-Maker:** Compare the "cost of safety" (reward drop) against the "cost of failure" (violation rate).

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---