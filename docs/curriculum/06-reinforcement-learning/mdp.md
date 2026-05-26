```yaml
---
title: Markov Decision Process
track: 06-reinforcement-learning
tags: [MDP, reinforcement learning, decision making, Markov property]
depth: foundational
prereqs: [markov-chain, bellman-equation]
updated: 2024-11-06
has_mvb: false
---
# Markov Decision Process
> **TL;DR:** A Markov Decision Process (MDP) provides a mathematical framework for modeling sequential decision-making in uncertain environments, crucial for developing reinforcement learning algorithms.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [Code & implementations](#code--implementations) | Understand how to use MDPs in practice |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Grasp the fundamental concepts of MDPs |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the formal definitions and equations |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Identify current research trends and key publications |

---

## What it is

Imagine a robot designed to assist in a home. Initially, it might perform tasks clumsily or even in ways that are unsafe. To improve, the robot needs a way to learn from human feedback, understanding what actions are helpful and harmless. This is where the concept of Markov Decision Processes (MDPs) and Reinforcement Learning from Human Feedback (RLHF) become essential.

A Markov Decision Process (MDP) is a mathematical framework for modeling decision-making in situations where outcomes are partly random and partly under the control of a decision-maker. It provides a structured way to represent the environment an agent interacts with, allowing the agent to learn optimal strategies through trial and error. The core idea is to formalize the agent's state, the actions it can take, and the probabilistic transitions between states based on those actions.

MDPs are characterized by the Markov property, which states that the future state depends only on the current state and action, not on the entire history of past states and actions. This simplifies the problem by allowing the agent to focus on the immediate situation when making decisions. The goal in an MDP is typically to find a policy, which is a mapping from states to actions, that maximizes the cumulative reward the agent receives over time.

## Why it matters at the frontier

Markov Decision Processes are fundamental to modern reinforcement learning, providing the theoretical foundation for algorithms that can learn complex behaviors in diverse environments. At the frontier, MDPs are essential for developing AI systems that can interact with the real world, make autonomous decisions, and adapt to changing conditions. This includes applications in robotics, game playing, and resource management.

One of the key challenges at the frontier is developing MDP models that can effectively incorporate and leverage human feedback to guide the learning process, especially in complex, real-world scenarios where reward functions are difficult to define. This is particularly relevant in the context of Reinforcement Learning from Human Feedback (RLHF), where human preferences are used to shape the behavior of AI agents. Advances in MDPs are crucial for creating AI systems that are not only intelligent but also aligned with human values and goals.

## Core concepts

- **State** — A representation of the environment at a particular point in time, providing the agent with the information needed to make decisions.
- **Action** — A choice the agent can make that influences the state of the environment.
- **Reward** — A scalar value that quantifies the immediate benefit or cost of taking a particular action in a particular state.
- **Policy** — A mapping from states to actions, specifying the agent's behavior in each state.
- **Transition Probability** — The probability of moving from one state to another after taking a specific action.
- **Markov Property** — The principle that the future state depends only on the current state and action, not on the past history.
- **Value Function** — A function that estimates the expected cumulative reward the agent will receive starting from a particular state and following a particular policy.
- **Discount Factor** — A value between 0 and 1 that determines the importance of future rewards relative to immediate rewards.

## Mathematical foundations

While there were no specific equations in the working memory, the core of MDPs can be expressed through the Bellman equation. The Bellman equation defines the optimal value function \(V^*(s)\) as:

\[
V^*(s) = \max_{a \in A(s)} \left( R(s, a) + \gamma \sum_{s' \in S} P(s' \mid s, a) V^*(s') \right)
\]

where \(V^*(s)\) is the optimal value function for state \(s\),
\(a\) is an action from the set of available actions \(A(s)\) in state \(s\),
\(R(s, a)\) is the immediate reward received after taking action \(a\) in state \(s\),
\(\gamma\) is the discount factor (0 ≤ \(\gamma\) ≤ 1),
\(s'\) is the next state,
\(P(s' \mid s, a)\) is the transition probability of moving to state \(s'\) from state \(s\) after taking action \(a\),
and \(S\) is the set of all possible states.

This equation says that the optimal value of a state is the maximum expected reward obtainable from that state, considering both the immediate reward and the discounted future rewards from the best possible actions in subsequent states. It forms the basis for many reinforcement learning algorithms that aim to find the optimal policy by iteratively updating the value function.

## Key algorithms / techniques

- **Value Iteration** — An iterative algorithm that computes the optimal value function by repeatedly applying the Bellman optimality operator until convergence.
- **Policy Iteration** — An algorithm that alternates between policy evaluation (computing the value function for a given policy) and policy improvement (updating the policy based on the value function).
- **Q-learning** — A model-free reinforcement learning algorithm that learns the optimal Q-function, which estimates the expected cumulative reward for taking a specific action in a specific state.
- **SARSA (State-Action-Reward-State-Action)** — Another model-free algorithm that learns a Q-function but uses the current policy to update the Q-values, making it an on-policy learning method.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Reinforcement Learning: An Introduction | 2018 | Sutton & Barto | Provides the foundational textbook definition and explanation of MDPs, covering all the core concepts. |
| Reinforcement Learning from Human Feedback | 2025 | Lally | Provides an overview of Reinforcement Learning from Human Feedback (RLHF). |
| RLHF Workflow: From Reward Modeling to Online RLHF | 2024 | Dong et al. | This paper provides a comprehensive practical alignment recipe of iterative preference learning, covering the entire RLHF workflow. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| A Markov Decision Process Formulation of Sequential Decision Problems | 1957 | Richard Bellman | Introduced the mathematical framework for Markov Decision Processes, laying the foundation for reinforcement learning. |
| Dynamic Programming | 1957 | Richard Bellman | Developed the dynamic programming approach for solving MDPs, including value iteration and policy iteration. |

## Current SotA

Current research focuses on scaling MDPs to high-dimensional state spaces and incorporating human feedback. Dong et al. (2024) provides a comprehensive practical alignment recipe of iterative preference learning, covering the entire RLHF workflow. Frick et al. (2024) introduces a new benchmark for evaluating reward models used in RLHF. Lang et al. (2024) investigates fine-tuning language models with reward learning on policy.

## What's happening now

Research frontiers are focused on developing more robust and sample-efficient RLHF algorithms that can effectively leverage both real and LLM-generated rollouts to learn complex tasks in dynamic environments. For example, Bai et al. (2022) explores the use of Reinforcement Learning from Human Feedback (RLHF) to align language models with human preferences. This includes exploring techniques for handling noisy or inconsistent human feedback and for generalizing learned policies to new situations.

Engineering and systems efforts are focused on deploying reinforcement learning models in real-world applications, such as robotics and autonomous driving. This involves developing scalable and efficient infrastructure for training and deploying RL models, as well as addressing challenges related to safety and reliability. NVIDIA's Hydra-MDP is an example of an end-to-end driving framework designed for scalable production deployment.

Open problems include developing MDP models that can effectively incorporate and leverage human feedback to guide the learning process, especially in complex, real-world scenarios where reward functions are difficult to define. How can we design reward functions that accurately reflect human preferences and values, and how can we ensure that RL agents learn to behave in a way that is both helpful and harmless? Munos et al. (2023) explores Nash Learning from Human Feedback.

## In production

- NVIDIA — Hydra-MDP, an end-to-end driving framework designed for scalable production deployment. — [https://developer.nvidia.com/blog/end-to-end-driving-at-scale-with-hydra-mdp/]
- AWS — Deploying reinforcement learning in production using Ray and Amazon SageMaker RL, enabling building, training, and deploying RL models at scale. — [https://aws.amazon.com/blogs/machine-learning/deploying-reinforcement-learning-in-production-using-ray-and-amazon-sagemaker/]

## Code & implementations

Official implementations are often integrated within reinforcement learning libraries:

- [OpenAI Gym](https://gymnasium.farama.org/) — A toolkit for developing and comparing reinforcement learning algorithms, providing a wide range of environments modeled as MDPs.
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) — While primarily for transformers, it integrates with RL libraries for tasks involving language and decision-making.

> *For a hands-on build with this concept, see the MVB on [[reinforcement-learning-from-human-feedback]].*

## What comes next

- [[reinforcement-learning]] — provides the algorithms that learn optimal policies within an MDP framework.
- [[bellman-equation]] — describes the core equation used to solve MDPs and find optimal value functions.

## Connected topics

- [Reinforcement Learning from Human Feedback (RLHF)](./rlhf.md) — RLHF builds upon reinforcement learning, which uses MDPs as a core concept.
- [Agent Architectures](../01-ai/agent-architectures.md) — MDPs are a fundamental framework for designing and understanding intelligent agents.
- [Classical Planning](../01-ai/classical-planning.md) — Classical planning can be viewed as a special case of MDPs with deterministic actions.
- [Foundation Models in Robotics](../11-robotics-embodied-ai/foundation-models-robotics.md) — MDPs are often used to model the decision-making process of robots.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Backpropagation is used in some RL algorithms that solve MDPs, like policy gradients.
- [Reinforcement Learning from Human Feedback (RLHF)](./rlhf.md) — RLHF uses reinforcement learning, which is based on the MDP framework.


## Further reading

- Sutton & Barto (2018) — "Reinforcement Learning: An Introduction" — Provides a comprehensive introduction to reinforcement learning, including a detailed discussion of Markov Decision Processes.
- Lilian Weng's survey on Reinforcement Learning (lil'log, 2018) — Offers an intuitive walkthrough of key RL concepts, including MDPs.
- David Silver's Reinforcement Learning Course (UCL, 2015) — A series of lectures covering the fundamentals of reinforcement learning, including MDPs and dynamic programming.
- OpenAI Spinning Up in Deep RL (OpenAI, 2018) — A guide to deep reinforcement learning, including practical advice on implementing and training RL algorithms in MDP environments.
```