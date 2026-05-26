```yaml
---
title: Proximal Policy Optimization (PPO)
track: 06-reinforcement-learning
tags: [reinforcement learning, policy optimization, machine learning, PPO, RLHF]
depth: foundational
prereqs: [policy-gradients, actor-critic-methods]
updated: 2024-07-01
has_mvb: false
---
# Proximal Policy Optimization (PPO)
> **TL;DR:** Proximal Policy Optimization (PPO) is a reinforcement learning algorithm that stabilizes policy updates, making it a practical choice for training agents in complex environments, particularly with large language models.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) | Understand the practical algorithms |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is

Imagine you're teaching a language model to answer complex questions, but it keeps getting stuck in loops or providing nonsensical responses. Standard methods often struggle to guide the model toward the right answers, leading to frustratingly slow progress. Reinforcement learning, specifically Proximal Policy Optimization (PPO), offers a way to reward the model for correct answers and penalize it for errors, guiding it toward more accurate and reliable responses. This approach has shown promise in improving the reasoning capabilities of large language models.

PPO is a policy gradient method that aims to train an agent to make decisions in an environment to maximize cumulative rewards. Unlike traditional policy gradient methods, PPO incorporates a mechanism to ensure that policy updates are small and controlled, preventing drastic changes that can destabilize training. This is achieved through a clipped surrogate objective function that penalizes significant deviations from the previous policy, making it more sample-efficient and robust.

PPO's stability and relative simplicity have made it a popular choice for a wide range of applications, from robotics and game playing to natural language processing. Its ability to handle continuous action spaces and its ease of implementation have contributed to its widespread adoption in both research and industry.

## Why it matters at the frontier

PPO has become a cornerstone in the development of advanced AI systems, particularly in the fine-tuning of large language models (LLMs). Its ability to align LLMs with specific goals has enabled significant improvements in reasoning, text generation, and even the emergence of new capabilities. By providing a stable and efficient way to train these models, PPO has unlocked new possibilities in natural language understanding and generation.

However, challenges remain in designing PPO-based algorithms that are robust to the variance in reward signals, especially in environments with sparse or delayed rewards. Addressing this issue is crucial for advancing the capabilities of LLMs and other AI systems, enabling them to tackle more complex and real-world tasks. Research is focused on improving the sample efficiency and stability of PPO, as well as exploring new ways to incorporate it into the training of LLMs.

## Core concepts

-   **Policy** — A strategy that the agent uses to decide which action to take based on its current state.
-   **Reward** — A numerical value that the agent receives after taking an action, indicating the desirability of that action.
-   **Advantage** — A measure of how much better an action is compared to the average action in a given state.
-   **Surrogate Objective** — A function that approximates the expected return of a policy, used to optimize the policy in PPO.
-   **Clipping** — A mechanism in PPO that limits the change in the policy during each update, preventing overly large updates that can destabilize training.
-   **Policy Gradient** — A method for updating the policy by following the gradient of the expected return.
-   **Trust Region** — A region around the current policy within which the updated policy is expected to perform well.

## Mathematical foundations

The clipped surrogate objective function used in PPO is defined as:
\[ L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right] \]
where \(\theta\) is the policy parameters, \(r_t(\theta)\) is the probability ratio, \(A_t\) is the advantage estimate, and \(\epsilon\) is a hyperparameter.
This equation calculates the clipped surrogate objective function used in PPO to update the policy, preventing overly large policy updates.

The probability ratio is calculated as:
\[ r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{old}}(a_t|s_t)} \]
where \(r_t(\theta)\) is the probability ratio, \(\pi_\theta(a_t|s_t)\) is the probability of taking action \(a_t\) in state \(s_t\) under the current policy \(\theta\), and \(\pi_{\theta_{old}}(a_t|s_t)\) is the probability under the old policy.
This equation calculates the probability ratio, which is a key component of the PPO objective function.

The Generalized Advantage Estimate (GAE) is calculated as:
\[ A_t = \sum_{k=0}^{\infty} (\gamma \lambda)^k \delta_{t+k} \]
where \(A_t\) is the advantage estimate, \(\gamma\) is the discount factor, \(\lambda\) is the eligibility trace, and \(\delta_{t+k}\) is the temporal difference error.
This equation calculates the Generalized Advantage Estimate (GAE), which is used to reduce the variance of the policy gradient estimates.

## Key algorithms / techniques

-   **Clipped Surrogate Objective (2017)** — Limits the change in the policy during each update, preventing overly large updates that can destabilize training.
-   **Generalized Advantage Estimation (GAE) (2016)** — Reduces the variance of the policy gradient estimates by using a combination of temporal difference errors.
-   **Trust Region Policy Optimization (TRPO) (2015)** — A predecessor to PPO that also aims to stabilize policy updates, but is more complex to implement.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Proximal Policy Optimization Algorithms | 2017 | Schulman et al. | Introduces the core PPO algorithm, including the clipped surrogate objective. |
| Bounded Ratio Reinforcement Learning | 2024 | Ao et al. | Explores reinforcement learning with bounded ratios. |
| PROXIMAL POLICY OPTIMIZATION IN PATH SPACE: A SCHRÖDINGER BRIDGE PERSPECTIVE | 2024 | Gong | Presents a Schrödinger Bridge perspective on Proximal Policy Optimization in path space. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| Proximal Policy Optimization Algorithms | 2017 | Introduced the PPO algorithm, a popular and effective method for reinforcement learning. |
| High-Dimensional Continuous Control Using Generalized Advantage Estimation | 2016 | Introduced Generalized Advantage Estimation (GAE), a technique for reducing the variance of policy gradient estimates. |
| Trust Region Policy Optimization | 2015 | Introduced Trust Region Policy Optimization (TRPO), a predecessor to PPO that also aims to stabilize policy updates. |

## Current SotA

Sequence-Level PPO (SPPO) achieves state-of-the-art performance on long-horizon reasoning tasks (2024). Benchmark evaluations for this area are not standardized as of 2024; the most widely cited comparison is qualitative description.

## What's happening now

Research frontiers are focused on improving the sample efficiency and stability of PPO, as well as exploring new ways to incorporate it into the training of LLMs. This includes developing new techniques for estimating the advantage function, as well as exploring different ways to clip the policy updates.

Engineering and systems efforts are focused on scaling PPO to larger and more complex environments, as well as developing more efficient implementations of the algorithm. This includes using distributed computing techniques to parallelize the training process, as well as developing specialized hardware accelerators to speed up the computation.

An open problem is: How can we design PPO-based algorithms that are more robust to the variance in reward signals, especially in environments with sparse or delayed rewards, while maintaining computational efficiency?

## In production

-   Salesforce — Modular SageMaker AI-based hosting framework — Deploys and manages large-scale, production-grade models (including LLMs) across multiple AWS Regions — [https://aws.amazon.com/blogs/machine-learning/how-salesforce-achieves-high-performance-model-deployment-with-amazon-sagemaker-ai/](https://aws.amazon.com/blogs/machine-learning/how-salesforce-achieves-high-performance-model-deployment-with-amazon-sagemaker-ai/)
-   iFood — Internal ML platform — Enables seamless creation, training, and deployment of models for both online (real-time) and offline — [https://aws.amazon.com/blogs/machine-learning/how-ifood-built-a-platform-to-run-hundreds-of-machine-learning-models-with-amazon-sagemaker-inference/](https://aws.amazon.com/blogs/machine-learning/how-ifood-built-a-platform-to-run-hundreds-of-machine-learning-models-with-amazon-sagemaker-inference/)
-   Superhuman — Inference platform — 200K+ QPS with sub-second P99 latency and 4 nines reliability — [https://www.databricks.com/blog/how-superhuman-and-databricks-built-200k-qps-inference-platform-together](https://www.databricks.com/blog/how-superhuman-and-databricks-built-200k-qps-inference-platform-together)
-   Crexi — ML models deployment pipeline — Rapid, production-grade model delivery — [https://aws.amazon.com/blogs/machine-learning/how-crexi-achieved-ml-models-deployment-on-aws-at-scale-and-boosted-efficiency/](https://aws.amazon.com/blogs/machine-learning/how-crexi-achieved-ml-models-deployment-on-aws-at-scale-and-boosted-efficiency/)

## Code & implementations

-   Official PPO implementation in TensorFlow: [https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ppo](https://github.com/tensorflow/agents/tree/master/tf_agents/agents/ppo)
-   Clean PPO implementation in PyTorch: [https://github.com/nikhilbarhate99/PPO-PyTorch](https://github.com/nikhilbarhate99/PPO-PyTorch)

## What comes next

-   [[Actor-Critic Methods]] — PPO is an actor-critic method, building upon the foundation of having both a policy (actor) and a value function (critic).
-   [[Reinforcement Learning from Human Feedback (RLHF)]] — PPO is commonly used as the policy optimization step in RLHF to align language models with human preferences.

## Connected topics

- [Markov Decision Process](./mdp.md) — PPO is a reinforcement learning algorithm that builds upon MDPs.
- [Reinforcement Learning from Human Feedback (RLHF)](./rlhf.md) — RLHF uses reinforcement learning, which includes algorithms like PPO.
- [Agent Architectures](../01-ai/agent-architectures.md) — PPO is often used within agent architectures for decision-making.
- [Imitation Learning](../11-robotics-embodied-ai/imitation-learning.md) — Imitation learning can be combined with PPO for robot control.
- [Optimization](../04-neural-networks-dl/optimization.md) — PPO uses optimization techniques to improve policy performance.


## Further reading

-   Schulman et al. (2017) — "Proximal Policy Optimization Algorithms" — [https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347) — This paper provides a detailed explanation of the PPO algorithm and its implementation.
-   Ao et al. (2024) — "Bounded Ratio Reinforcement Learning" — [https://arxiv.org/abs/2604.18578](https://arxiv.org/abs/2604.18578) — This paper explores reinforcement learning with bounded ratios.
-   Wang et al. (2024) — "SPPO: Sequence-Level PPO for Long-Horizon Reasoning Tasks" — [https://www.arxiv.org/pdf/2604.08865](https://www.arxiv.org/pdf/2604.08865) — This paper introduces Sequence-Level PPO (SPPO) for long-horizon reasoning tasks.
-   Gong (2024) — "PROXIMAL POLICY OPTIMIZATION IN PATH SPACE: A SCHRÖDINGER BRIDGE PERSPECTIVE" — [https://arxiv.org/pdf/2603.21621](https://arxiv.org/pdf/2603.21621) — This paper presents a Schrödinger Bridge perspective on Proximal Policy Optimization in path space.
-   Chen et al. (2024) — "Stabilizing Policy Optimization via Logits Convexity" — [https://arxiv.org/pdf/2603.00963](https://arxiv.org/pdf/2603.00963) — This paper explores stabilizing policy optimization through logits convexity.
-   Beukman et al. (2024) — "Preventing Learning Stagnation in PPO by Scaling to 1 Million Parallel Environments" — [https://arxiv.org/pdf/2603.06009](https://arxiv.org/pdf/2603.06009) — This paper investigates preventing learning stagnation in PPO by scaling to a large number of parallel environments.