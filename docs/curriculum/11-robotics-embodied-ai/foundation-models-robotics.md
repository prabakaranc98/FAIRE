```yaml
---
title: Foundation Models in Robotics
track: 11-robotics-embodied-ai
tags: [robotics, foundation models, VLA, embodied AI, zero-shot learning]
depth: applied
prereqs: [large-language-models, vision-language-models]
updated: 2024-11-06
has_mvb: false
---
# Foundation Models in Robotics
> **TL;DR:** Foundation models are transforming robotics by enabling robots to learn generalizable skills and adapt to new tasks with minimal task-specific programming.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is
Imagine a robot in a warehouse, tasked with sorting packages. Instead of being pre-programmed for each item, it understands natural language instructions like "put the red box on the top shelf." This robot doesn't just follow commands; it *understands* the task, the objects, and the space around it. This is the promise of foundation models in robotics. These models are trained on vast amounts of data, enabling robots to perform a wide range of tasks without needing specific programming for each one.

Traditional robotics relies on task-specific programming and extensive hand-engineering. Each new task requires significant effort to design controllers, perception systems, and task planners. Foundation models offer a paradigm shift by providing robots with a broad understanding of the world, enabling them to adapt to new tasks and environments with minimal additional training. This is achieved by leveraging large datasets of images, text, and action sequences to train models that can generalize across a wide range of scenarios.

The key to this generalization is the ability of foundation models to learn representations that capture the underlying structure of the world. These representations can then be used to reason about new tasks, plan actions, and control the robot's movements. By combining vision, language, and action, foundation models enable robots to understand instructions, perceive their environment, and execute complex manipulations with unprecedented flexibility.

## Why it matters at the frontier
Foundation models are poised to revolutionize robotics by addressing the long-standing challenge of generalization. Instead of training robots for specific tasks in controlled environments, foundation models enable robots to learn generalizable skills that can be applied to a wide range of scenarios. This is particularly important for robots operating in unstructured and dynamic environments, such as warehouses, hospitals, and homes.

The development of foundation models for robotics is driven by the need for robots that can seamlessly interact with humans and adapt to changing task requirements. A key open problem is: How can we develop a unified framework that effectively integrates the generalizable knowledge from foundation models with the dynamic modeling capabilities of world models to enable open-ended task solving in embodied environments, particularly in scenarios involving complex observations or domain gaps? Addressing this problem will unlock the potential for robots to perform complex tasks in collaboration with humans, leading to increased efficiency, productivity, and safety.

## Core concepts
- **Vision-Language-Action (VLA) Models** — Models that integrate visual perception, natural language understanding, and robotic action to enable robots to perform tasks based on multimodal instructions.
- **Zero-Shot Learning** — The ability of a robot to perform a task without any specific training for that task, relying instead on the general knowledge learned from foundation models.
- **Embodied Reasoning** — The process by which a robot uses its physical embodiment and sensory input to understand and interact with the world.
- **World Models** — Predictive models that learn to simulate the dynamics of the environment, enabling robots to plan and reason about the consequences of their actions.
- **Spatial Representations** — Encoding of spatial relationships between objects and the robot, enabling precise manipulation and navigation.
- **Generalist Robots** — Robots capable of performing a wide range of tasks without task-specific programming, leveraging the general knowledge learned from foundation models.
- **Modularity** — The design principle of building complex systems from independent, interchangeable modules, allowing for flexibility and extensibility.

## Mathematical foundations
While the specific equations vary depending on the architecture of the foundation model, a common underlying principle is the optimization of a loss function that encourages the model to learn a joint representation of vision, language, and action. For example, a contrastive loss might be used to align visual and textual representations:

\[
\mathcal{L} = \sum_{i=1}^{N} \max(0, 1 - \mathbf{v}_i \cdot \mathbf{t}_i + \mathbf{v}_i \cdot \mathbf{t}_j)
\]

where \(\mathbf{v}_i\) is the visual embedding of the \(i\)-th image, \(\mathbf{t}_i\) is the textual embedding of the corresponding caption, \(\mathbf{t}_j\) is the textual embedding of a negative caption, and \(N\) is the number of image-caption pairs. This loss function encourages the model to learn embeddings that are similar for corresponding image-caption pairs and dissimilar for non-corresponding pairs.

Another common approach is to use a sequence-to-sequence model to predict the robot's actions given a sequence of visual observations and natural language instructions:

\[
p(a_{1:T} \mid o_{1:T}, c) = \prod_{t=1}^{T} p(a_t \mid a_{1:t-1}, o_{1:t}, c)
\]

where \(a_{1:T}\) is the sequence of actions, \(o_{1:T}\) is the sequence of visual observations, \(c\) is the natural language instruction, and \(p(a_t \mid a_{1:t-1}, o_{1:t}, c)\) is the conditional probability of the action at time \(t\) given the previous actions, observations, and instruction. This equation says that the probability of a sequence of actions is the product of the conditional probabilities of each action given the history of actions, observations, and the instruction.

World models often incorporate a learned dynamics function:

\[
s_{t+1} = f(s_t, a_t)
\]

where \(s_t\) is the state of the world at time \(t\), \(a_t\) is the action taken at time \(t\), and \(f\) is a learned function that predicts the next state given the current state and action. This allows the robot to predict the consequences of its actions and plan accordingly.

## Key algorithms / techniques
- **Vision-Language Pre-training (VLP)** — Training models on large datasets of images and text to learn joint representations that can be used for various downstream tasks, including robotic manipulation.
- **Reinforcement Learning (RL)** — Training robots to perform tasks by rewarding desired behaviors and penalizing undesired behaviors, often used in conjunction with foundation models to fine-tune policies.
- **Imitation Learning (IL)** — Training robots to mimic human demonstrations, providing a way to bootstrap the learning process and guide the robot towards desired behaviors.
- **Transformer Networks** — A neural network architecture that excels at processing sequential data, making it well-suited for vision-language-action models in robotics.
- **Diffusion Models** — Generative models that learn to reverse a diffusion process, enabling robots to generate realistic images and actions.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| SpatialVLA: Exploring Spatial Representations for Visual-Language-Action Model | 2025 |  [2501.15830v5] | This paper proposes SpatialVLA, which explores spatial representations for robot foundation models, achieving strong results in zero-shot robot control. |
| Embodied-R1: Reinforced Embodied Reasoning for General Robotic Manipulation | 2025 |  [2508.13998v1] | This paper introduces Embodied-R1, a vision-language model designed for embodied reasoning and pointing, demonstrating robust zero-shot generalization in robotic manipulation. |
| Maestro: Orchestrating Robotics Modules with Vision-Language Models for Zero-Shot Generalist Robots | 2025 | Shi et al. | This paper presents Maestro, a system that orchestrates robotics modules with vision-language models for zero-shot generalist robots. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| 3D-VLA: A 3D Vision-Language-Action Generative World Model | 2024 | Zhen et al. | Introduces a 3D vision-language-action generative world model for robotic manipulation. |
| DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control | 2025 | Wen et al. | Presents DexVLA, a vision-language model with a plug-in diffusion expert for general robot control. |
| FOUNDER: Grounding Foundation Models in World Models for Open-Ended Embodied Decision Making | 2025 | Wang et al. | Introduces FOUNDER, a framework that integrates foundation models with world models for open-ended embodied decision-making. |

## Current SotA
SpatialVLA achieves strong results in zero-shot robot control by leveraging spatial representations [2501.15830v5] (2025). Embodied-R1 demonstrates robust zero-shot generalization in robotic manipulation through reinforced embodied reasoning and pointing [2508.13998v1] (2025). Maestro orchestrates robotics modules with vision-language models for zero-shot generalist robots (Shi et al., 2025).

## What's happening now
Research is actively exploring how to integrate foundation models with world models to enable robots to reason about the consequences of their actions and plan accordingly. This involves developing new architectures and training techniques that can effectively combine the general knowledge learned from foundation models with the dynamic modeling capabilities of world models.

Engineering efforts are focused on developing robust and scalable systems that can deploy foundation models on real-world robots. This includes optimizing the models for efficient inference, developing robust perception systems that can handle noisy and incomplete data, and designing control systems that can translate high-level plans into low-level motor commands. NVIDIA is developing an end-to-end workflow for training generalist robots using world foundation models (WFMs) and scalable synthetic data.

A key open problem is: How can we develop foundation models that effectively handle the complexities of real-world physics, including object interactions, material properties, and environmental dynamics, to achieve robust and reliable robotic manipulation in unstructured environments? This requires developing new representations and learning algorithms that can capture the underlying physics of the world and enable robots to reason about the consequences of their actions in a physically realistic manner.

## In production
- NVIDIA — End-to-end workflow for training generalist robots using world foundation models (WFMs) and scalable synthetic data. — [https://developer.nvidia.com/blog/r2d2-training-generalist-robots-with-nvidia-research-workflows-and-world-foundation-models/]
- NVIDIA — Digital-twin, simulation-first workflow to validate and optimize physical AI-driven operations for robot fleets in industrial facilities. — [https://developer.nvidia.com/blog/simulating-robots-in-industrial-facility-digital-twins/]

## Code & implementations
- [Official implementations of VLA models (if available)]
- [Hugging Face model hub for pre-trained robotics models (if available)]

## What comes next
- [[Robotics Simulation Environments]] — provides a platform for training and evaluating foundation models in simulated environments.
- [[Reinforcement Learning for Robotics]] — explores how reinforcement learning can be used to fine-tune foundation models for specific robotic tasks.

## Connected topics
- [Agent Architectures](../01-ai/agent-architectures.md) — Foundation models are often used to build advanced agent architectures.
- [Reinforcement Learning from Human Feedback (RLHF)](../06-reinforcement-learning/rlhf.md) — RLHF can be used to train foundation models for robotic tasks.
- [Diffusion Models](../02-generative-modeling/diffusion-models.md) — Diffusion models are used in foundation models for robotics, such as for generating robot actions.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers are a key architecture used in many foundation models for robotics.
- [Contrastive Learning](../03-representation-learning/contrastive-learning.md) — Contrastive learning is used to train representations for foundation models in robotics.
- [Classical Planning](../01-ai/classical-planning.md) — Foundation models can be integrated with classical planning for robotic task execution.


## Further reading
- Zhen et al. (2024) — "3D-VLA: A 3D Vision-Language-Action Generative World Model" — [https://arxiv.org/html/2403.09631] — Provides details on a specific architecture for vision-language-action models.
- Wen et al. (2025) — "DexVLA: Vision-Language Model with Plug-In Diffusion Expert for General Robot Control" — [https://arxiv.org/html/2502.05855v2] — Explores the use of diffusion models for robot control.
- Liao et al. (2025) — "Genie Envisioner: A Unified World Foundation Platform for Robotic Manipulation" — [https://arxiv.org/html/2508.05635] — Introduces a unified world foundation platform for robotic manipulation.
- Lilian Weng's survey on VLMs (lil'log, YYYY) — Offers a comprehensive overview of vision-language models and their applications.
```