---
title: Cognitive Architectures
track: 10-complexity-cognition
tags: [agentic-ai, reasoning, planning, symbolic-ai, cognitive-science]
depth: foundational
prereqs: [transformer, optimization]
updated: 2025-05-14
has_mvb: true
---

# Cognitive Architectures

> **TL;DR:** Cognitive architectures provide the structural blueprint for intelligent systems, integrating memory, perception, and reasoning to enable goal-directed behavior.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| Curious learner | [§What it is](#what-it-is) | Build intuition |
| CS student / tinkerer | [§Key algorithms](#key-algorithms--techniques) → [§MVB](#minimum-valuable-build) | Build something that works |
| Applied engineer | [§In production](#in-production) | Build reliable systems |
| Math/theory student | [§Core concepts](#core-concepts) → [§Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [§Current SotA](#current-sota) → [§Open questions](#open-questions) | Know where the open problems are |

---

## What it is

Imagine you are trying to navigate a complex, unfamiliar city without a map, relying only on your memory of similar cities you have visited. You might make good guesses about where the main roads are, but you would likely get lost the moment you encounter a unique intersection or a sudden road closure. Large Language Models (LLMs)—systems trained to predict the next word in a sequence based on massive datasets—operate much like this. They are excellent at predicting the next "turn" in a conversation, but they lack a persistent internal map or a "steering wheel" to handle multi-step planning and unexpected obstacles.

Cognitive architectures solve this by providing a structural blueprint for intelligent systems that mirrors human cognitive organization. They decompose intelligence into specialized modules—such as a working memory for immediate task state, long-term knowledge stores for facts and procedures, and decision-making cycles for deliberation—that interact to produce goal-directed behavior. This structural approach allows a system to maintain a coherent internal state while executing sequences of actions, effectively separating the "knowledge" of the world from the "control" logic that decides how to act.

The consequence is a shift from monolithic prediction—where the model attempts to solve a problem in a single, opaque forward pass—to modular orchestration. By embedding neural models within these architectures, researchers enable agents to pause, reflect, and re-plan. This framework provides the necessary scaffolding to move from reactive chatbots to autonomous agents capable of persistent reasoning.

## Why it matters at the frontier

Cognitive architectures represent the frontier of moving from simple chatbots to autonomous agents. While LLMs provide the linguistic substrate, they lack native mechanisms for persistent state management and recursive planning. Integrating these architectures with neural models is the primary bottleneck in building systems that can operate reliably in real-world environments.

The current paradigm of "prompt engineering" is fundamentally limited by the context window and the lack of an internal model of the world. By embedding LLMs within a cognitive architecture, researchers aim to create systems that can handle long-horizon tasks, learn from experience, and exhibit robust reasoning. This is the key tension in modern AI: balancing the flexibility of neural representations with the stability of symbolic control.

## Core concepts

- **Working Memory** — A short-term storage buffer that holds the current state, active goals, and immediate sensory inputs.
- **Production Rules** — Condition-action pairs that define the system's behavior by matching current memory states to specific operations.
- **Goal Stack** — A hierarchical structure that maintains the system's current objectives and sub-objectives to ensure task completion.
- **Long-term Memory** — A persistent store for declarative knowledge and procedural skills that can be retrieved based on current context.
- **Decision Cycle** — The fundamental loop of perception, deliberation, and action that drives the system's interaction with its environment.

## Mathematical foundations

The state of a cognitive architecture at time \(t\) can be represented as a tuple:
\[ S_t = \langle M_t, K, P \rangle \]
where \(M_t\) is the contents of working memory, \(K\) is the long-term knowledge base, and \(P\) is the set of production rules.

The transition function for the system is defined by the application of rules:
\[ S_{t+1} = \text{apply}(P, S_t) \]
where \(S_{t+1}\) is the next state, \(P\) is the set of production rules, and \(S_t\) is the current state.

The goal-directed behavior is governed by the selection function:
\[ a_t = \arg\max_{a \in A} V(S_t, a) \]
where \(a_t\) is the chosen action, \(A\) is the set of possible actions, and \(V\) is the value function evaluating the utility of action \(a\) given state \(S_t\).

## Key algorithms / techniques

- **STRIPS** — A foundational planning algorithm that uses state representations and operators to reach a goal state.
- **Soar** — A comprehensive cognitive architecture that uses production rules and "chunking" to learn from problem-solving experiences.
- **ACT-R** — A cognitive architecture that models human performance by integrating symbolic and sub-symbolic (neural) components.

## Open questions

> **Researcher:** How can we formally verify the stability of neuro-symbolic transitions when the underlying neural components are stochastic?

> **Engineer:** What are the optimal latency-throughput trade-offs for memory retrieval in agents operating at the scale of millions of tokens per second?

> **Open:** Can we derive a universal "cognitive loss function" that optimizes for both symbolic reasoning accuracy and neural linguistic fluency?

## Essential reading

| Paper | Year | Authors | Link |
|---|---|---|---|
| STRIPS: A New Approach | 1972 | Fikes & Nilsson | [PDF](https://cs.uky.edu/~sgware/reading/papers/fikes1972strips.pdf) |
| A Specification of Soar | 1992 | Laird et al. | [PDF](http://www.cds.caltech.edu/~mhucka/publications/Laird:ASpecificationOfTheSoarCognitiveArchitecture:1992.pdf) |
| 40 Years of Research | 2016 | Langley et al. | [arXiv](https://arxiv.org/abs/1610.08602v2) |
| Cognitive Foundations | 2025 | Zhu et al. | [arXiv](https://arxiv.org/abs/2511.16660v2) |

## Seminal papers & test-of-time

- **GPS: A Program that Simulates Human Thought** (Newell & Simon, 1961) — The first program to separate problem-solving from domain knowledge [Link](https://iiif.library.cmu.edu/file/Simon_box00064_fld04907_bdl0001_doc0001/Simon_box00064_fld04907_bdl0001_doc0001.pdf).
- **A Framework for Representing Knowledge** (Minsky, 1974) — Introduced "frames" as a way to structure knowledge representation [Link](https://dspace.mit.edu/bitstream/handle/1721.1/6089/AIM-306.pdf).

## Current SotA

The field is moving toward "Neuro-Symbolic" architectures. Zhu et al. (2025) provide a taxonomy of cognitive elements for reasoning, demonstrating that LLMs manifest these elements when structured within agentic frameworks. Empirical evaluations using the AgentBench suite (Liu et al., 2023, [arXiv:2308.03688](https://arxiv.org/abs/2308.03688)) indicate that architectures incorporating explicit memory modules outperform standard LLMs by 15-20% on long-horizon planning tasks.

## What's happening now

Research is currently focused on "LLM-as-a-Controller" architectures. Researchers are investigating how to use LLMs to generate production rules dynamically, allowing the architecture to "program itself" in response to novel tasks (Zhu et al., 2025). This addresses the rigidity of traditional symbolic systems by injecting neural flexibility into the control loop.

Engineering efforts are shifting toward "Extreme Co-design" platforms. These frameworks allow for the co-design of agentic systems, where the architecture is optimized for the specific latency and memory constraints of the underlying hardware. This is critical for deploying agents that require real-time deliberation.

The "Integration Gap"—the difficulty of grounding symbolic reasoning in high-dimensional neural representations—remains a primary challenge. Recent studies suggest that controlled evaluation of agent configurations is the only way to move from anecdotal success to a science of agentic scaling.

## In production

- **Edmunds** — Edmunds Mind — Multi-agent ecosystem for automotive research — [Databricks Blog](https://www.databricks.com/blog/lakehouse-digital-mind-architecting-multi-agent-ai-ecosystem-databricks-agent-bricks)
- **NVIDIA** — AI-Q Research Agent — LangGraph-based production agent — [NVIDIA Blog](https://developer.nvidia.com/blog/how-to-scale-your-langgraph-agents-in-production-from-a-single-user-to-1000-coworkers/)
- **Google** — Scalable Agent Framework — Large-scale evaluation of 180 agent configurations — [Google Research](https://research.google/blog/towards-a-science-of-scaling-agent-systems-when-and-why-agent-systems-work/)

## Minimum Valuable Build

**Goal:** Build a stateful agent using `gpt-4o-mini` that solves a multi-step logic puzzle.
**Compute:** Runs on free Colab T4 (16GB VRAM).

1. **Setup:** Install `langgraph` and `langchain-openai`.
2. **State:** Define `TypedDict` with `messages` and `plan` (list of strings).
3. **Nodes:** Create a `planner` node (LLM generates steps) and an `executor` node (LLM executes step).
4. **Graph:** Compile the graph with a `ConditionalEdge` that checks if the `plan` is empty.
5. **Run:** Execute on the "Tower of Hanoi" dataset (3-disk variant).
6. **Metric:** Success rate of reaching the goal state in < 10 steps.

*   **Curious learner:** Focus on the graph visualization (`graph.get_graph().draw_mermaid_png()`).
*   **CS student:** Implement a custom `MemorySaver` class to persist state to a local JSON file.
*   **Applied engineer:** Use `Redis` for the `Checkpointer` to handle concurrent state updates.
*   **Theory student:** Prove the state transition graph is acyclic for your specific task.
*   **Researcher:** Compare the performance of `gpt-4o-mini` vs `llama-3-8b-instruct` on the planning task.
*   **Frontier:** Implement a "reflection" node that evaluates the plan's validity before execution.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/langchain-ai/langgraph) is the only signal we collect.*

---

## Code & implementations
- [LangGraph](https://github.com/langchain-ai/langgraph) — Official framework for building stateful, multi-actor applications.
- [Soar Architecture](https://soar.eecs.umich.edu/) — Official repository for the Soar cognitive architecture.

## What comes next
Understanding cognitive architectures allows you to move beyond simple prompt-response loops into building persistent, goal-oriented systems.

- [[chain-of-thought]] — Provides the reasoning substrate that cognitive architectures organize into persistent loops.
- [[circuit-complexity]] — Provides the theoretical bounds on what these architectures can compute.
- [[bayesian-inference]] — Often used to model uncertainty within the decision cycles of these architectures.

## Further reading
- [Zhu et al. (2025)](https://arxiv.org/abs/2511.16660v2) — A comprehensive synthesis of cognitive science elements for modern LLM reasoning.
- [Langley et al. (2016)](https://arxiv.org/abs/1610.08602v2) — The essential survey for understanding the historical trajectory of cognitive architectures.
- [Fikes & Nilsson (1972)](https://cs.uky.edu/~sgware/reading/papers/fikes1972strips.pdf) — The foundational paper on operator-based planning.