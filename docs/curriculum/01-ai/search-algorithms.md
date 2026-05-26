---
title: Search Algorithms
track: 01-ai
tags: [search, heuristic-search, a-star, bfs, dfs, informed-search]
depth: foundations
prereqs: []
updated: 2026-05-25
---

# Search Algorithms
> **TL;DR:** Systematic methods for finding solutions in a state space — the original formulation of intelligent problem-solving, and still the backbone of planning, game-playing, and symbolic AI.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
[stub]

## Why it matters at the frontier
[stub]

## Core concepts
- **State space** — the set of all possible configurations of a problem
- **Search tree** — the unfolding of reachable states via actions
- **Heuristic function** — an estimate of the cost from a state to the goal
- **Admissibility** — a heuristic is admissible if it never overestimates the true cost
- **Completeness** — whether a search algorithm always finds a solution if one exists
- **Optimality** — whether the solution found is the best possible

## Mathematical foundations
[stub]

## Key algorithms / techniques
- **BFS** — explores all nodes at depth d before d+1; optimal for uniform costs
- **DFS** — depth-first exploration; memory-efficient but not optimal
- **A*** — uses f(n) = g(n) + h(n); optimal with admissible heuristic
- **Iterative Deepening A*** (IDA*) — memory-efficient variant of A*
- **MCTS** — Monte Carlo Tree Search; used in game-playing AI (AlphaGo, MuZero)

## Essential reading
> These papers are the minimum to understand this topic.

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [A Formal Basis for the Heuristic Determination of Minimum Cost Paths](https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/astar.pdf) | 1968 | Hart, Nilsson, Raphael | Introduces A* — the canonical heuristic search algorithm |

## Seminal papers & test-of-time
> Papers that defined the field and have held up over time.

| Paper | Year | Key contribution |
|---|---|---|
| A Formal Basis for Heuristic Determination | 1968 | A* algorithm and admissibility proof |

## Current SotA
> *Updated: 2026-05-25*

[stub]

## Code & implementations
[stub]

## Connected topics
- [Classical Planning](./classical-planning.md) — search as the engine of symbolic planning
- [Markov Decision Processes](../06-reinforcement-learning/mdp.md) — search in stochastic environments
- Muzero — modern learned search with MCTS

## Further reading
[stub]
