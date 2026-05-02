# FAIRE
### Frontiers in AI Research and Engineering

Open-ended, sprint-based experiments at the edge of what's understood in AI.

Not chasing breakthroughs — building, probing, and documenting what happens when you push on hard problems. Everything ships: working results, partial findings, and documented failures alike.

---

## What this is

A personal research program structured around **problem spaces** and **execution sprints**.

Each sprint starts with a precisely defined question and ends with a concrete result — working code, a finding, or a documented wall. The constraint is clarity, not novelty. The goal is to know something real by the end.

---

## Problem Spaces

The themes FAIRE operates across. Each sprint belongs to one.
All problems are in core Frontier AI Research and Engineering — no application domains.

| Space | Core question |
|---|---|
| **World Models** | How do causal structure and energy landscapes make models generalize beyond training distribution? |
| **Continual Learning** | Can we close the plasticity–stability gap without architectural overhead? |
| **Causal Representation Learning** | Under what conditions can latent causal variables be identified from observations alone? |
| **Post-Training & Alignment** | What do SFT, RLHF, and DPO actually do to a model's representations and capabilities? |
| **Systems & Efficiency** | Where are the real bottlenecks — training, inference, communication — and can they be moved? |
| **Foundation Model Science** | What can mechanistic analysis, probing, and scaling experiments tell us about what models learn? |

---

## Sprint Index

| # | Problem | Space | Status | Outcome |
|---|---|---|---|---|
| — | — | — | — | — |

*Status: `active` · `done` · `abandoned`*

---

## How a sprint works

```
sprints/
  NNN-problem-name/
    PROBLEM.md   ← define before building
    log.md       ← running notes: what was tried, what happened
    src/         ← code and notebooks
```

1. Copy `sprints/000-template/` → `sprints/NNN-name/`
2. Fill `PROBLEM.md` completely **before** writing any code
3. Run experiments, log as you go in `log.md`
4. Sprint ends when you hit an exit condition — result, wall, or time limit

Read [PRINCIPLES.md](PRINCIPLES.md) once before starting.

---

## Outcomes taxonomy

Every sprint ends with one of these:

- **`result`** — something was learned, code ran, a finding was documented
- **`wall`** — hit a real obstacle (compute, theory gap, wrong framing); documented why
- **`pivot`** — the question changed mid-sprint; original framing documented, new sprint opened
- **`abandoned`** — deprioritized; two sentences on why and what you'd try next
