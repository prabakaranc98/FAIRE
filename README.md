# FAIRE
### Frontiers in AI Research and Engineering

---

> *Effort without a spine doesn't compound. It just accumulates.*

The distinction that drives everything here:

| Arc of work | Proof of work |
|---|---|
| Thinking that builds on itself | Volume of accomplishments |
| Each project extends the last | Unconnected artifacts |
| Shows a mind that compounds | Shows a list |

FAIRE is the spine.

---

## The Structure

```
Curriculum          10 tracks — the knowledge substrate you draw from
      ↓
Research Arcs       6 areas — where you actually operate
      ↓
Projects            Chains of sprints within each arc, not standalone artifacts
      ↓
Capstone            Synthesized evidence of arc traversal
```

---

## Active Arcs

**Maximum 2 active at once.** Depth doesn't divide by six.

| Arc | Status | Phase | Current sprint |
|---|---|---|---|
| — | — | — | — |
| — | — | — | — |

*Phases: `Reproduce` → `Extend` → `Originate`*
*Status: `active` · `queued` · `dormant`*

---

## Arc Index

| # | Arc | Curriculum draws from | Status |
|---|---|---|---|
| 01 | [Post-training, Interpretability & Alignment](arcs/01-post-training-interpretability/) | AI · Neural Networks · RL | queued |
| 02 | [Generative Modeling & World Models](arcs/02-generative-worlds/) | Generative Modeling · RL | queued |
| 03 | [ML & AI Systems](arcs/03-ml-systems/) | Algorithms & Systems for AI | queued |
| 04 | [Advanced Deep Learning](arcs/04-advanced-deep-learning/) | Neural Networks · Representation Learning · Statistical ML | queued |
| 05 | [Causal & Statistical Inference](arcs/05-causal-inference/) | Causal & Statistical Inference · Statistical & Probabilistic ML | queued |
| 06 | [Complexity, Cognition & Natural Intelligence](arcs/06-complexity-cognition/) | Complexity · Cognition · Neuroscience | queued |

---

## Curriculum Tracks

The 10 knowledge tracks that feed the arcs. These are the substrate, not the work.

1. **AI** — LLMs, VLMs, alignment, mechanistic interpretability
2. **Generative Modeling** — autoregressives, VAEs, GANs, diffusion, flows
3. **Representation Learning** — self-supervised, contrastive, multimodal
4. **Neural Networks & Deep Learning** — CNNs, attention, transformers, optimization
5. **Statistical & Probabilistic ML** — Bayesian inference, graphical models, GPs
6. **Reinforcement Learning** — MDPs, policy gradients, world models, RLHF
7. **Attention, Memory, Reasoning & Continual Learning**
8. **Causal & Statistical Inference** — SCMs, causal discovery, counterfactuals
9. **Algorithms & Systems for AI** — parallelism, quantization, ML compilers
10. **Complexity, Cognition & Natural Intelligence** — dynamics, emergence, neuroscience

---

## How a Sprint Works

Each sprint lives inside an arc. It is one bird.

```
arcs/
  NN-arc-name/
    ARC.md              ← arc definition, phase, sprint index
    sprints/
      NNN-problem-name/
        PROBLEM.md      ← define before building
        log.md          ← what was tried, what happened (including failures)
        src/            ← code and notebooks
```

**To start a sprint:**
1. Check the arc's `ARC.md` — confirm the arc is active and which phase you're in
2. Copy `_templates/PROBLEM.md` + `_templates/log.md` into a new sprint folder
3. Fill `PROBLEM.md` completely before opening `src/`
4. Run experiments, log as you go — honest, not polished
5. Sprint ends when you hit an exit condition: result, wall, pivot, or abandoned
6. Update the sprint index in `ARC.md`

**The daily test:** *Does this artifact extend an active arc, or just describe FAIRE itself?*

---

## Arc Walking Methodology

Within each arc, work progresses in three phases:

**Reproduce** — take a key result from the literature and build it from scratch until your intuition matches theirs. Every step produces an artifact.

**Extend** — modify, ablate, or probe the thing you reproduced. Test a hypothesis. See what breaks.

**Originate** — add something that wasn't there. A new question, a new result, a new framing. Small is fine.

---

See [PRINCIPLES.md](PRINCIPLES.md) before starting anything.
