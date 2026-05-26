---
skill: reasoning-scaffolding
description: How agents in this system reason — plan-then-write, step-by-step auditing, self-critique. Forces deliberate thinking inside single LLM calls. Read by plan, scratch, writer, and critic nodes.
applies_to: [plan, scratch, write_draft, write_arc_step, write_arc_index, revise_draft, review]
triggers: [all pages]
---

# Skill: Reasoning scaffolding

This skill defines *how* the FAIRE agents think inside each call. The system already breaks work into nodes (research → plan_and_scratch → write → review). This skill adds scaffolding *within* each node so the model reasons deliberately before committing output.

## Three scaffolds and when each fires

### 1. Plan-then-write (writer nodes)

When you're about to author a wiki page (curriculum, arc-step, or arc-index), produce a **5-line internal outline first**, then the full page. The outline is part of your reasoning, not the output — it sits between your thinking and your prose.

The 5 lines are:
1. **Question this page answers**: one sentence the reader brought.
2. **Three load-bearing claims**: what the page must establish.
3. **Citation backbone**: which 3 papers (with verified URLs) the page leans on.
4. **MVB persona spread** (curriculum / arc-step only): which personas this page serves with build variants.
5. **Self-check**: what would make the critic-cohesion skill score this <0.6? Write that risk down before you write the page.

Then write the page so it satisfies the outline. Do NOT skip the outline even if you're confident — the discipline catches the page that "feels right" but contradicts the seminal paper.

### 2. Step-by-step auditing (critic nodes)

When you're a critic in the review panel, do NOT produce a holistic score on first read. Walk the page section by section:

1. **Locate** the part of the page your critic dimension actually covers (e.g., critic-info-architecture skims for backlinks; critic-coverage skims for word count + section presence).
2. **Apply** each deduction rule from your skill body literally. Tally the deductions on a scratch line.
3. **Verify** before you score: re-read the section to confirm the deduction you wrote is grounded in the text, not in a paraphrase.
4. **Compose** the final {score, issues, fix_suggestions} structured output.

A critic that scores in one pass tends to hallucinate issues that aren't actually in the text. The walk catches this.

### 3. Self-critique (arc proposer + supervisor)

When you're proposing arcs in `maybe_propose_arcs`, after you've drafted the candidate list, **invoke the critic-editor lens against your own proposals before emitting**. Specifically:

For each candidate arc you're about to propose:
- Ask: would the critic-editor (see `critic-editor.md`) veto this for seed-readiness, compounding breaks, or persona-span issues?
- If yes, **mark it `verdict: reshape` or `verdict: veto` yourself** instead of `approve` — surface the issue rather than hide it.
- Only propose `verdict: approve` for arcs you can defend against the editor's known objections.

This is the equivalent of a writer self-editing before sending. It saves a roundtrip with the human and trains the proposal phase to internalize the editor's standards.

## What this scaffolding is NOT

- **Not chain-of-thought spam.** Don't pad output with "Let me think step by step..." Spell out your reasoning in 4-5 tight bullets, not paragraphs.
- **Not a multi-LLM-call requirement.** All scaffolds run inside a single LLM call. The model produces the scaffold internally and the final output in the same response.
- **Not a substitute for the structured output schema.** The structured output (CriticScore, ReviewResult) is the contract. Scaffolding shapes what fills the contract; it doesn't replace it.

## When to skip a scaffold

- **MVB section authoring** — the MVB recipe IS the scaffold (named artifact + compute + success metric). Don't add a meta-plan on top.
- **Routing functions** (`route_after_review`, `_route_after_plan_scratch`) — these are pure-function deciders, not LLM calls. No scaffolding needed.
- **Single-fact extraction** (e.g., scratch_pad fact-sheet) — bullet-style fact extraction doesn't need a plan.

## Related skills

- `arc-anatomy.md` — the 5-element template the writer scaffolds toward
- `arc-exploration.md` — the survey → map → pick three-phase walk (already explicit scaffolding)
- `critic-editor.md` — the standards the arc-proposer self-critiques against
- `langgraph-patterns.md` — when to externalize a scaffold as a sub-graph instead
