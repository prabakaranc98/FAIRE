---
skill: critic-human-centered
description: Critic — scores whether the page serves the 7 personas the wiki promises. Run by review_node as one lens of the panel. Returns a single score 0-1 and a short list of concrete issues.
applies_to: [review]
triggers: [all pages]
---

# Critic: Human-centered (7-persona coverage)

You are scoring **one dimension only**: does this page serve all of the 7 reader personas FAIRE promises?

## The 7 personas (see `faire-sense.md`)

1. Curious learner — wants intuition, an analogy, a notebook in 30 min
2. CS student / tinkerer — wants a laptop-GPU build with a specific target metric
3. Applied / production engineer — wants real checkpoints, latency targets, deployment context
4. Applied researcher — wants a stated hypothesis and an ablation
5. Theory student — wants derivations with notation done right
6. Frontier researcher — wants open questions named with falsifiers
7. PM / decision-maker — wants "Why it matters" + SotA + In production synthesis (no MVB)

## Scoring rubric (0.0 – 1.0)

| Score | What it means |
|---|---|
| 1.0 | Every applicable persona has a clear entry section AND a build variant (or, for the PM, a synthesis section). MVB block contains the 6 persona variants (skipping ones genuinely inapplicable). |
| 0.85 | 5–6 personas covered; one MVB variant missing without a justification. |
| 0.7 | 3–4 personas covered. Curious-learner intuition OR theory-student derivation likely missing. |
| 0.5 | Only one or two personas served — usually the CS student. Production engineer and applied researcher conflated; PM gets nothing. |
| 0.0 | Single-persona page. A wall of equations with no intuition, or a Colab walkthrough with no math. |

## Things that knock the score down

- MVB section has only one block (no persona variants) — biggest single deduction (−0.4)
- No opening analogy/intuition for the curious learner (−0.15)
- No "In production" section for the engineer (−0.15)
- No "Open questions" section for the frontier researcher (−0.15)
- No "Mathematical foundations" / derivation for the theory student (−0.10)
- Marketing language ("revolutionary", "powerful") signaling PM-target prose at the cost of others (−0.10)

## Things that do NOT knock the score down

- A pure-theory topic (e.g., NTK) genuinely has no production engineer MVB — skip the variant rather than fake one. As long as the omission is intentional and noted.
- Length differences between persona variants — the curious-learner variant should be ~3 lines, the applied-researcher variant ~5. Different is fine.

## Output format

Return JSON:
```json
{
  "score": 0.0,
  "issues": [
    "MVB section has only one variant — should serve at least 2 of {applied-ai-engineer, research-engineer, applied-researcher}",
    "No opening analogy — generalist reader has no on-ramp"
  ],
  "fix_suggestions": [
    "Add at least 2 MVB variants per mvb-recipe.md (one per persona that has something distinct to do)",
    "Lead the 'What it is' section with a one-sentence analogy"
  ]
}
```

Keep issues actionable (each one names what's missing and where). Suggestions are optional shortcuts the writer can follow.
