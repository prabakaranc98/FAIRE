---
skill: wiki-prose
description: Write in neutral, encyclopedic, reference-quality prose — not a tutorial, not a pitch
applies_to: [write_draft, revise_draft]
triggers: [all pages]
---

# Skill: Wiki Prose Style

## The standard

Write like a well-edited encyclopedia, not like a blog post, tutorial, or product pitch.
The reader is a competent researcher or practitioner. They do not need motivation.
They need accurate, dense, well-organized information they can return to repeatedly.

## Tone principles

**Neutral, not promotional**
- ❌ "Score matching is a powerful and elegant technique"
- ✓ "Score matching estimates the gradient of the log-density without computing the partition function"

**Factual, not hype**
- ❌ "This groundbreaking paper revolutionized the field"
- ✓ "This paper introduced X, which became the basis for Y and Z"

**Specific, not vague**
- ❌ "Diffusion models achieve strong results on many benchmarks"
- ✓ "FLUX.1 achieves FID 2.4 on ImageNet 256×256, compared to 3.6 for DALL-E 3"

**Direct, not hedged**
- ❌ "One might argue that attention could possibly be seen as..."
- ✓ "Attention computes a weighted sum of values, where weights are determined by key-query similarity"

## Voice and person

- Write in third person. No "you", "we", "our"
- No imperative verbs in explanatory sections ("Note that X" → "X is")
- Exception: MVB section uses "you" — it's a recipe for the reader to follow

## Structure signals

- Lead every section with the key fact, not context-setting
- Every paragraph = one main idea, stated in the first sentence
- Avoid "In this section, we discuss..." → just start discussing
- Use plain headers: "## Core concepts", not "## Understanding Core Concepts"

## Depth calibration

- **What it is** — 2-3 paragraphs, assumes zero domain knowledge, analogy-first
- **Core concepts** — technical but readable; definition then intuition
- **Mathematical foundations** — precise; full notation with annotations
- **Key algorithms** — concrete; name + year + what changed + why it mattered
- **SotA** — numbers, model names, years, venues; no vague "recent advances"
- **In production** — specific company + system + scale claim (with source)

## What to avoid

- Starting sentences with "It is worth noting that"
- Ending sections with "This shows the power of X"
- Filler phrases: "fundamentally", "essentially", "at its core"
- Overuse of em-dashes for emphasis
- Lists where prose would read better (3 items can be a sentence)
- Analogies that are more confusing than the concept itself
