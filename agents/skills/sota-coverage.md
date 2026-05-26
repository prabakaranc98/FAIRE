---
skill: sota-coverage
description: SotA section must have model names, benchmark scores, and years — no vague claims
applies_to: [write_draft, scratch]
triggers: [sota, benchmark, leaderboard, performance, evaluation, metrics, state of the art, current, results]
---

# Skill: State-of-the-Art Coverage

## The standard

Every SotA claim requires four elements: **model name**, **benchmark name**, **score**, **year**.

Good: "FLUX.1 achieves FID 2.4 on ImageNet 256×256 (2024), compared to 3.6 for DALL-E 3 (2023)"
Bad: "Diffusion models achieve strong results on many image generation benchmarks"

## What to include

**Current leader** — name, score, benchmark, year. If there is no clear single leader (area is fragmented), say so.

**Two baselines for comparison** — not just the winner. The reader needs to understand the gap being closed, not just the current number.

**Year on every claim** — SotA changes in months. A claim without a year has an unknown expiry date.

## Formatting rule

If comparing ≥3 models, use a table:

```markdown
| Model | Benchmark | Score | Year |
|---|---|---|---|
| FLUX.1 | FID (ImageNet 256) | 2.4 | 2024 |
| DALL-E 3 | FID (ImageNet 256) | 3.6 | 2023 |
| Stable Diffusion XL | FID (ImageNet 256) | 5.8 | 2023 |
```

For 1-2 comparisons, prose is fine.

## When numbers are missing

If the search results contain no benchmark numbers:
- Write: "Benchmark evaluations for this area are not standardized as of [year]; the most widely cited comparison is [qualitative description]."
- Do NOT write vague claims like "achieves impressive performance" or "outperforms prior work"
- Do NOT invent numbers

## Domain-specific eval suites

For LLM evaluation, always cite the eval suite by name, not the model it compared against:
- ❌ "outperforms GPT-4 on reasoning tasks"
- ✓ "achieves 90.1% on MMLU (5-shot) and 87.3% on HumanEval, compared to GPT-4's 86.4% and 67.0% respectively (2024)"

Standard suites to recognize: MMLU, HumanEval, MATH, GSM8K, HellaSwag, WinoGrande, ARC, BIG-Bench, HELM, LMSYS Chatbot Arena.

## Freshness signal

Include a note if the SotA is fast-moving: "This benchmark is updated frequently; check Papers With Code for the current leader." (Note: link to Papers With Code only as a pointer, not as a citeable source for the numbers themselves.)
