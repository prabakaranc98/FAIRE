---
skill: source-policy
description: Enforce citation rules — approved domains, format, recency balance
applies_to: [write_draft, scratch]
triggers: [citation, source, reference, paper, read, links, essential reading, further reading, seminal]
---

# Skill: Source Policy

## Approved sources

Only cite sources from these domains. Any other domain is rejected.

| Domain | Examples | Use for |
|---|---|---|
| `arxiv.org` | arxiv.org/abs/2006.11239 | Papers — use exact arXiv ID |
| `*.edu` | cs.stanford.edu, proceedings.mlr.press | Papers, course notes |
| `huggingface.co` | huggingface.co/papers | Model cards, papers, datasets |
| `distill.pub` | distill.pub/2016/augmented-rnns | Visual explainers |
| `lilianweng.github.io` | lilianweng.github.io/posts/... | Verified survey posts only |
| Official library docs | pytorch.org, jax.readthedocs.io | API reference |
| Official company research | research.google, ai.meta.com, openai.com/research | Verified engineering posts |

## Banned sources

NEVER cite these, regardless of content quality:

- Medium (any subdomain)
- Towards Data Science
- Analytics Vidhya
- Personal blogs (non-company)
- Substack
- Wikipedia or any wiki
- Tutorial aggregators (machinelearningmastery.com, neptune.ai, etc.)
- Papers With Code (cite the original paper instead)

## Citation format

```
Author et al. (YEAR) — "Exact Title as It Appears in the Paper" — https://arxiv.org/abs/XXXX.XXXXX
```

Rules:
- If the arXiv URL did not appear in the search results → write `[URL NOT VERIFIED]`, never fabricate a URL
- Author list: first author only + "et al." unless ≤2 authors (then list both)
- Year: publication year, not preprint year if both exist

## Recency balance

Every `## Essential reading` and `## Seminal papers & test-of-time` section must have:
- At least 2 foundational papers (published before 2020)
- At least 1 recent paper (published within 2 years of the current date)

If no recent papers were found in search results, state that explicitly rather than stretching the definition of "recent."

## Fabrication rule

If a paper title sounds right but no URL was found in the search results, mark it `[UNVERIFIED]` in the scratch pad and do NOT include it in the final page. It is better to have 3 verified citations than 5 with fabricated URLs.
