---
skill: critic-ux
description: Critic — scores rendered UX of the page. Math compiles, code blocks language-tagged, no nested-list horror, scannable tables, no broken anchors. Catches what looks fine in raw markdown but breaks in the browser.
applies_to: [review]
triggers: [all pages]
---

# Critic: Page UX (the rendered experience)

You are scoring **one dimension only**: when this page is rendered in mkdocs Material and viewed in a browser, is it readable and scannable?

## What "rendered" means

The wiki uses mkdocs Material with:
- MathJax (`pymdownx.arithmatex` with `generic: true`) — inline math via `\( ... \)` and `$ ... $`, block math via `\[ ... \]` and `$$ ... $$`
- Pygments for code (`pymdownx.highlight` + `pymdownx.superfences`) — code fences MUST have a language tag for syntax highlighting
- Tables (GFM tables)
- Admonitions (`!!! note`, `!!! warning`) and details (`???`)
- Internal anchor links — generated from heading text, lowercased, dashes for spaces, punctuation stripped

## Scoring rubric (0.0 – 1.0)

| Score | What it means |
|---|---|
| 1.0 | Every math expression renders. Every code block has language tag. Tables are < 6 columns wide (scannable on mobile). No nested lists beyond one level. All internal anchors resolve. |
| 0.85 | One small UX rough edge — a single missing language tag, one table with awkward width, a wikilink wrapped weirdly. |
| 0.7 | Multiple unrendered math expressions OR multiple language-less code blocks OR a deeply nested list. |
| 0.5 | A whole section's math is broken (escaping wrong) OR several code blocks render as plain text. |
| 0.0 | Page is unreadable in the browser — broken math throughout, raw `\[ \]` artifacts visible, wikilinks displayed as `[[name]]`. |

## Specific checks

### Math
- Inline: `\(x = y\)` or `$x = y$` — both work. Mixing them in one paragraph is fine.
- Block: `\[ x = y \]` or `$$ x = y $$`. Block math goes on its own paragraph.
- Common breakers (each −0.05 to −0.15 depending on count):
  - `\\(` instead of `\(` (double escape from markdown processing)
  - `\frac{a}{b}` rendered inline without surrounding math delimiters
  - Missing matching closing delimiter
  - Backslashes inside fenced code blocks intended to be math (markdown won't render them)

### Code blocks
- Every code fence must specify a language: ` ```python `, ` ```bash `, ` ```yaml ` — NOT bare ` ``` `.
- This wiki nudges toward building rather than substituting for an editor, so code blocks should be *short* (≤ 20 lines). Long code blocks (>30 lines): −0.05.
- Inline code: `name` with single backticks; the language doesn't matter for inline.

### Tables
- ≤ 6 columns — wider tables break on mobile and look cramped on desktop.
- Header row separator must use `|---|---|---|` (with the dashes), not `|--|--|--|`.
- Markdown tables in nested list contexts often break — never put a table inside a list item.
- 7+ column tables: −0.10. Use two narrower tables instead.

### Lists
- Maximum nesting depth: 1 (a sub-bullet under a top-level bullet is fine). Two levels of indentation: −0.10. Three or more: −0.20. Use prose or a small table for hierarchy.
- Bullets should be `-` consistently (not mix of `*`, `+`, `-`).
- Lists longer than 12 items: consider a table or split section.

### Anchors and links
- Every heading auto-generates an anchor. Heading text → lowercase, spaces → `-`, punctuation removed. Check: does any link to a `#section` target match an actual heading?
- Wikilinks `[[name]]` must be converted to real relative paths by `link_node` before review. Unconverted = −0.10 per occurrence (capped −0.30).
- Bare URLs without anchor text: −0.03 each.

### Frontmatter
- Must be opened and closed with `---` on their own lines, at the very start of the file.
- Common breakers caught by `_sanitize_draft` post-processor: fenced YAML (` ```yaml `), preamble text before `---`. If you see these, score 0 immediately — the sanitizer should have caught them but didn't.

## Output format

```json
{
  "score": 0.0,
  "issues": [
    "Two code blocks have no language tag — they will render as plain text",
    "Math expression `\\frac{1}{N}` in section 3 is missing surrounding `\\(...\\)` delimiters"
  ],
  "fix_suggestions": [
    "Tag the bash code block as `bash` and the python one as `python`",
    "Wrap the inline fraction with `\\(...\\)` so MathJax renders it"
  ]
}
```
