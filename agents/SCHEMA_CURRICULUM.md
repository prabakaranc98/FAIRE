# Curriculum Page Schema

Curriculum pages are **reference wiki articles** — one per concept, organized by field.
They serve researchers, engineers, theorists, and curious learners simultaneously.
They contain NO builds. The build lives in the arc step pages.

---

## Sections (EXACT heading names — reviewer checks these)

1. **YAML frontmatter** between `---` delimiters:
   ```yaml
   title: [Concept Name]
   track: [track-slug]
   tags: [4–6 keywords]
   depth: foundational | intermediate | advanced
   prereqs: [2–4 prerequisite topic slugs]
   arc_refs: []           # filled by write_file_node from arc_context
   updated: YYYY-MM-DD
   has_mvb: false         # ALWAYS false for curriculum pages
   ```

2. `# [Title]`

3. `> **TL;DR:** [one sentence: what this is + why frontier researchers care]`

4. `## Who this page is for`
   Table with 4 rows, 3 columns. Every "Jump to" entry uses a real `#anchor` link
   to a heading on this page:
   ```
   | Persona | What you get | Jump to |
   |---|---|---|
   | Researcher | Open problems + current SotA + frontier citations | [§What's happening now](#whats-happening-now) |
   | Engineer / Practitioner | Key algorithms, production deployments, where to build | [§Key algorithms](#key-algorithms--techniques), [§In production](#in-production) |
   | Theory / Math student | Derivations, annotated equations, essential papers | [§Mathematical foundations](#mathematical-foundations) |
   | Curious generalist | Plain-English explanation, why it exists | [§What it is](#what-it-is) |
   ```

5. `## What it is`
   3 prose paragraphs. The FIRST sentence states the human problem or a surprising fact
   that makes the reader immediately feel WHY this concept exists. Never opens with a
   definition ("X is a..."). Paragraphs are causally connected: each flows from the prior.

6. `## Why it matters`
   2 paragraphs. Connects to frontier open problems, active lab priorities, and the
   adjacent concepts that depend on this one.

7. `## Core concepts`
   Flat bullet list. 5–8 items. Each: `**term** — one precise definition sentence.`

8. `## Mathematical foundations`
   3–5 equations. After EACH equation block, write:
   "where \(symbol\) is ..., \(symbol\) is ..." — annotate EVERY symbol.
   Then one intuition sentence: "This equation says that..."
   Use `\[...\]` for display math, `\(...\)` for inline. Never `$...$`.

9. `## Key algorithms / techniques`
   Flat list: `**Name** (Year) — 2 sentences: what it does, when you'd use it over alternatives.`

10. `## Essential reading`
    Table: `| Paper | Year | Authors | Why essential |`
    2–4 papers. Only verified arXiv/edu URLs. Each entry answers: what does reading this
    teach you that nothing else does?

11. `## Seminal papers & test-of-time`
    Table: `| Paper | Year | Key contribution |`
    Papers that reshaped the field AND held up. Cite with arXiv URL.

12. `## Current SotA`
    2–3 sentences. Named systems + specific benchmark numbers + years.
    Format: "[Model] achieves [metric] on [benchmark] ([year])."

13. `## What's happening now`
    3 prose paragraphs: Research frontiers / Engineering & Systems / Open problems.
    EVERY factual claim names a paper inline: "Author et al. (YEAR) showed [CLAIM] ([arXiv URL])."
    Vague language without a citation ("recent work suggests", "some approaches") is a violation.

14. `## In production`
    3–5 bullets: `Company — System — Scale (real number) — [Source](URL)`
    Source must be an approved engineering blog. No Wikipedia, no Medium.

15. `## Open questions`
    THREE admonition blocks — one per persona. Each question must be specific enough that
    a motivated person could design an experiment or write a paper to answer it:

    ```markdown
    !!! researcher "For researchers"
        [Theoretical or mathematical question no paper has cleanly answered.
         Must be specific enough to design a study around.]

    !!! engineer "For engineers"
        [Practical experiment or ablation no one has published.
         Should be runnable in under a day on a consumer GPU or free Colab.]

    !!! open "Think about this"
        [Conceptual puzzle that makes you question something you assumed was obvious.
         Phrased as a question, not a direction.]
    ```

16. `## This concept appears in`
    ← REQUIRED. At least one entry.
    Flat list: arc step pages that use this concept, with a one-sentence explanation
    of how they use it:
    ```
    - [Step N — Title](../../arcs/{arc}/step-NN-{topic}.md) — one sentence on the connection
    ```
    If no arc step has been generated yet, write: "Arc step pages for this concept are being generated."

17. `## Connected topics`
    3–5 cross-curriculum links. Each with ONE sentence on the mechanistic relationship —
    what specific concept, equation, or structure they share. Not "also related to".

18. `## Further reading`
    4–6 items: arXiv, *.edu, distill.pub, lilianweng.github.io only.
    One sentence per item on what it adds beyond this page.

---

## HARD RULES

- `has_mvb: false` in every curriculum page frontmatter — no exceptions
- No `## Minimum Valuable Build` section — if present, the reviewer rejects
- "What it is" must not open with a definition. Fail test: does it start with "[Topic] is a..."?
- Every prose section (What it is, Why it matters, What's happening now) must use causal
  connectives: "This is why...", "The consequence is...", "That led directly to...", etc.
- Tables may not be nested inside other tables
- Bullet lists may not be nested (no bullet inside a bullet in explanatory sections)
- Math: ONLY `\[...\]` for display, `\(...\)` for inline. Never `$...$` or `$$...$$`
- Banned URLs: medium.com, towardsdatascience.com, wikipedia.org, substack.com, youtube.com
- No motivational language: "you'll learn", "let's explore", "your journey", "feel free to"
- "This concept appears in" section is required — at least one link or a placeholder
