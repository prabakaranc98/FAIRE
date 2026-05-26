# Arc Index Page Schema

Arc index pages are **path pages** — one per arc destination, living at `docs/arcs/{arc-id}/index.md`.
They define WHERE the arc is going, HOW it gets there, and WHAT the reader will have built by the end.
They contain NO builds. Builds live in arc step pages.
They ARE the curated reading guide: selective, opinionated, test-of-time.

---

## Sections (EXACT heading names — reviewer checks these)

1. **YAML frontmatter** between `---` delimiters:
   ```yaml
   title: "Arc: [Arc Name] — [Destination]"
   arc: [arc-id]
   destination: "[one phrase: what you'll have built/understood by the end]"
   tracks:              # curriculum fields this arc draws from
     - 02-generative-modeling
     - 05-statistical-probabilistic-ml
   prereqs:             # curriculum slugs the reader should know before starting
     - variational-inference
     - backpropagation
   total_steps: N
   estimated_time: "[e.g., 6–8 weeks]"
   has_mvb: false       # arc INDEX has no MVB — steps have MVBs
   updated: YYYY-MM-DD
   ```

2. `# Arc: [Arc Name]`

3. `> **Destination:** [One sentence: what capability you will have built by the end of this arc.
   Stated as a concrete ability, not a topic. E.g., "You will have trained a conditional latent
   diffusion model from scratch and understand why it generates better samples than a GAN."]`

4. `## Why this arc exists`
   2 paragraphs. Paragraph 1: what problem or capability gap this arc addresses — the question
   a reader would have that makes them choose this arc over just reading papers randomly.
   Paragraph 2: why THIS sequence and destination, not some other path. What is opinionated
   about the ordering? What would you miss if you skipped to the end?

5. `## Prerequisites`
   Flat list: 3–5 curriculum page links, each with one sentence on what you need from that page.
   Format: `- [Concept Name](../../curriculum/{track}/{slug}.md) — one sentence on what you need`
   Be honest: what can the reader skip and still follow this arc?

6. `## The compounding trajectory`
   A table showing what each step builds and what it enables. Every step's artifact is the
   input to the next step's build.
   ```
   | Step | What you build | Artifact produced | Used by |
   |---|---|---|---|
   | 1 | VAE on MNIST | trained encoder E(x)→z, decoder D(z)→x | Steps 2, 4 |
   | 2 | β-VAE disentanglement | same VAE, interpretable z-dims | — |
   | 3 | Score field on 2D data | score network ∇_x log p(x) | Step 4 |
   | 4 | DDPM on CIFAR-10 | working diffusion model + checkpoint | Steps 5, 6, 7 |
   ```
   Every row must have a non-empty "Artifact produced". "Understanding X" is not an artifact.

7. One `## Chapter K — [Chapter Title]` block per chapter. Each chapter block contains:

   a. **Chapter overview** — 1 paragraph. What this chapter covers and what gap it fills.
      Must use causal connectives. Opens with what the previous chapter left unresolved.

   b. **Curated readings** — a table:
      ```
      | Reading | Type | Why this, why now |
      |---|---|---|
      | [Paper/Blog Title](URL) | seminal paper | one sentence on what this teaches at THIS point in the arc |
      | [Paper Title](URL) | test-of-time | one sentence |
      | [Blog Title](URL) | practitioner | one sentence |
      ```
      2–4 readings per chapter. Types: `seminal paper` | `test-of-time` | `sota model` | `practitioner`.
      "Why this, why now" must be specific to the reader's position in the arc, not generic.
      Only verified arXiv/edu/official URLs — no Medium, no Wikipedia.

   c. **Steps in this chapter** — flat list:
      ```
      - [Step N — What You're Building](./step-NN-{topic}.md) — one sentence on what the build demonstrates
      ```
      Every step must link to a real (or planned) step page.

8. `## The reading order`
   One paragraph explaining HOW to use this arc: read the chapter overview, do the curated
   readings, then do the step builds in order. State explicitly what is optional vs. required.
   Do NOT say "feel free to" or "explore at your own pace" — be opinionated.

9. `## Key figures`
   Flat list: 3–5 researchers whose work defines this arc. Each:
   `- **Name** (Affiliation) — one sentence on their specific contribution to THIS arc's scope`

10. `## Where this arc leads`
    2–3 sentences. After completing this arc, what OTHER arcs become accessible?
    What frontier research problems are now within reach?
    Format: one sentence per follow-on arc, naming the arc and the specific dependency.

---

## HARD RULES

- `has_mvb: false` in frontmatter — arc indexes have no builds
- `destination` field must be filled — one concrete capability phrase, not a topic
- Every chapter must have curated readings (no empty chapters)
- Curated readings must use only: arXiv, *.edu, distill.pub, lilianweng.github.io,
  official engineering blogs (ai.meta.com, research.google, openai.com/research)
- Never: medium.com, towardsdatascience.com, wikipedia.org, substack.com, youtube.com
- "The compounding trajectory" table must have non-empty "Artifact produced" for every step
- Step links in chapter blocks must be real relative paths (or contain a TODO comment)
- No motivational language: "your journey", "feel free to", "explore at your own pace"
- Chapters must be causally ordered: each chapter opens by naming what the previous left unresolved
- The arc index does NOT reproduce curriculum page content — it links to it
