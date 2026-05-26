# Arc Step Page Schema

Arc step pages are **build pages** — one per arc step, organized inside an arc directory.
They live inside `docs/arcs/{arc-id}/step-NN-{topic}.md`.
They contain the Minimum Valuable Build. The curriculum page contains the theory.

---

## Sections (EXACT heading names — reviewer checks these)

1. **YAML frontmatter** between `---` delimiters:
   ```yaml
   title: "Step N — [What You're Building]"
   arc: [arc-id]
   step: N               # integer — step number within the arc
   total: M              # total steps in this arc
   chapter: K            # integer — chapter this step belongs to (1-based)
   chapter_title: "[Chapter Title]"
   topic_refs:           # curriculum pages this step draws from
     - curriculum/02-generative-modeling/diffusion-models
   compounding_artifact: "[one-sentence description of what this step produces]"
   prev_artifact: "[one-sentence description of what the previous step produced]"
   has_mvb: true         # ALWAYS true for arc step pages
   updated: YYYY-MM-DD
   ```

2. **Arc navigation breadcrumb** (injected by write_file_node — writer should include it):
   ```markdown
   > **Arc:** [Arc Title](../index.md) — Step N of M
   > ← [Previous Step Title](./step-NN-topic.md) &nbsp;&nbsp; [Next Step Title →](./step-NN-topic.md)
   ```
   On the first step, omit the "←" back-link. On the last step, omit the forward arrow.

3. `# Step N — [What You're Building]`
   Title states the artifact being built, not the concept being learned.
   Good: `# Step 2 — Train a DDPM on 2D Toy Data`
   Bad: `# Step 2 — Understanding Diffusion Models`

4. `> **TL;DR:** After this step you will have [specific artifact] that demonstrates [falsifiable claim].`

5. `## Why this step exists`
   2 paragraphs. Paragraph 1: name the PREVIOUS step explicitly and state the specific artifact
   it produced (use `prev_artifact` from frontmatter). Then state what gap that artifact leaves —
   what you now have but cannot yet do with it.
   Paragraph 2: what THIS step adds, what artifact it produces (`compounding_artifact`), and why
   this ordering is the right one. Must use causal connectives.

6. `## The idea you're testing`
   1–2 paragraphs. State the conceptual claim the build will validate as a FALSIFIABLE assertion.
   Bad: "DDPM is a diffusion model that uses noise prediction."
   Good: "DDPM claims that learning to denoise in 300 small steps is easier for a neural
   network than learning to generate in one step. This build tests that claim: if the
   model converges and generates clean samples, the claim holds on this toy data."
   The claim must be specific enough that running the build either confirms or falsifies it.

7. `## The theory you need`
   2–3 paragraphs. Minimal, targeted theory — exactly what the reader needs to understand
   the recipe. Link out to the curriculum page for full derivations — do not reproduce them.
   Include the 1–2 key equations needed to implement the recipe, fully annotated.
   Each equation: annotate EVERY symbol with `where \(x\) is ..., \(y\) is ...`
   Then one intuition sentence: "This equation says that..."

8. `## Minimum Valuable Build`

   **What you're building:** [1 sentence — specific artifact with real observable output]

   **Why this:** [1–2 sentences — what the build demonstrates that reading could not show you]

   **Stack:**
   - **Model:** [exact class name or HuggingFace model ID]
   - **Dataset:** [exact HuggingFace dataset ID or sklearn/torchvision source]
   - **Framework:** [library + version]
   - **Compute:** Colab T4 free / consumer GPU ≤16GB

   **Estimated time:** [realistic wall-clock time including data download]

   ### The recipe
   Numbered steps. Each step: ONE operation + ONE sanity check.
   Every step must include either an `assert` or a `print` statement so the reader
   immediately knows whether it worked.

   ### Expected output
   Specific: shapes, ranges, qualitative description.
   One sentence on what SUCCESS looks like. One sentence on what FAILURE looks like.
   Example: "Success: the scatter plot shows two clean clusters by epoch 50.
   Failure: all points collapse to a single cluster — check your learning rate."

   ### Common failure modes
   3–5 bullets. Each: `[Symptom] → [Diagnosis] → [Fix]`
   These must be drawn from real failure patterns, not invented hedges.

9. `## Stretch goals`
   2–3 extensions for readers who completed the build.
   Each: one sentence on what to try, one sentence on what you would learn from it.
   Stretch goals must be achievable on the same hardware as the build.

10. `## What this unlocks`
    2 paragraphs. Paragraph 1: what capability you now have that you didn't before,
    stated as a concrete ability ("you can now train a model that...").
    Paragraph 2: how this capability is the prerequisite for the NEXT arc step —
    the specific conceptual dependency, not just "you're ready for step N+1".

11. `## Open questions`
    THREE admonition blocks — one per persona. Each question must be specific enough
    that a motivated person could design an experiment or write a paper to answer it.
    The questions should be inspired by the build — things the reader will wonder
    AFTER they run the code and see results.

    ```markdown
    !!! researcher "For researchers"
        [Theoretical question raised by the build — something observable that no paper
         has cleanly explained. Specific enough to design a study around.]

    !!! engineer "For engineers"
        [Practical experiment: what happens if you change X? Should be runnable
         in under a day on the same hardware as the build. No one has published
         a clean ablation.]

    !!! open "Think about this"
        [Something the build makes concrete that was abstract before. Phrased
         as a question that makes you question something you assumed was obvious.]
    ```

12. `## Go deeper`
    3–5 links to curriculum pages, each with one sentence on what deeper content
    they provide for this step specifically.
    ```
    - [Concept Name](../../curriculum/{track}/{slug}.md) — one sentence on the specific
      theory from that page that illuminates what you just built
    ```
    At minimum, link to the primary curriculum page for this step's main concept.

---

## HARD RULES

- `has_mvb: true` in every arc step frontmatter — no exceptions
- `compounding_artifact` and `prev_artifact` must be present and non-empty
- "Why this step exists" must name the previous step's artifact verbatim (from `prev_artifact`)
- Every recipe step includes either an `assert` or a `print` statement
- "The idea you're testing" section must contain exactly ONE falsifiable claim
- "Expected output" must give specific numbers or shapes — not "a plot" but "shape: (1000, 2)"
- Compute must fit on free Colab T4 (≤15GB GPU RAM) — if it doesn't fit, redesign the build
- No nested lists anywhere
- No definition-first openings ("Diffusion models are..." → FAIL)
- "Why this step exists" must reference the PREVIOUS step explicitly
- Open questions must be phrased as questions, not as directions ("investigate X" → FAIL)
- "Go deeper" must link to at least one curriculum page
- Arc breadcrumb must be present immediately after frontmatter
- Title format: `# Step N — [artifact name]`, not `# Step N — [concept name]`
- Math: ONLY `\[...\]` for display, `\(...\)` for inline. Never `$...$` or `$$...$$`
- Banned URLs: medium.com, towardsdatascience.com, wikipedia.org, substack.com, youtube.com
- No motivational language: "you'll learn", "let's explore", "your journey", "feel free to"
