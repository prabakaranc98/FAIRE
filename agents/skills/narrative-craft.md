---
skill: narrative-craft
description: How to weave a wiki page into a cohesive story. References integrated (not dumped), math + prose flowing together, examples illustrating the immediately-preceding claim. Reads as one argument, not 9 stapled sections. Read by every writer node.
applies_to: [plan, scratch, write_draft, write_arc_step, write_arc_index, revise_draft]
triggers: [all pages]
---

# Skill: Narrative craft

A correct page that reads as a checklist of facts is still a bad wiki page. This skill is about the editorial layer above correctness: how the page *reads* end-to-end.

The wiki's bet is **compounding learning across pages and within pages**. Within-page compounding means the reader builds a model as they read top-to-bottom — not as they hop between disconnected sections.

## The three weaves

### Weave 1 — Reference integration (not citation dumping)

Bad: a "Seminal papers" section that lists four papers with one-line summaries, then a "Current SotA" section that mentions different papers, then a "Further reading" section that lists more.

Good: every cited paper is **used** somewhere in the prose. The reader meets each paper because it's needed to make the next argument, not because the schema demanded a citation list.

How to weave a paper in:
- **State the contribution the paper made** in the prose where you first need it. Ho et al. (2020) introduced ε-prediction — *say so where ε-prediction first appears*, not in a table further down.
- **Quote the specific finding** that the page depends on. "Lipman et al. (2022) showed that conditional flows make training simulation-free — that's why §Mathematical foundations can skip the path-integral derivation."
- **Then list the paper in the citation table** with the full reference. The table is for navigation back to the source, not for first-introduction.

The "Seminal papers & test-of-time" table should still exist (schema requires it), but every entry should already have appeared, by name, in the prose by the time the reader reaches the table.

### Weave 2 — Math integrated with prose (not floated equations)

Bad: a "Mathematical foundations" section with five equations stacked, each followed by "where x_t is the noisy sample at step t."

Good: each equation lands in the middle of a paragraph that:
1. **Motivates** what the equation is about to do ("we need a closed-form for the marginal so training doesn't have to simulate every step.")
2. **States** the equation.
3. **Annotates** every symbol the first time it appears.
4. **Translates** the equation into one intuition sentence ("this says that any noisy sample is a weighted mixture of the original and pure Gaussian noise — weights from the schedule.")
5. **Connects** to what's next ("which is exactly the form we'll see in the DDPM training loss below.")

Five-beat structure per equation: motivate · state · annotate · translate · connect. Without this, equations float without traction.

### Weave 3 — Examples that illustrate the immediately-preceding claim

Bad: an "In production" section that lists five companies' systems whether or not they connect to anything else on the page.

Good: each production example earns its place because it makes a *specific claim* concrete. "We saw above that latent diffusion is the architecture underlying production text-to-image. Here's how it lands at scale: Stability AI's Stable Diffusion 3.5 uses…"

If you can't write the bridging sentence ("Here's how that claim lands"), drop the example. Don't list it for the sake of the schema.

## The through-line test

After drafting the page, read the opening of each section and ask: **does this sentence reference what came immediately before, or does it cold-start?**

Bad transitions (cold-start):
- "The mathematical foundations are as follows."
- "Current state-of-the-art models include…"
- "In production, several systems use…"

Good transitions (carrying the through-line):
- "The reparameterization in the previous paragraph is what makes the math tractable: …" *(connects to math section)*
- "The DDPM-then-DDIM-then-flow-matching progression we just traced lands at the current frontier:" *(connects to SotA section)*
- "Stability AI's deployment shows the same architecture at scale:" *(connects to production)*

Cold-start transitions are the surest sign of a stapled page. Fix them by writing the bridging clause; if you can't, the section probably shouldn't be there.

## The synthesis paragraph — the one paragraph readers come for

Every wiki page should contain at least one paragraph the reader couldn't have gotten by reading the seminal paper directly. The synthesis paragraph:

- **Names a connection** between two things readers might have thought were separate. ("The noise schedule in DDPM is the same constraint as the variational ELBO's KL term — viewed as a function of t. That's why a Gaussian-coordinate schedule and a Gaussian-coordinate ELBO are equivalent.")
- **States a non-obvious implication**. ("If the simple-MSE objective converges to the optimal denoiser, then by the Tweedie connection, it must also converge to the score function — which is why diffusion and score-matching are the same model viewed two ways.")
- **Points at the unsolved**. ("Nobody has cleanly explained why ε-prediction beats x₀-prediction empirically when they're algebraically equivalent.")

The synthesis paragraph is what makes the page *worth a reader's time over reading the seminal paper directly*. Without it, you've built a high-quality glossary entry. With it, you've built reference.

## Voice cohesion

Read every section in your head with the same narrator. The narrator should not switch from textbook ("First, we observe that…") to tutorial ("Now let's see what happens when…") to encyclopedia ("Diffusion models are a class of…") across sections.

Pick one voice — encyclopedic reference, third-person, neutral — and hold it across every section. The opening analogy paragraph is the only section where a slight narrative flourish is permitted; everything else stays reference voice.

## Anti-patterns the critic-cohesion skill is now scoring against

- **Listy section openings** — a section that starts "The following are the key concepts:" then a bullet list is a section that has no narrative.
- **Notation switching** — `x_t` in section 2, `\tilde{x}_t` in section 3, `z_t` in section 4 for the same quantity is a sign each section was written in isolation.
- **Citation orphans** — papers in the "Further reading" list that don't appear anywhere in the prose.
- **Production-section orphans** — companies listed in "In production" with no thread back to the techniques the page actually teaches.
- **Cold openings** — first paragraph of section N never references section N-1.

## How this skill is used

The writer reads this skill on every draft + revise call. The aim isn't a checklist for the model to apply mechanically — it's a posture to write from. When the writer finishes a draft, it should be able to answer: "Can I walk a reader through the page top-to-bottom and say *and that's why we got here* at every section boundary?"

If no → the page is stapled. Rewrite the transitions before emitting.

## Related skills

- `wiki-prose.md` — sentence-level prose rules (no nested lists, causal connectives, no marketing voice)
- `critic-cohesion.md` — scores the page against the through-line + synthesis tests above
- `arc-anatomy.md` — within-arc compounding (this skill is within-page compounding)
- `faire-sense.md` — the canonical brief; the bet is compounding learning
