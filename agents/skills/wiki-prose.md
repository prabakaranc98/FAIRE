---
skill: wiki-prose
description: Write as a mentor walking through the idea — primer-quality from sentence one. Accurate as an encyclopedia in WHAT it says, primer-paced in HOW it says it. The reader should be able to follow the whole page without bouncing.
applies_to: [write_draft, revise_draft, write_arc_step, write_arc_index]
triggers: [all pages]
---

# Skill: Wiki Prose Style

## The standard

Write as **a mentor walking a smart peer through the idea**, not as an encyclopedia entry. Two readers should leave satisfied:

1. The newcomer who has only heard the topic name — they finish the page able to paraphrase the concept and the trade-offs.
2. The practitioner who already knows the area — they finish the page with concrete numbers, named papers, and a build to attempt.

These are not in tension. A good primer is a good reference too, because the connective prose that helps the newcomer also names every concept precisely enough for the practitioner.

The wiki's whole nudge-to-build thesis rests on this: if reading is hard, the reader bounces before reaching the MVB, and the build never happens. **Primer tone is not a style choice; it is the product.**

## Tone principles

**Neutral, not promotional**
- Reject: "Score matching is a powerful and elegant technique."
- Accept: "Score matching estimates the gradient of the log-density without computing the partition function — which is exactly the term that makes likelihood-based generative models hard."

**Factual, not hype** — same rule, with the WHY attached.
- Reject: "This groundbreaking paper revolutionized the field."
- Accept: "Ho et al. (2020) introduced epsilon-prediction, which let diffusion models train with a simple MSE loss instead of the variational bound earlier work used. That single change is why DDPM became the dominant generative architecture for the next three years."

**Specific, not vague**
- Reject: "Diffusion models achieve strong results on many benchmarks."
- Accept: "FLUX.1 reaches FID 2.4 on ImageNet 256×256 — for comparison, DALL-E 3 sits at 3.6 and the original DDPM was 5.6."

**Direct, but with the intuition first**
- Reject: "Attention computes a weighted sum of values, where weights are determined by key-query similarity."
- Accept: "Attention is the model's way of saying 'when reading this token, the relevant context lives over here, here, and here, not anywhere else'. Mechanically, it computes a weighted sum of values where each weight measures how strongly a query matches a key."

## Voice — mentor walking through

Write as if the reader is sitting next to you and you are explaining the idea. The voice is third-person about the content, but the rhythm is conversational about the *explanation*. Both can coexist.

- The content stays in third person. NOT "we apply softmax" — "the model applies softmax".
- But the explanation can address the reader's likely confusion: "the natural question is why ..."; "if this looks like coordinate descent, that intuition is correct"; "the part that trips most readers is ...".
- Imperative verbs are allowed when guiding the reader through a derivation or build — "notice that ...", "consider what happens when ...". They are not allowed as filler ("note that X is important").
- "You" is allowed in the Build it section and in places where you are nudging the reader toward an action they will take. Otherwise stay third-person.

## Structure — gradual build, never definition-dump

- A section opens with **a sentence framing why we are about to read it**, then the substance. Never opens with an equation, never opens with a bullet list, never opens with "X is defined as...".
- The first time a technical term is used in a page, it gets a one-line intuition in the same sentence or the next: "the KL divergence — informally, how surprised you would be if you used the wrong distribution — measures ...". Subsequent uses do not need re-explaining.
- Acronyms are spelled out on first use, even if they are common in the field. "RLHF (reinforcement learning from human feedback)" — once. "RLHF" thereafter.
- Equations land with a translation sentence on either side: the sentence before says what the equation is about to express, the sentence after says which term in it does the work.
- Connective tissue is required across paragraphs and sections: "the consequence is ...", "here is where it gets interesting ...", "this is the key tension ...", "the trade-off the field is still resolving is ...". A page without these markers reads as stapled fragments.

## What every section is for (depth calibration)

- **Hook (no heading, ~150 words)** — open with the human problem the concept solves. A scenario, a misconception, a vivid image. Not a definition. Hook earns the reader's next paragraph.
- **The territory (~300 words)** — where this idea sits in the field, what family of techniques it belongs to, what problem it answers. End by setting up the mechanism: "how does it actually work?".
- **How it works (~800–1500 words)** — the mechanism narrated. Math embedded in prose with annotations. Worked examples and failure modes belong here. Build from the simplest version of the idea to the modern one.
- **Where the field is now (~400 words)** — named papers in prose ("the 2025 result from Yu et al. (DAPO) showed ..."). One research frontier, one engineering frontier. Specific benchmark numbers.
- **What's still open (~250 words)** — the named open questions, each specific enough that a researcher could write a paper on it.
- **Where to read next** — a paragraph of pointers, not a bullet dump.
- **Build it** — the MVB section. See `mvb-recipe.md` for the gates.

## What to avoid

- Opening sentences with "It is worth noting that ..." / "This shows the power of ..." — both are filler.
- Filler intensifiers: "fundamentally", "essentially", "at its core", "ultimately".
- Bullet lists as the spine of a section. Bullets are for the MVB Stack + recipe + per-persona variants, and nowhere else.
- Equations stacked without English between them. Every equation needs a translation sentence.
- "Mathematical foundations" as a fenced-off section disconnected from the explanation. Math goes where the explanation needs it.
- Marketing voice: "powerful", "elegant", "groundbreaking", "lightning-fast", "state-of-the-art" used as adjectives. State the specific result instead.
- Reference dumps in "Where the field is now" — three sentences naming five papers with no connecting argument. Each paper should be cited because it makes a point you are arguing.
- Acronyms before they are spelled out.
- More than three consecutive paragraphs without a connective phrase. The reader loses the through-line.
