---
skill: mvb-recipe
description: Write the Minimum Valuable Builds section — multi-variant, one per persona, sharing the same concept but diverging in artifact, time, and success metric. No code dumps; this wiki nudges toward building, not toward tutorials.
applies_to: [write_draft, mvb_recipe, write_arc_step]
triggers: [MVB, build, implementation, recipe, HuggingFace, persona]
---

# Skill: Minimum Valuable Builds (by persona)

## Why this section exists

The MVB is the difference between *"I understand this concept"* and *"I can actually build with it."* But "build" means different things to different readers. A research engineer reproducing a Llama-3 RLHF table on 1×H100 and an applied AI/ML engineer shipping a LoRA-tuned 7B model at 80 ms p95 latency need different recipes. **Both deserve to be served on the same page.**

This wiki rejects code dumps and tutorial introductions. The MVB section is **a structured invitation** — for each persona, a named artifact and a named success metric. The reader picks the variant that matches their day.

## Personas served — the canonical three

Every MVB targets exactly **one** of these. Same arc step can carry multiple MVBs (one per persona that needs a different shape).

| # | Persona | What "their MVB" looks like | Time bound |
|---|---|---|---|
| 1 | **Applied AI/ML engineer** (forward-deployed) | Fine-tune a real model on a custom dataset, ship it behind an endpoint, measure latency / throughput at a stated batch size | Half a day — one working day |
| 2 | **Research engineer** | Reproduce a specific table/figure from a named paper on commodity hardware, hit the number within ±5% | One working day — three days |
| 3 | **Applied researcher** | Run a hypothesis-driven experiment with a falsification criterion and one plot | Two days — one week |

A page typically carries 1–3 MVBs (one per persona that has something distinct to do). Pure-theory pages may carry only one. Above 3 MVBs the section becomes a wall — don't pad just to hit a count.

## The MVB quality bar (writer must clear all five)

A page fails the reviewer if its MVB is missing any of these:

1. **A real, ship-able artifact** — a finetuned checkpoint, a deployed endpoint, an instrumented experiment plot, a reproduced benchmark table. Never "run the official notebook and look at the output."
2. **A concrete time-to-ship** — "1 working day on a single A10", not "a few hours."
3. **Real HF model + dataset IDs** that load via `from_pretrained()` today. No placeholder names; no org-less IDs.
4. **A specific success metric** — "FID < 60", "reward gain ≥ +0.4 vs SFT", "p95 latency ≤ 80 ms at batch 4". Never "results should improve."
5. **Hardness in the middle** — the recipe MUST do at least one of: fine-tune (not just inference), reproduce a paper's number, run an ablation with a falsifier, or deploy with a measured latency target. A `pip install` + `pipeline()` call is not an MVB — it's the model card.

The fifth bar is what makes MVBs FAIRE's USP. A reader can copy from any blog. They come to FAIRE because the MVB is the smallest valuable thing that *takes effort*, with the success metric named upfront so they know when they're done.

## Required structure (one block per persona served)

```markdown
## Minimum Valuable Builds

### 1. For the applied AI/ML engineer (1 working day · single A10 / Colab Pro A100)
**Build:** [ship-it variant — fine-tune real model on custom data, behind a real endpoint]
**Artifact:** [e.g., "fine-tuned `mistralai/Mistral-7B-v0.3` on a 5k support-ticket corpus, served via TGI at p95 ≤ 80ms (batch 4)"]
**Success:** [latency or throughput or accuracy number on the deployed endpoint]
**Stack:** [verified HF IDs + framework + serving lib]

### 2. For the research engineer (1–3 working days · 1×H100 / 2×A100)
**Build:** [reproduce a specific table or figure from a named paper, hit ±5% of the number]
**Artifact:** [e.g., "Table 3 of Llama 3 RLHF paper reproduced on HelpSteer2, reward gain within ±5%, with TensorBoard log"]
**Success:** [the exact metric from the paper, with the ± tolerance you actually hit]
**Stack:** [paper reference + verified HF IDs + the eval harness used]

### 3. For the applied researcher (2–7 days · 1×A100 small)
**Build:** [stated hypothesis + falsifier + the smallest ablation that resolves it]
**Artifact:** [a comparison plot or table — 2–3 conditions, not 30]
**Success:** [the evidence threshold that would change your mind — falsification criterion]
**Stack:** [paper section + minimal training/inference loop + plotting]
```

Every variant: 3–5 lines, no more. The whole MVB section should fit in ~15 lines. **Density over breadth.**

## Hard rules

- **One concept, up to three artifacts.** All variants share the same mechanism; they diverge in scale, compute, and success metric — not in the technique.
- **Named artifacts only.** "Train a small model" fails. "Train a 4M-param UNet DDPM on MNIST 32×32 and hit FID ≤ 20" passes.
- **Real HuggingFace IDs.** `meta-llama/Llama-3-8B-Instruct`, not "Llama 3." `stabilityai/stable-diffusion-xl-base-1.0`, not "SDXL."
- **No code blocks unless naming a config.** The wiki nudges toward building; it doesn't paste training loops. Link to the library quick-start instead.
- **No marketing voice.** No "lightning-fast", "state-of-the-art", "powerful" — strip them.
- **Drop the variant rather than fake it.** If a pure-theory page has nothing for an applied AI engineer, skip that block — don't invent one.

## Three gates every MVB must clear — sensible · valuable · feasible

Each variant the writer drafts must answer YES to all three. The reviewer enforces these explicitly; the writer should self-check before emitting.

### 1. Sensible — matches the persona's actual day

| Persona | Compute realistic for them | Build SHAPE expected |
|---|---|---|
| Applied AI/ML engineer | Single A10 / L4 / Colab Pro A100 — half a day to one working day | Fine-tune + serve a real model; includes a deploy or latency step, not just training |
| Research engineer | 1×H100 or 2×A100 — one to three working days | Reproduce a specific paper table or figure, hit the number within ±5%, with the eval harness instrumented |
| Applied researcher | 1×A100 small — two days to one week | Test ONE stated hypothesis with ONE falsifier and ONE plot — 2–3 conditions, not 30 |

If a variant gives an applied engineer an A100 cluster build, or gives an applied researcher a 30-condition sweep, it fails the sensibility gate.

### 2. Valuable — running it changes what the reader believes

The artifact must be the difference between "I read about this" and "I now know what's actually true." Concrete tests:

- Can the reader paraphrase the central idea differently after the build than before? (Not "I trained the model" but "the noise schedule's curvature determines mode coverage.")
- Does the build force the reader to touch the part of the concept that's hard to grasp from reading? (For diffusion: the `q(x_t | x_0)` reparameterization. The build that doesn't exercise this is just an API tutorial.)
- Does the success metric distinguish right from wrong behavior? (FID ≤ 60 on a held-out set is meaningful; "loss decreases" is not — every training run shows that.)

### 3. Feasible — the persona can actually finish it with the named resources

Verify before recommending:

- **Compute:** does the named hardware fit the named model in memory? (Mistral-7B LoRA fine-tune on a single A10 = ~22 GB → ok. 70B full fine-tune on A10 = broken.)
- **Time:** is the wall-clock realistic? (CIFAR-10 DDPM fine-tune in 1 day on A10 — realistic. SDXL full retrain in 1 hour on free Colab — not.)
- **Data:** is the named dataset accessible and sized correctly? (`HelpSteer2` for an RLHF reproduction — fine. ImageNet-22k from scratch on a single A10 — not.)
- **Library/version:** does the named library version actually support the named operation? (`diffusers >= 0.27` for SDXL Turbo — verify.)
- **Metric:** is the named success metric measurable on the named compute? (FID needs 10k+ samples; if the budget can't generate them, pick a smaller proxy.)

**This is the most common LLM failure mode** — confidently named model + dataset + GPU combinations that don't actually fit. The deterministic MVB stack-verifier in `tools.py::verify_mvb_stack` catches HF-ID typos; the human-readable feasibility check catches the rest.

### Common failure modes and their fixes

| Failure | Example | Fix |
|---|---|---|
| Over-spec'd compute | "Applied engineer: pretrain a 7B model on a single 4090" | Either raise the compute (A100 cluster) or shrink the build (LoRA fine-tune the 7B) |
| Under-spec'd metric | "Reproduce the paper and see if numbers look close" | Name the exact table + the ± tolerance |
| Phantom model ID | "`openai/gpt-3.5-turbo` from HuggingFace" (it isn't there) | Verify on huggingface.co; substitute a real ID |
| Mismatched persona/build | "Applied researcher: ship 70B inference endpoint" | Replace with the smallest hypothesis-test instead |
| Underwhelming — pure inference | "Load the pretrained model and run `pipeline()`" | Add the fine-tune or the reproduction or the ablation — make the middle hard |
| Vague artifact | "Applied researcher: explore the space" | Name the hypothesis + the falsifier + the plot |

## Arc step pages — one MVB, one persona

Arc steps are not concept pages. **Each arc step has exactly ONE MVB**, because each step produces a single compounding artifact for the next step to consume. Declare which persona the step targets in the frontmatter (`for_persona: applied-ai-engineer`). The arc-index lists every step + its persona so the reader sees the lane: "this arc walks the applied-ai-engineer lane for steps 1–4, then opens up to the research-engineer lane in step 5."
