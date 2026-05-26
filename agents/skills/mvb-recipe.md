---
skill: mvb-recipe
description: Write the Minimum Valuable Builds section — multi-variant, one per persona, sharing the same concept but diverging in artifact, time, and success metric. No code dumps; this wiki nudges toward building, not toward tutorials.
applies_to: [write_draft, mvb_recipe, write_arc_step]
triggers: [MVB, build, implementation, recipe, HuggingFace, persona]
---

# Skill: Minimum Valuable Builds (by persona)

## Why this section exists

The MVB is the difference between *"I understand this concept"* and *"I can actually build with it."* But "build" means different things to different readers. A CS student building MNIST diffusion in a notebook and a production engineer serving SDXL Turbo at p99 latency need different recipes. **Both deserve to be served on the same page.**

This wiki rejects code dumps and tutorial introductions. The MVB section is **a structured invitation** — for each persona, a named artifact and a named success metric. The reader picks the variant that matches their day.

## Personas served (and skipped)

| # | Persona | MVB shape | Skip? |
|---|---|---|---|
| 1 | Curious learner | A Colab/browser thing they can *see* in 30 min | — |
| 2 | CS student / tinkerer | A small training run on a laptop GPU, hit one metric | — |
| 3 | Applied / production engineer | Load real checkpoint, serve, measure latency or throughput | — |
| 4 | Applied researcher | One focused experiment with a stated hypothesis | — |
| 5 | Theory student | A derivation verified on toy data; one plot | — |
| 6 | Frontier researcher | A probe of an open question, with a falsification criterion | — |
| 7 | PM / decision-maker | (no MVB — they get "Why it matters" + SotA + "In production" only) | **skip** |

Six variants per pivotal page. Skip variants when the topic genuinely doesn't have one (e.g., a pure-theory page may have no production engineer MVB — leave it out rather than fake one).

## Required structure (one block per persona served)

```markdown
## Minimum Valuable Builds

### 1. For the curious learner (30 min · free tier)
**Build:** [one sentence — what they'll see]
**Artifact:** [the named, sharable thing — e.g., "a Colab notebook visualizing the forward diffusion of 2D Gaussians"]
**Success:** [observable signal — e.g., "the cloud collapses to N(0, I) by step 1000"]
**Stack:** [link or model/dataset ID]

### 2. For the CS student / tinkerer (1 day · RTX 4070 / M-series)
**Build:** [...]
**Artifact:** [...]
**Success:** [a specific number — e.g., "FID ≤ 20 on MNIST 32×32"]
**Stack:** [verified HuggingFace IDs and library versions]

### 3. For the applied / production engineer (1 week · A10 or cloud)
**Build:** [shipping-shaped — quantization, serving, latency target]
**Artifact:** [e.g., "a vLLM endpoint serving SDXL Turbo at p50 < 1.5s on A10"]
**Success:** [latency, throughput, or cost number]
**Stack:** [...]

### 4. For the applied researcher (3 days · A100)
**Build:** [stated hypothesis + ablation]
**Artifact:** [a comparison table or curve]
**Success:** [evidence that confirms or falsifies the hypothesis]
**Stack:** [...]

### 5. For the theory student (1 day · CPU)
**Build:** [a derivation followed by numerical verification]
**Artifact:** [a plot or table showing theory matches simulation]
**Success:** [residual below threshold; or correct closed-form match]
**Stack:** [paper section referenced + minimal Python]

### 6. For the frontier researcher (1 week+ · A100 cluster)
**Build:** [probe of a named open problem from the "Open questions" section]
**Artifact:** [evidence the proposed answer holds or fails]
**Success:** [a falsification criterion — what would change your mind]
**Stack:** [...]
```

Every variant: 3–5 lines, no more. The full MVB section is ~25 lines, not 100. Density over breadth.

## Hard rules

- **One concept, six artifacts.** All variants share the same underlying mechanism. They diverge in scale, compute, and success metric — not in the technique being demonstrated.
- **Named artifacts only.** "Train a small model" is wrong. "Train a 4M-param UNet DDPM on MNIST 32×32 and hit FID ≤ 20" is right.
- **Real HuggingFace IDs.** `meta-llama/Llama-3-8B-Instruct`, not "Llama 3." `stabilityai/stable-diffusion-xl-base-1.0`, not "SDXL."
- **No code blocks unless naming a config.** This wiki nudges toward building; it doesn't substitute for the user's editor. Link to the library's quick-start instead of pasting 20 lines of training loop.
- **No marketing voice.** "Lightning-fast," "state-of-the-art," "powerful" — strip them.
- **If a variant doesn't fit the topic, drop it.** Don't invent a "production engineer MVB" for a pure-theory page like NTK.

## The MVB quality bar — sensible · valuable · feasible

Every variant the writer proposes must pass all three of these gates. Critics check this; the writer should self-check before emitting.

### 1. Sensible — does the build match the persona's day?

- Curious learner → browser / free Colab, < 1 hr, no installation required.
- CS student → consumer GPU (RTX 3060 to 4090, M-series Mac), < 1 day, `pip install` is fine.
- Applied engineer → cloud GPU (A10/L4/A100), 3 days to 1 week, includes a deployment step (serving, quantization, monitoring) not just training.
- Applied researcher → A100 × small, 3 days to 1 week, the build is an ABLATION with a stated hypothesis — not a model train.
- Theory student → CPU / notebook, < 1 day, the build is a VERIFICATION OF A DERIVATION on toy data — not a training run.
- Frontier researcher → A100 cluster, 1+ week, the build PROBES AN OPEN QUESTION with a falsification criterion stated.

If a variant gives a curious learner an A100 cluster build, or gives a frontier researcher a Colab notebook, it fails the sensibility gate.

### 2. Valuable — does running the build teach the concept?

The artifact must be the difference between "I read about this" and "I now know what's actually true." Concrete tests:

- Can the reader **paraphrase the central idea differently** after the build than before? (Not just "I trained the model" but "the noise schedule's curvature determines mode coverage.")
- Does the build force the reader to **touch the part of the concept that is hard to grasp from reading**? (For diffusion: the closed-form `q(x_t | x_0)` reparameterization. The build that doesn't exercise this is just an API tutorial.)
- Does the success metric **distinguish the right behavior from the wrong behavior**? (FID ≤ 20 on MNIST is meaningful; "loss decreases" is not — every training run shows loss decreasing.)

A build that produces an artifact but doesn't teach the concept = fails the value gate.

### 3. Feasible — can the named persona actually complete it with the named resources?

Verify before recommending:

- **Compute:** does the named hardware actually fit the named model in memory? (124M GPT pre-train on a single 4090 = ~24GB VRAM, fits with gradient checkpointing — ok. Pre-training a 7B model on a 4090 = won't fit — broken.)
- **Time:** is the named wall-clock realistic for the named compute? (CIFAR-10 DDPM in 1 day on a 4070 — realistic. Stable Diffusion fine-tune in 1 hour on a free Colab — not realistic.)
- **Data:** is the named dataset publicly accessible and reasonably sized for the named compute? (HF `wikitext-103-raw-v1` for a 124M pre-train baseline — fine. ImageNet-22k from scratch on consumer hardware — not feasible.)
- **Library:** does the named library/version actually support the named operation? (`diffusers >= 0.27` supports SDXL Turbo distillation — verify version before naming.)
- **Metric:** is the named success metric measurable on the named compute? (FID requires 10k+ generated samples — feasible if the budget allows it; otherwise pick a smaller proxy like sample-quality eyeballing on 64 generations.)

A build that the persona literally cannot complete with the resources you named = fails the feasibility gate. **This is the single most common failure mode of LLM-generated MVBs** — confidently named model + dataset + GPU combinations that don't actually fit.

### Self-check before emitting

For each variant the writer drafts, mentally answer:

1. **Sensible?** Would this persona, on a typical day, choose this build?
2. **Valuable?** Will running this build change what the reader believes about the concept?
3. **Feasible?** Have I verified the compute / time / model / dataset / metric quadruple is consistent?

If any answer is "I'm not sure," cut the variant or rewrite until you can answer yes to all three.

### Common failure modes and their fixes

| Failure | Example | Fix |
|---|---|---|
| Over-spec'd compute | "Curious learner: train SDXL from scratch on H100" | Drop to a Colab visualization of one forward pass |
| Under-spec'd metric | "CS student: train and see what happens" | Name a benchmark + a target number |
| Phantom model ID | "Use `openai/gpt-3.5-turbo` from HuggingFace" (it isn't there) | Verify on huggingface.co; substitute a real ID |
| Mismatched persona/build | "Theory student: build a 70B inference endpoint" | Replace with a derivation + toy verification |
| Time/compute incoherent | "Applied engineer: pretrain GPT-2 on a single 3060 in 1 day" | Either raise the compute or shrink the model |
| Vague artifact | "Frontier researcher: try something new" | Name the open question + the falsifier |

## Calibrating difficulty (per-variant)

| Variant | Time | Compute | Artifact size |
|---|---|---|---|
| Curious learner | 30 min – 1 hr | Free / browser | 1 notebook |
| CS student | 4 hr – 1 day | 1 consumer GPU | 1 checkpoint + 1 plot |
| Applied engineer | 3 days – 1 week | 1 cloud GPU | 1 deployed endpoint + latency report |
| Applied researcher | 3 days – 1 week | 1–4 cloud GPUs | 1 table or curve, 1 written conclusion |
| Theory student | 4 hr – 1 day | CPU | 1 derivation + 1 verification plot |
| Frontier researcher | 1+ week | cluster | 1 evidence artifact + falsifier statement |

## Why this beats one-MVB-per-page

A single MVB silently picks one persona as "the" reader. That used to be the CS student by default — which excluded everyone shipping at work and everyone running ablations. The 6-variant block makes inclusion explicit and lets every reader find their lane.

## Arc step pages — one MVB, declared persona

Arc steps are different from curriculum pages. **Each arc step has ONE MVB**, because each step has a single compounding artifact. Declare which persona the step is shaped for in the frontmatter (`mvb_persona: ml-tinkerer`). The arc index lists every step + its persona so a reader can see the lane: "this arc walks the CS student lane for steps 1–5, then opens up to the applied researcher lane in step 6."
