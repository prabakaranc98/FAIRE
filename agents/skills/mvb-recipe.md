---
skill: mvb-recipe
description: Write the Minimum Valuable Build section — a working implementation recipe, not pseudocode
applies_to: [write_draft, mvb_recipe]
triggers: [MVB, build, implementation, recipe, HuggingFace, code]
---

# Skill: Minimum Valuable Build (MVB)

## Purpose

The MVB is the difference between "I understand this concept" and "I can actually build with it."
It is a concrete, reproducible recipe for a working implementation that demonstrates the core idea.

It is NOT:
- A code dump
- A tutorial introduction
- Pseudocode that won't run
- A list of "next steps"

It IS:
- A specific task ("Train a denoising diffusion model on CIFAR-10")
- A verified stack (real HuggingFace model IDs, real library versions)
- A step-by-step recipe (numbered, each step does one thing)
- A working snippet that a practitioner can run today

## Structure

```markdown
## Minimum Valuable Build

**What you're building:** [one sentence — specific artifact]
**Why this build:** [one sentence — what it demonstrates about the concept]
**Stack:** [library names + versions, HuggingFace model IDs]
**Estimated time:** [realistic — "30 min" or "2-3 hours"]

### The recipe

1. **[Step title]**
   [What to do, as a command or code block]

2. **[Step title]**
   [...]

### Expected output
[What success looks like — specific numbers, shapes, qualitative description]

### Common failure modes
- [Failure] → [Fix]
```

## HuggingFace model IDs

Always use real, verified model IDs from the search results. Format:
- `meta-llama/Llama-3-8B-Instruct` (not "llama-3" or "Meta Llama")
- `google/flan-t5-base` (not "T5" or "Flan T5")
- `stabilityai/stable-diffusion-2-1` (not "SD 2.1")

## Calibrating difficulty

| Topic type | Build target |
|---|---|
| Core concept (e.g. attention) | Implement from scratch in < 100 lines, validate on toy data |
| Architecture (e.g. transformer) | Fine-tune on a standard benchmark (SST-2, CIFAR-10) |
| Algorithm (e.g. PPO) | Train an agent on a standard env (CartPole, MuJoCo Hopper) |
| System (e.g. data parallelism) | Reproduce a training run with DDP on 2+ GPUs |
| Frontier method (e.g. flow matching) | Train on CIFAR-10 or CelebA; measure FID |

## Code quality rules

- All code blocks must be labeled with language: ` ```python `
- Use real imports, not `from model import Model`
- Include `device = torch.device("cuda" if torch.cuda.is_available() else "cpu")`
- Include model loading with explicit dtype for large models
- Every snippet should be self-contained (can run from top to bottom)
- Include one assertion or sanity check per major step
