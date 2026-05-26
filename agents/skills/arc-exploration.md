---
skill: arc-exploration
description: The explorer playbook — given a curriculum seed, survey the branching space, map against the diagonal pattern, then pick + outline the top arcs. Three steps. Refuses to pick prematurely. Read by the supervisor's arc-proposal phase.
applies_to: [supervisor]
triggers: [arc, exploration, survey, planning]
---

# Skill: Arc exploration playbook

A single curriculum seed almost always has multiple valid arc destinations. Example: the seed "world models" can plausibly become any of —

- A Gaussian-Splatting world model arc (Kerbl et al. 2023 → 3DGS → scene-aware planning)
- A JEPA world model arc (LeCun 2022 → V-JEPA 2024 → predictive coding for action)
- A diffusion-based world model arc (Du et al. 2024 → DDPM-WM → diffusion policy)
- A predictive-action-network arc (PAN-style → action-conditioned dynamics)

A naive proposal step picks the first plausible arc the LLM generates and commits to it. That's not an arc; that's a guess. **The explorer playbook forces survey before selection.**

## The three steps — survey → map → pick

### Step 1 — Survey (Exa-driven)

Given seed `S`:

1. Issue an `exa_search_sota` call with `query="active research directions in {S} 2024 2025"` and `start_published_date="2024-01-01"`. Get 6 hits.
2. From the top 2 hits, call `exa_find_similar(url, num_results=5)` to widen the net. (See `exa-search-deep.md` for the find_similar idiom.)
3. Read the titles and highlights. Inventory the distinct branching directions. Aim for ≥ 3 branches; fewer suggests the field is too narrow for an arc.
4. Format the inventory as a table:

| Branch | Anchoring paper(s) | Cost class | Persona fit | Compounding spine |
|---|---|---|---|---|
| Gaussian-Splatting WM | Kerbl 2023 (3DGS), Tao 2024 (4D-GS) | A100 × few | applied-researcher, frontier-researcher | scene → dynamics → planning |
| JEPA WM | LeCun 2022, V-JEPA 2024 | A100 × few | theory-student, frontier-researcher | mask-predict → temporal JEPA → action |
| Diffusion WM | Du 2024 (DDPM-WM) | A100 × medium | applied-researcher | DDPM → conditional → diffusion-policy |
| PAN-based WM | (named paper) | A10 × few | applied-engineer | action → next-state → planning loop |

If the survey returns fewer than 3 distinct branches, **stop**. Return verdict
`needs-curriculum`: the seed isn't broad enough to support a real arc yet,
and the right move is to queue more curriculum pages first.

### Step 2 — Map against the diagonal pattern

For each surviving branch, evaluate the diagonal-arc shape from `arc-anatomy.md` (specialized tool → broader frame → capability → frontier intersection):

| Check | If fails |
|---|---|
| Branch has a clean diagonal — column 1 ≠ column 4 | Drop branch (vertical isn't an arc) |
| Specialized tool (col 1) maps to an EXISTING approved curriculum page | Drop branch (prereq is stubbed — queue prereq instead) |
| Frontier intersection (col 4) maps to a real active research area | Drop branch (no frontier landing → not arc-worthy) |
| Compounding spine has ≥ 4 named compoundable artifacts | Drop branch (chain too loose) |
| `mvb_persona` can be assigned to ≥ 75% of intended steps | Drop branch (persona-fit is unclear) |

A branch that survives all 5 checks goes to step 3. Drop the others to a "deferred" list with the gating reason — the supervisor surfaces these so the user can decide to queue prereq curriculum work.

### Step 3 — Pick + construct (via `arc-selection.md`)

Apply the 7-dimension impact rubric from `arc-selection.md` to each surviving branch. Compute EV/$ per the cost formula there. Pick the top 1–2 such that:

- `Σ cost ≤ remaining_budget × 0.5` (leave headroom for curriculum maintenance + revisions)
- `K ≤ 2 − active_arcs` (the 2-active-arcs ceiling from `arc-anatomy.md`)

For each picked branch, **construct the arc outline** using `arc-anatomy.md`'s 7-element template:
1. arc_id + track + destination
2. Prereqs (4–6 curriculum pages, with status: solid/adequate/stub/missing)
3. Compounding trajectory table
4. Chapters (3–4 chapters of 2–3 steps each)
5. Per-chapter curated readings (1–3 papers, labeled seminal/test-of-time/SotA/production reference)
6. Per-step `mvb_persona` tag
7. Per-chapter open question

If any element comes up empty during construction, mark the arc `verdict: reshape` and surface the missing element to the human. **Do not paper over gaps.**

## Hard rules

- **Do not pick the first branch the model generates.** Even if it looks great, finish the survey of ≥ 3 branches before scoring.
- **Do not pick a branch where the diagonal is broken.** Vertical arcs ("transformer → … → transformer at scale") aren't arcs.
- **Do not pick a branch whose specialized-tool prereq is a stub.** Queue the curriculum work first; come back next cycle.
- **Do not propose more than 2 new arcs in one cycle.** The 2-active-arcs ceiling is a hard limit, even when EV/$ looks great.

## Output format

Write the survey + map + pick to `docs/system/arc-proposals.md`. See `arc-selection.md` for the full output template — this skill just defines the *process* that fills it in.

If `verdict: needs-curriculum`, write:

```markdown
## Exploration deferred — seed "{S}"

The survey returned < 3 distinct branching directions. Queue more curriculum
pages first. Suggested prereqs: {list of stubbed concepts the survey touched}.
```

## How this is invoked

Two callers:

1. **Autonomous mode**: `supervisor.maybe_propose_arcs()` runs this skill when its three preconditions hold (coverage ≥ 60%, mode = full, active_arcs < 2). The seed is picked from the curriculum tracks with the highest coverage.

2. **CLI mode**: `uv run python generate.py explore <seed>` runs this skill on the seed the user names. Optional `--track <track>` constrains the survey domain.

Both modes produce the same `arc-proposals.md` output. Both require the human to pick before any arc-step is queued.

## Related skills

- `arc-anatomy.md` — the diagonal pattern + 5-bundle anatomy this skill builds toward
- `arc-selection.md` — the scoring rubric step 3 applies
- `critic-editor.md` — the editor-in-chief who can veto an exploration's chosen arc
- `exa-search-deep.md` — the survey-phase Exa idioms
- `agent-runtime.md` — budget gate and the `active_arcs` count
