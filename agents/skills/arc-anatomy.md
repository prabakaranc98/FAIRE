---
skill: arc-anatomy
description: What a good arc looks like — five things bundled into one syllabus. Worked example using "Pre-training" as the arc. Read by the writer when producing arc-index pages and by the supervisor when proposing arcs.
applies_to: [plan, scratch, write_arc_index, supervisor]
triggers: [arc, arc-index, syllabus, pre-training, planning]
---

# Arc anatomy — the five things bundled

## The shape: the diagonal arc pattern

Before listing what an arc must contain, name what shape it should have. A good arc is **diagonal**, not vertical:

```
              specialized              broader              capability             frontier
              tool you can     →       frame that      →    that uses        →    intersection
              already touch            connects to a        both                   where two
                                       research community                          fields meet
```

A vertical arc ("go deep on diffusion models") is a textbook chapter — useful but not transformative. A diagonal arc crosses domains: it starts somewhere a reader already has footing, broadens to a frame that exposes them to a community, synthesizes a capability that depends on both, and lands at an intersection where two active research areas collide.

**Worked patterns (same shape, different topics):**

| Specialized tool | Broader frame | Capability | Frontier intersection |
|---|---|---|---|
| Causal state-space models | Causal representation learning | Counterfactual reasoning | Causal RL |
| Transformer | Efficient attention variants | Long-context retrieval | Agentic memory systems |
| RLHF | Reward modeling | RL fine-tuning | Agent RL with tool use |
| Diffusion models | Score matching | Conditional generation | Inverse problems / scientific simulation |
| VAE | Information bottleneck | Representation learning | World models |
| MCTS | Policy improvement | Imitation learning | Self-improvement loops |

Why this shape works:

1. **Specialized tool** — gives the reader an immediate, low-barrier entry point. Something they can already touch with their existing skill.
2. **Broader frame** — pulls them out of "one technique" and into "a community of thinking." They start reading the right papers.
3. **Capability** — forces synthesis. The reader has to combine the specialized tool with the broader frame to do something neither alone can do.
4. **Frontier intersection** — where the wiki's promise pays off. The reader lands in a place where two active research areas overlap, and the open questions are real.

When the supervisor proposes an arc, it should be able to fill in this four-cell row clearly. If a proposed arc has the same column-1 and column-4 (e.g., "transformer → ... → transformer at scale") it is not diagonal — it is vertical. Reshape or veto.

## The five things every arc must bundle

A good arc is **not** a list of pointers. Beyond the diagonal shape, it is five things bundled into one editorial syllabus:

1. **Essentials tied back to curriculum** — every step links to its prereq curriculum pages so a reader can fix a knowledge gap without leaving the arc.
2. **Arc-specific information** — the arc index itself has substantial content: why this arc exists, the compounding-trajectory table, key figures, where it leads.
3. **Seminal reads (curated)** — 1–3 papers per chapter, picked because they are *necessary to understand what the next build is testing*. Not exhaustive.
4. **Related reference material** — beyond the seminal: implementation references (HuggingFace model cards, official library docs), lab tech reports (DeepMind/Anthropic/Meta engineering blogs where they exist).
5. **MVBs + open questions per step** — every step ends in a persona-tagged build AND opens a specific question the reader can ponder/probe in the next sprint.

If a proposed arc doesn't deliver all five, it's a reading list, not an arc.

---

## Worked example — "Pre-training" arc inside Track 01 (AI)

The user named this as a model arc. Let's specify it precisely so the agents can produce arcs of this shape on demand.

### Arc identity
- **arc_id:** `pre-training`
- **track:** `01-ai`
- **destination:** "Pretrain a 124M-parameter GPT-2-class model from scratch, end-to-end, with a clean data pipeline and a defensible loss curve."
- **total_steps:** 8
- **estimated_time:** 4–6 weeks, 6 hr/week

### Prereqs (curriculum)
- [transformer architecture](../../curriculum/07-attention-memory-reasoning/transformer.md)
- [optimization](../../curriculum/04-neural-networks-dl/optimization.md)
- [scaling-laws](../../curriculum/04-neural-networks-dl/scaling-laws.md)
- [data-parallelism](../../curriculum/09-algorithms-systems-ai/data-parallelism.md) *(soft prereq for step 6+)*
- [kv-cache](../../curriculum/09-algorithms-systems-ai/kv-cache.md) *(soft prereq for step 8)*

A reader missing any of these can click out, read, and come back without losing the thread.

### Compounding trajectory

| Step | Build | Artifact | Used by |
|---|---|---|---|
| 1 | BPE tokenizer on 1B-token corpus | trained tokenizer.json + vocab.json | Steps 2, 3, 4, 5, 6, 7, 8 |
| 2 | Tiny GPT (4-layer, 64-dim) on Shakespeare | small_gpt.pt checkpoint | Step 3 (for sanity comparison) |
| 3 | Data pipeline: deduplication + quality filtering | filtered_corpus.parquet (5B tokens) | Steps 4, 5, 6, 7 |
| 4 | Loss-curve baseline: 124M model, single-GPU, no parallelism | baseline_124m.pt + loss_curve.csv | Step 5 (for comparison) |
| 5 | Add learning-rate schedule + warmup; re-run | tuned_124m.pt + tuned_loss_curve.csv | Step 6 |
| 6 | Switch to 2× data parallelism | dp_124m.pt + dp_throughput.csv | Step 7 |
| 7 | Add gradient checkpointing + mixed precision | optimized_124m.pt | Step 8 |
| 8 | Eval suite: perplexity on WikiText-103 + HellaSwag zero-shot | eval_report.md (a numbered table) | — |

This is **literal compounding** — step 7's checkpoint is the file step 8 loads. Step 1's tokenizer is the function steps 2–8 import. A reader who walks the arc end-to-end finishes with one defensible pipeline and one named metric.

### Chapters

**Chapter 1 — Tokenization & data substrate (Steps 1–3)**
- Why tokenization is upstream of everything: a bad tokenizer caps loss reduction before training starts.
- Curated readings:
  - Sennrich et al., 2016 — Neural Machine Translation of Rare Words with Subword Units (BPE seminal)
  - Penedo et al., 2024 — The FineWeb Datasets (data quality + filtering, test-of-time-emerging)
  - HuggingFace `tokenizers` library docs (implementation reference)
- Open question raised: "Does deduplication at the 13-gram level still beat MinHash deduplication for downstream perplexity at the 124M scale?"

**Chapter 2 — Loss curve mechanics (Steps 4–5)**
- Why the baseline loss curve is the contract: every subsequent optimization is measured against it.
- Curated readings:
  - Kaplan et al., 2020 — Scaling Laws for Neural Language Models (seminal)
  - Loshchilov & Hutter, 2017 — Decoupled Weight Decay Regularization (AdamW, test-of-time)
  - Llama 3 tech report (Meta, 2024) — for the actual LR schedule used at modern scale (production reference)
- Open question raised: "At what model size does the cosine schedule decisively beat the linear schedule on downstream zero-shot tasks?"

**Chapter 3 — Throughput engineering (Steps 6–7)**
- Why parallelism is when training stops being a single-GPU experiment.
- Curated readings:
  - Rajbhandari et al., 2020 — ZeRO: Memory Optimizations Toward Training Trillion Parameter Models (seminal)
  - Chen et al., 2016 — Training Deep Nets with Sublinear Memory Cost (gradient checkpointing, test-of-time)
  - PyTorch FSDP docs (implementation reference)
- Open question raised: "For the 124M scale, is FSDP's communication overhead amortized in fewer than 10K steps? When does it pay off?"

**Chapter 4 — Evaluation (Step 8)**
- Why eval is not a final step but a design constraint: choosing eval first changes what you train on.
- Curated readings:
  - Zellers et al., 2019 — HellaSwag (eval seminal)
  - Hendrycks et al., 2021 — MMLU (eval test-of-time)
  - EleutherAI lm-eval-harness docs (implementation reference)
- Open question raised: "Does perplexity on WikiText-103 correlate with HellaSwag zero-shot above some inflection point — and what is it?"

### MVB per step (persona-tagged)

| Step | `mvb_persona` | Build target |
|---|---|---|
| 1 | ml-tinkerer | Train a 32K-vocab BPE on TinyStories; check coverage on held-out text |
| 2 | curious-learner | Run nanoGPT on Shakespeare in a Colab notebook; observe the loss curve |
| 3 | applied-engineer | Build a parquet pipeline that filters by language + perplexity threshold |
| 4 | ml-tinkerer | Train a 124M model on a single RTX 4090 overnight; capture the baseline loss curve |
| 5 | applied-researcher | A/B-test LR schedules (linear vs cosine vs WSD) on the 124M baseline; produce comparison table |
| 6 | applied-engineer | Migrate to 2× A100 DDP; measure tokens/sec and step latency |
| 7 | applied-engineer | Add gradient checkpointing + bf16; measure memory + throughput vs step 6 |
| 8 | applied-researcher | Run lm-eval-harness on WikiText-103 + HellaSwag; produce evaluation report |

Note the **persona walk:** ml-tinkerer for the early "get something running" steps, applied-engineer for the systems/data steps, applied-researcher for the experimental/evaluation steps. This is the right shape for a pre-training arc — it walks the reader from "I can train a tiny GPT" to "I can run an ablation and an eval suite."

### Where this arc leads

After finishing pre-training, a reader can credibly walk into:
- The **RLHF** arc (already in the repo) — they have a base model to apply DPO/GRPO to.
- The **systems-for-scale** arc — they have the throughput baseline to optimize against.
- A new **post-training** arc — supervised fine-tuning, instruction tuning, alignment.

The arc destination naturally feeds the next arc's entry point. That's the inter-arc compounding.

---

## Use this template

When the supervisor proposes a new arc, it must produce a draft of:
1. arc_id + track + destination
2. Prereqs (4–6 curriculum pages, with status: solid/adequate/stub/missing)
3. Compounding trajectory table (3-column: Step / Artifact / Used by)
4. Chapters (3–4 chapters of 2–3 steps each)
5. Per-chapter curated readings (1–3 papers, labeled seminal/test-of-time/SotA/production reference)
6. Per-step MVB persona tag
7. Per-chapter open question for the reader to ponder

If any of those seven elements is missing, the arc isn't ready to propose — the supervisor reshapes or vetoes via `critic-editor`.

The "Pre-training" arc above is the canonical example. The next 5 arcs the system proposes should match this anatomy.
