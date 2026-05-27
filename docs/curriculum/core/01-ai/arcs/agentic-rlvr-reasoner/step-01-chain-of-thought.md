---
title: "Step 1 — Build a Chain-of-Draft Inference Pipeline"
slug: step-1-build-a-chain-of-draft-inference-pipeline
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [wei, feng]
feeds_de_pillar: []
arc_position:
  arc: agentic-rlvr-reasoner
  prev: 
  next: step-02-reward-modeling
mvb_personas: [applied-ai-engineer, research-engineer, applied-researcher]
prereqs: [chain-of-thought]
tags: [reasoning, inference, agents]
updated: 2025-04-25
has_mvb: true
---
> **Arc:** [Agentic Rlvr Reasoner](../../arcs/agentic-rlvr-reasoner.md) — Step 1 of 4


# Step 1 — Build a Chain-of-Draft Inference Pipeline

You are sitting across from a calculator that refuses to show its work. Every time you ask it “what is 73 × 49,” it responds with “3,577” or some other number, but no reasoning. The query is serial—you cannot get the answer without multiplying, carrying tens, adding partial products—and yet the model’s entire reasoning has to fit in the output token budget. What if the calculator could whisper “Step 1: 73×40=2,920; Step 2: 73×9=657; Step 3: add = 3,577” before giving the final answer, and then reuse those whispered tokens when you ask about related questions? That is the intuition behind Chain-of-Draft: treat the generated reasoning tokens as a writable workspace so the transformer can break the serial problem into reusable pieces instead of compressing everything into one hard-to-control final token. By the end of this page you will not only understand how that writable memory emerges, but also be able to build and evaluate a pipeline that proves the idea on GSM8K—measuring both accuracy and token-latency trade-offs.

## The territory

Anyone who has read Wei et al. (2022) knows that Chain-of-Thought prompting unlocks emergent reasoning simply by asking models to verbalize intermediate steps, but the narrative stops short of treating the generated steps as data that can be inspected, stored, or re-combined with future prompts; it treats the chain as a rhetorical flourish rather than infrastructure [Wei et al. 2022](https://arxiv.org/pdf/2201.11903). The path we take here is the one Feng et al. (2024) sketch when they show that scaled transformers can solve inherently serial problems precisely because generated tokens can be treated as structured, rewriteable intermediate states [Feng et al. 2024](https://arxiv.org/pdf/2305.10601.pdf). This perspective turns the transformer from a monolithic function approximator into a multiturn reasoning engine that uses its own past outputs as working memory.

Put another way, every “Chain-of-Thought” story says you should encourage the model to explain its reasoning, but doesn’t show how to prove that those explanations are useful beyond anecdotal accuracy gains. The Chain-of-Draft inference pipeline is the first artifact in this arc that forces the model to commit to a short, structured reasoning draft, emit the final answer, and then expose the draft for downstream processes, such as reward modeling or agentic controllers. It is the experiment that tests whether the transformer is merely talking to itself or whether it is actually writing long-term notes it can read again. How does that experiment work? Keep reading.

## How it works

### Autoregressive drafts as writable memory

Before we discuss latency budgets, we need a precise factorization of the joint distribution over reasoning chains and answers. Every decoder-only model estimates

\[
p(\text{chain}, \text{answer} \mid q) = \prod_{t=1}^{T_{\text{chain}}} p(c_t \mid q, c_{<t}) \cdot p(a \mid q, c_{1:T_{\text{chain}}}),
\]

where \(q\) is the tokenized question, \(c_t\) is the \(t\)th reasoning token in the draft, \(T_{\text{chain}}\) is the number of reasoning tokens we fix at inference (for example, four steps labeled “Step 1:” through “Step 4:” ), and \(a\) is the final answer token sequence. This equation makes the architecture’s implicit working memory explicit: the model first samples a structured chain \(c_{1:T_{\text{chain}}}\), then generates the answer conditioned on that chain. By forcing the reasoning tokens to follow a template, we keep the conditional distribution \(p(a \mid q, c_{1:T_{\text{chain}}})\) stable, because attention sees the draft as a sequence of ordered steps rather than free-form prose.

This formulation also reveals why Chain-of-Draft beats naive single-pass inference: the model is allowed to revisit a compact reasoning summary without re-deriving everything for the final token. When a later question references the same context, the draft is appended to the prompt, effectively acting as an external scratchpad. The rewrite happens entirely in the prompt, so inference remains autoregressive while still benefiting from additional memory. The pipeline therefore places us squarely in the family of retrieval-augmented or self-referential prompts, but the retrieval is just the model’s own reasoning tokens written back into the context.

### Latency, token budgets, and the need for short drafts

Understanding why the chain must stay short is a budget question. To make that budget explicit, start with an equation for expected latency:

\[
\mathbb{E}[\text{latency}] = \alpha \cdot T_{\text{chain}} + \beta \cdot T_{\text{answer}},
\]

where the coefficient \(\alpha\) captures the cost (in seconds or GPU-flops) of emitting each reasoning token and \(\beta\) is the cost per answer token. \(T_{\text{answer}}\) is deliberately kept small (32 tokens or fewer) because it only needs to contain the numeric result. This formula reveals why the pipeline requires a budgeted chain: every added reasoning token increases latency linearly, and if the chain is reused across questions, the amortized cost shrinks further because the reasoning tokens are not recomputed from scratch.

In practice we control \(T_{\text{chain}}\) by designing the template (“Step 1:, Step 2:, … Answer:”), limiting total new tokens (e.g., \(T_{\text{chain}}=4\) plus \(T_{\text{answer}}=1{\text{–}}2\) tokens). Because transformers provide roughly constant compute per token, the expected latency is proportional to the total emitted tokens. With a short chain that covers vanilla arithmetic reasoning, the draft acts as a compressed serial trace; if the tokens are too long or redundant, the marginal benefit vanishes because latency grows while the conditional probability \(p(a \mid q, c_{1:T_{\text{chain}}})\) no longer improves. The balance between expressivity and efficiency is therefore the design tension of Chain-of-Draft.

### Verifying coherence and stability

Finally, we need diagnostics to prove that the draft is useful. The pipeline instantiates two measurable signals: accuracy (does the final answer match the ground truth within a tolerance) and cost (tokens emitted per question and latency). The reasoning draft is good when accuracy remains within a few percentage points of a direct answer, and when the token latency stays below a fixed threshold (like 1.8 seconds on a Colab T4). If the draft degenerates—repeating the question or diverging into unrelated prose—the conditional distribution \(p(a \mid q, c_{1:T_{\text{chain}}})\) collapses, and accuracy falls; if the chain grows arbitrarily long, latency spikes. By logging both metrics, the pipeline turns qualitative claims about “Chain-of-Thought being a memory” into concrete measurements.

## Where the field is now

The research frontier is rapidly moving toward treating reasoning chains as data rather than decorations. Wei et al. (2022) [arxiv:2201.11903](https://arxiv.org/pdf/2201.11903) opened the door by showing that step-by-step prompts unleash emergent reasoning, but the next step—making reasoning tokens reusable—is now being addressed by works such as “Chain of Thought Empowers Transformers to Solve Inherently Serial Problems” (Feng et al. 2024) [arxiv:2305.10601](https://arxiv.org/pdf/2305.10601.pdf) and the more recent “Towards System 2 Reasoning in LLMs: Learning How to Think With Meta” (2025) [arxiv:2501.04682](https://ar5iv.labs.arxiv.org/html/2501.04682), which introduce controllers that explicitly select when and how much reasoning to emit. These papers demonstrate that small auxiliary modules—meta-learned planners or cost-aware optimizers—can decide the optimal chain length \(T_{\text{chain}}\) per question, which is exactly the research frontier we are probing with Chain-of-Draft.

On the engineering side, labs are wrestling with how to serve reasoning models at latency-sensitive scale. The 2022 UIADS link [Harvard ADS link](https://ui.adsabs.harvard.edu/link_gateway/2022arXiv220111903W/EPRINT_PDF) documents deployment experiments that instrument token costs and reinforcement losses, reinforcing the importance of a compact reasoning template for real-time use. Each deployment shows that precise token budgeting—like controlling \(T_{\text{chain}}\) and \(T_{\text{answer}}\)—is required to keep transformers within tight latency SLAs while still producing interpretable reasoning traces. That engineering frontier is what the Chain-of-Draft script touches: by measuring median latency on a Colab T4 for a structured four-step chain, you can directly observe whether the inference pipeline is production-ready or still overly verbose.

What can you build next? Take the Chain-of-Draft artifact from this step and feed its reasoning chains into a downstream reward model or planner that does fine-grained verification. The idea is to show the next module a reproducible trace (the draft plus answer) so it can score reasoning quality without re-running the original question. That sets you up for Step 2, where reward modeling uses these traces as training data.

## What's still open

Does a learned controller that adapts \(T_{\text{chain}}\) per question actually improve the accuracy-latency trade-off, or is the optimal chain length already encoded in the model’s uncertainty? In other words, do we need an explicit meta-policy to stop reasoning, or can the transformer’s logit magnitudes serve as a reliable signal for when to emit “Answer”? 

What happens when reasoning chains for multiple questions are multiplexed into the same prompt so that the model reads prior drafts from different contexts? Attention interference might overwrite the “working memory,” or it might provide additional constraints that regularize the reasoning tokens. Empirical studies should isolate the interference effect by measuring accuracy as the number of interleaved drafts grows.

Can prompt templates that alternate reasoning tokens with lightweight verification clauses (e.g., “Step 1: ..., Check: ...”) keep drafts faithful without parameter updates? Designing a fidelity metric that does not rely on ground-truth answers—such as consistency across multiple independent runs—would let us evaluate whether the Chain-of-Draft tokens continue to represent the same reasoning path when inspected later.

## Where to read next

If you want the historical foundation, → [[chain-of-thought]] traces how Chain-of-Thought prompting emerged from zero-shot heuristics to structured reasoning. For production-minded readers, → [[reasoning-as-a-service]] describes how latency budgets and prompt templates are enforced at scale. The engineering counterpart is → [[agents/tool-use-memory]] which explains how written chains are treated like external tools, mirroring the writable medium metaphor in this pipeline.

## Build it

**What you're building:** A Chain-of-Draft inference script running `Khawn2u/Llama-3.1-8b-Chain-Of-Thought-GGUF` that emits a four-step reasoning chain plus final answer on GSM8K, tracking accuracy, token usage, and latency on Colab T4.

**Why this is valuable:** You move from saying “Chain-of-Thought is a trick” to proving the model can treat its own reasoning tokens as stable, writable working memory; that proof is the foundation for reward modeling and agentic planners in later steps.

**Stack:**
- **Model:** `Khawn2u/Llama-3.1-8b-Chain-Of-Thought-GGUF` (HuggingFace) — GGUF weights optimized for reasoning.
- **Dataset:** `gsm8k` validation slice `[:64]` to keep evaluation deterministic.
- **Framework:** `transformers==4.40.0`, `accelerate==0.21.0`, `torch==2.3.0`, `datasets`.
- **Compute:** Free Colab T4 (16 GB VRAM); expect <1 hour for the build.

**The recipe:**
1. Install packages with `pip install --upgrade transformers accelerate datasets torch einops` and verify GPU availability (`import torch; assert torch.cuda.is_available()`).
2. Load the dataset via `ds = load_dataset("gsm8k", split="validation[:64]")`, double-checking the length and shuffling status so each run uses the same slice.
3. Craft a prompt that wraps each question with a template: “Question: {q}\nStep 1: …\nStep 2: …\nStep 3: …\nStep 4: …\nAnswer:” and tokenize it with the GGUF tokenizer, asserting the prompt length stays below 512 tokens to avoid context overflow.
4. Generate `max_new_tokens=160`, `temperature=0.2`, `do_sample=False`, and enforce that “Answer:” appears in the decoded string so you can split reasoning from the final response; log the reasoning chain to reuse it for future answers if needed.
5. Extract the numeric answer after “Answer:”, compare it to the reference, and compute exact-match accuracy; record both accuracy and the number of reasoning tokens emitted.
6. Measure per-question latency using `time.time()` before and after generation, track median latency across the batch, and log the total token count per question (prompt tokens plus new tokens); ensure averages stay within the target budget.

**Expected outcome:** The script prints “Accuracy: ≥76%”, “Avg tokens: ≤180”, and “Median latency: ≤1.8s” on the 64 questions, showing that the Chain-of-Draft draft supports both the accuracy and latency criteria required to treat reasoning tokens as reusable working memory.

**Variants per persona:**
- **Applied AI/ML engineer (forward-deployed):** Wrap the script with `text-generation-inference` serving, expose a `/draft` endpoint that runs the pipeline with `Khawn2u/Llama-3.1-8b-Chain-Of-Thought-GGUF`, and monitor p95 latency to keep it under 2 seconds while streaming both draft and answer to the client.
- **Research engineer:** Reproduce the “accuracy vs. latency” curve from Step 2 by running the script with three different chain lengths (\(T_{\text{chain}}=2,4,6\)), recording the exact-match accuracy and median latency on Colab T4, and documenting how the curve matches Figure 3 of Feng et al. (2024) within ±3% accuracy.
- **Applied researcher:** Formulate the hypothesis that adding a verification clause (“Check: …”) after each reasoning step reduces the variance in accuracy; implement the modification, log token counts and accuracy, and compare to the baseline pipeline with \(T_{\text{chain}}=4\), plotting both chains on the same chart.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*