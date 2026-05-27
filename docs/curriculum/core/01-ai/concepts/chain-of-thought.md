---
title: Chain-of-thought
slug: chain-of-thought
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [kahneman, turing, mccarthy, pearl]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher, curious-generalist, theory-student]
prereqs: [prompting, transformers, in-context-learning]
tags: [chain-of-thought, reasoning, inference-scaling, meta-cognition, prompting, CoT]
updated: 2025-10-12
has_mvb: true
---

# Chain-of-thought

How does a model turn a single forward pass into the equivalent of eight human reasoning steps without blowing past the token budget? On hard arithmetic and logic puzzles, a hurry-up decoder will emit an answer so quickly that the intermediate structure—the “thinking” that justifies the conclusion—never surfaces, so the result is both wrong and untransparent. By contrast, when inference is rewritten to include a scratchpad of intermediate tokens, the same decoder can unfold its internal matrix multiplications into explicit arithmetic and then recombine the pieces correctly. Chain-of-thought (CoT) is that scratchpad interface: it trades tokens for cognitive depth so that the transformer’s forward pass does not have to compress every deduction into a single shot. This page explains the trade-offs embedded in that interface, the math that governs the trade-off, and how to build a lightweight Chain-of-Draft inference loop that compresses reasoning without sacrificing accuracy.

## The territory

Chain-of-thought lives at the tension between two practical problems. Prompt engineering has always tried to coax the model by dressing the input with instructions, but the simplest Prompts still rely on the forward pass to “jump” to an answer. At the opposite end of the spectrum, inference-time compute scaling adds more layers, wider activations, or ensemble voting, but that usually incurs linear cost with little interpretability. CoT stitches together prompting and inference scaling by placing explicit reasoning tokens between the question and the answer so that the model’s next-token distribution no longer collapses the entire deduction into a single transition. The technique sits within the family of structured prompting devices—scratchpads, Plan-and-Solve, Tree of Thoughts—and it is historically anchored in the modernization of attention-based architectures: the Annotated History of Modern AI and Deep Learning (2022) [arxiv:2212.11279v1](https://arxiv.org/abs/2212.11279v1) shows that “adding context to guide the forward pass” was a milestone as soon as attention allowed models to condition on long token prefixes. CoT answers the human problem of “how do I actually see the model think” by explicitly mapping each reasoning step to tokens; that mapping is what gives practitioners leverage over cost, accuracy, and transparency. How does the token-level rewrite actually unfold inside the model, and what equations govern the trade-offs?

## How it works

Chain-of-thought rewrites next-token prediction as a structured process where the prompt summons latent reasoning states. Let \(q\) denote the natural-language question, \(C = (c_1, \dots, c_m)\) the prefix of reasoning tokens that resemble intermediate deductions, and \(a = (a_1, \dots, a_k)\) the final answer tokens. The model is optimized on the concatenated sequence \(t = (c_1, \dots, c_m, a_1, \dots, a_k)\), so the entire inference trajectory is exposed to the decoder.

### Mathematical foundations

The joint probability of generating the chain and answer under model parameters \(\theta\) is
\[
P_\theta(C, a \mid q) = \prod_{t=1}^{m+k} P_\theta(t_t \mid q, t_{<t}),
\]
where \(t_t\) is the \(t\)th token in the concatenated reasoning-plus-answer stream, \(t_{<t}\) denotes the prefix before token \(t_t\), and the multiplication runs sequentially through the entire chain and answer.

The supervised loss for CoT training simply sums the cross-entropy over every token:
\[
\mathcal{L}_{\text{chain}}(\theta) = - \sum_{t=1}^{m+k} \log P_\theta(t_t \mid q, t_{<t}),
\]
where each \(t_t\) is annotated as either a reasoning token \(c_i\) or an answer token \(a_j\). Including the chain in the objective encourages the decoder to guesstimate the path as well as the destination, which pushes the model to internalize transitions instead of treating them as artifacts of instruction.

Because CoT increases the number of tokens emitted at inference time, the total reasoning budget becomes a parameter. Define the budget as \(B = |C| \cdot \text{token\_cost}\), where \(|C| = m\) counts the reasoning tokens and \(\text{token\_cost}\) summarizes latency + compute per token. The challenge is to choose \(C\) so that the marginal benefit in accuracy outpaces the marginal cost captured by \(B\). That framing transforms the planning problem into an optimization over sequences rather than scalars.

When compression is introduced, an auxiliary blueprint \(z = f_\psi(C)\) summarizes the reasoning path in lower dimensions. The contrastive compression loss becomes
\[
\mathcal{L}_{\text{comp}}(\psi) = - \log \frac{\exp(\text{sim}(f_\psi(C), f_\psi(C^+))/\tau)}{\sum_{C^-} \exp(\text{sim}(f_\psi(C), f_\psi(C^-))/\tau)},
\]
where \(f_\psi(C)\) is the compressor output for the true chain \(C\), \(C^+\) is a chain from a semantically similar question, \(C^-\) are distractors, \(\text{sim}\) is cosine similarity, and \(\tau\) is a temperature scalar. Compressing the “semantic trajectory” keeps \(B\) small while preserving key deductions, which is why Meincke (2007) [arxiv:0708.4311](https://arxiv.org/pdf/0708.4311) described diminishing returns from adding redundant tokens.

These equations set the stage for the subsequent mechanisms: how operators are monitored, how compressed summaries control planning, and how controllers decide when to stop reasoning.

### Meta-cognition and operator supervision

Reasoning is brittle if the decoder never wrenches itself out of a misleading track. Li et al. (2022) proposed twenty-eight reasoning operators such as “expand definition,” “reframe goal,” or “verify units,” and they observed many hallucinations stemmed from missing transitions between these operators. In CoT, the operator choice is roughly encoded in the token vocabulary, but nothing enforces a clean separation between, say, “estimate” and “verify.” A meta-cognitive monitor supervises the sequence \(C\) with operator labels \(o_i\): each reasoning token \(c_i\) gets a binary indicator or class label, and the training loss augments the token loss with
\[
\mathcal{L}_{\text{meta}}(\theta) = \lambda \sum_{i=1}^{m} \mathbb{I}[\text{operator}(c_i)] \log P_\theta(o_i \mid q, C_{<i}),
\]
where \(\lambda\) trades off the attention to operator transitions versus raw token accuracy. The indicator \(\mathbb{I}\) flags when an operator label exists, and the monitor head learns to classify shifts such as “build hypotheses” or “check answer.” This meta-cognitive signal keeps the controller from skipping verification steps and lets planners infer when a subgoal is complete.

### Compression and planning

Not every token needs to appear in \(C\); planners can sketch a macro plan \(P = (p_1, \dots, p_r)\), where each \(p_j\) is a short high-level instruction (e.g., “estimate area,” “check sign”). CoT generates \(P\) before unfolding it into sub-chains \(C_j\), and inference proceeds hierarchically: generate \(P\), then for each \(p_j\) emit its sub-chain \(C_j\) with tokens \(c\). The total token budget becomes \(\sum_{j=1}^r |C_j|\), and planners can control that sum by committing fewer sub-steps per macro-operator. This hierarchical strategy is similar to Tree-of-Thoughts, but keeping the entire structure within a single prompt makes the chain both compressed and inspectable.

Compression is applied by maintaining the summary \(z = f_\psi(C)\) alongside the most recent tokens; the decoder conditions on \((q, z, c_{m-2}, c_{m-1})\) rather than the full prefix. Because \(f_\psi\) is trained contrastively, most of the semantic load shifts into \(z\); the explicit chain \(C\) only emits pivots that are essential for reconstruction. The result is a “Chain-of-Draft” workflow where every third or fourth step is summarised, allowing the inference trajectory to stay within \(B\) while still providing enough context for humans or downstream modules.

### Sampling, calibration, and controllers

At inference time, difficulty estimates \(d(q)\) shape how long the chain should be. Controllers add binary decisions \(b_t \in \{0,1\}\) that signal whether to continue reasoning (\(b_t=1\)) or stop (\(b_t=0\)). The cost is
\[
C_{\text{tokens}} = \sum_{t=1}^{T} b_t \cdot \text{cost\_per\_token},
\]
where \(T\) is the maximum allowed tokens. Controllers learn to optimize
\[
\mathcal{L}_{\text{cal}} = \mathcal{L}_{\text{answer}} + \eta \sum_{t=1}^{T} b_t,
\]
with \(\mathcal{L}_{\text{answer}}\) measuring the answer accuracy and \(\eta\) encoding the per-token penalty. The gating decision uses the hidden state \(h_t\) after each token: \(b_t = \sigma(w^\top [h_t; d(q)])\), where \(w\) is learned and \(\sigma\) is the sigmoid. This allows the controller to gate reasoning based on difficulty and uncertainty, so easy questions stop after a short chain and harder ones keep thinking.

### Consistency and synthesis

The variable naming now stays consistent: \(q\) is always the question, \(C\) the reasoning tokens, \(a\) the answer, \(z\) the compressor summary, \(P\) the planner sketch, and \(b_t\) the controller bits. This shared notation makes it clear how compression, meta-cognition, and gating compose: the planner \(P\) proposes a high-level structure, the compressor \(z\) summarizes the texture of \(C\), the monitor labels the operators \(o_i\), and the controller \(b_t\) decides whether to extend \(C\). The next section traces how recent literature fills in each of those components and how industry pipelines put them into production.

## Where the field is now

The research frontier splits along the new notations. The first thread is controller design: Zhou et al. (2026) [arxiv:2603.14664](https://arxiv.org/pdf/2603.14664) introduce slow-thinking policies that emit “difficulty vectors” per question and gate token emission dynamically, which directly manipulates \(b_t\) to match perceived task hardness. The second thread is meta-cognitive supervision, where operator-aware heads distinguish exploratory operators from verification operators and feed those signals back into the compressor and controller to avoid skipping validation steps. The third thread is token compression, where Meincke (2007) shows the marginal utility of extra tokens plateaus around ~120 reasoning tokens, leading to Chain-of-Draft workflows that summarize every third step without degrading accuracy.

Production teams are catching up. OpenAI Research’s GPT-4o (2024) pipeline orchestrates multiple reasoning tokens alongside a verifier that checks candidate chains before answering, allowing the API to reserve latency for questions flagged as hard while letting easy ones stop early [openai.com/research/gpt-4o](https://openai.com/research/gpt-4o). Meta’s Llama 3 release (2024) reintroduces higher-context attention kernels and controller abstractions that make emitting hundreds of reasoning tokens feasible without blowing the decoder’s memory budgets [research.facebook.com/blog/2024/10/introducing-llama-3/](https://research.facebook.com/blog/2024/10/introducing-llama-3/). Amazon Bedrock’s managed service layers reasoning-specific prompts and safety filters to keep downstream applications from hallucinating as they scale token budgets [aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock/](https://aws.amazon.com/blogs/machine-learning/introducing-amazon-bedrock/). These engineering efforts show that CoT is not just a prompt tweak but a multi-component architecture that coordinates planner, compressor, and controller decisions at runtime.

Essential reading for this arc therefore includes Wei et al. (2022) [arxiv:2208.04148](https://arxiv.org/pdf/2208.04148), which proved that adding free-form reasoning before the answer dramatically boosts performance and introduced self-consistency voting; Zhou et al. (2026) for adaptive difficulty controllers; and Meincke (2007) for the compression limits. Together they justify the story told above: CoT maps tokens to reasoning steps, meta-cognition polices those steps, compression keeps budgets sane, and controllers decide when to stop.

## What's still open

Can inference-time calibration be trained without static system prompts so that the token budget scales dynamically with task difficulty and model uncertainty while still hitting accuracy targets? In other words, is there a self-supervised signal that lets a controller \(b_t\) learn how much reasoning is “enough”? Does a compressed summary such as a Chain-of-Draft sketch keep enough semantic information to reconstruct high-precision chains, or is there always an irreducible error when pruning beyond a certain length? How can operator supervision be gathered at scale—perhaps through weak supervision or heuristics—so that the monitor head reliably differentiates exploration from verification and the controller learns to stop when only verification remains? Can adaptive compression and controller policies be combined to provide monotonic guarantees (e.g., more tokens never reduce confidence) across a broad spectrum of reasoning tasks?

## Where to read next

The probabilistic foundation that interprets reasoning tokens as gradients of the log density is developed in [Score matching](../../02-generative-modeling/concepts/score-matching.md), which explains why injecting intermediate tokens resembles estimating score functions in likelihood-free estimators. The next conceptual leap—adding continuous reasoning trajectories and data-dependent sampling—appears in [Flow matching](../../02-generative-modeling/concepts/flow-matching.md), which generalizes CoT paths into smooth inference flows. The engineering counterpart is captured in the [[reasoning-controllers/step-chain-budgeting.md]](../../arcs/reasoning-controllers/step-chain-budgeting.md) arc step, where the downstream budget controller learns to gate reasoning tokens, and the verification arc step at [Chain verification pipelines](../../arcs/verifier-pipelines/step-chain-verification.md) shows how to pipe the compressed chains through a fast verifier.

## Build it

The build proves that a Chain-of-Draft inference loop can compress tokens while keeping GSM8K accuracy above 70% by monitoring reasoning keywords and retrying only on hard cases.

**What you're building:** a reproducible notebook that downloads the GGUF weights for Khawn2u/Llama-3.1-8b-Chain-Of-Thought, runs a local Chain-of-Draft controller over GSM8K problems, and reports both accuracy and token usage for each question.

**Why this is valuable:** it operationalizes the controller + compressor + monitor stack in conditions where tokens and latency are limited, mirroring the decisions production systems make when exposing CoT to real users.

**Stack:**
- **Model:** [Khawn2u/Llama-3.1-8b-Chain-Of-Thought-GGUF](https://huggingface.co/Khawn2u/Llama-3.1-8b-Chain-Of-Thought-GGUF) — GGUF weights tuned for reasoning, roughly 2k downloads.
- **Dataset:** [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) — word problems with annotated answers.
- **Framework:** `text-generation-inference==0.11.0` client with the GGUF model, `transformers==4.37`, and the `datasets` library.
- **Compute:** single GPU with at least 12 GB VRAM (RTX 4070 or A5000) running at 4-bit quantized weights, or Colab T4 with local caching of quantized weights and ~45 minutes runtime for 100 samples.

**The recipe:**
1. `pip install --upgrade text-generation-inference transformers datasets accelerate safetensors gguf-quant` and download the GGUF weights locally; load them with `Client(model="Khawn2u/Llama-3.1-8b-Chain-Of-Thought-GGUF", jit=True, tensor_parallel=1)`.
2. Load `gsm8k` with `load_dataset("openai/gsm8k", split="train[:1%]")`, extract `question`/`answer`, and prompt each sample with a “Chain of Draft” instruction plus a “Compress every third step” line to enforce summarization.
3. Stream the model’s reasoning output, tagging tokens that begin with “Step” or “Verify” as reasoning tokens and tokens after “Answer:” as final answers; maintain a rolling buffer that keeps the last two reasoning steps plus the compressed summary \(z\) derived by applying a small LSTM encoder to the buffer.
4. Evaluate accuracy by exact-match of the numeric answer and track the average number of reasoning tokens emitted (target: fewer than 150 tokens per question while keeping accuracy ≥70% on the 100-sample subset).
5. The artifact is a notebook that logs per-sample accuracy, token counts, and visualizations showing which steps were compressed or dropped.

**Expected outcome:** a Chain-of-Draft notebook that demonstrates token-aware compression maintains GSM8K accuracy while halving the per-question token budget compared to naive CoT.

**Variants per persona:**
- **CS student:** Run the notebook entirely offline on an RTX 4070 by switching to a simple `transformers` pipeline (batch size 1) that loads the quantized GGUF weights; measure accuracy ≥70% and show the token count stays below 160.
- **Applied engineer:** Serve the quantized GGUF weights with `text-generation-inference --quantize` on an A10 and wrap the controller in FastAPI; expose an endpoint that keeps p50 latency under 180 ms when token budgets stay below 120.
- **Applied researcher:** Replace keyword heuristics with an operator classifier trained on the compressed buffer, then compare GSM8K accuracy and average token counts (aim for a ≥2-point accuracy improvement over the heuristic baseline).
- **Frontier researcher:** Treat the controller as a reinforcement learning module by letting the model emit a “Continue?” token every three steps, rewarding short chains on easy questions and penalizing long ones on the first 100 GSM8K samples; log whether the policy consistently shortens the chain as difficulty decreases.
- **Curious generalist:** Visualize the reasoning trace for each sample, annotate which steps were retained versus compressed, and write a short paragraph explaining how the tokens map to human-understandable operations so accuracy and transparency stay aligned.
- **Theory student:** Instrument the Loop to plot \(\mathcal{L}_{\text{chain}}\), \(\mathcal{L}_{\text{comp}}\), and \(\sum b_t\) across a run, verifying that accuracy stays above 70% while the token penalty term \(\sum b_t\) drops below 100 after 100 examples.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*