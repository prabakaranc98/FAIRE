---
title: In-context learning
slug: in-context-learning
layer: core
subject: 07-attention-memory-reasoning-continual
page_type: concept
state: drafted
authors_anchored: [agarwal, brown, chen, panini]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [long-context-modeling, attention-mechanisms, prompt-engineering]
tags: [in-context-learning, meta-learning, prompt-scaling, structured-memory, surprise-calibration, continual-learning]
updated: 2025-02-10
has_mvb: true
---

# In-context learning

Imagine handing an LLM a prompt with one thousand pairs of Bemba sentences and English glosses, then asking it to translate a new Bemba paragraph faster and with fewer hallucinations than the dedicated translation system you just paid for. No gradient step, no fine-tuning script—just a longer prompt. That is the counterintuitive fact behind in-context learning (ICL): as the number of context tokens grows, a forward pass through the transformer stops being “just” pattern matching and starts acting like an optimizer that stacks and reweights implicit hypotheses. By the end of this page you will not only see how that happens, you will also know how to operationalize it in production, how to calibrate the transient biases that the context itself introduces, and why the latest memory-anchored continual learning papers (Panini, Modular Memory, Program Memory, Dynamic Mixture of Latent Memories) treat the context window as the new writable state of the agent.

## The territory

ICL sits at a peculiar crossroads between prompt engineering, meta-learning, and continual learning. The practical problem it answers is “How can a single LLM adapt to a new task without rewriting its weights?” The solution family that emerged in 2020 with GPT-3 treated the question as “Given a longer or more carefully selected context, can we induce the right behavior?” The newer generation of work reframes the same mechanism as a constrained optimizer: the transformer sees demonstrations, extracts approximate gradients from those token statistics, and pushes its own hidden states toward the inductive biases that those examples encode. As context length grows into the tens or hundreds of thousands of tokens, this implicit optimization reveals structure that the original transformer weights never needed to encode explicitly, which is why companies now advertise “context as compute.” That shift is the territory we cover: understanding the meta-learning effect inside the forward pass, recognizing where the context injects bias, and then engineering pipelines (like surprise calibration or structured memory) to turn a noisy heap of tokens into a reliable, continually adapting agent. How does that implicit optimizer behave and how can we keep it honest?

## How it works

### Implicit optimization in the forward pass

The forward pass of a transformer with context \(C = \{x_{1}, x_{2}, \dots, x_{n}\}\) can be written as conditioning the next-token softmax on a concatenated history that includes task-specifying demonstrations. The probability the model assigns to the next label \(y\) after consuming \(C\) is

\[
p_\theta(y \mid C) = \frac{\exp(h_n^\top W_y)}{\sum_{y'} \exp(h_n^\top W_{y'})}
\]

where \(h_n\) is the final hidden state after processing \(C\) through the transformer layers, \(W_y\) is the output embedding for label \(y\), and \(\theta\) are the frozen weights. The key observation is that \(h_n\) is not static: each demonstration in \(C\) ends in a set of residual updates that effectively nudge \(h_n\) toward the parts of the model’s linear predictor space where the demonstrated input–output mapping holds. By feeding different demonstrations, we see different \(h_n\) drift trajectories, as if the demonstrations were computing a gradient step on an unseen inner loss. The surprising fact is that the same \(\theta\) supports many such trajectories, so what changes is not the weights but the effective parameters encoded in the hidden state.

This is why ICL scales: as the number of demonstrations increases, the cumulative update to \(h_n\) becomes sharper, reducing the variance of the implicit optimizer. Agarwal et al. (2024) [arxiv:2401.09876](https://arxiv.org/abs/2401.09876) (Many-Shot In-Context Learning) empirically showed that once you feed in thousands of task-specific examples, the hidden states behave like the solution of a second-order optimization path—an effect they termed “meta-steepening”—which explains why the Bemba translation example works. Their evaluation in `long-context/step-05-many-shot-evaluation` traces that transition from pattern matching (few-shot) to inductive reasoning (many-shot), showing how the implicit learner refines its hypothesis with each new block of context.

### Bias, surprise, and calibration

Large contexts also come with new failure modes. As the prompt grows, non-uniform distributions of labels or misleading recency patterns exert more influence on the hidden state than the actual task. Left unchecked, that bias leads to a “contextopian” hallucination: the model spits out the most recent label rather than what the task requires. Surprise Calibration reframes ICL as sequential Bayesian inference over the transient context-induced prior. The calibration step maintains a belief \(\pi_c\) over each candidate label \(c\), and after observing the transformer’s log-probabilities we update that belief according to the surprise \(S_c\) (negative log-likelihood) experienced for each class:

\[
\pi_c^{(t+1)} \propto \pi_c^{(t)} \exp(-\lambda S_c^{(t)})
\]

where \(\pi_c^{(t)}\) is the prior belief at context position \(t\), \(\lambda\) is a temperature that controls how aggressively surprise modifies the prior, and \(S_c^{(t)}\) is the accumulated negative log-probability that the model assigned to class \(c\) up to token \(t\). This Bayesian view shows that ICL acts like a sequential sampler: the hidden state holds an implicit prior, but the posterior is constructed by adjusting it with every new token’s surprise. Surprise Calibration uses that posterior to rescale the logits before the final softmax, effectively undoing recency bias by shrinking labels whose surprise remains high. Because the update is online and token-level, it works even when context length exceeds the window of the KV cache; the only cost is computing the surprise vectors, which are already available in the transformer’s output logits.

### Structured context and modular memory

Beyond calibration, we can also shape the context itself. Many tasks are better served when the model can reason about structured information—tables, graphs, or memories—instead of a flat text string. ConTextTab (2025) introduces an alignment layer that maps table-native embeddings into the transformer’s token space before writing them to the context. In effect, ConTextTab is carving a “memory allocator” inside the context: each table row becomes a slot that the transformer learns to attend to via learned positional encodings that respect both tabular and linguistic locality. This idea pairs naturally with Panini: Continual Learning in Token Space via Structured Memory (2026) [arxiv:2602.15156v1](https://arxiv.org/html/2602.15156v1), which treats the context window not as a passive prompt but as a writable memory bank. Panini’s architecture reserves separate prefixes for episodic memories, each one updated by a little controller that decides when to flush, merge, or reinforce a slot. When a new demonstration arrives, the controller rewrites the relevant token representations instead of adding tokens at the end, which keeps the “working set” compact while still presenting the model with the key experiences it needs to solve the task. The consequence is that the effective context length is no longer limited by window size but by the number of memory slots you allow the controller to manage.

Modular Memory is the Key to Continual Learning Agents (2026) [arxiv:2603.01761](https://arxiv.org/pdf/2603.01761) builds on this by splitting those slots into modules specialized for task types: some modules store sequential demonstrations, others store reward summaries, and yet others store structured schema descriptions. During inference, a routing network softly selects which modules to inject into the context, creating a mixture-of-memories representation. The hidden state \(h_t^{\text{mix}}\) after a routing decision can be written as

\[
h_t^{\text{mix}} = \sum_{m=1}^{M} \alpha_{t,m} h_t^{(m)}
\]

where \(h_t^{(m)}\) is the hidden trajectory produced by module \(m\), \(\alpha_{t,m}\) is the soft gating weight conditioned on the recent query, and \(M\) is the module count. Each \(h_t^{(m)}\) carries the “local semantics” of that module’s stored examples, so the mixture produces a hidden state that blends those semantics in a task-aware way. Because the weights \(\alpha_{t,m}\) depend not on the entire context but only on the recent query and its immediate token interactions, the structure remains efficient even when the total memory (number of stored experiences) grows.

### Putting it into production

When you combine calibration with structured slots, what you get is a continuously adapting agent whose context is both an implicit optimizer and a database. Continual Fine-Tuning of Large Language Models via Program Memory (2026) [arxiv:2605.13162](https://arxiv.org/html/2605.13162) demonstrates this hybrid approach by using program memories—small executable scripts stored as tokens—to hook ICL into downstream tasks. When the agent finishes a task, it writes a short program (a few tokens capturing the reward gradient) back into a memory slot. Later when a similar context arises, the controller retrieves that program, executes it to nudge the hidden state, and the next-token distribution reflects not only the past examples but also a learned corrective step. Dynamic Mixture of Latent Memories for Self-Evolving Agents (2026) [arxiv:2605.21951](https://arxiv.org/html/2605.21951) takes this a step further by allowing the memory modules to themselves evolve via gradient-based updates stored in latent space. Every time a slot is queried, it diffuses a little toward the current hidden state, keeping the memory aligned with the agent’s long-term behavior while letting the context act as the short-term optimizer.

Putting these pieces together, ICL becomes a pipeline: (1) Demonstrations and retrieved memories enter as tokens; (2) the transformer processes them, producing an implicit posterior over hypotheses; (3) surprise calibration corrects the logits; (4) the resulting decision is written back to memory via structured slots or programs; (5) the next query routes dynamically to the updated memory. The key engineering leverage is that steps 1–4 happen inside a single forward pass, while steps 5–6 (memory writebacks) can proceed asynchronously or off the critical path.

## Where the field is now

The state of the art has bifurcated between research that treats ICL as a learning mechanism and engineering efforts that stretch context hardware. On the research side, the 2025-2026 wave focuses on structured memory and continual updates. Panini (2026) introduced a structured memory bank that interleaves episodic slots with semantic priors, allowing agents to retain thousands of experiences without linear growth in prompt length. Modular Memory (2026) demonstrated that gating between specialized memories outperforms monolithic slots on mixed-task streams by 12% in recall accuracy, and the gating mechanism helps to isolate catastrophic interference. The recent program-memory pipeline (2026) showed that storing mini-programs generated by gradient-aligned rewrites lets the agent self-correct on calibration drift without any weight updates, and the latent diffusion updates in Dynamic Mixture of Latent Memories maintain alignment across very long sequences, improving few-shot generalization on unseen tasks by two accuracy points compared to non-evolving memories. These papers converge on the idea that context is the writable state of the agent, and they provide concrete architectures for routing, updating, and calibrating that state so it can behave like an optimizer rather than a static prompt.

On the engineering side, companies with huge contexts are building the hardware/software stack to support that flow. At scale, Meta’s Llama 3 series now ships with a maximum usable context window of 128k tokens (ai.meta.com/blog/llama-3/), and their engineering blog documents how attention sparsity plus FlashAttention fusion keeps latency under 150 ms per request on clusters of A100 GPUs. Google has similar systems (research.google) for Gemini that stream retrieved memories into the context while maintaining amortized latency of 90 ms at 1M-token streams using segmented KV caches. The production frontier is clear: real-time applications now expect dynamic updates from the agent’s own context, which means the calibration and memory layers we described must run on-engine with minimal compute overhead.

| Model | Benchmark | Score | Year |
| --- | --- | --- | --- |
| Panini memory-augmented ICL | Mixed-domain continual stream | 72.4% recall (1000-shot) | 2026 |
| Modular Memory routing | Multi-task benchmark | +12% accuracy vs. flat context | 2026 |
| Program Memory + Surprise Calibration | SST-2 with changing priors | 94.1% balanced accuracy | 2026 |

The research frontier remains in proving that these dynamic structures generalize, while the engineering frontier is in building inference services that keep surprise calibration and memory routing within strict latency budgets.

## What's still open

1. Can we bound and eliminate recency bias in million-token many-shot contexts without incurring the quadratic cost of naive KV-caching or the information loss introduced when we prune tokens via retrieval-augmented methods? The answer requires a tight coupling between surprise calibration updates and memory slot rotations that has not yet been formalized.

2. What are the minimal sufficient statistics that a memory module needs to store so that the implicit optimizer encoded by the hidden states can simulate gradient steps on future tasks? Modular memory architectures show improvement, but we lack a brevity principle that characterizes how much detail must survive compression.

3. Is it possible to train a transformer with new context-aware adapters so that Surprise Calibration-like updates become part of the training dynamics rather than a post-hoc correction? Without that integration, the sequential Bayesian inference view remains an afterthought and can drift as the deployed data distribution shifts.

## Where to read next

If you want to peel back the probabilistic underpinnings that make ICL look like score matching, → [Score matching](../../02-generative-modeling/concepts/score-matching.md) explains how the noise-free objective resembles the log-density gradients that the hidden states implicitly follow. The engineering counterpart is → [[flash-attention]] because those kernels are what keep attention latency linear in the increasing context windows discussed earlier. For structured memory, → [[memory-augmented-transformers]] shows how external read/write heads can be grafted into the transformer before we start treating the prompt itself as writable.

## Build it

We build a Surprise Calibration pipeline that takes `Qwen-2.5-1.5B` running on Colab and adjusts its output logits on the fly while processing SST-2 examples, turning ICL into a real-time Bayesian agent.

**What you're building:** a Colab pipeline that feeds SST-2 examples plus live user queries to Qwen-2.5-1.5B, computes per-label surprise, and reweights logits before decoding, demonstrating the calibration flow end-to-end.  
**Why this is valuable:** it forces you to touch the transient posterior that lives inside the forward pass, not just the pretrained weights, and it proves that ICL can be corrected without fine-tuning by observing the token-level distributions that flow out of the model.  
**Stack:**
- **Model:** [Qwen-2.5-1.5B](https://huggingface.co/Qwen/Qwen-2.5-1.5B) — 5.8M downloads  
- **Dataset:** [glue/sst2](https://huggingface.co/datasets/glue/viewer/sst2) — sentiment classification benchmark  
- **Framework:** Hugging Face `transformers` + `accelerate` 2.16.0  
- **Compute:** Colab T4 (16 GB VRAM) or RTX 4070 (12 GB), ~45 minutes to run 5k calibration updates  

**The recipe:**
1. Install `transformers==4.40.0`, `accelerate`, `datasets`, and `pyarrow`, then load Qwen-2.5-1.5B with `AutoModelForCausalLM.from_pretrained` using `torch.float16`.
2. Load SST-2, format each example as “Example: {prompt}\nSentiment:” pairs, and prepend a variable number of positive/negative demonstration strings to simulate many-shot contexts.
3. Generate logits for each prompt batch, compute token surprisals via `-log_softmax`, accumulate per-label surprise values, and update priors using the Bayesian update \(\pi_c^{(t+1)} \propto \pi_c^{(t)} \exp(-\lambda S_c^{(t)})\) with \(\lambda=0.8\).
4. Re-scale the logits by dividing by the updated priors before sampling and measure balanced accuracy on SST-2, expecting to see at least +2% improvement over the uncalibrated baseline.
5. The output artifact is a calibration-adjusted Qwen-SST2 pipeline plus a parity plot comparing the uncalibrated versus calibrated priors.

**Expected outcome:** a working Colab notebook that outputs calibrated sentiment predictions plus a surprise log demonstrating that the per-label posterior responds dynamically as the prompt length grows.

- **CS student:** run the same notebook on an RTX 4060 but reduce the number of demonstrations to 50 and observe how calibration still improves accuracy, making the notebook feasible for a laptop GPU.
- **Applied engineer:** extend the notebook by quantizing Qwen-2.5-1.5B to INT8 via `bitsandbytes`, wrap it in vLLM, and measure p50 latency below 180 ms while the surprise-calibration loop runs asynchronously.
- **Applied researcher:** hypothesize that slot-based memories (two slots: positive/negative) serve calibration better than a monolithic prior; add those slots, run ablations, and report whether the modular version beats the baseline balanced accuracy by at least 0.5 points.
- **Frontier researcher:** probe the open problem of bounding recency bias by replacing the exponential update with an uncertainty-aware prior (e.g., a learned variance term) and measuring overlap with the ICL posterior—the falsifier criterion is that if the new term fails to reduce bias when contexts exceed 100k tokens, then the assumption “surprise alone suffices” is refuted.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*