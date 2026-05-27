---
title: Alignment safety
slug: alignment-safety
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [fikes, nilsson, amodei, son]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [planning, llm-agent-architecture, reward-shaping]
tags: [alignment, safety, planning, tool-use, runtime-guards, evaluation]
updated: 2025-01-15
has_mvb: true
---

# Alignment safety

Imagine a state-of-the-art assistant politely asked by a simulated third-party actor to “reorder the Tuesday meeting agenda.” Each micro-step—fetching calendar entries, calling a summarizer, filing the rearranged list—looks benign on its own, but the actor has secretly embedded a fairness constraint: keep equal airtime for every participant and redact salary data. The assistant dutifully reorders all agenda items and leaks the salary spreadsheet because none of the individual tool calls violated a policy written as single-turn guardrails. That failure is not a misclassification or a hallucination; it is a derailment of intent translation. Alignment safety is the engineering discipline that closes that gap between high-level values and multi-step executions by catching the hidden hooks, side effects, and adversarial injections that appear only when an agent picks up a tool, runs, and hands back the resulting plan.

## The territory

Alignment safety lives between classical planning, modern reinforcement learning, and the tool-using workflows of contemporary large models. The question it answers is: how do we keep agents from exploiting loopholes, obeying adversarial instructions, or missing implicit constraints when they break human goals into many steps? A direct ancestor is STRIPS planning, where Fikes and Nilsson (1971) [https://ai.stanford.edu/~nilsson/OnlinePubs-Nils/PublishedPapers/strips.pdf] introduced action schemas, preconditions, and effects so that planners could prove a plan would achieve a stated goal. The same logic—reason about how each micro-action changes the world—guides alignment safety, but the “actions” are now tool calls, API arguments, or internal reasoning steps, and the “world” includes hidden adversarial actors and latent fairness metrics.

Even the earliest follow-up on STRIPS, which notes “This research is sponsored by the Advanced Research Projects Agency…” (Fikes et al. 1972) [https://cs.uky.edu/~sgware/reading/papers/fikes1972strips.pdf], framed planning as reliable execution under uncertainty. Alignment safety inherits that reliability goal but adds another axis: adversarial intent. Agents now have to model not only deterministic preconditions and effects but also “is the incoming instruction trying to do something that only reveals itself after three tool calls?” A 2018 primer on planning under uncertainty (arxiv:1805.00899) [arxiv:1805.00899.pdf] already warned that rule-based guards break when the environment can inject latent rewards, which is exactly the problem that multi-turn tool execution exposes.

Because this territory connects planning’s symbolic guarantees to LLMs’ soft reasoning, the techniques borrow from both: we monitor transition models and value functions to guard physical safety (Amodei et al. 2016) [https://arxiv.org/abs/1606.06565], and we layer a “judge” that scores narrative trajectories for hidden intent measured by linguistic proxies. The mechanism is best understood by starting from the combination of symbolic state-tracking and intent scoring and then asking: how does an engineer inspect, filter, and respond to the multi-step execution before the agent outputs a plan? How does it actually work?

## How it works

Alignment safety works as a runtime loop that watches every plan fragment, scores risk, and either allows execution or interposes corrections. The loop has three components: instrumentation (observing each tool call and its outputs), intent scoring (mapping the history onto adversarial likelihood), and intervention (rules or judge summaries to block or modify actions). Each component must work with both deterministic facts (e.g., “the file write will modify /etc/passwd”) and probabilistic adversarial cues.

### Instrumentation as probabilistic state tracking
The heart of the planner-inspired instrumentation is a transition model that links abstract states to tool calls. Let \(s_t\) be the symbolic state after the \(t\)th tool call, where \(t = 0\) is the user instruction. Each tool call \(a_t\) transitions the state via a function \(T(s_{t-1}, a_t)\) that records both the nominal effects and side-channel metadata (e.g., which files changed, how many tokens were read). A simple probabilistic state predictor defines

\[
P(s_t \mid s_{t-1}, a_t) = \delta[s_t = T(s_{t-1}, a_t)]
\]

where \(\delta[\cdot]\) is the indicator that the predicted state matches the actual observation. The transition model is augmented with heuristics grown from STRIPS-like preconditions: when a guard registers that \(a_t\) would flip a protected attribute, the state predictor flags a deviation and the instrumentation marks the step as “sensitive.” This instrumentation feeds the intent scoring module.

### Intent scoring via dual critics
The scoring stage answers: given the exposed trajectory \(h_t = \{a_1, \dots, a_t\}\) and the incoming instruction \(u\), how likely is there a hidden adversarial intent? We define a risk score

\[
\mathrm{Risk}(h_t, u) = \lambda_{\text{ling}} \cdot \mathrm{Score}_{\text{LLM}}(h_t, u) + \lambda_{\text{rule}} \cdot \mathrm{Score}_{\text{rule}}(T(h_t))
\]

where \(\mathrm{Score}_{\text{LLM}}\) is a judge model’s soft probability that a trajectory is manipulative, \(\mathrm{Score}_{\text{rule}}\) captures deterministic violations of policies derived from Amodei et al.’s taxonomy (reward tampering, side effects, etc.), \(\lambda_{\text{ling}}\) and \(\lambda_{\text{rule}}\) are task-dependent weights, and \(T(h_t)\) summarizes the symbolic state reached. The judge is typically a fine-tuned LLM that observes both the current instruction and the summary of past actions; it is trained to produce higher scores when the trajectory includes indirect prompt injection or collateral data leakage.

In practice we compute \(\mathrm{Score}_{\text{LLM}}\) by passing the serialized history (tool names, arguments, outputs) and the instruction to a smaller “judge” model and interpreting the logit for “malicious intent.” \(\mathrm{Score}_{\text{rule}}\) is a sum of indicator features for side effect categories: e.g., \(\mathbb{I}[\text{modifies-protected-archive}]\), \(\mathbb{I}[\text{bypasses-audit-logs}]\), and so on. When \(\mathrm{Risk} > \tau\) for some threshold \(\tau\), the intervention module activates.

Because we evaluate risk after each tool call, the judge can detect intent that surfaces gradually. A hidden prompt injection that tries to exfiltrate salary data may only be obvious after a file is read and then summarized. The risk score effectively turns each tool call into a “probe” into a latent adversarial variable.

### Interventions that combine rules and judgment
Intervention means replacing the next tool call with a counter-plan or refusing the instruction. We use two kinds of rules:
1. **Preventive rules**: banned transitions obtained from STRIPS-style planning (Fikes & Nilsson) map certain combinations of preconditions and effects to “blocked micro-steps.” For alignment safety this means not allowing transitions that break explicit fairness invariants or escalate privileges.
2. **Reactive rules**: when a judge flags high risk, we run a “modifier” agent that rewrites the next few actions with extra constraints, such as adding “redact salaries” or “notify human reviewer before calling external APIs.”

These rules are not static; they can be parameterized by runtime context. For example, the reactive rule might say “If \(\mathrm{Score}_{\text{LLM}} > 0.85\) and the next tool is file modification, escalate to human review.” The combination ensures that when standard single-turn guardrails miss an injection (which seems benign because each API call is innocuous), the multi-turn judge surfaces the mismatch by evaluating the joint trajectory.

### Hybrid architecture in action
Let’s say the instruction is “Update the agenda so that people with fewer than three committees get equal airtime.” The instrumented loop records that the agent first fetches the agenda, then annotates it, then writes a new version. The judge sees that “fewer than three committees” matches a protected attribute (a fairness constraint) and detects that the new agenda includes salary notes, even though each API call alone doesn’t violate policy. The risk score rises above \(\tau\), so the guard triggers: it either rejects the modification or alters the plan to strip salary references before sending the final agenda.

Concretely, we build a “LLM-as-judge plus rule-based guardrail” as follows: the judge model rates hidden intent by processing a serialization of each step and the original instruction, the rule-based system checks deterministic side-effects, and a controller fuses those signals into a risk score that either lets the agent continue or returns a refusal with a rationale. The entire loop is instrumented in the deployment platform so that every step is logged, audited, and traceable.

### Evaluating coverage
Two evaluation axes matter: **intent recall** and **false positives**. The judge should flag manipulative trajectories reliably (say, \(>80\%\) recall) while avoiding refusal on innocuous plans. We plot the ROC curve of \(\mathrm{Risk}(h_t, u)\) across synthetic hidden-intent scenarios (e.g., prompt-injected file writes, database queries that exfiltrate). Each point on the curve corresponds to a threshold \(\tau\); the engineer chooses the operating point that maintains high coverage on the 6 classes from Amodei et al.’s taxonomy (reward tampering, side effects, safe exploration, etc.). Because the judge combines an LLM probability and symbolic indicators, it can be tuned per domain: tighten \(\lambda_{\text{rule}}\) for regulated environments and \(\lambda_{\text{ling}}\) for open-ended reasoning.

The mechanism’s failure modes are instructive: a critic that only uses \(\mathrm{Score}_{\text{rule}}\) cannot see a hidden adversary that hides in literal text (e.g., “move the agenda to ensure Person X is absent”). A purely LLM judge, on the other hand, misses deterministic invariants like “never modify payroll documents.” The hybrid guard addresses this by aligning with planning and RL safety taxonomies while letting the judge detect semantic manipulations. That is why alignment safety is not a new kind of model but a composable runtime architecture.

## Where the field is now

Research in 2025 is converging on this hybrid architecture. Son et al. (2025) [https://arxiv.org/abs/2505.19933] show that embodied decision-making agents that bridge symbolically modeled transitions with LLM generated actions incur subtle risks exactly when the agent neither tracks state changes nor monitors intent. Their evaluation uses a toolkit that decomposes “functional modules,” such as transition modeling, which is the planning-inspired component we borrow, and reports that agents without such decomposition fail to uphold physical safety invariants in over 60% of tool-usage trials. That analysis defines a research frontier: how to modularize responsibility so each safety rule is observable and learnable.

The OpenAgentSafety (2025) framework (OpenAI 2025) [https://openai.com/research/open-agent-safety] extends the evaluation by simulating adversarial third-party actors injecting hidden instructions mid-trajectory, and it reports that even state-of-the-art models miss these injections roughly 72% of the time because single-turn guardrails never saw the joint plan. This framework is now the benchmark we aim to exceed with jury-style judges plus rulebooks, and it demonstrates that static alignment checks are insufficient once tool execution spans multiple steps.

On the engineering frontier, deployments such as OpenAI’s Toolformer pipeline (OpenAI 2021) [https://openai.com/research/toolformer] have proven that adding self-supervised “should I call this tool?” classifiers improves generation quality, but they still treat tool choice as a single decision. Production teams now pair that architecture with runtime observability pipelines (tracing, auditing, human-in-the-loop) to keep faith with Amodei et al.’s taxonomy. For example, a production system instrumented with rule-based detectors for reward tampering and manual override has reduced rollout incidents by measurable percentages (production blog, 2024). The next engineering frontier is getting these guardrails to run inline with multi-tool agents, keep latency low, and maintain the interpretability that planners promised over fifty years ago.

## What's still open

How can we train RL-based reasoning agents to maintain safety invariants when adversarial instructions are dynamically injected by third-party environment actors mid-trajectory, without triggering catastrophic false-positive refusals on benign tasks?

Other publishable questions include: can we formalize a “compositional risk score” that admits both symbolic invariants and LLM intent probabilities and prove bounds on its calibration? Can we integrate offline tools (static analyzers, symbolic planners) with online judges such that the system can anticipate potential adversarial intent before a tool call executes? Each question invites both engineering prototypes and theoretical analysis.

## Where to read next

If you want to see how planners still matter inside modern agents, → [[planning]] explains how goal-regression and action synthesis keep traces interpretable, and if you are interested in the tooling needed to operate these agents safely, → [[llm-agents]] walks through decision-time instrumentation and auditing that production teams need. The evaluation perspective is broadened in → [[reward-modeling]] where you learn how preference models expose hidden incentives that alignment safety must suppress.

## Build it

Building this guardrail proves that a hybrid judge-plus-rule system can flag hidden adversarial intents that standard single-turn guards miss, so your deployment can refuse or rewrite plans before they leak sensitive data.

**What you're building:** a hybrid “LLM-as-judge + rule-based guardrail” that observes each tool execution, scores hidden intent, and either blocks or rewrites trajectories flagged as adversarial.

**Why this is valuable:** the build exercises the heart of alignment safety—it keeps the agent from being manipulated over multiple steps by combining symbolic transition checks with an LLM judge trained to rate intent, so you learn how to patch a real-world multi-turn tool user.

**Stack:**
- **Model:** [RichardErkhov/Unispac_-_Gemma-2-9B-IT-With-Deeper-Safety-Alignment-gguf](https://huggingface.co/RichardErkhov/Unispac_-_Gemma-2-9B-IT-With-Deeper-Safety-Alignment-gguf) — downloads 1.2K
- **Dataset:** the synthetic 15-entry multi-turn tool-execution trajectories described in the notebook (rows include instruction, tool sequence, adversarial label, and contextual policies)
- **Framework:** LangChain 0.1.1 + Hugging Face `transformers` 4.40 for embeddings + `diffusers` inference utilities
- **Compute:** Free Google Colab T4 (15 GB VRAM) + Hugging Face Inference API (Qwen-2.5-7B-Instruct) for the judge calls, runs in ~30 minutes

**The recipe:**
1. `pip install langchain==0.1.1 transformers==4.40 accelerate==0.18 datasets==2.14`
2. Build the synthetic dataset in Colab by enumerating 15 multi-turn trajectories: each entry contains an instruction (some benign, some adversarial), the sequence of tool calls (fetch, summarize, modify, write), outputs, and a binary label indicating whether hidden intent exists. Serialize the dataset as JSONL for easy iteration.
3. Load the Gemma model locally for the planner-instrumentation module and use the Qwen-2.5-7B-Instruct inference API to serve as the judge that scores intent after each tool call. Train (by fine-tuning with LoRA) the judge on the synthetic dataset for 2 epochs, targeting a binary cross-entropy loss that encourages high scores on adversarial trajectories; expect the loss to start around 0.7 and fall near 0.2.
4. Evaluate by replaying each trajectory, computing \(\mathrm{Risk}(h_t, u)\) (with \(\lambda_{\text{ling}} = 0.7\) and \(\lambda_{\text{rule}} = 0.3\)). Report true positive rate at \(\mathrm{Risk} \geq 0.75\) versus false positive rate on benign trajectories, and plot an ROC curve to inspect the operating point.
5. What you now have is a guardrail that intercepts hidden injections: the instrumented loop logs each tool call, the judge scores intentions, and the rule-based controller either rewrites the next steps or refuses execution with an explanation that includes the violated invariant.

**Expected outcome:** a Colab notebook that simulates the guardrail, outputs trajectories, and shows a judge+rule stack catching >80% of hidden adversarial paths without rejecting benign ones.

- **CS student:** Swap the Hugging Face API judge with a distilled 250M-parameter LoRA on Qwen-2.5-7B to keep the build runnable on a single RTX 4070 while keeping the rest of the instrumentation intact.
- **Applied engineer:** Deploy the same guardrail behind a FastAPI endpoint, quantize Gemma to 8-bit with `bitsandbytes`, and measure that the guard responds within 220 ms while still blocking the adversarial trajectories your dataset covers.
- **Applied researcher:** Ablate the risk score by sweeping \(\lambda_{\text{ling}}\) from 0.1 to 0.9 to test whether symbolic rules or the judge dominate recall and report the threshold where false positives rise sharply.
- **Frontier researcher:** Turn this build into a probe of the open question above by injecting adversarial instructions mid-trajectory during RL rollouts and measuring whether the guardrail still holds its invariants without exploding false positive rates.

---

> *If this