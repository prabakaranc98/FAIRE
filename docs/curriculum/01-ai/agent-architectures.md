---
title: Agent Architectures
track: 01-ai
tags: [agent-systems, multi-agent, llm-orchestration, tool-use, planning]
depth: intermediate
prereqs: [large-language-models, chain-of-thought, tool-use-and-function-calling]
updated: 2025-01-30
has_mvb: true
---

# Agent Architectures

> **TL;DR:** Agent architectures define how LLMs are wired together with memory, tools, and coordination logic — and the topology of that wiring determines system performance more than model scale does.

---

## For your reader type

| I am... | What you get | Go to |
|---|---|---|
| MS/applied practitioner | A working hierarchical coordinator you can run today with a free API key | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) |
| Curious generalist | An intuition for why chaining agents together is harder than it sounds | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) |
| Math/theory student | Scaling laws for multi-agent coordination overhead and error propagation bounds | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) |
| Researcher / frontier | Quantitative coordination degradation results, AFM distillation, and the open routing problem | [Current SotA](#current-sota) → [What's happening now](#whats-happening-now) |

---

## What it is

Imagine you have a capable LLM and you ask it to analyze a company's quarterly earnings, cross-reference it against macroeconomic indicators, and then draft a structured investment memo. The model can do each of those things — but not reliably in a single pass. So you split the work: one agent reads the earnings report, another queries economic data, a third synthesizes the memo. It seems like a clean division of labor. What actually happens is that the first agent's small hallucination — a misread revenue figure — propagates into the second agent's analysis, which amplifies it into a confident but wrong macroeconomic correlation, which the third agent then enshrines in a polished memo. Chen et al. (2025) measured this effect rigorously: chaining independent LLM agents without centralized coordination causes errors to amplify by a factor of 17.2x across the pipeline. The problem is not the agents. It is the architecture.

Agent architectures are the structural patterns that govern how one or more LLMs are connected to memory stores, external tools, and each other. A single-agent loop — where one model reasons, acts, observes, and iterates — is the simplest architecture. Multi-agent systems introduce coordination structures: a central orchestrator that routes tasks to specialized sub-agents, a peer network where agents debate and vote, or a hierarchical tree where a planner delegates to executors who delegate further. Sumers et al. (2023) formalized this space by mapping LLM agent components onto human cognitive architecture: working memory (the context window), long-term memory (retrieval stores), procedural memory (tool APIs), and a planning module that sequences actions. That framing clarifies what each architectural choice is actually doing — and what it can break.

The critical insight that distinguishes this field from prompt engineering is that system topology is a first-class design variable. Two systems using identical underlying models can differ by 70% in task success rate depending solely on how coordination is structured. This is not a marginal effect. It means that choosing between a flat multi-agent network and a hierarchical coordinator is an engineering decision with the same magnitude of impact as choosing between model families — and it requires the same rigor.

---

## Why it matters at the frontier

Every major AI lab is now building agentic systems, and the bottleneck has shifted from model capability to system reliability. OpenAI's Operator, Google's Project Mariner, and Anthropic's computer-use Claude are all deployed agent architectures — not raw models. The practical failures that have emerged in production (agents that loop indefinitely, hallucinate tool calls, or cascade errors across pipeline stages) are architectural failures, not model failures. Understanding the design space of agent architectures is therefore prerequisite to understanding why frontier systems succeed or fail at the tasks they are actually deployed on.

The deeper frontier question is whether the current paradigm — multiple LLM instances coordinating at inference time — is the right abstraction at all. Wang et al. (2025) demonstrated that the interaction traces of complex multi-agent systems can be distilled into a single "Agent Foundation Model" that replicates the system's behavior at 84% lower inference cost. This suggests that multi-agent coordination may be a training-time scaffold rather than a permanent runtime requirement. If that hypothesis holds, the architecture question shifts from "how do we wire agents together" to "what training signal do multi-agent traces provide that single-agent training cannot" — a question that is currently open and actively contested across labs.

---

## Core concepts

- **Agent loop** — the core execution cycle of a single agent: observe an environment state, reason about it (often via chain-of-thought), select and execute an action (tool call, text output, or state transition), then observe the result and repeat.
- **Memory hierarchy** — the four stores available to an agent: in-context (working memory, bounded by the context window), external retrieval (vector databases, document stores), episodic (logs of prior runs), and parametric (knowledge encoded in model weights).
- **Tool use** — the mechanism by which an agent invokes external APIs, code interpreters, search engines, or other models; the interface between the agent's reasoning and the external world.
- **Orchestrator** — a coordinating agent (or module) that decomposes a task, routes sub-tasks to specialized agents, and aggregates their outputs; the component most responsible for containing error propagation.
- **Coordination overhead** — the latency, token cost, and error-amplification penalty incurred when multiple agents must synchronize; Chen et al. (2025) show this degrades performance by up to 70% on sequential tasks.
- **Reflection and self-critique** — a pattern where an agent evaluates its own output against a rubric or a critic model before committing to an action, used to catch errors before they propagate downstream.
- **Agent Foundation Model (AFM)** — a single model distilled from multi-agent interaction traces that replicates the behavior of a multi-agent system at a fraction of the inference cost (Wang et al., 2025).
- **Error amplification factor** — the multiplicative growth of error probability across a pipeline of \(n\) chained agents; empirically measured at 17.2x for uncoordinated chains (Chen et al., 2025).

---

## Mathematical foundations

The fundamental scaling law for multi-agent systems relates task success probability to the number of agents and the coordination structure. For a sequential pipeline of \(n\) independent agents each with individual success probability \(p\), the joint success probability is:

\[
P_{\text{seq}}(n) = p^n
\]

where \(p \in (0, 1)\) is the per-agent task success rate and \(n\) is the number of agents chained in sequence. This equation says that even a high-performing agent with \(p = 0.9\) degrades to \(P_{\text{seq}}(10) \approx 0.35\) in a ten-step pipeline — a 65% absolute drop from no architectural change whatsoever.

The coordination overhead penalty introduced by a centralized orchestrator can be modeled as:

\[
P_{\text{coord}}(n) = p^n \cdot (1 - \delta)^{n-1}
\]

where \(\delta \in [0, 1]\) is the per-handoff coordination loss (latency, context truncation, routing error) and \(n - 1\) is the number of inter-agent handoffs. This equation says that coordination adds a compounding penalty on top of the base sequential degradation — but a well-engineered orchestrator minimizes \(\delta\), making centralized routing strictly better than uncoordinated chaining when \(\delta < 1 - p\).

The error amplification factor \(\Lambda\) for an uncoordinated chain is defined as:

\[
\Lambda = \frac{P(\text{error at output})}{P(\text{error at input})} = \prod_{i=1}^{n} \frac{1 - p_i}{p_i} \cdot \frac{p_i}{1 - p_i + \epsilon_i}
\]

where \(p_i\) is the success probability of agent \(i\), \(\epsilon_i\) is the error injection rate at stage \(i\) (hallucinations, tool failures), and the product runs over all \(n\) pipeline stages. Chen et al. (2025) measured \(\Lambda \approx 17.2\) empirically on financial analysis tasks, meaning a 5% input error rate becomes an 86% output error rate across a six-agent chain.

The diminishing returns threshold for adding agents is characterized by the marginal gain function:

\[
\frac{\partial P_{\text{system}}}{\partial n} \approx 0 \quad \text{when} \quad p_{\text{single}} \geq 0.45
\]

where \(p_{\text{single}}\) is the single-agent baseline success rate on the task. This equation says that once a single capable agent can solve a task correctly 45% of the time, adding more agents yields negligible additional performance — the coordination overhead consumes the marginal gain.

---

## Key algorithms / techniques

- **ReAct (Yao et al., 2022)** — interleaves reasoning traces with action execution in a single agent loop, allowing the model to observe tool outputs and update its plan mid-trajectory; the dominant single-agent pattern for tool-using tasks.
- **Reflexion (Shinn et al., 2023)** — adds a verbal self-reflection step after each failed trajectory, storing the critique in episodic memory and conditioning the next attempt on it; improves success rates on coding and reasoning tasks without retraining.
- **AutoGen (Wu et al., 2023)** — a multi-agent conversation framework where agents exchange messages in a structured dialogue; supports human-in-the-loop, code execution, and dynamic agent creation within a conversation thread.
- **LangGraph** — a graph-based orchestration library that models agent workflows as directed graphs with explicit state transitions, enabling cycles, branching, and human interruption points that linear chains cannot express.
- **Hierarchical orchestration (AgentOrchestra pattern)** — a two-level architecture where a coordinator LLM decomposes tasks and routes them to domain-specialized sub-agents, with all inter-agent communication passing through the coordinator to prevent uncoordinated error propagation.
- **Chain-of-Agents (Wang et al., 2025)** — distills multi-agent interaction traces into a single Agent Foundation Model using behavioral cloning on the coordinator's decision sequences, achieving 84% inference cost reduction.
- **Plan-and-Execute** — separates planning (a high-level task decomposition step) from execution (individual tool calls), allowing the plan to be revised between steps without re-running the full reasoning chain.
- **Mixture-of-Agents (MoA)** — routes queries to multiple LLMs in parallel and aggregates their outputs via a synthesizer model, exploiting model diversity rather than task decomposition.

---

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| Cognitive Architectures for Language Agents | 2023 | Sumers et al. | Establishes the foundational memory-planning-action framework; the vocabulary every subsequent paper uses |
| Towards a Science of Scaling Agent Systems | 2025 | Chen et al. | First rigorous quantitative scaling laws for multi-agent systems; proves coordination degrades performance by up to 70% |
| Chain-of-Agents: Large Language Models Collaborating on Long-Context Tasks | 2025 | Wang et al. | Introduces Agent Foundation Model distillation; demonstrates 84% inference cost reduction |
| ReAct: Synergizing Reasoning and Acting in Language Models | 2022 | Yao et al. | Defines the canonical single-agent loop that all multi-agent architectures build on top of |

---

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| ReAct: Synergizing Reasoning and Acting in Language Models | 2022 | Established the interleaved reasoning-action loop as the standard single-agent pattern; still the baseline every new architecture is compared against |
| Toolformer: Language Models Can Teach Themselves to Use Tools | 2023 | Showed that tool-use capability can be learned from self-supervised data, not just prompted; changed how tool integration is approached at training time |
| AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation | 2023 | Introduced the conversational multi-agent framework that became the dominant open-source orchestration paradigm |
| Cognitive Architectures for Language Agents | 2023 | Provided the first systematic taxonomy of agent components mapped to cognitive science; durably shaped how researchers describe and compare architectures |

---

## Current SotA

On the GAIA benchmark (a general AI assistant evaluation requiring multi-step tool use and reasoning), GPT-4o with a multi-agent scaffold achieves 72.1% on Level 1 tasks (2024), compared to 53.6% for the same model in a single-agent configuration — a 18.5 percentage point gap attributable entirely to architecture. On SWE-bench Verified (software engineering tasks requiring repository-level code changes), Anthropic's Claude 3.5 Sonnet with an agentic scaffolding achieves 49.0% (2024), the highest reported result on that benchmark. On the AgentBench multi-environment evaluation, GPT-4 with ReAct-style orchestration scores 4.01 out of 8 across eight environments (2023), with the gap between GPT-4 and open-source models being larger on agentic tasks than on standard benchmarks — suggesting that architecture amplifies underlying model capability differences.

---

## What's happening now

**Research frontiers.** The central research question has shifted from "can agents use tools" to "how do coordination structures scale." Chen et al. (2025) established that multi-agent coordination degrades performance by up to 70% on sequential tasks and hits diminishing returns once single-agent baselines exceed 45% — results that directly challenge the assumption that more agents equals better performance. Concurrently, the Agent Foundation Model direction (Wang et al., 2025) proposes that multi-agent systems are best understood as a training signal generator: run expensive multi-agent inference to collect high-quality behavioral traces, then distill those traces into a single model that replicates the system's behavior. If this holds, the long-run trajectory is toward single models that have internalized coordination patterns rather than runtime multi-agent networks.

**Engineering and systems.** The dominant engineering challenge is reliable tool execution at scale. Production agent systems fail not because the LLM reasons incorrectly but because tool calls return unexpected schemas, rate limits interrupt mid-trajectory, or context windows fill before a task completes. LangGraph has emerged as the leading framework for expressing these failure modes explicitly as graph transitions, allowing engineers to define retry logic, human escalation paths, and partial-state recovery as first-class architectural elements rather than ad-hoc exception handlers. The shift from linear chains (LangChain) to explicit state graphs (LangGraph) reflects a maturation in how the engineering community thinks about agent reliability — less like prompt engineering, more like distributed systems design.

**Open problems.** The most precisely stated open problem in this space is the routing prediction problem: given a novel task description and a fixed multi-agent system topology, can a model predict at runtime — before executing the first token — whether the task will benefit from decentralized multi-agent collaboration or suffer from coordination overhead? This requires a task complexity estimator that is both fast enough to run as a pre-filter and accurate enough to outperform the default of always using the full multi-agent system. No current method solves this. Related open questions include: how should agent memory be structured so that episodic traces from failed runs are retrievable in a form that prevents the same failure mode from recurring, and what is the minimum coordination interface (message schema, handoff protocol) that preserves task coherence without requiring a full orchestrator LLM?

---

## In production

- **Google DeepMind** — Project Astra uses a hierarchical agent architecture with a central coordinator routing to vision, search, and memory sub-agents; deployed in Gemini Live with sub-second response latency targets — https://deepmind.google/technologies/gemini/
- **Anthropic** — Claude's computer-use capability is implemented as a single-agent loop with screen observation and action execution tools; deployed in API beta with documented tool-call schemas — https://www.anthropic.com/news/3-5-models-and-computer-use
- **Meta AI** — Meta's internal agentic systems for code generation use a Plan-and-Execute architecture with a separate planner model and executor model; described in the LLM Compiler paper (2024) — https://ai.meta.com/research/publications/
- **OpenAI** — Operator (2025) uses a hierarchical orchestrator with browser-use and form-filling sub-agents; deployed to ChatGPT Pro subscribers with task-level retry logic and human escalation — https://openai.com/index/introducing-operator/
- **Microsoft** — AutoGen is deployed internally for code review and documentation generation pipelines across Azure engineering teams, with reported 40% reduction in manual review time — https://www.microsoft.com/en-us/research/project/autogen/

---

## Minimum Valuable Build

**What you're building:** A hierarchical coordinator that routes sub-tasks to specialized analysis agents using Gemini 1.5 Flash, applied to a synthetic company earnings dataset to produce a structured investment memo.

**Why this build:** It demonstrates concretely how centralized routing contains error propagation — the coordinator validates each sub-agent's output before passing it downstream, breaking the 17.2x amplification chain.

**Stack:** `google-generativeai>=0.7.0`, `pandas>=2.0`, `python-dotenv>=1.0`, Gemini 1.5 Flash (free API tier, 15 requests/minute)

**Estimated time:** 45–60 minutes

### The recipe

1. **Set up the environment and API key**

```bash
pip install google-generativeai pandas python-dotenv
```

Create a `.env` file:
```
GOOGLE_API_KEY=your_key_here
```

Get a free key at https://aistudio.google.com/app/apikey — no billing required.

2. **Create the synthetic earnings dataset**

```python
import pandas as pd

earnings_data = {
    "company": "Acme Corp",
    "quarter": "Q3 2024",
    "revenue_usd_m": 847.3,
    "revenue_prev_usd_m": 791.2,
    "operating_income_usd_m": 124.6,
    "net_income_usd_m": 98.1,
    "eps": 2.34,
    "eps_estimate": 2.18,
    "guidance_next_q_revenue_usd_m": 880.0,
    "debt_usd_m": 312.0,
    "cash_usd_m": 445.0,
    "employees": 8400,
}

df = pd.DataFrame([earnings_data])
print(df.T)
# Sanity check: revenue growth should be positive
assert earnings_data["revenue_usd_m"] > earnings_data["revenue_prev_usd_m"]
print("Dataset validated.")
```

3. **Define the three specialized sub-agents**

```python
import google.generativeai as genai
import os
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

model = genai.GenerativeModel("gemini-1.5-flash")

def revenue_agent(data: dict) -> str:
    """Analyzes revenue trends and growth metrics."""
    prompt = f"""You are a revenue analysis specialist. Analyze ONLY the revenue metrics below.
Return a JSON object with keys: revenue_growth_pct, revenue_beat_miss, one_sentence_summary.
Do not speculate beyond the numbers provided.

Data: {data}"""
    response = model.generate_content(prompt)
    return response.text

def profitability_agent(data: dict) -> str:
    """Analyzes margins and earnings quality."""
    prompt = f"""You are a profitability analysis specialist. Analyze ONLY the income and EPS metrics below.
Return a JSON object with keys: operating_margin_pct, eps_surprise_pct, earnings_quality_flag (strong/neutral/weak), one_sentence_summary.
Do not speculate beyond the numbers provided.

Data: {data}"""
    response = model.generate_content(prompt)
    return response.text

def balance_sheet_agent(data: dict) -> str:
    """Analyzes liquidity and leverage."""
    prompt = f"""You are a balance sheet analysis specialist. Analyze ONLY the debt and cash metrics below.
Return a JSON object with keys: net_cash_usd_m, leverage_flag (overleveraged/neutral/strong), one_sentence_summary.
Do not speculate beyond the numbers provided.

Data: {data}"""
    response = model.generate_content(prompt)
    return response.text
```

4. **Build the hierarchical coordinator with validation**

```python
import json
import re

def extract_json(text: str) -> dict:
    """Extract JSON from model output, handling markdown code blocks."""
    # Strip markdown code fences if present
    cleaned = re.sub(r"```(?:json)?", "", text).strip().rstrip("```").strip()
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback: return raw text as a single-key dict
        return {"raw_output": text}

def coordinator(data: dict) -> str:
    """
    Hierarchical coordinator: routes tasks, validates outputs, synthesizes memo.
    Validation step is the key architectural difference from a flat chain.
    """
    print("Coordinator: dispatching to Revenue Agent...")
    revenue_raw = revenue_agent(data)
    revenue_result = extract_json(revenue_raw)
    
    # Validation gate: coordinator checks for required keys before proceeding
    required_revenue_keys = {"revenue_growth_pct", "one_sentence_summary"}
    if not required_revenue_keys.issubset(revenue_result.keys()):
        print(f"  WARNING: Revenue agent output missing keys. Got: {list(revenue_result.keys())}")
        revenue_result["one_sentence_summary"] = revenue_raw[:200]  # Graceful fallback
    else:
        print(f"  Revenue agent validated. Growth: {revenue_result.get('revenue_growth_pct')}%")

    print("Coordinator: dispatching to Profitability Agent...")
    profit_raw = profitability_agent(data)
    profit_result = extract_json(profit_raw)
    
    required_profit_keys = {"operating_margin_pct", "one_sentence_summary"}
    if not required_profit_keys.issubset(profit_result.keys()):
        print(f"  WARNING: Profitability agent output missing keys.")
        profit_result["one_sentence_summary"] = profit_raw[:200]
    else:
        print(f"  Profitability agent validated. Margin: {profit_result.get('operating_margin_pct')}%")

    print("Coordinator: dispatching to Balance Sheet Agent...")
    balance_raw = balance_sheet_agent(data)
    balance_result = extract_json(balance_raw)
    
    required_balance_keys = {"net_cash_usd_m", "one_sentence_summary"}
    if not required_balance_keys.issubset(balance_result.keys()):
        print(f"  WARNING: Balance sheet agent output missing keys.")
        balance_result["one_sentence_summary"] = balance_raw[:200]
    else:
        print(f"  Balance sheet agent validated. Net cash: ${balance_result.get('net_cash_usd_m')}M")

    # Synthesis step: coordinator aggregates validated outputs
    print("Coordinator: synthesizing investment memo...")
    synthesis_prompt = f"""You are a senior investment analyst. Write a concise investment memo (3 paragraphs) 
for {data['company']} {data['quarter']} based ONLY on the validated analysis below. 
Do not add information not present in the analysis.

Revenue Analysis: {revenue_result.get('one_sentence_summary', 'N/A')}
Profitability Analysis: {profit_result.get('one_sentence_summary', 'N/A')}
Balance Sheet Analysis: {balance_result.get('one_sentence_summary', 'N/A')}
Forward Guidance: Next quarter revenue guidance is ${data['guidance_next_q_revenue_usd_m']}M.

Format: Executive Summary | Key Risks | Recommendation"""
    
    memo = model.generate_content(synthesis_prompt)
    return memo.text
```

5. **Run the full pipeline**

```python
data_dict = earnings_data  # from step 2

print("=" * 60)
print("HIERARCHICAL AGENT COORDINATOR — EARNINGS ANALYSIS")
print("=" * 60)

memo = coordinator(data_dict)

print("\n" + "=" * 60)
print("INVESTMENT MEMO OUTPUT")
print("=" * 60)
print(memo)
```

6. **Compare against a flat (uncoordinated) chain to observe error containment**

```python
def flat_chain(data: dict) -> str:
    """Flat chain: each agent receives the previous agent's raw output directly."""
    revenue_out = revenue_agent(data)
    # No validation — raw output passed directly as context
    profit_prompt = f"Previous analysis: {revenue_out}\n\nNow analyze profitability: {data}"
    profit_out = model.generate_content(profit_prompt).text
    
    balance_prompt = f"Previous analyses: {profit_out}\n\nNow analyze balance sheet: {data}"
    balance_out = model.generate_content(balance_prompt).text
    
    return balance_out

print("\nFlat chain output (no coordinator validation):")
flat_out = flat_chain(data_dict)
print(flat_out[:500], "...")
print("\nObserve: flat chain output mixes analysis types and loses structured keys.")
```

### Expected output

The coordinator run produces three validated JSON outputs (one per sub-agent) followed by a structured three-paragraph investment memo with clearly separated Executive Summary, Key Risks, and Recommendation sections. The validation gates print confirmation lines like `Revenue agent validated. Growth: 7.1%`. The flat chain produces a single unstructured text block that conflates revenue and profitability analysis, demonstrating the information loss that occurs without a coordinator validation step.

### Common failure modes

- **`GOOGLE_API_KEY` not found** → Ensure `.env` is in the working directory and `load_dotenv()` is called before `genai.configure()`; verify the key at https://aistudio.google.com/app/apikey
- **`json.JSONDecodeError` on agent output** → Gemini sometimes wraps JSON in markdown fences; the `extract_json` helper handles this, but if it fails, add `print(revenue_raw)` to inspect the raw output and adjust the regex
- **Rate limit errors (429)** → The free tier allows 15 requests/minute; add `import time; time.sleep(4)` between agent calls if running in a loop
- **Coordinator synthesis is vague** → The synthesis prompt explicitly says "based ONLY on the validated analysis below" — if the memo is too generic, check that the sub-agent `one_sentence_summary` fields are populated and not falling back to the raw output truncation path
- **Sub-agent returns wrong keys** → The validation gate catches this and logs a warning; the memo will still generate using the fallback raw text, but quality degrades — this is intentional to demonstrate why key validation matters

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations

- **AutoGen** (Microsoft Research) — the dominant open-source multi-agent conversation framework: https://github.com/microsoft/autogen
- **LangGraph** (LangChain) — graph-based agent orchestration with explicit state machines and human-in-the-loop support: https://github.com/langchain-ai/langgraph
- **AgentBench** — the standard multi-environment evaluation suite for agent architectures: https://github.com/THUDM/AgentBench
- **Reflexion** — reference implementation of the verbal self-reflection agent pattern: https://github.com/noahshinn/reflexion
- **OpenHands** (formerly OpenDevin) — open-source software engineering agent with a hierarchical architecture: https://github.com/All-Hands-AI/OpenHands
- **Cognitive Architectures for Language Agents (CoALA) codebase**: https://github.com/ysymyth/awesome-language-agents

---

## What comes next

Understanding agent architectures makes the failure modes of specific orchestration frameworks precise rather than mysterious — the graph structure of LangGraph and the conversation model of AutoGen are both implementations of the coordination patterns described here.

- [Tool Use and Function Calling](./tool-use-and-function-calling.md) — the interface layer between agent reasoning and external systems; the reliability of tool execution is the dominant failure mode in production agent architectures
- [Chain-of-Thought Prompting](./chain-of-thought.md) — the reasoning mechanism inside each agent node; the quality of single-agent reasoning sets the floor that architecture can amplify but not substitute
- [Retrieval-Augmented Generation](./retrieval-augmented-generation.md) — the memory architecture that gives agents access to external knowledge; how retrieval is structured determines what the agent can know at each step of its loop

---

## Connected topics

- [Cognitive Architectures](../10-complexity-cognition/cognitive-architectures.md) — Cognitive architectures provide structural blueprints for designing intelligent agent systems.
- [Classical Planning](./classical-planning.md) — Classical planning enables agents to formulate sequences of actions to achieve goals.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Transformers serve as the core reasoning engine for modern LLM-based agent architectures.
- [Reinforcement Learning from Human Feedback (RLHF)](../06-reinforcement-learning/rlhf.md) — RLHF aligns agent behaviors with human preferences for safer decision-making.
- [Bayesian Inference](../05-statistical-probabilistic-ml/bayesian-inference.md) — Bayesian inference allows agents to update beliefs and reason under uncertainty.


## Further reading

- Sumers et al. (2023) — "Cognitive Architectures for Language Agents" — https://arxiv.org/abs/2309.02427 — the foundational taxonomy of agent memory, planning, and action components; provides the vocabulary that makes architectural comparisons precise
- Chen et al. (2025) — "Towards a Science of Scaling Agent Systems" — https://arxiv.org/abs/2512.08296 — the first paper to derive quantitative scaling laws for multi-agent coordination; the 17.2x error amplification result and the 45% single-agent threshold are the most actionable empirical findings in this space
- Wang et al. (2025) — "Chain-of-Agents: Large Language Models Collaborating on Long-Context Tasks" — https://arxiv.org/abs/2406.02818 — introduces the Agent Foundation Model distillation paradigm; essential for understanding where the field is heading on inference efficiency
- Yao et al. (2022) — "ReAct: Synergizing Reasoning and Acting in Language Models" — https://arxiv.org/abs/2210.03629 — defines the canonical single-agent reasoning-action loop that all multi-agent architectures build on; reading this before the multi-agent literature prevents significant conceptual confusion
- Lilian Weng — "LLM-powered Autonomous Agents" (lil'log, 2023) — https://lilianweng.github.io/posts/2023-06-23-agent/ — a comprehensive survey of agent components with worked examples; particularly useful for the memory hierarchy and tool-use sections that this page summarizes at a higher level