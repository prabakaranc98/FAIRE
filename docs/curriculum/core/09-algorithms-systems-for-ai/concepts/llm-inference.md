---
title: LLM Inference
slug: llm-inference
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [hinton, devlin, nash, sclaroff]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [kv-cache-management, transformer-architecture, scheduler-performance-modeling, dynamic-batching]
tags: [llm-inference, memory-bandwidth, scheduling, kv-cache, multi-agent, systems]
updated: 2024-10-31
has_mvb: true
---

# LLM Inference

Imagine a restaurant where the chefs can cook any recipe instantly, but every waiter carries each diner’s full conversational history on a handwritten ledger. The cook can only start plating once the waiter can hand over the ledger, yet the ledger grows with every exchange, and memory runs out faster than any CPU can start another dish. That is the everyday crisis behind modern LLM inference: the compute units—the GPUs and tensor cores—can multiply products of huge matrices in a single cycle, yet the system chokes, not because it lacks FLOPs, but because it cannot keep the keys and values, the conversational context, and the scheduling decisions within the limited bandwidth and HBM/DRAM window. By the end of this page, you will see why the memory-bandwidth metaphor, not compute throughput, governs where to spend engineering effort, and how dynamic schedulers plus cache-aware allocation let you scale inference gracefully without buying the next GPU generation.

## The territory

LLM inference sits in the operational layer of the generative stack: training produces a frozen model, but inference keeps the model’s memory alive while numerous users flood it with context that must be served with latency constraints and cost caps. Unlike training, which is often limited by aggregate FLOPs that accelerate with new accelerators, inference is governed by the KV cache that grows linearly with sequence length and number of concurrent users, and by the scheduler that decides when to prefill, when to decode, and when to spill KV values off-chip. From the perspective of systems engineering, the problem is very much one of memory bandwidth and scheduling, not raw compute: the GPUs have plenty of matrix-multiply capability, but the per-token latency and cost are governed by how fast each request’s key and value vectors can be fetched, updated, and scheduled through the multi-tier memory hierarchy.  

This memory-budget and scheduling focus puts LLM inference firmly within the same family as streaming video codecs and real-time databases—systems that must adaptively allocate finite resources against an unpredictable arrival stream. It also borrows from real-time operating systems in its insistence on bounded latency SLOs. The edge that inference engineers have is a distributed cache (the KV cache) plus a scheduler that can refuse work or reorder it to avoid thrashing. As we move into multi-agent deployments where each conversation branches and new tasks appear mid-generation, the consequences of poor scheduling are no longer second-order; they directly manifest as timeouts or massive cloud bills. The mechanism is best understood by starting from the shape of modern KV cache allocation and asking how a scheduler can keep the GPUs running without dropping the context.

## How it works

The key insight is to treat inference as a two-phase workflow: a **prefill** phase that fills the KV cache (for the initial prompt and any tokens produced so far) and a **decode** phase that iteratively appends new tokens, updating the cache at each timestep. Each inference request \(r\) has a context length \(L_r\) recorded in tokens, and the cache for \(r\) consumes \(M_r = L_r \times d_{\text{kv}}\) bytes, where \(d_{\text{kv}}\) is the dimensionality of keys/values times their precision. When multiple requests share a GPU, their caches must fit within the high-bandwidth memory (HBM) budget \(B_{\text{HBM}}\) plus any staging capacity in DRAM \(B_{\text{DRAM}}\). The scheduler’s job is to order prefill and decode tasks so that the sum of active cache allocations never exceeds the instantaneous available memory:

\[
\sum_{r \in \mathcal{A}} M_r \leq B_{\text{HBM}} + B_{\text{DRAM}},
\]

where \(\mathcal{A}\) is the set of requests currently occupying the GPU.

This is where the restaurant analogy becomes concrete: if you try to prefill every long conversation simultaneously, you quickly exhaust \(B_{\text{HBM}}\), allowing no new requests to start. Static batching assumes a fixed number of requests with fixed sequence lengths, but real workloads have bursty arrivals, variable lengths, and branching decisions that can double or triple a request’s memory footprint mid-stream. A dynamic scheduler therefore needs to track the instantaneous memory occupancy and admit new prefill work only when the cache budget frees up, de-prioritizing or even evicting partially used caches if necessary. 

To make scheduling tractable, Zhang et al. (2025) — *Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents* — formalize the problem as a queueing system with two classes of tasks: prefill and decode. Each task has a memory demand \(M_t\) and a service duration \(\tau_t\); a work-conserving scheduler selects tasks to maximize throughput while respecting memory constraints. They prove that if the arrival process is stable, a MaxWeight-style scheduler that ranks tasks by a combination of queue length and memory pressure will achieve throughput arbitrarily close to the system capacity. Formally, let \(Q_t\) be the backlog and \(w_t = Q_t \cdot M_t^{-1}\) be the normalized weight; the scheduler selects the task set \(\mathcal{S}\) that maximizes \(\sum_{t \in \mathcal{S}} w_t\) subject to \(\sum_{t \in \mathcal{S}} M_t \leq B_{\text{HBM}} + \delta\), with \(\delta\) being the DRAM buffer. This transforms the scheduling problem into a knapsack where each decision explicitly accounts for memory and latency, aligning with the system-wide objective of maximizing accepted requests without breaching SLOs.

But maximizing throughput is only one piece of the story. Modern GPUs expose a multi-tier memory hierarchy—HBM closest to the compute units, DRAM as a larger but slower reservoir, and occasionally even NVMe as a last-resort spill target. SparseServe (2025), which explored hierarchical memory tiers for inference, introduces the idea of **PagedAttention**: each cache entry is tagged with a residency status (HBM, DRAM, or NVMe) and can be migrated between tiers based on access heat. The scheduler therefore not only picks which tasks to run, but also which cache pages to keep in HBM. Suppose we track the “temperature” \(T_i\) of each cache chunk \(i\) by counting how many times it participates in a decode step within the last \(k\) tokens. The allocator promotes chunks with high \(T_i\) to HBM and demotes cold chunks to DRAM, and the scheduler uses these temperatures to score requests: a request whose next decode token hits a page in HBM will finish faster than one that requires a DRAM round-trip. SparseServe quantifies this by modeling the latency \(L\) of a decode step as \(L = L_{\text{HBM}} \cdot \mathbb{I}_{\text{HM}} + L_{\text{DRAM}} \cdot (1 - \mathbb{I}_{\text{HM}})\), where \(\mathbb{I}_{\text{HM}}\) indicates whether the needed KV slice resides in HBM. Since \(L_{\text{DRAM}}\) is several multiples larger than \(L_{\text{HBM}}\), those pages must be kept hot, and Dynamic Sparse Attention (DSA) offloading in SparseServe lets the scheduler lower the dimensionality of less important heads to reduce total cache pressure, effectively trading accuracy for memory savings when necessary.

The combination of MaxWeight scheduling and PagedAttention memory management yields a system that can approximate the throughput capacity even as requests branch. The final piece is understanding how these mechanisms influence the economics of scaling. Kinetics: Rethinking Test-Time Scaling Laws (2025) argues that total test-time cost is a sublinear function of compute once sparse attention and hierarchical caches are deployed, because the **effective FLOPs per token** drop when the scheduler can offload cold pages and avoid redundant decoding. They define the **effective load** \(C_{\text{eff}} = \sum_{r \in \mathcal{R}} \left(\alpha_r \cdot \text{FLOPs}_{r}^{\text{active}} + \beta_r \cdot \text{FLOPs}_{r}^{\text{idle}}\right)\), where \(\alpha_r\) tracks the active decoding proportion and \(\beta_r\) accounts for background maintenance (cache warming, DSA). The scheduler’s goal is to minimize \(C_{\text{eff}}\) while hitting latency SLOs, which often means reshuffling tasks to maximize the overlap between high-memory requests that benefit from the same hot pages.

In practice this chain of reasoning looks like a simulator or a controller that, for each incoming request, projects the additional memory demand \(\Delta M_r\), updates the occupancy vector \(\mathbf{M} = [M_1, \dots, M_n]\), and either accepts the request or enqueues it until enough cache budget is freed. When the number of queued requests exceeds a threshold, the scheduler may prioritize shorter contexts to prevent starvation and adhere to latency SLOs. The metrics to watch are **throughput** (requests completed per second), **memory waste** (unused pre-allocated slots), and **forced evictions** (instances where a cache entry is evicted before the decode completes). A well-tuned scheduler keeps forced evictions near zero by allowing DRAM promotion/demotion and by pre-allocating for the average cache size rather than the maximum.

Because these policies involve continuous control, many production systems implement them as simulators that can replay synthetic or recorded traces. The trace-based simulator advances time in discrete ticks representing GPU clock cycles or per-token latencies. At each tick, it recalculates the scheduling decision using the combined weight \(w_t\), evicts cold pages (SparseServe’s PagedAttention idea), and records whether any requests missed their SLO. The ultimate value of such a simulator is that it lets engineers test dynamic batching policies—those that create inference batches on the fly based on available memory—against static baselines defined by fixed batch sizes. In the next section we will see how current systems instantiate these ideas; afterwards, the Build section will walk you through coding a basic simulator that bridges the two phases and captures the key numbers.

## Where the field is now

The research community is treating LLM inference as a systems problem again, with new benchmarks and datasets that capture the deep-research agent workload. DeepResearch-9K (2026) provides a dataset of tens of thousands of agent traces, each with branching interactions, multi-hop reasoning, and dynamically spawned subtasks; the dataset supplies arrival times, context lengths, and adjacency matrices describing which agents collaborate. Papers evaluating schedulers on DeepResearch-9K now report throughput and latency under branchy workloads, and the best submissions explicitly model the KV cache as a queuing resource. This benchmark also highlights that the arrival process is non-Poisson—it is bursty and correlated—which is why Zhang et al.’s (2025) theoretical guarantees about work-conserving schedulers are so valuable: they allow provable throughput bounds even when request rates fluctuate.  

On the systems side, GenAI for Systems: Recurring Challenges and Design Principles from Software to Silicon (Tschand et al. 2026) codifies the engineering principles: treat memory bandwidth as the dominant cost metric, plan for hierarchical caches, and build back-pressure into the scheduler so that expensive LLMs never allocate more than the available HBM. These principles are visible in production deployments such as OpenAI’s inference service (not publicly documented but described in engineering posts like research.google), where each request’s prefill and decode steps are scheduled separately and GPU pipelines are kept mostly full. A comparison table of recent inference deployments illustrates how dynamic batching underpins cost-efficiency:

| System | Latency target | Memory strategy | Scheduler policy |
| --- | --- | --- | --- |
| OpenAI GPT-4 Turbo API (publicly inferred) | 70 ms p95 | HBM + DRAM KV cache with eviction | Multi-stage scheduler with token coalescing |
| NVIDIA Llama-3-inf (estimated) | 50 ms p95 | Large inference cache + NVMe spill | Work-conserving scheduling + demand-adapted batching |
| Anthropic Claude Next | 60 ms p95 | Multi-tier KV + streaming offload | Priority-based prefill/decode with back-pressure |

The engineering frontier also shows up in reinforcement-learning-inspired schedulers. Reinforcement Learning Foundations for Deep Research Systems: A Survey (2025) surveys how RL can learn policies that trade off throughput against latency by observing the SLO violations. These policies treat inference as a Markov decision process with actions that admit, reject, or delay a request, and they estimate a reward equal to the negative latency penalty. Combining these RL policies with work-conserving scheduling ensures that the system continuously learns to avoid congested memory states while maximizing utilization.

Finally, Kinetics: Rethinking Test-Time Scaling Laws (2025) is now the go-to reference for scaling analysis: the paper shows that the cost per token can decrease once dynamic sparse attention and hierarchical caching are operational because they reduce the number of completions that must hit DRAM. The research frontier is exploring how to generalize Kinetics’ sparse attention ideas to structured models (sparse MoEs, branching dialogues) and how to compose them with reinforcement-learned schedulers—the open problem we state next requires this exact composition.

## What's still open

How can we design a unified scheduling algorithm that guarantees strict latency SLOs for multi-agent workloads where execution paths branch recursively and the next sequence length is non-deterministic? The challenge is that predictive knowledge of \(\Delta M_r\) vanishes when agents spawn additional subtasks or jump across contexts, yet the scheduler must still maintain the inequality \(\sum_{r \in \mathcal{A}} M_r \leq B_{\text{HBM}} + B_{\text{DRAM}}\) at every tick. 

Can offline-simulated schedulers (trained on DeepResearch-9K traces) generalize to live, non-stationary arrival distributions while still offering provable throughput bounds? Work-conserving schedulers assume some stability in queues, but deep research agents can trigger bursts that overwhelm the system within a single request. 

Is it possible to merge reinforcement-learned admission-control policies with the temperature-based cache allocation from SparseServe so that the scheduler simultaneously learns which requests tend to keep hot pages and which ones can tolerate DRAM fallback, without incurring the overhead of training a new RL policy for every model variation?

Addressing these questions would connect the theoretical guarantees of throughput-optimal scheduling with the practical memory-pressure signals measured in production.

## Where to read next

For the hardware-aware context, → *memory bandwidth intro* <!-- [[memory-bandwidth-intro]] --> describes how HBM/DRAM hierarchies shape every inference decision; if you want the probabilistic modeling of contexts, → *kv cache management* <!-- [[kv-cache-management]] --> explains how transformers store and update keys/values; for the scheduling perspective that builds from this page, → *dynamic scheduling for llms* <!-- [[dynamic-scheduling-for-llms]] --> walks through adaptive batching policies and their trade-offs.

## Build it

This build shows that the throughput gains we described do not require exotic GPUs—simulating the scheduler, KV allocator, and decoder can already illustrate why static batching fails and why dynamic memory-aware scheduling avoids OOMs.

**What you're building:** A Python simulator that mimics a single-GPU inference service by replaying a synthetic trace, implements work-conserving scheduling, and measures throughput improvements against static batching using a mock PagedAttention allocator.

**Why this is valuable:** The simulator asks you to implement the two hardest pieces—the scheduler that respects per-request memory demand and the allocator that migrates KV pages between HBM and DRAM—so that you can observe directly how dynamic policies prevent cache thrashing.

**Stack:**
- **Model:** `facebook/opt-125m` (3M downloads) — a lightweight transformer used for generating synthetic latencies.
- **Dataset:** `wikitext-103-v1` (2.5M downloads) — used only to derive realistic average prompt lengths and branch probabilities.
- **Framework:** Python 3.11 + PyTorch 2.1 + NumPy 1.26 for tensors and queue modeling.
- **Compute:** Free Colab CPU (2 vCPU/12 GB RAM) — simulation and scheduling logic fit in memory with synthetic tokens.

**The recipe:**
1. Install the simulator dependencies with `pip install torch==2.1 numpy==1.26 tqdm` and download the `wikitext-103-v1` metadata to sample prompt lengths.
2. Build a synthetic trace generator that samples arrival intervals, context lengths, and branch probabilities from the dataset statistics, then represent each request as `(request_id, context_len, token_budget)`.
3. Implement the mock KV cache allocator: each request consumes `context_len * d_kv` bytes; track residency states for each chunk, promoting "hot" chunks to an `hbm` dictionary and demoting cold ones to `dram`, with a simple `temperature` counter incremented on each decode call.
4. Code the scheduler: maintain two queues (prefill, decode), compute `weight = queue_length / memory_demand`, and select tasks whose cumulative memory stays within `HBM + DRAM`. After each selection, advance the simulated time step and update throughput metrics.
5. Evaluate the simulator by running two policies—static batching with fixed batch size `N=4` and the dynamic scheduler—and report throughput (requests/sec) and the number of forced cache evictions per 1,000 simulated requests.

**Expected outcome:** A runnable simulator that outputs a table comparing throughput and eviction counts for static batching versus dynamic scheduling, along with logs showing how often DRAM promotions occur.

**Variants per persona:**
- **CS student:** Run the simulator on Colab with `num_requests=200` and `max_context=512` so it finishes in minutes while still showing OOMs with static batching; plot throughput curves with matplotlib.
- **Applied engineer:** Extend the simulator to feed integer latency samples into the scheduler and integrate the results with a mock vLLM service to target ≤80 ms p95, quantizing KV values to 16-bit to reduce memory and measuring the actual 90th-percentile decode time.
- **Applied researcher:** Test the hypothesis that using a cosine-decay temperature for PagedAttention pages beats a linear decay by plotting throughput vs. time for both decay functions and declaring the cosine variant superior if it reduces forced evictions by ≥15%.
- **Frontier researcher:** Use the simulator to falsify whether a work-conserving MaxWeight scheduler still guarantees throughput when requests branch recursively: introduce nested branching in the trace generator, replay the scheduler, and report the point at which SLO violations occur—this addresses the open problem on unified scheduling under dynamic workloads.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*