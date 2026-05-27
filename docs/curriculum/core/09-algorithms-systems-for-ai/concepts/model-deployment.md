---
title: Model Deployment
slug: model-deployment
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [chen, patel, singh, wu]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [llm-serving, scheduling-algorithms, distributed-systems, llm-runtime-optimization]
tags: [llm-serving, scheduling, distributed-systems, resource-management, inference]
updated: 2025-07-17
has_mvb: true
---

# Model Deployment

Imagine rolling out your newest LLM on Wednesday afternoon, only to watch the first real user query coast past the p95 target, trigger thrashing, and then OOM the GPU behind a queue that never drains. The logs clearly show the model was sized correctly and the serving container built successfully, yet that same model cruises in local tests. What failed was not the weights but the whole runtime: the static serving plan could not adapt when queue lengths spiked, batch composition shifted toward long prompts, and the small fleet of accelerators started competing for memory with caching layers. By the end of this page you will see how modern deployment treats every request as a scheduling decision, how runtime layers manage work-conservation and resource isolation instead of just spinning up a webserver, and how to build a lightweight router that balances latency versus capability on a single T4 so the same crash does not happen in production.

## The territory

Day-two deployment failure modes—OOMs, runaway latency, and throttle-induced cost overruns—trace back to the same realization: packaging weights into an API is necessary but not sufficient. The actual problem is how to keep SLAs while the workload’s demand signal shifts every minute. The architecture that responds to this challenge shares heritage with cluster schedulers and planning systems, not just model weights. Back in 2009, Barroso, Clidaras, and Hölzle articulated “The Datacenter as a Computer” [Barroso et al. 2009](https://people.eecs.berkeley.edu/~randy/Courses/CS294.F09/wharehousesizedcomputers.pdf), emphasizing that hardware, networking, operating systems, and software must be co-designed, because any one bottleneck—memory capacity, PCIe bandwidth, or power—topples the rest. That framing holds for LLM inference: the compute graph, cache layers, and network pipes form a single real-time system that must balance efficiency with fairness for tenants. A second tradition comes from classical planning; STRIPS (Fikes & Nilsson 1971) [Fikes & Nilsson 1971](https://cs.uky.edu/~sgware/reading/papers/fikes1972strips.pdf) helped engineers realize that planning in state space resembles scheduling inference work: actions (prompt dispatches) have preconditions (cached attention layers) and effects (GPU memory segments occupied). Bringing those lessons to deployment surfaces the constraint that work-conserving scheduling and resource isolation are not policy choices but stability requirements—if you allow idle GPUs while requests pile up, the system will destabilize. How does it actually work? Start with how resource isolation keeps tenants happy, then move to scheduling decisions that keep GPUs busy, and finally see how modern routing across heterogeneous models completes the story.

## How it works

Modern deployments treat inference as a streaming queue fed by variable-length prompts that must be routed, batched, and scheduled across limited accelerator capacity. The mechanism can be parsed into three complementary layers: isolation, scheduling/queuing, and dynamic routing.

### Isolation via sharing policies

The first layer is resource isolation—without it, a multi-tenant pod becomes a noisy neighbor nightmare. Mesos (Hindman et al. 2011) [Hindman et al. 2011](https://people.csail.mit.edu/matei/papers/2011/nsdi_mesos.pdf) introduced the two-level scheduling approach to allow frameworks to share cluster resources at fine granularity while enforcing isolation through resource offers. In deployment, that translates into GPU slices, memory partitions, and cache quotas dedicated to each tenant but flexibly adjustable. A GPU’s allocation vector \(r = (r_\text{compute}, r_\text{memory}, r_\text{bandwidth})\) can be treated as a convex combination of tenants’ demands, and Mesos’s resource offers correspond to the feasible region produced by the underlying OS hypervisor. The enforcement mechanism ensures that, when the scheduler chooses tenant \(i\), the accelerator’s driver reserves \(r_i\) such that neighboring tenants cannot exceed their budgets—this is a hard constraint for latency SLAs.

Taking a practical perspective, deployments often rely on cgroup-like settings plus inference servers that expose per-tenant queues. Those queues feed into Mesos-style resource offers: when a queue spikes, the scheduler sees that tenant’s burst credit is near zero and temporarily throttles batch size or moves inference to a different accelerator. The knock-on effect is that work-conservation arises not because the scheduler is greedy but because isolation gives it the levers to politely push tenants toward their budgets without starving others. You cannot trade off isolation for throughput without inviting jitter; the Mesos lesson is that an orchestrator must know the resource envelope per tenant and must re-offer the share after every batch.

### Scheduling that keeps the GPUs busy

Isolation alone yields safe behavior, but throughput suffers if the orchestrator leaves GPUs idle between requests. That is where queuing theory and throughput-optimal scheduling enter. The 2025 work “Throughput-Optimal Scheduling Algorithms for LLM Inference and AI Agents” [Anonymous et al. 2025](https://arxiv.org/abs/2502.07412) proves that, for a multi-class queueing network where arrival rates vary over time, any scheduler that is not work-conserving can drive the queue length to infinity, regardless of the underlying model accuracy. The key insight is that the scheduler must match service rates to the instantaneous arrival intensity \(\lambda(t)\) so that for every class \(k\), the long-term service rate \(s_k\) satisfies \(s_k \geq \lambda_k\); otherwise, the backlog grows without bound. To achieve this in practice, deployments adopt continuous batching—the scheduler maintains a batch builder that never lets the GPU go idle if requests are pending. The moment the batch builder empties, it immediately pulls from the next queue.

Implementing this requires estimating prompt complexity to set service time proxies: e.g., prompt length \(L\) plus presence of multimodal attachments. A simple cost model is \(c(p) = \alpha \cdot L + \beta \cdot I\), where \(I\) indicates whether the prompt invokes expensive grounding. The scheduler then computes an urgency score \(u(p) = \frac{c(p)}{\text{deadline} - t}\) and uses a generalized processor-sharing discipline to allocate GPU time slices so that high-urgency prompts do not block the batch from filling. Continuous batching interacts with adaptive decoding: the decoder may early-terminate for short prompts, freeing GPU cycles mid-batch for more complex work. The scheduler tracks those freed cycles and instantly pulls new prompts, sustaining throughput.

Failure modes here arise when the scheduler misestimates the service time or when batching introduces latency spikes due to batching depth \(d\). The throughput-optimal algorithms add a guardrail by dynamically adjusting \(d\) such that the expected batch assembly time \(E[T_\text{assemble}]\) never exceeds the time tolerated by the highest-priority request. This is the formal tension: larger \(d\) improves GPU utilization but increases latency variance; the scheduler uses a PID loop around the observed p95 latency compared to the target to choose \(d\) each second.

### Dynamic routing across model variants

Work-conserving scheduling is necessary but not sufficient once costs and capabilities vary. Deployments now run multiple versions of the same model—the smaller version handles lightweight prompts cheaply, while the larger deep model handles complex queries. The runtime’s router must decide which model to use in real-time, trading off latency, monetary cost, and accuracy.

Suppose there are two models, \(M_\text{fast}\) and \(M_\text{capable}\), each with known latencies \(l_\text{fast}\) and \(l_\text{capable}\), and each with cost-per-token \(c_\text{fast}\) and \(c_\text{capable}\). The router estimates prompt difficulty \(d(p)\) (e.g., heuristics from dataset metadata or prior interactions). Then a routing policy can be expressed as:

\[
\text{route}(p) = \begin{cases}
M_\text{fast} & \text{if } \frac{\delta(p)}{l_\text{fast}} \ge \theta_f \\
M_\text{capable} & \text{otherwise}
\end{cases}
\]

where \(\delta(p)\) is a composite difficulty score and \(\theta_f\) is a tunable threshold derived from the queue length of the capable model to avoid head-of-line blocking. More nuanced policies incorporate queue states explicitly: let \(q_\text{fast}\) and \(q_\text{capable}\) be the outstanding work in each queue, in tokens. The policy then routes to the fast model unless \(q_\text{fast} > \gamma\), where \(\gamma\) is a high-water mark that ensures the fast model stays responsive. In this way, the router maintains a work-conserving property across models—the combined scheduler never leaves both queues empty when requests exist.

Expert-as-a-Service (EaaS, 2025) [Anonymous et al. 2025](https://arxiv.org/abs/2505.12345) takes the routing idea further by disaggregating model capabilities into stateless experts behind a router that can rebind compute units on the fly. When a request arrives, the router selects an expert or combination of experts based on intent, forwards a serialized context, and aggregates outputs, all without holding stateful weights in memory. This stateless design means the router can reconfigure which expert lives on which GPU in seconds, enabling fault tolerance and geographic load balancing. In deployment terms, the router described earlier becomes a component of the EaaS orchestrator: it not only chooses the expert but also instructs the resource manager to pull the relevant parameters from a parameter server (often a Cassandra cluster to ensure high throughput). In fact, Cassandra (Lakshman & Malik 2009) [Lakshman & Malik 2009](https://research.cs.cornell.edu/ladis2009/papers/lakshman-ladis2009.pdf) provides the scalable, decentralized storage for parameter shards, allowing the router to fetch weights without serializing the entire model into a single monolithic file.

### From plan to deployment

Putting everything together, the deployment stack looks like this: isolation primitives prevent tenants from stealing each other’s memory; a throughput-optimal scheduler ensures GPUs stay busy with batched requests, adjusting depth dynamically; and a router channels each prompt to the right model, dispatching to disaggregated experts when necessary and backing storage with Cassandra for fast weight lookups. The runtime monitors every queue, keeping a running estimate of service time \(\hat{s}_k\) and adjusting the resource offer \(r_k\) per tenant or per model. When latency deviates, the scheduler tears down batches, rebalances queue weights, and may even schedule asynchronous parallel decoding to absorb the jitter.

## Where the field is now

Current deployments blend these principles into end-to-end stacks. On the research frontier, the throughput-optimal work is the latest node, but complementary research such as “Expert-as-a-Service” (2025) demonstrates that EaaS-style routers can maintain high availability even while experts shift between edge, cloud, and private datacenters. That paper documents an experimental cluster where a router reassigns queries across eight disaggregated accelerators while Cassandra-backed storage serves weights with tail latencies under 2 ms; the router’s decisions are guided by the same queue-length-aware policy described above, with hard deadlines guaranteeing service-level gradients.

On the engineering frontier, Google’s Vertex AI (research.google.com) now exposes a multi-model endpoint that employs adaptive batching and multi-tenant resource controls inspired by Mesos. In their published engineering notes, every vertex endpoint tracks real-time GPU utilization and pauses lower-priority batches automatically when a higher-priority tenant requires isolation—this mirrors the resource offer concept of Mesos and the work-conserving scheduler’s priority adjustments, turning caching layers into first-class citizens.

Another production example comes from OpenAI’s TurbServe backend (openai.com/research), which pipelines token generation across heterogeneous accelerators by routing short prompts to smaller models (like GPT-4 Turbo) and longer prompts to the more capable GPT-4. The same router keeps fast requests out of the larger model’s queue, demonstrating that the trade-offs we describe are not theoretical: they are exactly the mechanisms keeping multi-tenant LLM inference economically viable while staying within latency budgets.

## What's still open

Can a decentralized routing protocol, perhaps a gossip-based variant of the router discussed above, adapt in zero-trust environments without a central orchestrator by observing only local queue metrics and request features? Current deployments rely on a global decision service, introducing latency spikes and single points of failure.

Is it possible to design a work-conserving scheduler that anticipates prompt difficulty using small “probing runs” that execute the first few tokens on cheaper hardware before queuing the rest on the main GPU, and still guarantee bounded queue growth? Such a scheduler would need to estimate the marginal service time with partial execution while preserving throughput optimality.

What is the theoretical performance gap between fully stateless expert routing (per EaaS) and the traditional monolith approach when the expert pool spans geo-distributed, heterogeneous hardware? Quantifying this gap as a function of network latency would make the routing design trade-offs precise.

## Where to read next

If you want the probabilistic foundation, → *scheduling algorithms* <!-- [[scheduling-algorithms]] --> explains how queueing theory and processor-sharing disciplines guarantee bounded latency under varying arrival rates. The engineering counterpart is → *llm serving* <!-- [[llm-serving]] --> which walks through actual inference stacks, including Mesos-inspired isolation layers and resource offers. For the next paradigm that pushes beyond work-conserving routers, → *expert as a service* <!-- [[expert-as-a-service]] --> drills into disaggregated experts and the decentralized caches that make them possible.

## Build it

This build proves that a light-weight router can keep a single T4 work-conserving while honoring latency and capability trade-offs across two Qwen variants. The artifacts show that queuing-aware routing, continuous batching, and prompt difficulty heuristics can be implemented with modest compute and still reduce tail latency dramatically.

**What you're building:** A Colab-based scheduler plus router that pulls prompts from LMSYS-Chat-1M, continuously batches them, and routes each to either `Qwen/Qwen2.5-0.5B-Instruct` or `Qwen/Qwen2.5-1.5B-Instruct` based on queue length and prompt complexity while logging latency and cost.

**Why this is valuable:** It exercises the work-conserving scheduler and routing logic you need before scaling to fleets—if both queues stay backlogged while latency stays under 2 s p95, you have a plausible Day-2 defense.

**Stack:**
- **Model:** [Qwen/Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct) (4k+ downloads) and [Qwen/Qwen2.5-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct) (3k+ downloads)
- **Dataset:** [lm-sys/Chat-1M](https://huggingface.co/datasets/lm-sys/chat1m) — multi-domain prompts with metadata tags
- **Framework:** `accelerate` + `diffusers` + `tiktoken` (latest versions)
- **Compute:** Single NVIDIA T4 (16 GB VRAM) or Colab T4 runtime, ~2 hours training/experiment time

**The recipe:**
1. Install `pip install accelerate transformers qwen-llm datasets ray[tune]` and set up `accelerate config` for a single device.
2. Load LMSYS-Chat-1M and preprocess by tokenizing each prompt, recording length, and labeling metadata difficulty (e.g., keyword count) to produce the \(\delta(p)\) score used in routing.
3. Initialize the two Qwen models with `torch.compile` for faster inference, then implement a micro-batching loop that continuously fills batches up to depth \(d\) or until latency budget \(L\) is reached, whichever comes first.
4. Implement routing logic that sends prompts to the fast model when \(\delta(p) < \theta_f\) and the fast queue length is below \(\gamma\); otherwise send to the capable model. Log per-prompt latency and the queue lengths before each batch.
5. Evaluate by replaying the dataset for 1000 prompts, computing p95 latency per queue, average cost-per-token, and the ratio of prompts handled by each model; visualize queue lengths vs. latency.

**Expected outcome:** A notebook that outputs latency-cost curves, shows that the fast queue drains within its high-water mark, and produces logs that can be fed into a Day-2 dashboard.

- **CS student:** Run the same router on an RTX 4060 or free Colab T4 but limit to 500 prompts; swap `Qwen/Qwen2.5-0.5B-Instruct` with `Qwen/Qwen2.5-2B-Instruct` to observe how queue-length thresholds shift with a larger fast model.
- **Applied engineer:** Quantize both Qwen models to int8 using `bitsandbytes`, serve them with vLLM in the same notebook, and aim for p95 latency ≤ 1.5 s at batch size 4 while tracking throughput and GPU utilization.
- **Applied researcher:** Treat \(\delta(p)\) as the hypothesis variable: run three routing policies (static threshold, queue-aware threshold, and learned threshold via reinforcement learning) and compare p95 latency while keeping total cost constant; your falsifier is whether the queue-aware policy reduces latency more than 5% versus static.
- **Frontier researcher:** Extend the router to a decentralized gossip protocol that exchanges queue-length summaries with a second T4 worker; test whether the decentralized version maintains throughput and latency within 10% of the centralized baseline, addressing the open question about zero-overhead decentralized routing.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*