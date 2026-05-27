---
title: Model Deployment
slug: model-deployment
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [chen, patel, singh, wu, rivera]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, research-engineer, applied-researcher, frontier-researcher, systems-architect]
prereqs: [model-compression, llm-serving, distributed-systems, scheduling-algorithms]
tags: [model-serving, compression, hardware-software-co-design, multi-tenant, latency, energy-efficiency]
updated: 2025-06-01
has_mvb: true
---

# Model Deployment

When a new reasoning model is ready, the last mile is not "torchscript it and ship it" — the last mile is the datacenter floor, where each request competes for scarce latency, memory, and energy. If a CFO were to compare serving an LLM to shipping freight, the bill of materials would list not just the trained weights but the racks of accelerators, the power drawn by those cards, the cooling needed to keep temperature budgets, and the tail latency that decides whether a customer gets an answer in time. Model deployment is the orchestrated answer to that bill of materials: it answers “how do we keep the reasoning quality, keep the latency below its SLA, and keep the environmental and cost tail under control?” By the end of this page you will be able to explain why deployment is a hardware-software co-design challenge, how compressed deltas, adaptive schedulers, and multi-tenant evaluation anchor this challenge, and what a runnable, resource-aware serving stack looks like on accessible hardware.

## The territory

Deploying a model today means running dozens or thousands of tenant-specific versions on a single accelerator pod. Without careful planning, a single inference can either pinch latency budgets (if every tenant has an independent copy of the weights) or waste capacity (if there is slack because the accelerator sits idle while waiting for long reasoning chains). The community responded along two dimensions: compression-aware model variants and resource-aware runtime design. Compression-aware model variants shrink the working set through low-bit quantization, delta fine-tuning, and structured sparsity so that multiple versions can share a base model. Runtime design orchestrates token prefill, decoding, and cache warming around the accelerator’s geometry so that latency, throughput, and energy stay within budget even as reasoning chains stretch. GenAI for Systems (Anand et al. 2026) [https://arxiv.org/html/2602.15241v1] distills these recurring principles: instead of wrapping a trained model in a web API, practitioners now co-design schedulers, energy monitors, and compression knobs. DeepResearch-9K (Lin et al. 2026) [https://arxiv.org/html/2603.01152] emerged to evaluate this combined reality by replaying multi-tenant traces with token-level reasoning difficulty and tenant budgets baked in. The territory is therefore the overlap between multi-tenant compression, resource-aware scheduling, and trace-driven validation that keeps reasoning quality visible. How does that overlap work in practice?

## How it works

Contemporary deployment rests on three layers: (1) delegating tenant-specific behavior to compact deltas, (2) exposing adaptive schedulers that choose precision per request, and (3) anchoring validation to realistic traces and energy accounting. Each layer introduces equations and interfaces that engineers must reason about before they deploy.

### Layer 1: Multi-tenant deltas and shared bases

When an LLM is fine-tuned per tenant, the final weights \( \theta_i \) differ from the base \( \theta_{\text{base}} \) by a small perturbation \( \Delta_i \). DeltaZip (Yao et al. 2025) constructs deployments where \( \theta_i = \theta_{\text{base}} + \Delta_i \) and the \( \ell_2 \) norm \( \|\Delta_i\|_2 \) is deliberately kept an order of magnitude smaller than \( \|\theta_{\text{base}}\|_2 \). This separation enables a serving engine to keep \( \theta_{\text{base}} \) resident in fast shared memory and to stream the sparse, quantized \( \Delta_i \) per tenant on demand. Because \( \Delta_i \) occupies a fraction of the memory footprint, the working set during inference is
\[
W = \text{size}(\theta_{\text{base}}) + \text{size}(\Delta_i),
\]
where \( \text{size}(\cdot) \) counts the compressed byte size of the shared base and the tenant delta. At runtime \( W \) is dramatically smaller than \( \text{size}(\theta_i) \) without compression, which makes it feasible to host tens of tenants on one accelerator.

In addition to memory savings, the delta view simplifies caching and migration. Service engineers treat \( \theta_{\text{base}} \) as a resident kernel and the \( \Delta_i \) as per-tenant patches: when a request from tenant \( i \) arrives, the server pulls \( \Delta_i \), decompresses it, and adds it to the base (either in fused kernels or on-the-fly). The cache only stores the last few \( \Delta_i \), so the eviction policy depends on request frequency rather than the full-parameter load time. The consequence is that tenants with similar deltas share most of the working set, which is why aggressive quantization, structured sparsity, and LoRA-style updates are all common in delta computation.

### Layer 2: Scheduling precision and reasoning tokens

Deltas reduce the memory constant, but latency and energy are dictated by how reasoning tokens flow through the accelerator. The total latency of processing a reasoning query with \( T_r \) tokens can be written as
\[
\text{Latency}(Q, R) = \sum_{t=1}^{T_r} L_t(Q_t, R_t),
\]
where \( Q_t \) is the quantization level (e.g., 8-bit, 4-bit, ternary) used when decoding token \( t \), \( R_t \) captures the reasoning state (cache fills, attention context, and the difficulty label for that token), and \( L_t \) is the time to execute the kernel on the accelerator given those settings. The arithmetic term of \( L_t \) shrinks with lower \( Q_t \), but tokens that are part of a critical rationale may demand higher precision to avoid cascading accuracy drops. The When Reasoning Meets Compression benchmark (Lu et al. 2025) [https://arxiv.org/abs/2504.02] explores this trade-off by measuring how 1.58-bit quantization affects energy per token \( E_t \) and reasoning fidelity: pruning attention heads reduces \( E_t \), but if the scheduler continues with that low precision into the final tokens, the chain needs more steps \( T_r \) to reach the same answer. The benchmark finds an optimal quantization schedule that fits latency while keeping accuracy losses under 2% on DeepResearch-9K.

Rather than fixing \( Q_t \) ahead of time, modern serving stacks expose a scheduler \( \pi(s_t) \) that observes a state \( s_t \) and chooses both the quantization level and the reconstruction strategy for \( \Delta_i \). The state \( s_t \) includes current accelerator occupancy, thermal readings, query difficulty scores, and remaining energy quota. In Reinforcement Learning Foundations for Deep Research Systems (Santos et al. 2025) [https://export.arxiv.org/pdf/2509.06733] these schedulers are modeled as partially observable Markov decision processes where the reward is
\[
r_t = -\alpha \cdot \text{Latency}_t - \beta \cdot \text{AccuracyLoss}_t,
\]
with scalars \( \alpha, \beta \) that encode the service-level objective’s preference between tight latency and fidelity. The scheduler can thus lower \( Q_t \) when \( R_t \) labels a token as “low difficulty” and temporarily raise precision for tokens flagged as critical. Practical implementations expose those knobs through high-level APIs such as vLLM’s control structure and Triton’s asynchronous batches, translating policy decisions into quantized kernels and delta reconstruction orders.

A synthesis sentence bridges Layer 2 to Layer 3: the scheduler’s policy must be validated on traces that reflect multi-tenant token mixes, otherwise a well-tuned policy for synthetic inputs will still break fairness and energy budgets when the tokens and tenant weights change.

### Layer 3: Evaluation anchored in realistic workloads

Evaluation consumes a trace \( \mathcal{T} = \{(q_i, u_i, d_i)\}_{i=1}^M \) where \( q_i \) is a query, \( u_i \) is a tenant, and \( d_i \) is a difficulty tag. DeepResearch-9K (Lin et al. 2026) defines this trace across 9,000 queries with multi-hop math, code tasks, and policy critique, each annotated with token weights. The deployment reports two primary metrics per tenant \( u \): the 95th percentile latency \( \mathcal{L}(u) \) computed over all \( q_i \) with \( u_i = u \), and the reasoning accuracy \( \mathcal{R}(u) \) such as code BLEU or math correctness. When a scheduler drops precision, \( \mathcal{R}(u) \) is linked to the reconstruction error of \( \Delta_u \), so accuracy drops can be traced back to the compressed deltas or to low-precision tokens.

Energy accounting is the third axis. A Decade of Deep Learning: A Survey on The Magnificent Seven (Patel et al. 2024) [https://arxiv.org/html/2412.16188] charts how the dominant AI methods (transformers, diffusion, retrieval, reasoning models, etc.) expand energy usage as parameters grow. Modern deployment inherits those patterns by enforcing \( \mathcal{E}(u) \), the energy per query averaged over the trace, to stay within policy budgets. A practical measurement couples hardware telemetry (nvidia-smi power draw, current via IPMI, or a wall-clock × TDP proxy) with DeepResearch-9K’s trace so that the scheduler can react if a tenant begins to thrash the accelerator.

The three layers compose into a single narrative: compressed deltas shrink resident memory, adaptive schedulers steer quantization along reasoning chains, and trace-driven validation certifies that latency, accuracy, and energy align with the service-level objectives. That is the mechanism of contemporary deployment.

## Where the field is now

Research continues to knit those layers together. DeltaZip (Yao et al. 2025) introduces delta-aware retrieval and asynchronous decompression so that tens of tenant deltas share a base model while hitting 5 ms reconstruction targets. When Reasoning Meets Compression (Lu et al. 2025) demonstrates that 1.58-bit quantization paired with dynamic scheduler adjustments can preserve DeepResearch-9K accuracy within 1.6% while yielding ~3× energy savings compared to static 8-bit inference. GenAI for Systems (Anand et al. 2026) surveys multiple production stacks and identifies recurring patterns—multi-tenant quotas, energy tracking, and scheduler APIs—showing why modern deployments are co-designed from chip to scheduler.

### In production

Many teams now adopt versions of this multi-layer story. NVIDIA’s Triton Inference Server schedules quantized kernels and multi-tenant models with a priority queue that resembles the scheduler \( \pi(s_t) \) described in the RL survey (NVIDIA Triton Inference Server). vLLM exposes a production-mode scheduler with hooks for difficulty scores and precision knobs, enabling the per-token adjustments recommended by When Reasoning Meets Compression (vLLM project). Hugging Face Endpoints pairs those runtimes with per-deployment compute quotas and latency SLAs, letting engineers enforce tenant budgets without redeploying the model (Hugging Face Endpoints). Together these production systems illustrate how compression-aware storage, adaptive scheduling, and quota-aware APIs have become standard pieces of the deployment stack.

## What's still open

Can a scheduler pre-commit a portion of its energy budget to a request, dynamically reallocating high-precision tokens mid-flight based on observed token difficulty, without violating latency and accuracy SLAs? A concrete falsifier is whether such reallocation ever increases the 95th percentile latency by more than 5% on DeepResearch-9K traces.

How do shared delta caches remain privacy-safe when the same base \( \theta_{\text{base}} \) is used across tenants? The structured sparsity introduced by compression could, in principle, leak tenant-specific signals. The question is whether differential privacy or randomized rounding can bound that leakage without blowing up reconstruction latency.

What minimal telemetry (token counters, queue lengths, temperature) is required to predict the next token’s difficulty so the policy \( \pi(s_t) \) can rehearse the precision mode before the decode stage runs? The experiment is to add instrumentation incrementally and measure whether a scheduler with only two telemetry features matches the accuracy of a fully instrumented one.

Can RL-based schedulers generalize to bursty traffic (e.g., product launches) without retraining? A potential metric is whether a scheduler trained on smooth traces maintains fairness (maximum latency delta across tenants) within 10% when the arrival rate spikes to ×3.

## Where to read next

Where this concept appears: the model deployment core sits at the junction of the [[09-algorithms-systems-for-ai]] arc and the [[llm-serving-architecture]] strand, connecting to compression, scheduling, and resource monitoring nodes. Connected topics include [[model-compression]] for understanding delta noise and quantization theory, [[scheduling-algorithms]] for the MDP framing of \( \pi(s_t) \), and [[energy-aware-ops]] for telemetry and policy enforcement, each explaining a different cut of the co-design story; if you want the engineering side of production runtimes, → [[llm-serving-architecture]] walks through pipeline parallelism and caching, while the evaluation counterpart is → [[benchmarking-ml-infrastructure]] which shows how DeepResearch-9K–style traces are constructed and measured.

## Build it

This build returns a working, delta-compressed, multi-tenant serving stack on a single Colab T4, letting you measure latency, accuracy, and energy on a slice of DeepResearch-9K.

**Artifact:** a FastAPI-based server that hosts `togethercomputer/Qwen-1.5B` as a shared base, loads three tenant deltas, and replays a synthetic DeepResearch-9K trace with per-tenant latency, accuracy, and energy telemetry.

**Value:** the artifact makes the hardware-software co-design constraints tangible—each log shows the latency, accuracy, and energy of a tenant request alongside the scheduler’s precision choice.

**Stack:**
- **Model:** [togethercomputer/Qwen-1.5B](https://huggingface.co/togethercomputer/Qwen-1.5B) — 2.3M downloads.
- **Dataset element:** [huggingface/datasets: DeepResearch-9K](https://huggingface.co/datasets/deep-research-9k) — annotated multi-tenant traces.
- **Framework:** `transformers==4.38`, `accelerate==0.23.0`, `bitsandbytes==0.41.0`, `fastapi==0.99`, `uvicorn==0.23`.
- **Compute:** Colab T4 (16 GB GPU RAM) for ~2h fine-tuning + 30m evaluation.

**The recipe:**
1. Install the environment with:
   ```bash
   pip install transformers accelerate bitsandbytes fastapi uvicorn datasets torch torchdata
   ```
   Ensure the Colab runtime has GPU enabled and download `togethercomputer/Qwen-1.5B` using `accelerate launch` so the weights land in shared storage.
2. Load the DeepResearch-9K subset, group prompts into three tenant buckets, and synthesize a trace by tagging each prompt with tenant ID and difficulty scores from the metadata.
3. Fine-tune the base for each tenant with LoRA (rank 8, alpha 16, dropout 0.05) on 200 prompt-completion pairs, saving only the delta tensors. Compress each delta into 4-bit values plus a sparse mask and confirm reconstruction latency stays below 5 ms on the T4.
4. Build a FastAPI endpoint that loads the shared base into a pinned CUDA context, reconstructs the requested tenant’s weights on demand, and routes requests through a scheduler that raises precision for the hardest difficulty scores while using lower precision for easy ones. Measure per-tenant 95th percentile latency and DeepResearch-9K accuracy.
5. Replay the synthetic trace, log per-tenant latency/throughput, and compute energy using `nvidia-smi --query-gpu=power.draw --format=csv` when available or, if not, a wall-clock × TDP proxy with the T4’s 70 W rating; state this proxy in your logs so readers understand the accuracy caveat. Report whether each tenant stays below its latency SLA and whether accuracy loss remains under 2% relative to the uncompressed baseline.

**Expected outcome:** A runnable Colab notebook that demonstrates delta compression, scheduler-guided precision, and energy-aware evaluation with logs showing per-tenant latency, accuracy, and energy.

### What can you build next
Extend the server with a second scheduler that pre-fetches deltas for anticipated tenants, add a difficulty estimator that feeds back into \( \pi(s_t) \), and compare the new outcome against the base scheduler to demonstrate a measurable improvement in tail latency or energy.

**Variants per persona:**
- **CS student:** Swap Colab T4 for an RTX 4070 laptop GPU, reduce batch size to 1, and cut fine-tuning to 30 minutes to prioritize reconstruction latency measurements over throughput.
- **Applied engineer:** Layer the server behind vLLM + Triton, enable 8-bit quantization for the scheduler’s low-power mode, and expose per-tenant compute quotas via Prometheus while holding p95 latency ≤ 120 ms at batch size 2.
- **Research engineer:** Reproduce Table 2 from When Reasoning Meets Compression by hitting the reported 3× energy savings within ±10% on the same DeepResearch-9K split, instrumenting the scheduler to log token-level quantization.
- **Applied researcher:** Hypothesize that scheduler-triggered high-precision decode on the final three tokens improves accuracy by ≥1.5% compared to a static scheduler that never ups precision, while keeping average latency increase under 10%; plot both metrics across control and experiment runs with at least 100 queries each.
- **Frontier researcher:** Add adaptive reasoning token allocation so the scheduler decides how many high-precision tokens each request deserves using a quick difficulty estimator, then falsify by checking whether this controller degrades overall accuracy by more than 0.5% compared to the baseline.
- **Systems architect:** Design a tenant admission controller that rejects new tenants when cumulative energy per query exceeds a policy threshold and demonstrate this controller via simulation on the collected logs.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*