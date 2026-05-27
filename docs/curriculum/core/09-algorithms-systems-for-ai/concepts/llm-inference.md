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
updated: 2025-01-10
has_mvb: true
---

# LLM Inference

Here’s a puzzle: modern GPUs bench hundreds of teraFLOPs, yet a single user prompt to ChatGPT can still take tens to hundreds of milliseconds because nothing is being computed — the chip is waiting on its own memory. Every autoregressive token requires streaming a growing KV cache, slicing the schedule into a latency-critical prefill phase and a latency-headache decode phase, and the faster the decoder runs the more it waits on bytes from HBM. By the end of this page you will see how the “memory wall” — the fact that a GPU spends over 99% of its time simply moving keys, values, and schedule metadata between HBM, DRAM, and the compute engines for every token — forces inference engineers to treat scheduling, cache placement, and task orchestration as their only true levers. You will also leave with a hands-on Speculative Decoding engine that turns that memory-bound reality into measurable latency savings and acceptance statistics.

## The territory

LLM inference lives after training but before the user sees words. Training hardened a model’s weights under massive parallelism, but inference must keep those weights cold while keeping an ever-growing context window hot across multiple concurrent conversations. The thing missing in most explanations is that inference is fundamentally a **memory-bandwidth** problem with a scheduling heart. Every new prompt inserts a request for the prefill phase, where the request’s initial tokens are batched and the KV cache is grown, followed by many decode steps, where each token reads and writes the KV cache that now holds the whole context. Within a single GPU the compute units sit idle while HBM churns bytes; across a cluster, the scheduler becomes the bottleneck as it balances latency SLOs, fairness, and memory usage.

That perspective places LLM inference within the same operational family as streaming video encoders and real-time databases, because like them it must “shape” resource usage against unpredictable arrivals. It also borrows from real-time OS design: inference schedulers must be work-conserving and predictable enough to meet service-level objectives. This is why the most successful deployments today are not pure new chips but **engineered systems** that stitch together KV cache management, prefetching, speculative decoding, and multi-tenant scheduling heuristics. The mechanism is best understood by starting from how a single request moves through prefill and decode and how that movement bottlenecks on bandwidth rather than arithmetic.

## How it works

The first insight is that every token in an autoregressive decode has two different cost components: the prefill cost that loads the initial \(L_{\text{prefill}}\) tokens, and the decode cost that generates future tokens. Expressing this as a per-token latency helps highlight why memory dominates:  
\[
T_{\text{token}} = T_{\text{prefill}} + T_{\text{decode}}.
\]
Here \(T_{\text{prefill}}\) is the cost of computing all attention layers for the client’s prompt and streaming their resulting KV pairs to HBM, while \(T_{\text{decode}}\) is the incremental cost of running attention on new tokens and moving the KV cache updates back into memory. The compute cost inside each attention block is \(O(nd^2)\), but the **data movement** per token is \(2 L N_{\text{head}} d_{\text{head}} \cdot 4\) bytes (for both reads and writes) where \(L\) is the current context length, \(N_{\text{head}}\) is the number of heads, \(d_{\text{head}}\) is the head dimension, and 4 bytes represent fp32 or bf16. As \(L\) grows, this linear term quickly dwarfs the compute, so GPUs spend their cycles waiting on the KV cache streaming to and from their HBM — the “memory wall.”

Prefill and decode therefore meet entirely different bottlenecks. Prefill can be throughput-bound because it processes many tokens in a batch where the attention operations can saturate the matrix units. Decode, in contrast, adds one token at a time, and the GPU becomes **memory-bandwidth-bound** because each incremental token requires touching the entire KV cache. The scheduler must therefore orchestrate when to launch batches, how to shard contexts across GPUs, and when to spill old KV entries to DRAM or flash. This is the stage where **Throughput-Optimal Scheduling Algorithms** (Arora et al. 2025) [arxiv:2504.07347](https://arxiv.org/abs/2504.07347) enters the story: the paper proves that within the queuing model for inference requests, any work-conserving scheduler that stabilizes the total queue length (i.e., keeps arrival rate below service rate) is necessary to avoid infinite latency spirals under adversarial agent workloads. In practice that means schedulers like Orca and Sarathi-serve must be aware of both memory pressure and decoding latency and never allow the system to idle when there is runnable work.

Once the scheduler ensures steady service, the engineering focus shifts to **cache-aware execution**. The KV cache is distributed across HBM and DRAM, and beyond a certain context length a single GPU simply cannot hold everything in HBM. SparseServe (Lee et al. 2025) [arxiv:2509.24626](https://arxiv.org/abs/2509.24626) introduces hierarchical offloading: the most recent tokens stay in HBM while older segments spill to DRAM and slower tiers. SparseServe also highlights that the prefetcher must rehydrate KV frames before decode to avoid stalls; the system needs to compute a **memory-horizon** \(M = K\cdot B\) where \(K\) is the number of cached tokens a GPU can service at full HBM bandwidth and \(B\) is the per-token bytes. The scheduler keeps the working set inside \(M\); when requests exceed it, the scheduler must either drop requests, lower fidelity (e.g., reduce context length), or offload KV pages and pay the latency tax. Appendix-level telemetry then informs decisions like “spill tokens 128 steps older than the most recent key” to maintain throughput.

A third lever is **speculative decoding**, which uses a small “draft” model to run multiple candidate continuations ahead of time while the full model is still catching up on the previous token. The key math is in the acceptance probability \(p\) that the target model will agree with the draft sequence. If the draft model generates \(k\) tokens in speculation, the expected cost per token becomes
\[
\mathbb{E}[T_{\text{spec}}] = \frac{T_{\text{draft}} \cdot k + (1 - p) \cdot T_{\text{full}}}{p \cdot k},
\]
where \(T_{\text{draft}}\) is the draft model’s decode latency per token, \(T_{\text{full}}\) is the full model’s per-token latency, and \(p\) is the probability that the full model accepts the draft continuation. When \(p\) is high and \(T_{\text{draft}} \ll T_{\text{full}}\), the speculation effectively amortizes the full-model latency across many tokens, hiding memory stalls. The scheduler must still manage the KV cache because the draft model needs access to the same context, so the inference pipeline duplicates the KV cache snapshot across both models, incurring additional memory overhead. System designers therefore tune the speculation depth \(k\) and acceptance threshold to trade additional cache pressure for reduced decode latency.

Another important mechanism is **multi-tenant batching with dynamic token windows**, which is where the GenAI for Systems survey (Miller et al. 2026) [arxiv:2602.15241](https://arxiv.org/html/2602.15241v1) becomes actionable: the paper catalogs design principles for running large language models across software stacks and hardware, emphasizing that inference stacks must gracefully degrade by pausing low-priority requests, enforcing per-tenant fairness budgets, and accounting for heterogeneity in GPU types and memory capacity. Practically, that means adding a `priority_queue` of requests and associating each one with a `memory_budget` so the scheduler can pre-reserve KV capacity before admitting admission. When a high-priority request arrives, the queue temporarily halts lower tiers and drains their KV pages back to DRAM, letting the HBM and compute focus on the urgent decode steps.

An inference system also needs observability that ties hardware counters to queue metrics. The per-GPU **pipeline efficiency** \(E\) can be expressed as the ratio of compute time to wall-clock time:
\[
E = \frac{T_{\text{compute}}}{T_{\text{compute}} + T_{\text{memory}} + T_{\text{stall}}},
\]
with \(T_{\text{memory}}\) being time spent waiting on KV transfers. Efficiency collapses whenever \(T_{\text{memory}} \gg T_{\text{compute}}\), which happens in inference outside of batched prefill. That’s why schedulers monitor \(T_{\text{memory}}\) and throttle new prefill work until the current decode pipeline finishes. The RL survey (Reinforcement Learning Foundations for Deep Research Systems, Chen et al. 2025) [arxiv:2509.06733](https://export.arxiv.org/pdf/2509.06733) explains how reinforcement learning can be used to tune such priority policies in multi-agent scenarios, training policies that learn which requests to speculatively execute and when to spill KV segments to slower storage while still satisfying cumulative latency budgets.

Finally, real inference systems must keep an eye on **energy**. The “How Hungry is AI?” analysis (2025) reports that reasoning models like DeepSeek-R1 consume an order of magnitude more energy per token than smaller edge models under identical workloads because they keep massive KV caches warm and operate at high throughput continuously. That study emphasizes practical levers: quantization, operator fusion, and scheduling heuristics that allow GPUs to enter low-power states between bursts. Without scheduling discipline, inference deployments risk turning every request into a “hot compute spike,” so the system alternates between high-throughput prefill phases and energy-aware decode phases that reuse warmed caches while throttling new arrivals.

## Where the field is now

The last two years have seen inference move from ad-hoc scaling to formally analyzed schedulers and memory hierarchies. “Throughput-Optimal Scheduling Algorithms” (Arora et al. 2025) [arxiv:2504.07347](https://arxiv.org/abs/2504.07347) gave the scheduling community a queuing-theoretic foundation, proving that work-conserving schedulers are the only way to maintain bounded latency under adversarial workloads and predictable memory pressure. This insight is reflected in the latest versions of Orca and Sarathi-serve, each instrumented with queue-length feedback loops to adjust batching, spilling, and speculative decoding dynamically. SparseServe (Lee et al. 2025) [arxiv:2509.24626](https://arxiv.org/abs/2509.24626) extends this work with a multi-tier memory hierarchy that stores the most recent \(M\) tokens in HBM while older tokens drift to DRAM and flash, giving deployments the ability to offer 32k+ token contexts without excessive memory cost.

On the engineering frontier, Hugging Face’s Text Generation Inference (TGI) stack now embodies these design principles at scale. The TGI documentation [https://huggingface.co/docs/text-generation-inference/main/en/overview](https://huggingface.co/docs/text-generation-inference/main/en/overview) explains how the service bundles KV cache pooling, async batching, and dynamic quantization modes to deliver latency below 80 ms for 4-token bursts on H100 clusters. The stack integrates logging from NVML to inform per-GPU scheduling heuristics and exposes knobs to trade off fairness and throughput, showing that inference at scale is no longer about brute compute but about orchestrating memory, caches, and scheduling signals across multi-tenant clusters.

Benchmarking is catching up too. DeepResearch-9K (Garcia et al. 2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152) provides a challenging benchmark of deep-research agents with long, branching prompts, enabling inference engineers to test scheduler choices under realistic multi-agent workloads. Because the dataset comprises 9,000 prompt-response sequences with varying lengths, it surfaces regimes where speculative decoding fails (low acceptance rates) or where prefill latency dominates. Together, these benchmarks and systems show that inference progress is now measured not in raw TFLOPs but in how many requests per second can be satisfied while keeping \(T_{\text{prefill}}\) and \(T_{\text{decode}}\) within SLOs, how little KV data spills to DRAM, and how many tokens can be served without energy hogging.

## What's still open

Can inference schedulers be fully decentralized and still maintain the savings of local execution? Today's orchestration layers assume central controllers to route requests to the least-loaded GPU, but as deployments move into disaggregated, heterogeneous clusters, communicating queue lengths over the network introduces milliseconds of latency that erase the gains of local optimization. The question becomes: **Can we design a zero-overhead scheduling algorithm that dynamically partitions autoregressive decoding steps and KV cache shards across heterogeneous hardware while keeping inter-node communication smaller than the local execution latency?**

A second question is whether blocking speculative decoding on a fixed acceptance probability \(p\) is optimal. When requests vary in difficulty, a single \(p\) either wastes compute on long sequences or causes misprediction for hard-generated tokens. **Can we learn request-aware speculative policies that adapt \(k\) (speculation depth) and \(p\) per request using lightweight on-device telemetry, yet still guarantee latency SLOs under adversarial arrival patterns?**

Finally, the energy studies remind us that inference is now an environmental concern. **Can we formalize an energy-latency frontier and embed it in the scheduler so that deployments can trade off power (Watts), latency (ms/token), and accuracy in a single multi-objective policy?** This would let inference systems report expected CO₂ per prompt as easily as they report median latency.

## Where to read next

If you want the scheduler theory behind stable latency, the work on → *scheduler performance modeling* <!-- [[scheduler-performance-modeling]] --> formalizes the queueing models and priority policies this page calls upon; the engineering companion is → *kv cache management* <!-- [[kv-cache-management]] --> because the KV cache is the object that every scheduler manipulates; for the runtime layer, → *dynamic batching* <!-- [[dynamic-batching]] --> shows how batching decisions are tuned when every GPU stitch is about memory bandwidth.

## Build it

Speculative decoding makes memory-bound inference measurable, and this build proves it by pairing a tiny draft model with a real LLaMA-3.2-1B target in a PyTorch pipeline running entirely on a free Google Colab T4.

**What you're building:** A speculative decoding engine that runs `meta-llama/Llama-3.2-1B` for verification but uses `EleutherAI/gpt-neo-125m` to draft continuations, measures latency before and after speculation, and plots token acceptance versus latency for prompts from DeepResearch-9K.

**Why this is valuable:** You touch the full inference stack: KV cache duplication, asynchronous speculation, and scheduler metrics, turning the abstract “memory wall” into a chart of acceptance rates.

**Stack:**
- **Model:** `meta-llama/Llama-3.2-1B` — 5.3M downloads; real 1B-parameter model with official quantized weights
- **Dataset:** `deepresearch/DeepResearch-9K` — documented benchmark of long-reasoning prompts for multi-agent systems
- **Framework:** PyTorch 2.1 + `accelerate` 0.27 + `transformers` 4.38
- **Compute:** Google Colab T4 (16 GB VRAM) — spec: 8.1 TFLOPs Tensor and enough memory for both KV caches; training time ~45 minutes for the full pipeline

**The recipe:**
1. `pip install accelerate==0.27 transformers==4.38 torch==2.1 matplotlib` and use `accelerate config` to target the Colab T4 runtime.
2. Download five representative prompts from DeepResearch-9K, tokenize them with the model’s tokenizer, and pad each to 1024 tokens; store the tokens and attention masks so you can reproduce the same KV cache snapshots.
3. Load `meta-llama/Llama-3.2-1B` with quantized weights, cache the prefill output for each prompt, and then load `EleutherAI/gpt-neo-125m` as the draft; implement speculation by generating \(k=4\) tokens from Neo per decode step, compare them against the LLaMA-3.2-1B logits, and commit the draft if the top-1 token matches. Track \(p\) as the ratio of accepted drafts.
4. Measure latency by timing seven decode steps per prompt both with and without speculation, logging the per-step latency, KV bytes moved, and CPU/GPU memory occupancy; plot the per-token latency versus the acceptance probability.
5. You now have a dashboard-style artifact: a checkpointed speculative decoder, the latency/acceptance plot, and exported metrics that show how memory bandwidth and scheduler choices affect actual throughput.

**Expected outcome:** A runnable speculative decoder notebook that reports decode speedups of 1.4x–2x depending on \(p\), along with plots of latency, acceptance, and KV cache pressure for DeepResearch-9K prompts.

- **CS student:** Run the same pipeline on an RTX 4070 (or free Colab A100) by reducing context length to 512 tokens; this keeps the KV cache within 11 GB so the maintainers can explore the inference regime where compute begins to matter again.
- **Applied engineer:** After the base build, quantify latency under `text-generation-inference`-style batching by exporting the pipeline as ONNX, running it in TGI with 4-bit quantization, and using `triton` to enforce 80 ms p95 latency; report the new \(p\) and CPU memory footprints.
- **Applied researcher:** Use this setup to test the hypothesis “draft models with similar attention heads (Neo-125M) have higher acceptance than smaller random decoders.” The falsifier is a plot showing acceptance drop below 0.5 for a randomly initialized draft, proving structure matters.
- **Frontier researcher:** Extend the build to multi-node simulation: emulate a decentralized scheduler where each node shares KV metadata via a lightweight gossip protocol and measure whether communication latency ever exceeds the local gain predicted in the open question.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*