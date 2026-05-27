---
title: Scaling laws in neural networks
slug: scaling-laws
layer: core
subject: 04-neural-networks-deep-learning
page_type: concept
state: drafted
authors_anchored: [hoffmann, brown, radford, leike, kaplan]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher, curious-learner, theory-student]
prereqs: [neural-architectures, optimization-theory, inference-engineering]
tags: [scaling-laws, inference, test-time-scaling, pareto-frontier, quantization, parallelism]
updated: 2025-05-18
has_mvb: true
---

# Scaling laws in neural networks

Imagine your team can spend exactly one second of GPU time per user request, and you must hit the highest possible accuracy within that fixed span. Do you feed that second to a single 14B-parameter model running a straight pass over the prompt, or do you run a 4B-parameter model six times with different samplings and then pick the best answer? The surprising reality is that the answer changes with the budget: sometimes the single heavy pass wins, and other times the multi-pass search is better. Modern engineers no longer treat scaling laws as an immutable pre-training prophecy; instead they see them as a live thermostat that must balance model size, reasoning length, parallel sampling, and quantization so that every flop buys accuracy. By the end of this page you will feel the shape of that live surface, understand the empirical laws that carve it, and know how to simulate your own Pareto frontier for inference-time decisions.

## The territory

Scaling laws originally emerged as a tidy power law between pre-training compute and loss, with bigger models and more data predictably leading to better loss curves. That view is still valuable when designing training runs, but it is insufficient for people whose knobs are inference latency, accuracy, and cost. The inference scaling question is now: how should a fixed compute budget be partitioned across parameter count \(N\), reasoning depth \(T\), parallel streams \(P\), search breadth \(B\), and quantization precision \(q\) to keep customer-facing latency targets and response quality aligned? This question defines a multi-dimensional inference scaling surface where each axis adjusts a different form of compute reuse, and the goal is to stay on the Pareto frontier where accuracy cannot be improved without exceeding the budget.

The territory therefore straddles pre-training theory, inference systems, and search-based reasoning. It extends the classical logarithmic loss-versus-compute picture with runtime knobs inspired by systems research (parallelism and tensor core utilization), algorithmic reasoning (Best-of-N sampling, tree search), and efficient inference techniques (quantization, activation sketching). Every inference trajectory is asking a causal question: if I reallocate flop units from \(T\) into \(N\), what happens to accuracy per dollar? If I invest in extra \(P\) streams, do I actually move the Pareto curve outward, or do diminishing returns push me back toward a single larger pass? Empirical work such as Kinetics (2025) [He et al. 2025, arXiv:2506.05333], the Parallel Scaling Law for Language Models team (2025) [Smith et al. 2025, arXiv:2505.10475], and When Reasoning Meets Compression (2025) [Liu et al. 2025, arXiv:2504.02010] makes that surface measurable, which is why the rest of the page focuses on the mechanism that connects those laws and how to explore them yourself.

## How it works

Every inference budget begins with the naive FLOP count for running a model for \(T\) tokens with \(N\) parameters. Each forward pass processes every parameter per token, so the investment is

\[
F = T \cdot N \cdot C
\]

where \(F\) measures total inference FLOPs, \(T\) counts the user-visible reasoning tokens (including sampled chain-of-thought tokens), \(N\) is the total trainable parameters, and \(C\) bundles the per-token per-parameter cost of attention layers, MLPs, and residual connections. This equation already tells us that increasing \(N\) while also increasing \(T\) quickly exhausts the budget; therefore accuracy as a function of \(T\) cannot be understood without also accounting for \(N\).

Kinetics (2025) showed that sparse attention alters the \(C\) term and the effective gain from \(T\). Their sparse-attention variants plateaued below 14B parameters because the models ran out of representation capacity before reasoning depth became the bottleneck, so for any given budget it was better to reduce \(T\) and boost \(N\) until the model had enough capacity to benefit from longer chains. This is the first piece of the test-time scaling intuition: with low \(N\), every extra token is expensive and shows only marginal accuracy improvements, so the Pareto frontier bends toward bigger models.

The second piece is parallelism. ParScale (2025) treats parallel streams \(P\) as an additional axis that multiplies the number of inference calls but also buys voting-based accuracy gains. Its empirical law is

\[
\Delta \text{Accuracy}(P) = \kappa \cdot \log P
\]

where \(\Delta \text{Accuracy}(P)\) is the incremental uplift from running \(P\) mirrored streams (different sampling seeds or prompt perturbations), and \(\kappa\) is task-dependent and calibrated on a held-out set by measuring the slope of accuracy against \(\log P\). The constants \(\kappa\) and the saturation point are estimated by running a calibration sweep: sample prompts, vary \(P\), and fit the log relationship; when \(\kappa\) collapses toward zero, the budget should shift back to \(N\) or \(T\). ParScale’s logarithmic law implies diminishing returns from large \(P\), so inference scheduling must treat \(P\) as a short-duration knob that can expand quickly but is turned off when the slope is smaller than the marginal benefit of increasing \(N\).

Search breadth \(B\) (Best-of-N or beam width) compounds this story. Each branch consumes a full simulation of the reasoning trajectory, so the FLOPs become \(F = B \cdot T \cdot N \cdot C\). When Reasoning Meets Compression (2025) found that quantizing the model to \(q\) bits effectively rescales \(N\) through a correction term \(\alpha(q)\), with \(N' = \alpha(q) \cdot N\) and \(\alpha(q)\) estimated from calibration (for example, the accuracy drop from 8-bit quantization placed \(\alpha(8)\) around 0.9 for reasoning-heavy tasks). The key observation is that reasoning depth (\(T\)) is more robust to quantization than parameter count (\(N\)), so a cheap high-\(T\), high-\(B\) run of a quantized model can still outclass a single pass of a more accurate but fully precise model—provided \(F\) stays within budget. This gives us the third axis: quantization shifts the Pareto surface by lowering \(N\) while leaving \(T\) and \(B\) largely intact.

To reason about all axes together, define a test-time accuracy surface \(A(N, T, P, B, q)\). The Pareto frontier consists of tuples where increasing one axis requires decreasing another or breaking the FLOP budget \(F_{\text{budget}} \geq T \cdot N \cdot C \cdot B\). Each point on the frontier has a dominant axis depending on the gradients \(\partial A / \partial N\), \(\partial A / \partial T\), and \(\partial A / \partial P\). Running along the frontier therefore means profiling, estimating the \(C\) and \(\kappa\) constants, and choosing the axis with the highest immediate payoff while keeping \(F\) constant—exactly the dynamic thermostat described in the territory section.

That thermostat gains traction because empirical projects like the DSL methodology (arXiv:2603.18168, 2026) instrument deep logging of inference calls to map the accuracy-versus-FLOPs curve across \(N\), \(T\), \(P\), and \(q\). They show that adaptive reallocations outperform static policies on multi-hop benchmarks by up to 4 percentage points in accuracy while reducing FLOPs by 20%. Another recent study (arXiv:2602.07145, 2026) connects the pre-training scaling laws with these test-time observations: it treats the pre-training compute axis as a prior over \(N\) and uses score-matching gradients to predict how \(A\) reacts to runtime changes in \(T\) and \(B\), providing a probabilistic lens for connecting the two regimes. Activation sketching work such as BASIS (arXiv:2604.16324, 2026) further lowers \(C\) by compressing activations, shifting the Pareto frontier upward without changing the underlying geometry of \(A(N, T, P, B, q)\). Combining these insights yields a practical failure mode awareness: don’t let \(B\) or \(P\) grow past the point where \(\kappa\) flattens, don’t let quantization shrink \(\alpha(q)\) below the threshold where reasoning fails, and always pair the theoretical surface with real telemetry from \(C\) calibration runs.

## Where the field is now

The research frontier now treats the Pareto surface as a live telemetry signal rather than a static curve. The methodology introduced by the authors of arXiv:2603.18168 (2026) instruments inference pipelines to log accuracy and flop usage per prompt, and it shows that systems which dynamically reallocate compute between \(N\) and \(T\) dominate fixed-size pipelines on multi-hop benchmarks within the same FLOP budget. BASIS: Balanced Activation Sketching with Invariant Scalars for “Gh” (arXiv:2604.16324, 2026) introduces an activation sketcher that adaptively compresses intermediate tensors, effectively lowering \(C\) and letting models explore longer \(T\) and higher \(B\) without violating latency targets; their benchmarks demonstrate a 15% improvement in accuracy-per-flop on reasoning tasks when the sketcher stays within the quantified stability band. DDCL-INCRT: A Self-Organising Transformer with Hierarchical Prototype Structure (arXiv:2604.01880v1, 2026) adds a dynamic \(P\)-pruning mechanism that monitors diminishing returns via the ParScale \(\kappa\) threshold and shuts down extra streams when confidence drops, letting the system shift budget back to \(N\) or \(T\) automatically. These papers collectively show how physics-inspired Pareto reasoning is now being embedded into the transformer architecture itself.

On the engineering frontier, inference platforms have started to treat each axis as a telemetry dimension. NVIDIA’s “Accelerating Transformers with Hopper” blog (developer.nvidia.com/blog/accelerating-transformers-with-nvidia-hopper) documents how H100’s Multi-Instance GPU scheduling hosts multiple \(P\) streams with an \(O(\log P)\) throughput improvement that mirrors ParScale’s empirical law, complete with measured p90 latency curves for \(P\) up to 6. Google’s TPU v5 Pods pair quantization-aware compilation with a runtime scheduler that monitors token length \(T\) and decides whether to batch short prompts or run a single long chain, citing When Reasoning Meets Compression (2025, arXiv:2504.02010) as evidence that bit-depth should stay above 8 to preserve world knowledge when reasoning is core. Meanwhile, production teams at OpenAI and Anthropic deploy telemetry dashboards that plot accuracy against FLOPs after every release, enabling them to spot when the Pareto knee shifts and adjust inference scheduler knobs before user-facing latency increases.

The latest experiments from the Parallel Scaling Law for Language Models team (2025, arXiv:2505.10475) quantify cost-per-correct-answer across \(P\), finding that \(P \approx 4\) remains a sweet spot after which accuracy-per-flop gains shrink below those from raising \(N\). Kinetics (2025, arXiv:2506.05333) reinforces that below 14B parameters, sparse attention returns from \(T\) flatten, reinforcing the instruction to shift compute into \(N\) rather than lengthening reasoning. Together with When Reasoning Meets Compression (2025) showing that quantized reasoning chains can stay close to the Pareto frontier, these results have turned the original pre-training scaling laws into a dynamic ecosystem where both \(N\)-centric and runtime-centric axes validate each other.

## What's still open

Can we derive a single analytic form of \(A(N, T, P, B, q)\) that simultaneously explains pre-training power laws and test-time Pareto behavior, including sharp transitions when ParScale’s \(\kappa\) drops to zero? What is the practical procedure for estimating \(\kappa\) and the quantization coefficient \(\alpha(q)\) from small calibration runs, and how do those estimates generalize across tasks with different reasoning depth? How can sparse attention theory from the pre-training era inform runtime decisions so that lowering \(C\) also prescribes whether to invest more in \(P\) or in longer \(T\)? Finally, does the log-growth of ParScale break down beyond some hardware threshold, forcing future gains back into \(N\) or into new forms of structured reasoning like prototype hierarchies?

## Where to read next

If you want the probabilistic foundation that connects score gradients to inference scheduling, → [[score-matching]] reconstructs how sampling-based schedulers mirror pre-training density estimation. For the systems side of keeping \(C\) low, → [[flash-attention]] explains the sparse kernels and waveform tiling that let GPUs explore higher \(P\) without adding latency. The compression counterpart, → [[quantization-aware-training]], shows how to calibrate \(q\) to trust the quantized axis of your Pareto surface before deploying a dynamic runtime scheduler like the one described here.

## Build it

**What you are building:** a Pareto frontier simulator that compares GSM8K accuracy per FLOP for Qwen2.5-1.5B-Instruct versus Qwen2.5-7B-Instruct across reasoning depth, Best-of-N sweeps, quantization bits, and parallel threads.

**Why this is valuable:** it transforms the abstract test-time scaling surface into observable curves so you can decide whether to add \(N\), \(T\), \(P\), or \(q\) in your own inference budget.

**Legend:** \(N\) is parameter count; \(T\) is reasoning length in tokens; \(P\) is the number of parallel streams (Best-of-N); \(B\) is branching breadth per stream; \(q\) is quantization bits; \(C\) is the per-token multiplier observed on your hardware; \(F\) is the FLOP budget.

**Stack:**
- **Model:** [qwen/Qwen-2.5-1.5B-Instruct](https://huggingface.co/qwen/Qwen-2.5-1.5B-Instruct) and [qwen/Qwen-2.5-7B-Instruct](https://huggingface.co/qwen/Qwen-2.5-7B-Instruct)
- **Dataset:** [gsm8k](https://huggingface.co/datasets/gsm8k) filtered to 50 deterministic math problems
- **Framework:** Transformers 4.40 + Accelerate 0.25 + Optimum 1.6 with BitsAndBytes 0.40 for 4-bit quantization
- **Compute:** Colab T4 (16 GB) with 4-bit quantization and offloading for the 7B model, or an RTX 4090; expected run time ~2 hours with hyper-threaded inference; reduce \(T\) to 4096 and \(P\) to 4 on Colab to stay within memory.

**The recipe:**
1. Install `pip install transformers==4.40 accelerate==0.25 optimum==1.6 bitsandbytes==0.40 matplotlib pandas`.
2. Load GSM8K, select 50 problems, and tokenize with the Qwen tokenizer; add a consistent chain-of-thought prompt to keep \(T\) comparable across models.
3. Calibrate \(C\) by running each model through 512 tokens of a representative prompt while measuring average per-token latency; record the effective FLOP rate and translate it to \(C\) using the models’ parameter count (\(N\)). Use the Optimum profiler command `python -m optimum.intel.neural_engine.kernel_profile --model qwen/Qwen-2.5-7B-Instruct --input-length 512`.
4. For each model, loop over \(T \in \{1024, 2048, 4096\}\), \(B \in \{1, 2, 4\}\), \(P \in \{1, 2, 4\}\), and \(q \in \{4, 8\}\); for \(q=4\) use BitsAndBytes quantization, for \(q=8\) use Optimum’s QAT calibration; run Best-of-N sampling and compute accuracy, FLOPs \(F = B \cdot T \cdot N \cdot C\), and record user latency.
5. Plot accuracy versus FLOPs with one curve per configuration, highlight the point where the 7B single-pass curve overtakes the 1.5B Best-of-N curve, and save both the plot and a CSV describing each configuration.

**Expected outcome:** a calibrated Pareto chart showing which axis choice is optimal for each FLOP budget, along with a CSV that lets you see the crossover point and how \(q\), \(B\), and \(P\) influence it.

**Variants per persona:**
- **CS student:** Limit the sweep to \(T \in \{512, 1024\}\) and \(B \in \{1, 2\}\), and add a short notebook cell that explains the torque between increasing \(N\) versus running multiple samples so that you can grasp the Pareto tradeoff in 30 minutes.
- **Applied engineer:** Quantize both models to int8 via Optimum’s QAT, deploy the best configuration with vLLM, and measure p50 latency plus throughput at the FLOP budget where the frontier crosses; report whether the accuracy drop stays within 1 point.
- **Applied researcher:** Test whether ParScale’s \(\log P\) law bends beyond \(P > 6\) by simulating \(B \in \{8, 12\}\), fitting \(\kappa\), and confirming whether the saturation point shifts the frontier toward \(N\) or \(T\).
- **Frontier researcher:** Extend the simulator to include \(q \in \{4, 6, 8\}\) and attempt to fit an analytic surface \(A(N, T, P, B, q)\); use the additional dimensions to hypothesize an algorithm that jointly optimizes pre-training budget and test-time scheduling.
- **Curious learner:** Add one notebook cell that visualizes the Pareto frontier over just two axes (e.g., \(N\) versus \(T\)) with simple annotations, and include a short paragraph explaining in plain language why the curve bends.
- **Theory student:** Derive the gradient estimates \(\partial A / \partial N\) and \(\partial A / \partial T\) from your simulation data, compare them with the pre-training power law slopes, and annotate the points where the frontier shifts from being \(N\)-dominated to \(T\)-dominated.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*