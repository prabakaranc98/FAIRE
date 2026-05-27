---
title: Mixed-precision training
slug: mixed-precision-training
layer: core
subject: 09-algorithms-systems-for-ai
page_type: concept
state: drafted
authors_anchored: [micikevicius, huang, brown, wu]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [numerical-stability, tensor-cores, scaling-training]
tags: [mixed-precision, tensor-cores, loss-scaling, memory-efficiency, training-stability, hardware-aware]
updated: 2024-11-27
has_mvb: true
---

# Mixed-precision training

Imagine you kick off a training run on a new mini-GPT and, forty minutes later, your loss is `NaN`. The gradients were marching toward the right direction, your optimizer and scheduler were sane, and yet the backward pass suddenly delivered zero updates because a single partial sum fell below \(5.96 \times 10^{-8}\), the smallest non-zero value representable in FP16. Tens of millions of parameters have a minimum dynamic range, and by naively casting everything to the half-precision world, the hardware chews through throughput but silently throws the model off the cliff as soon as gradients underflow. Mixed-precision training is the discipline of keeping that machine humming at Tensor-Core speed while policing its numerical precision so it never drops into `NaN` land. By the end of this page you will see how loss scaling, master weights, and manual casting combine to tame that cliff, and you will know how to build a training loop that measures exactly what you gain and what you risk when you switch from FP32 to reduced precision.

## The territory

Most modern generative stacks demand both more memory and more FLOPs than the hardware in front of you can deliver. The GenAI for Systems: Recurring Challenges and Design Principles from Software to Science survey (2026) [arxiv:2602.15241v1](https://arxiv.org/html/2602.15241v1) paints the same picture from a system-design view: the software needs to coordinate across scheduler, compiler, and observability layers so that every platform opportunity—Tensor Cores, matrix multiplication units, data movement pipelines—is converted into real throughput. Mixed precision is the lever in that coordination. Instead of treating precision reduction as a binary “use FP16 everywhere” command, the discipline adds finesse: it lets you keep expensive values such as master weights or accumulated gradients in FP32 while the inner products still run in FP16, and it inserts loss scalers so the gradients never underflow.

That finesse matters because the datasets and workloads have only gotten harder. DeepResearch-9K: A Challenging Benchmark Dataset of Deep-Research Agent (2026) [arxiv:2603.01152](https://arxiv.org/html/2603.01152) demonstrates that agents now need to model ten thousand distinct research tasks, each with its own signal-to-noise ratio and gradient scale. The only way to keep these agents trainable within a reasonable budget is to recover a large chunk of FP32’s memory and bandwidth costs through lower-precision arithmetic without throwing away stability. A Decade of Deep Learning: A Survey on The Magnificent Seven (2024) [arxiv:2412.16188](https://arxiv.org/html/2412.16188) catalogs the last ten years of architectures and notes that what has changed is not simply the model size but the willingness of practitioners to reallocate which tensors stay high precision and which are allowed to shrink. Mixed precision sits at the intersection of numerical stability and hardware systems; it is the technique that lets us insert more parameters into the same GPU budget by judicious casting.

How does it actually work? The mechanism is best understood by starting from the way gradients are computed and then seeing how precision interacts with that computation.

## How it works

Proper mixed precision training has three parts: (1) letting the accelerator’s low-precision matrix hardware handle the bulk of the multiplication, (2) keeping a high-precision shadow copy of the parameters so the optimizer still sees accurate gradients, and (3) scaling the loss so the gradients do not underflow. Each part is a guardrail on the numerical cliff that naive casting walks straight over.

### The Tensor-Core promise and its numeric risk

Tensor Cores double or quadruple throughput when they operate on FP16 (half-precision) or BF16 data compared to FP32. The hardware pipelines consume two, four, or eight operands simultaneously, so training a single batch in FP16 can be multiple times faster and leave more headroom for larger batches. But Tensor Cores view the gradient accumulation as a high-volume, low-precision reduction. If the loss \(\mathcal{L}\) is computed with FP16 activations and the gradients \(\nabla_\theta \mathcal{L}\) are also stored in FP16, then any value smaller than \(2^{-24} \approx 5.96 \times 10^{-8}\) is rounded to zero before the update. The result is stagnation or explosion; the same gradient that would produce a benign small update in FP32 becomes literally zero in FP16.

To illustrate, suppose we compute a loss \(\mathcal{L}\) on a batch \(x\) and pass it through a scalar function \(f\). The gradient w.r.t. a parameter \(\theta\) is

\[
g = \frac{\partial \mathcal{L}}{\partial \theta} = \frac{\partial \mathcal{L}}{\partial f} \cdot \frac{\partial f}{\partial \theta}.
\]

In FP16, the smallest representable non-zero value is \(2^{-14}\) for mantissa bits, but the effective range for gradients after accumulation is even smaller. When the gradient magnitude \(|g|\) drops below the precision floor, the entire update becomes zero, and the optimizer stops learning. This is exactly the collapse you saw in the hook.

### Master weights and accumulation

Micikevicius et al. (2017) [arxiv:1710.03740](https://arxiv.org/abs/1710.03740) introduced the machinery that rescues the optimizer: master weights. The idea is simple but powerful. Instead of updating the model weights \(\theta\) directly in FP16, you keep two copies—a master copy \(\theta_{32}\) in FP32 and a working copy \(\theta_{16}\) in FP16 that feeds the forward and backward passes. During each iteration, you cast \(\theta_{32}\) to FP16 before the forward pass, and after computing the gradient \(\nabla_{\theta_{16}} \mathcal{L}\) in FP16, you cast it back to FP32, apply the update to \(\theta_{32}\), and then copy it back to \(\theta_{16}\). The optimizer sees high-precision gradients, so it never falls into the underflow trap, while the heavy multiplications still happen in FP16.

This matters because the gradient step becomes

\[
\theta_{32}^{(t+1)} = \theta_{32}^{(t)} - \eta \cdot \mathrm{cast}_{32}(\nabla_{\theta_{16}} \mathcal{L}),
\]

where \(\eta\) is the optimizer step size and the cast ensures the optimizer never sees the truncated FP16 gradients. When we copy \(\theta_{32}^{(t+1)}\) back into \(\theta_{16}\), the model weights used in the next forward pass keep benefiting from the fidelity of the master weights without piling more memory cost on the GPU.

### Dynamic loss scaling

Even with master weights, gradients can still underflow because the gradient tensor itself passes through FP16 before being cast. The cure is loss scaling. You define a scaling constant \(S > 1\) and compute the scaled loss \(\mathcal{L}_S = \mathcal{L} \cdot S\). The gradient with respect to \(\theta\) becomes

\[
g_S = \frac{\partial \mathcal{L}_S}{\partial \theta} = S \cdot \frac{\partial \mathcal{L}}{\partial \theta}.
\]

Because you divide the scaled gradient by \(S\) when transferring it to the FP32 master weights, the effective update is identical to the original \(\partial \mathcal{L} / \partial \theta\). However, while the gradient is being computed, its magnitude is enlarged by \(S\), so it is less likely to underflow. The key implementation detail is that \(S\) is not static; it is adjusted at runtime. After every backward pass, you check whether any of the FP16 gradients overflowed to infinity or NaN. If so, you reduce \(S\) (e.g., \(S \leftarrow S / 2\)) and repeat the backward pass. If gradients have been stable for a few iterations, you increase \(S\) (e.g., \(S \leftarrow S \times 2\)). This is dynamic loss scaling.

Roughly speaking, the update in code looks like:

```
loss = compute_loss(model, batch)
scaled_loss = loss * S
scaled_loss.backward()
for param in model.parameters():
    if param.grad is not None:
        param.grad.data = param.grad.data / S
optimizer.step()
```

The division by \(S\) after the backward pass ensures that the optimizer receives the correct update magnitude.

### Monitoring memory and throughput

Mixed precision is not a binary decision; it is an observable trade-off. To understand it, practitioners frequently measure: (a) the peak memory allocation in manual mixed precision vs FP32; (b) how many tokens or samples per second the GPU can deliver; (c) whether the loss curve diverges or converges more slowly. On PyTorch, you can inspect \( \text{torch.cuda.max_memory_allocated()} \) before and after the backward pass to measure memory savings. Throughput can be measured with simple timers around the forward/backward/step calls, and the convergence can be inspected by comparing FP16 and FP32 losses on a held-out set.

Because of those observables, mixed precision training is rarely an automatic global cast. Production code relies on patterns such as:

- `auto_cast` contexts where certain layers (LayerNorm, Softmax) remain in FP32 because their operations are sensitive to precision.
- Gradient clipping that happens after unscaling to ensure the gradient norm remains stable.
- `torch.cuda.amp.GradScaler` or manual scaling routines when libraries cannot fully capture the desired behavior.

Together, these patterns keep the numerical stability on rails while letting you pass the heavy matrix multiplications to the accelerator at FP16 speed.

## Where the field is now

If the territory is about maintaining stability while hitting Tensor-Core throughput, the current battleground is about automating which tensors deserve high precision and which can, in fact, be shrunk down to 8 bits. Reinforcement Learning Foundations for Deep Research Systems: A Survey (2025) [https://export.arxiv.org/pdf/2509.06733](https://export.arxiv.org/pdf/2509.06733) emphasizes that RL-based agents working in deep research settings suffer from amplified instability because reward signals are sparse and gradients are noisy, and therefore these deployments double down on mixed precision as a stability lever; massive RL agents now routinely run on BF16 because full FP32 would bottleneck memory bandwidth during distributed rollouts.

On the research frontier, teams are experimenting with structured precision re-allocation. Huang et al. (2024) in their SliM-LLM paper demonstrated that rather than uniformly casting every layer’s weights to FP16, the best-performing models assign different bit widths to attention matrices vs feedforward layers, guided by per-layer saliency metrics. Later in 2024, the CompleteP analysis extended that idea by showing that depth-wise hyperparameter transfer with \(\alpha = 1\)—a parameterization that keeps the ratio of learning rates between layers constant—keeps the mixed-precision training stable even as bit allocations change during pre-training. These lines of thought point toward the larger open question of automatically varying precision mid-training without human tuning.

On the engineering frontier, NVIDIA’s Hopper architecture blog (developer.nvidia.com/blog/hopper-architecture-whitepaper) describes how the \(800\) GB/s H100 HBM2e memory and the HBM-side Tensor Cores trigger right-sizing of mixed precision across pipeline stages. They report that large-scale deployments at Meta and Google Cloud run their language model pre-training jobs in BF16 on H100s and that the observed throughput gains are around \(1.8\times\) compared to FP32 while also reducing the required number of data parallel workers. These system-level implementations keep the same numerical constructs—loss scaling and master weights—but wrap them in telemetry that watches per-layer overflows and automatic resets.

Together, these research and engineering advances illustrate that mixed precision is not just about casting tensors; it is about building a feedback system that monitors gradients, scales losses, re-allocates precision, and treats throughput as one more variable in the optimization.

## What's still open

Can we design a mixed-precision scheduler that, during training, estimates the bit-width sensitivity of each tensor and re-assigns precision without a brute-force search? This would require inexpensive gradient sensitivity estimates and a stability predictor that can simulate a precision change before committing to it.

Is it possible to derive a differentiable loss scaling schedule whose parameters are trained alongside the model weights, rather than heuristically tuned thresholds of “how many consecutive safe steps” before scaling grows? Such a schedule would view scaling constants as learnable hyperparameters with gradients defined via automatic differentiation.

How do we build mixed-precision algorithms that are aware of the entire distributed stack—data parallel, tensor parallel, optimizer state sharding—so that low-precision tensors do not become the bottleneck in communication without sacrificing convergence guarantees?

Lastly, can we unify the RL-style gradient variance analyses with the mixed-precision stability story so that sparse reward signals automatically trigger more conservative precision settings, while dense reward phases are allowed to run aggressively low-precision?

## Where to read next

If you want the numerical stability story that motivates validation of every cast, → [[numeric-stability]] unpacks the calculus that makes rounding error formal. The engineering counterpart is → [[tensor-cores]] which details how the hardware pipelines shape the latency vs precision trade-off. For the Bayesian and probabilistic take on gradient underflow, → [[loss-scaling]] gives the ELBO-style derivation that connects loss scaling with variance control.

## Build it

This build proves that manual mixed-precision training—without relying on `torch.cuda.amp`—can be instrumented fully on a free Colab T4 and that the observed memory, throughput, and loss curves are both predictable and measurable when you implement master weights and custom dynamic loss scaling.

**What you're building:** a minimal GPT-style transformer trained on WikiText-2 using manual mixed precision, with logging of memory usage and throughput contrasted against a full FP32 run.  
**Why this is valuable:** you see how scaling constants, master weights, and scaled gradients interact, rather than letting an automated scaler hide the mechanics, so you can debug the moment NaNs appear in production.  
**Stack:**
- **Model:** `hf-internal-testing/tiny-gpt2` — minimal GPT-like transformer with tens of thousands of parameters for quick experiments.
- **Dataset:** `wikitext-2-raw-v1` — the standard small-language modeling benchmark from Hugging Face, clean and fast to download.
- **Framework:** PyTorch 2.2 with GPU support (torchvision 0.15.2, torchtext 0.15.2) running on CUDA 11.8.
- **Compute:** Colab T4 (16 GB VRAM, Tensor Cores) — expect 2–3 hours to finish the full run and under 30 minutes for a shorter ablation.

**The recipe:**
1. Install the stack via `pip install torch==2.2.0 torchvision==0.15.2 torchtext==0.15.2 datasets==2.17.0 accelerate==1.33.0` and verify CUDA availability with `torch.cuda.is_available()`.
2. Load WikiText-2 with `datasets.load_dataset("wikitext", "wikitext-2-raw-v1")`, tokenize with a byte-level BPE tokenizer, and pad sequences to 128 tokens; convert to `torch.float16` inputs but keep targets in `torch.int64`.
3. Define the tiny GPT model with LayerNorm and attention implemented in FP16, but create a master weight copy `master_params = [p.clone().float() for p in model.parameters()]`; wrap the forward pass in `with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=False):` to control casting manually.
4. Train for 5 epochs with AdamW (learning rate \(5\times 10^{-4}\)), computing the loss in FP32; scale the loss by a dynamic scalar starting at \(2^{10}\), check for `torch.isfinite()` on every gradient tensor, adjust the scaler by \(\times 0.5\) or \(\times 2\) depending on overflow, divide the gradients by the scaler before copying into the master weights, and log `torch.cuda.max_memory_allocated()` plus samples-per-second every 100 steps.
5. Evaluate unscaled perplexity on the validation split and record the FP32 baseline run with identical architecture but no casting to quantify the delta in memory and throughput.

**Expected outcome:** a checkpoint representing the mixed-precision-trained transformer, a CSV of per-step memory/throughput metrics, and a comparison table showing perplexity vs aggregated gradient norm relative to the FP32 baseline.

- **CS student:** Run the same training loop but limit to 3 epochs on a Google Colab T4, log both GPU memory and per-step time, and aim to reproduce a 1.4× throughput gain over FP32 while neatly handling one overflow event per run.
- **Applied engineer:** Deploy the trained checkpoint through a vLLM endpoint (e.g., using `vllm` with OpenVINO quantization) and serve at 80 ms p95 latency on an A10, reporting the throughput/latency before and after enabling mixed precision and verifying you can drop `max_memory_allocated` by at least 30%.
- **Applied researcher:** Treat the magnitude of the dynamic loss scaler as the hypothesis variable: run two jobs with different scaler warm-up strategies (fixed \(2^{10}\) vs. auto-scaling) on WikiText-2, chart gradient norm and perplexity, and record whether auto-scaling converges faster by at least 10% fewer steps.
- **Frontier researcher:** Probe the open question of dynamic precision scheduling by instrumenting per-layer gradient variance and designing a simple controller that switches a layer from FP16 to BF16 when the variance exceeds a threshold; falsify the hypothesis if the controller introduces more instability than a fixed FP16 configuration.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*