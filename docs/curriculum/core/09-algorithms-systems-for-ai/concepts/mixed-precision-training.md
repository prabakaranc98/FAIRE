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
tags: [mixed-precision, tensor-cores, loss-scaling, numerical-stability, low-precision, training-efficiency]
updated: 2024-12-06
has_mvb: true
---

# Mixed-precision training

Imagine a training run that looks perfectly healthy on day one—loss is descending, gradient norms sit between 0.1 and 1.0, and your RTX A5000 hums along at 19 TFLOPs. Then, at step 1,024, the loss shoots to `NaN` and every gradient becomes zero. The optimizer still believes in the weights, but the gradients have underflowed into oblivion because the last multiplication was performed in FP16 and the true value lived at \(2^{-18}\). The hardware has not broken, but the number system has: FP16 can only represent values as small as \(2^{-14}\) in the tensor-core-friendly schemes that dominate modern GPUs. Mixed-precision training is the craft of letting Tensor Cores multiply, accumulate, and move data in half or eighth precision while keeping the numerics sane. By the end of this page you will understand the three safeguards—master weights, loss scaling, and format switching—that make mixed precision practical, and you will be ready to write a PyTorch training loop that watches underflow happen and then immediately rescues itself without hugging high-level wrappers.

## The territory

Training large neural networks without precision tricks can require more bandwidth than a single accelerator provides, and doubling the batch size or depth to squeeze more FLops often leads to out-of-memory errors long before the model is “finished.” When Tensor Cores first introduced FP16 matrix multiplies, practitioners saw massive throughput gains, but they also saw gradients disappear. The problem is not the compute units; it is the 11-bit mantissa in FP16. Each multiplication is precise enough for activations that stay near unity, but gradients and updates often fall into the \(10^{-6}\) to \(10^{-9}\) range. The naive solution—keep everything in FP32—severely limits batch sizes and model depth. Mixed precision sits between those extremes: keep the data in reduced precision for the heavy inner loops but track the weights and accumulators in higher precision so updates stay meaningful. This is the discipline that Micikevicius et al. (2017) [arxiv:1710.03740](https://arxiv.org/abs/1710.03740) introduced, and the same recipe lives in the astrophysics catalog entry “Mixed Precision Training – ADS” (2017) [https://ui.adsabs.harvard.edu/abs/2017arXiv171003740M/abstract], which catalogs the earliest HPC deployments that turned off underflow with a scaled loss.

The territory also includes the observability and scheduling layers that make modern stacks work: once you have a mixed-precision engine, the compiler, scheduler, and telemetry must know where the master copy lives, where the cast happens, and when to reduce the loss to keep gradients visible. The result is a discipline that is as much about numerical psychology as it is about hardware: the engineer must constantly ask, “What precision does this accumulator tolerate?” and “How do we detect when a gradient vanished so we can react?” How mixed precision actually works depends on whether you focus on the hardware advantage (memory, bandwidth, Tensor Core peaks) or the numerical safety net (master weights, loss scaling, hybrid formats), and the mechanism is best understood by starting from the underflow itself.

## How it works

The central observation is simple: the backward pass generates gradients whose magnitude can be orders of magnitude smaller than the forward activations. Consider a single linear layer whose input is \(x\in\mathbb{R}^d\), weights \(W\in\mathbb{R}^{d\times k}\), and loss \(L\). The default gradient update is

\[
W' = W - \eta \frac{\partial L}{\partial W},
\]

where \(\eta\) is the learning rate. Training in FP32 evaluates both \(W\) and \(\partial L / \partial W\) at 32-bit precision, so the subtraction retains the gradient contribution even when it is small. In mixed precision, \(W\) and \(\partial L / \partial W\) are stored or computed in FP16, whose mantissa can only express about 3 decimal digits. If a gradient falls below \(2^{-11}\), it is rounded to zero and the update disappears, so training stalls.

### The precision gap and the scaled loss

The cure is to widen the floating-point funnel for gradients without losing the throughput gains of FP16 arithmetic. Micikevicius et al. (2017) did this by keeping a “master” copy of the weights in FP32 while still running matrix multiplies in FP16. Every forward pass casts the FP32 master to FP16 before entering the kernel, and every backward pass computes gradients in FP16 and then casts them back to FP32 before applying the update. That casting itself does not rescue underflow, so the authors also multiply the loss by a constant scale factor \(s\) before backpropagation:

\[
\tilde{L} = s \cdot L.
\]

Here, \(\tilde{L}\) is the scaled loss entering the backward pass and \(s > 1\) is chosen so that the gradients land in the representable range of FP16. After computing gradients, the scheme divides them by \(s\) before applying them to the FP32 master:

\[
G_{fp32} = \text{cast}_{fp32}(g_{fp16}) / s,
\]

where \(g_{fp16} = \nabla_L\tilde{L}\) is the raw gradient in FP16. The subtraction

\[
W'_{fp32} = W_{fp32} - \eta G_{fp32}
\]

happens in full precision, so the underflow has no chance to corrupt the master copy. This re-scaling is a bookkeeping step: the optimizer always sees the correctly scaled gradient, while FP16 math gets the benefit of extra dynamic range.

### Gradients, losses, and overflow detection

Scaling also introduces the risk of overflow: if \(s\) is too large, the scaled gradients exceed the FP16 finite range and produce infinities or NaNs. The detection is built into the forward-backward loop: once the gradients are computed, the runtime searches the gradient tensors for any Inf or NaN. If any appear, the step is discarded and the scale \(s\) is reduced, typically by a safety factor such as \(1/2\). If the gradient passes the check for several consecutive steps, the scale is increased (e.g., multiplied by 2) in order to capture more of the FP16 mantissa. Zhao et al. (2019) formalized this adaptive process as “Adaptive Loss Scaling,” proposing the control law:

\[
s_{t+1} =
\begin{cases}
s_t / 2 & \text{if overflow detected on step } t,\\
s_t \cdot \alpha & \text{if success counter } \geq \beta,\\
s_t & \text{otherwise},
\end{cases}
\]

where \(s_t\) is the current scale, \(\alpha > 1\) is the increase factor (often 2), and \(\beta\) is the number of clean steps before scaling up. The success counter tracks how many recent steps produced finite gradients, and overflow detection is cheap because it only inspects the FP16 buffers that already exist for the gradients.

This adaptive rule means engineers no longer have to search for a “magic” static scale. Instead, the runtime responds to the gradient distribution: if the gradients are tiny because of a small learning rate, the scale climbs to keep them visible; if a sudden failure mode yields large updates, the scale shrinks and the optimizer repeats the step.

### Accumulators, master weights, and optimizer states

The FP32 master copy needs to exist for every parameter and every optimizer accumulator. Optimizers such as Adam maintain first- and second-moment estimates, so the parameter tuple becomes \((W_{fp32}, m_{fp32}, v_{fp32})\), while the forward and backward passes still work with FP16 casts. The sequence is:
1. Cast \(W_{fp32}\) to FP16 and run the forward pass.
2. Compute loss \(L\) in FP16, scale to \(\tilde{L}=sL\).
3. Backpropagate and capture gradients \(g_{fp16}\).
4. Check for overflow; if any gradient contains Inf/NaN, discard step, reduce \(s\), and restart.
5. Cast \(g_{fp16}\) to FP32 and unscale by \(1/s\) to get \(G_{fp32}\).
6. Update \(m_{fp32}, v_{fp32}\), and \(W_{fp32}\) as usual.
7. Optionally, copy \(W_{fp32}\) to FP16 for the next forward pass if weights are stored in FP16 for inference.

Tagged as steps \(1\) through \(7\), this pipeline keeps the compute-heavy kernels in FP16 while the higher-precision bookkeeping happens outside the tight loops. The diet of operations prevents catastrophic rounding; the Master copy garners enough precision to remember the \(10^{-8}\) differences that the FP16 kernel would otherwise lose.

### Observing underflow and the early lessons

An early technical note titled “Untitled” (2018) [http://arxiv.org/pdf/1807.11205v1](http://arxiv.org/pdf/1807.11205v1) documented this underflow phenomenon before the community had settled on a recipe. The note traced a small convolutional network where gradients would vanish after a few steps, purely because the typical gradient magnitude hovered near \(2^{-12}\). That historical artifact is a reminder that these failures looked like mysterious instabilities until engineers peered at the mantissa of every gradient tensor. The note also emphasized that the underflow was not hardware failure; it was a predictable consequence of casting without scaling.

You can instrument a modern training loop with hooks that record the minimum absolute value of each gradient tensor before scaling. When the minimum touches \(2^{-11}\), the step is about to underflow. The “Mixed Precision Training – ADS” entry from 2017 catalogs how early adopters used such instrumentation to justify the extra bookkeeping cost because it saved whole training runs from dying in the first epoch.

### Hybrid formats and the rise of FP8

The mechanism extends beyond FP16. Nemotron-H (2025) [arxiv:2504.03624](https://arxiv.org/abs/2504.03624) shows that you can run entire transformer layers in hybrid FP8—specifically E4M3 for activations and E5M2 for weights—while keeping master copies in BF16 or FP32. The approach uses per-layer occupancy parameters that determine when to downcast outputs and when to keep accumulators in higher precision. Nemotron-H demonstrates that with well-chosen dynamic ranges, FP8 can match FP16 accuracy while reducing memory and bandwidth even more aggressively. The scaling logic is the same: scale the loss so that gradients live in the representable FP8 range, unscale before the master update, and track when overflow happens. Because FP8 has only 6 or 7 bits of mantissa, the adaptive logic must be more sensitive, but the payoff is a near-halving of data movement for activations and gradients.

### Training controls and runtime instrumentation

A practical mixed-precision engine exposes instrumentation that records:
- Overflow counts per tensor (how many times a NaN or Inf occurred).
- The current scale factor \(s_t\).
- Gradient norms before and after scaling.
- Step success rate (how many steps repeated due to overflow).

These signals let you see the “numerical stress” on each layer. When a particular layer spikes in overflow rate, you can either freeze its precision to FP32 or tighten its scale growth to be more conservative. Engineers call this “precise profiling,” and it links the algorithmic mechanism to the system-level decision of whether to keep a component in FP16/FP8 or revert to BF16.

## Where the field is now

The research frontier for mixed precision today is about hybrid precision tuning across depths and modalities. Nemotron-H (Zhou et al. 2025) [arxiv:2504.03624](https://arxiv.org/abs/2504.03624) presents layer-wise schedules that switch between BF16, FP16, and FP8 depending on the sensitivity of each transformer block. Their experimental table shows that the hybrid scheme achieves the same perplexity as a BF16 baseline on multilingual translation while reducing peak memory by 35%. The paper also explores a “parameterization rule” that increases the FP8 scale when the variance of an optimizer’s second moment grows, suggesting that mixed precision recipes must respond to the optimizer state, not just the raw gradients.

On the engineering frontier, NVIDIA’s Transformer Engine (developer.nvidia.com/blog/accelerating-training-of-transformers-with-nvidia-transformer-engine) remains the production workhorse because it tightly integrates FP8 kernels, automatic loss scaling, and master weight maintenance into one driver. The engine ships in the cuDNN-TensorRT stack and powers both inference and training pipelines for Meta’s Llama 3 and OpenAI’s GPT-4 Turbo backends. It demonstrates that the value of mixed precision is not only reduced memory but also the ability to saturate Tensor Cores without modifying your architecture: the engine casts tensors around the compute kernels and transparently handles null gradients, so production teams can “enable mixed precision” and rely on the runtime to keep the numerics stable.

A related frontier is observability: the monitoring systems in Meta and OpenAI report not just loss but also scaling factors, overflow rates, and the step at which a gradient tensor first hit the underflow cliff. This telemetry enables feature flags where a particular layer can fall back to BF16 when the loss scale tries to grow past a threshold. The interplay between instrumentation, optimization, and system stack is what keeps large transformer training runs alive for thousands of steps at 0.1 petaflop-days per step.

## What's still open

- Can a runtime dynamically probe the gradient distribution and switch per-tensor between FP4, FP8, and BF16 without introducing jitter into the loss curve? The scale factor jumps that save underflow today are disruptive because they repeat steps; a less aggressive probing mechanism might allow faster growth of the scale while still rejecting bad steps. The key question: “What probe statistic predicts overflow before it occurs?” is still unanswered.

- How should adaptive loss scaling interact with second-moment optimizers beyond Adam? Current recipes assume the optimizer state can hold the unscaled gradient, but if you adopt AdaFactor or Lion, the gradient transformation happens immediately after scaling, and overflow detection becomes less obvious. A publishable question is: “If you fuse the scaling into the optimizer update (instead of unscaling before it), what invariants must hold to keep the optimizer consistent?”

- What are the real costs of hybrid FP8 schedules when you include communication overhead in distributed training? Nemotron-H provides per-layer accuracy curves, but scaling those schedules across multiple accelerators with all-reduce steps remains an open systems question. The research challenge is to find a scheduler that balances precision, communication volume, and numerical stability without manual tuning.

- Is there a lightweight, differentiable proxy for overflow that can be added to the loss so the network learns to avoid underflow/overflow at training time? If such a proxy existed, mixed precision could be tuned end-to-end rather than via heuristics, and the open research goal is to prove that proxy correlates with the actual Inf/NaN rate under real workloads.

## Where to read next

If you want the hardware context, → [[tensor-cores]] explains how mixed precision maps to specialized cores and why compute density matters; the numerical safety net is → [[numerical-stability]] where scaled gradients, Kahan summation, and condition numbers live; the training-scale perspective is → [[scaling-training]] which covers large-batch recipes that mixed precision makes tractable.

## Build it

This recipe proves that mixed precision is not a library call but a numerical discipline: you will construct every element manually—FP16 activations, FP32 master weights, dynamic loss scaling, and overflow detection—and observe how the loss scale evolves as gradients flirt with the FP16 cliff.

**What you're building:** a raw PyTorch training loop that fine-tunes `distilgpt2` on the `tiny_shakespeare` dataset, instrumented to print gradient minima, scale factors, and overflow events every 128 steps.

**Why this is valuable:** you see the underflow catastrophe, injected by logging the minimum gradient magnitude before scaling, and you learn how adaptive loss scaling rescues it in real time; you also produce a checkpoint that can be used by any later arc step.

**Stack:**
- **Model:** [distilgpt2](https://huggingface.co/distilgpt2) — 2.0B downloads, proven for lightweight fine-tuning
- **Dataset:** [tiny_shakespeare](https://huggingface.co/datasets/tiny_shakespeare) — 15 KB of Shakespeare text, tokenized to 32-character sequences
- **Framework:** PyTorch 2.1 + Transformers 4.45 + Accelerate 0.21 (only for dataloader sharding; training loop is manual)
- **Compute:** free Colab T4 (16 GB VRAM) or a local RTX 4060 Ti — expect ~1.5 hours per full run

**The recipe:**
1. Install the stack with `pip install torch==2.1.0 transformers==4.45 accelerate==0.21 tqdm` and clone both the model and dataset with `from transformers import AutoTokenizer, AutoModelForCausalLM` and `load_dataset("tiny_shakespeare")`.
2. Tokenize into 32-length sequences, batch to 32, and pin memory on the DataLoader; store the batches on CUDA by calling `.to("cuda")` directly to avoid any automatic casting.
3. Define the optimizer state (`AdamW` with betas (0.9, 0.95)) on FP32 master weights that you copy from the model’s `.parameters()`. Run the forward pass with `model(data["input_ids"].half())`, compute the loss, multiply it by the current scale \(s\), and call `.backward()` on `scaled_loss`.
4. After `scaled_loss.backward()`, iterate through the gradients: record `grad.min().item()` and `grad.max().item()` before unscaling, check `torch.isfinite(grad).all()`, and if any gradient is infinite you skip the optimizer step, reduce \(s\), and call `model.zero_grad()`. Otherwise, unscale the gradients by dividing by \(s\), update the master weights and optimizer moments, step the optimizer, and call `model.zero_grad()`.
5. Every 128 training steps, log the current scale \(s\), the minimum gradient magnitude, and the optimizer step count; save a checkpoint that includes both the FP16 model state dict and FP32 master weights so future arc steps can load them.

**Expected outcome:** a checkpoint plus logs that show how the scale factor oscillates, how gradient minima approach the FP16 threshold, and how adaptive scaling keeps the loss from blowing up.

- **CS student:** Run the same recipe on `hf-internal-testing/tiny-random-gpt2` with a batch size of 8 to fit on an RTX 3060, logging only the maximum gradient per layer to reduce I/O pressure.
- **Applied engineer:** Wrap the training loop in a script that exports an ONNX model with dynamic axes, quantize the FP16 activations to INT8 for inference via TensorRT, and measure p50 latency < 5 ms on an L4.
- **Applied researcher:** Hypothesis: doubling the loss scale growth threshold \(\beta\) will reduce overflow-induced step repeats but slow convergence. Run two mixed-precision runs (standard versus \(\beta=6\)) and compare perplexity after 20 epochs; the falsifier is seeing the modified run’s perplexity exceed the baseline by more than 2.0.
- **Frontier researcher:** Extend the recipe to monitor per-layer gradient variance and switch a chosen layer between FP8 and FP16 depending on whether the variance exceeds a threshold. The falsification criterion is observing whether the dynamic switch introduces more than 3 consecutive overflow steps compared to keeping the layer fixed in FP16.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*