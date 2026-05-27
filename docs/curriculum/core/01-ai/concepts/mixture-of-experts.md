---
title: Mixture of experts
slug: mixture-of-experts
layer: core
subject: 01-ai
page_type: concept
state: drafted
authors_anchored: [hinton, jordan, shazeer, dai]
feeds_de_pillar: []
mvb_personas: [cs-student, applied-engineer, applied-researcher, frontier-researcher]
prereqs: [mlp, conditional-computation, transformers, optimization-techniques]
tags: [sparsity, routing, conditional-computation, transformers, scale, efficiency]
updated: 2024-11-01
has_mvb: true
---

# Mixture of experts

Imagine a hospital where every arriving patient—scraped knee, urgent stroke, the whole gamut—must walk through every department, every diagnostic machine, every specialist, and only after that receives a treatment plan. Dense neural networks act the same way: whatever the input is, it is forced through every neuron, every parameter, every layer, no matter how trivial the computation needed to resolve it. Mixture-of-experts (MoE) drops a triage desk at the network’s entrance. The gate asks “which specialists actually matter for this case?” and routes the input to two or three expert subnetworks, leaving the rest asleep until some other token pulls them awake. The result is a parameter budget measured in the hundreds of billions while the FLOP count per token stays within a single dense layer’s footprint. This page shows how MoE gates learn the routing policy, how auxiliary losses keep every expert useful, and what it takes to implement a Top-2 gated MoE layer in PyTorch so the inference cost stays steady even as capacity rockets.

## The territory

Modern language, vision, and multi-modal models keep increasing parameters because scale keeps buying capability. But every token still earns its way through the whole network, which makes deployment more expensive than the training gains merit. That inefficiency led early researchers back in Jacobs et al. (1991) “Making associative learning competitive” [https://people.eecs.berkeley.edu/~jordan/papers/mixtures-of-experts.pdf] to propose a “divide-and-conquer” structure: a gate predicts a responsibility vector over predictor modules, and each module specializes on the partition it receives. The conditional computation idea—where computation is a function of the input rather than an immutable pipeline—was later argued by Bengio et al. (2013) “Conditional computation in neural networks” [http://arxiv.org/pdf/1312.4314v3] as the lever that scales capacity without linear compute growth. Hinton et al. (1993) “Hierarchical mixtures of experts and the EM algorithm” [https://www.cs.toronto.edu/~hinton/absps/hme.pdf] made specialization learnable through probabilistic layering and EM-style updates, showing that gating can itself be trained rather than hand-designing experts for every case. MoE inherits these intuitions inside Transformers: each token looks up a sparse subset of expert MLPs chosen by a learned gate, so the total capacity is the sum of all experts while the active FLOPs equal only the few experts answering each call. How does the gate commit to two or three experts without collapsing to a single specialist, and how is this routing implemented cheaply? The mechanism is best understood by starting from the gate’s math and the Top-2 sparse dispatch it uses.

## How it works

The Mixture-of-Experts block splits into two cooperating pieces: the gate that computes attention-like weights, and the experts—usually small MLPs—who do the heavy lifting. Start with an input token embedding \(x \in \mathbb{R}^d\). A gating network applies a linear projection \(W_g x + b_g\) to produce logits \(l \in \mathbb{R}^E\), where \(E\) is the number of experts. A naive softmax \(p = \text{softmax}(l)\) gives a probability vector over experts, but the trick that makes modern MoEs practical is to activate only the top \(k\) experts per token (typically \(k=2\)). This keeps the per-token compute proportional to \(k\) instead of \(E\). The clean gating formulation is

\[
g(x) = \text{TopK}\bigl(\text{softmax}(l + n)\bigr)
\]

where \(n \sim \mathcal{N}(0, \sigma^2 I)\) is noise added per token before top-\(k\) selection to nudge the gate away from deterministically favoring a single expert.
In this equation, \(x\) is the token embedding entering the block, \(W_g \in \mathbb{R}^{E \times d}\) and \(b_g \in \mathbb{R}^E\) are gating parameters, \(\text{TopK}(\cdot)\) zeroes out all but the \(k\) highest entries, and \(\sigma\) is a tunable scale that ensures exploration.

The gated dispatch and combine steps mimic attention. After computing \(g(x)\), the dispatcher scatters \(x\) to each expert \(e\) that received a non-zero gate score, forming \(k\) independent mini-batches. Each expert \(F_e\) (an MLP with its own parameters) processes only the tokens routed to it:

\[
y_e = F_e(x)
\]

and the block’s output is the weighted sum of these expert outputs:

\[
y = \sum_{e \in \text{TopK}} g_e(x) \cdot y_e
\]

where \(g_e(x)\) is the gate score for expert \(e\). Because only \(k\) experts are active per token, the FLOPs per token are \(k \cdot \text{MLP-FLOPs}\), regardless of \(E\). For example, with \(E=256\) and \(k=2\), each token uses \(2 \times\) the FLOPs of a single expert rather than \(256 \times\). The inactive experts keep their weights but pay no compute price.

Noisy top-k gating is the route that keeps gradients flowing through the discrete expert selection. Without noise, the gate would hard-select one expert and the gradient would vanish for the rest. During backprop, the gate’s gradient is computed via the softmax before the top-k mask, and the noise term \(n\) is sampled once per token at forward pass, so the gate still learns which experts to prefer on average.

Routing collapse—where every token chooses the same expert—looks small in the loss but fatal for capacity. Shazeer et al. (2017) “Outrageously large neural networks” [https://arxiv.org/abs/1701.06538] introduced auxiliary balancing losses for this reason. The importance of expert \(e\) is the aggregate gate mass:

\[
\text{Importance}_e = \sum_{x \in \mathcal{B}} g_e(x)
\]

and the load is the expected number of tokens dispatched to \(e\). The load-balancing loss pairs these two terms:

\[
\mathcal{L}_{\text{balance}} = \lambda \cdot E \cdot \sum_{e=1}^E \text{Importance}_e \cdot \text{Load}_e
\]

where \(\lambda\) scales the auxiliary objective relative to the downstream loss, \(E\) is the expert count, and the product encourages both importance and load to be uniform across experts. This loss keeps the gating network from collapsing and ensures idle experts still receive traffic. In practice, Shazeer et al. found \(\lambda = 0.01\) works well, and the expert-local losses are combined with the main classification or language-modeling loss.

Training also requires efficient dispatch kernels. The dispatcher reshapes the per-token gate scores into a sparse matrix that maps the mini-batch axes to expert-local mini-batches. During the backward pass, gradients propagate through the same sparse gather/scatter path, preserving the Top-2 structure. When MoE sits inside a Transformer layer, the MoE block replaces the standard feed-forward network (FFN) while keeping the multi-head attention and residual structure intact; the rest of the model treats the MoE output like any other feed-forward output.

Modern MoE stacks add refinements on top of this core. DeepSeekMoE (Dai et al. 2024) [https://arxiv.org/abs/2401.0606] introduces a two-tiered structure of “shared experts” that handle common patterns and “dedicated specialists” that handle hard, long-tail features. A routing head first decides whether to use shared or dedicated experts, which prevents redundant parameter allocation when multiple regions of the data share the same computation. That paper also segments experts according to finer-grained signal (e.g., syntax vs. semantics) and trains the segmentation with contrastive regularizers so that each expert only competes with a limited subset of peers. The resulting gating map has wider coverage: with 48 dedicated experts and 8 shared ones, an 80B-parameter DeepSeekMoE model still routes only two experts per token, keeping per-token FLOPs similar to a 12B dense model while providing the expressivity of an 80B parameter budget.

In addition to gating and load balancing, MoEs must manage communication between experts. Multi-expert communication occurs either through all-gather operations across expert ranks (common in TPU mesh) or through batched dispatch kernels on GPUs. When training across devices, each expert typically occupies a separate slice of device memory, and the dispatcher handles cross-device transfers, which is why MoE training frameworks often expose fused kernels that flatten the Top-2 gate into contiguous dispatch/load operations.

In summary, MoE works because: (1) a lightweight gate selects \(k\) experts via noisy top-k softmax; (2) the dispatch operation routes each token to those experts and returns a weighted sum; and (3) auxiliary losses keep the gate from collapsing while per-token compute stays constant. The gating losses therefore anchor the MoE block’s behavior, and implementing them efficiently is the crux of using MoEs in practice.

## Where the field is now

Research frontier: DeepSeekMoE (Dai et al. 2024) [https://arxiv.org/abs/2401.0606] pushes MoE scaling by explicitly carving experts into shared and dedicated roles. Their reported results on open benchmarks show that a 54B-dedicated-parameter variant beats the same-size dense baseline by 2.4 points on MMLU while operating with only two active experts per token, so the throughput matches a 12B dense model even though the parameter budget is four times larger. Latency-sensitive tasks like long-context classification benefit from the shared experts’ ability to diffuse common computations, meaning the model can reduce tail perplexity by ~3% on C4 without re-training the gate.

Engineering frontier: Google Research’s Mixture-of-Experts deployments, first described for the GLaM family (Du et al. 2021) [https://ai.google/research/pubs/pub50629], built a 1.2-trillion-parameter model that routes to two experts per token so inference cost equals that of a 1.7-billion-parameter dense model, achieving the same latency as a much smaller Transformer while supporting massive parameter counts. The engineering blog reports that Google deploys these MoE models inside search and Gemini-style assistants to trade off accuracy and compute in production, scaling to TPU v4 pods by placing each expert on a separate chip slice and letting the dispatcher handle fast transfers.

Combined, the research and engineering frontiers show the MoE story today: architectural innovation keeps per-token compute low while expert budgets expand, and TPU- and GPU-based service stacks manage the sparse communication patterns needed for production latency targets.

## What's still open

1. Can routing collapse be avoided without auxiliary load-balancing losses by explicitly modeling routing entropy? Current loss terms need careful tuning to avoid oscillation; a principled entropy regularizer that keeps the gate spruce without extra hyperparameters would make MoEs easier to deploy.

2. How can dedicated experts learn curriculum-style access patterns without manual expert segmentation? DeepSeekMoE uses contrastive losses to separate signals, but a data-driven emergent segmentation that organizes experts by syntactic and semantic properties would remove reliance on hand-crafted shared/dedicated splits.

3. Is it possible to compress MoEs for on-device inference by distilling the sparse dispatch into a dense block while preserving the “divide-and-conquer” behavior? Distillation today either replicates the gating or collapses back to a dense model; we need a method that preserves expert specialization in a single inference pass.

4. What scheduling policies can co-optimize throughput and energy on GPUs where the dispatcher’s scatter/gather ops compete with tensor cores? A routing-aware scheduler that batches tokens for shared experts while keeping memory copies minimal could unlock MoE deployment on latency-critical platforms beyond TPUs.

## Where to read next

If you want the probabilistic foundation that explains why the gate can be trained as part of a variational bound, → [[conditional-computation]] digs into the EM and likelihood-ratio interpretations. The engineering counterpart is → [[transformers]] for how MoE layers replace Transformer feed-forward blocks without reworking the attention stack. For broader scalability concerns, → [[sparsity-and-pruning]] contrasts MoE’s conditional activation with structured pruning’s static sparsity.

## Build it

MoE inference cost looks like a dense MLP, but the parameter budget is multiplied by the expert count; this build proves that Top-2 gating keeps the runtime per token stable while exposing hundreds of millions of expert parameters. You will replace a Transformer MLP block with a custom Top-2 MoE layer, train it on a lightweight sequence classification dataset, and log the expert load statistics to confirm the balancing loss is doing its job.

**What you're building:** a Transformer encoder with a Top-2 gated MoE block that classifies sequences from the GLUE/SST-2 dataset while logging expert loads and validation loss.

**Why this is valuable:** It forces you to implement dispatch + combine kernels, noisy gating, and the balancing loss, which are the pieces that let MoEs scale without collapsing.

**Stack:**
- **Model:** `google/bert_uncased_L-4_H-256_A-4` (over 300k downloads) — lightweight Transformer encoder you can fork instead of building a tokenizer.
- **Dataset:** `glue/sst2` — binary sentiment classification dataset with 67k training examples; each sentence is treated as a token sequence.
- **Framework:** `PyTorch 2.1` with `torch.distributed.fsdp` for parallel expert storage; use `transformers 4.38` for the base encoder.
- **Compute:** Google Colab T4 (16 GB VRAM), ~90 minutes for 3 epochs with gradient checkpointing and mixed precision.

**The recipe:**
1. `pip install torch==2.1.0 torchvision torchaudio transformers datasets accelerate --upgrade && pip install einops tokenizers`; load GLUE/SST-2 via `datasets.load_dataset("glue", "sst2")` and tokenize with the BERT tokenizer at 128 tokens.
2. Build a Top-2 MoE module that takes the FFN input, applies a gating linear layer to produce logits for \(E=64\) experts, adds Gaussian noise (std = 1.0) to the logits, selects the top 2 experts, dispatches the tokens, and recombines the expert outputs with the gate scores.
3. Train: fine-tune for 3 epochs with batch size 32, learning rate \(5 \times 10^{-5}\), AdamW weight decay 0.01, gradient clipping 1.0, warmup 10% of steps, and include the balancing loss \(0.01 \cdot E \cdot \sum_e \text{Importance}_e \cdot \text{Load}_e\); track training/validation loss curves and expert load means.
4. Evaluate: report accuracy on the SST-2 validation split plus the uniformity of gate distribution (coefficient of variation of \(\text{Importance}_e\)); expect accuracy ≥87% with load imbalance CV <0.25.
5. What you now have: a checkpoint of the MoE-augmented Transformer, expert statistics plots, and a script that logs gate distributions each gradient step.

**Expected outcome:** A Top-2 gated MoE Transformer block ready for downstream SST-2 inference, with recorded expert utilization demonstrating the balancing loss in action.

- **CS student:** Run the same recipe on an RTX 4070 with `batch_size 16` and replace SST-2 with the smaller `sst2-small` split; this keeps everything on one consumer GPU while still forcing you to log gate statistics.
- **Applied engineer:** After training, quantize the MoE block with ONNX Runtime + QDQ and serve it through vLLM’s custom MoE dispatcher, targeting p50 latency <120 ms on a single T4 while measuring p95 due to dispatch jitter.
- **Applied researcher:** Hypothesize that Top-1 gating with per-layer load loss matches Top-2 gating; run the exact same training loop with \(k=1\) and compare validation accuracy and load CV to test whether the extra expert helps BLEU-like metrics.
- **Frontier researcher:** Probe the open question of load balancing without auxiliary loss by replacing the loss with an entropy regularizer on the gate distribution and measuring whether routing collapse (all tokens to a single expert) occurs on SST-2 and on the auxiliary SST-2 hard subset; log the same metrics as the main recipe to falsify whether entropy alone suffices.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*