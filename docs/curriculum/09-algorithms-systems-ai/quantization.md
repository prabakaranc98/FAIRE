---
title: Quantization
track: 09-algorithms-systems-ai
tags: [quantization, int8, int4, post-training-quantization, qlora, efficiency]
depth: applied
prereqs: []
updated: 2026-05-25
---

# Quantization
> **TL;DR:** Reducing the precision of model weights and activations (FP32 → INT8 → INT4) to shrink memory footprint and accelerate inference — making large models deployable on consumer hardware.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Neural network weights are stored in FP32 (32 bits) by default. Quantization maps these to lower-precision formats (FP16, BF16, INT8, INT4) with minimal accuracy loss. A 70B model in FP32 requires ~280GB of memory; in INT4, it fits in ~35GB — deployable on a single GPU. GPTQ and AWQ are the leading post-training quantization methods for LLMs.

## Core concepts
- **Post-training quantization (PTQ)** — quantize after training; no retraining required
- **Quantization-aware training (QAT)** — simulate low precision during training; better quality
- **Weight-only vs. activation quantization** — quantize weights only, or weights + activations
- **GPTQ** — approximation using second-order optimization; weights quantized layer by layer
- **AWQ** — activation-aware weight quantization; preserves weights important for salient activations
- **QLoRA** — fine-tune quantized (INT4) base model with LoRA adapters in FP16

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [GPTQ: Accurate Post-Training Quantization for Generative Pre-trained Transformers](https://arxiv.org/abs/2210.17323) | 2022 | Frantar et al. | GPTQ — production INT4 quantization for LLMs |
| [QLoRA: Efficient Finetuning of Quantized LLMs](https://arxiv.org/abs/2305.14314) | 2023 | Dettmers et al. | QLoRA — fine-tune 65B in 48GB with INT4 |

## Current SotA
> *Updated: 2026-05-25*
AWQ (Lin et al. 2023) currently matches or outperforms GPTQ. FP8 (supported on H100) is increasingly used for training and inference — better than INT8 with less overhead than BF16. The frontier is W4A8 (INT4 weights, INT8 activations) for maximum throughput.

## Connected topics
- [[peft]] — QLoRA uses quantization + LoRA for memory-efficient fine-tuning
- [[inference-serving]] — quantization dramatically improves inference throughput
- [[kv-cache]] — KV cache can also be quantized (KVQuant)

## Further reading
- [A Survey of Quantization Methods for Efficient Neural Network Inference](https://arxiv.org/abs/2103.13630) — Gholami et al. 2021
