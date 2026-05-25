---
title: Data Parallelism & Distributed Training
track: 09-algorithms-systems-ai
tags: [data-parallelism, distributed-training, allreduce, ddp, zero, fsdp]
depth: applied
prereqs: [backpropagation, gpu-architecture]
updated: 2026-05-25
---

# Data Parallelism & Distributed Training
> **TL;DR:** Training a model across multiple GPUs by splitting the data — each device holds a full model copy, processes a data shard, and synchronizes gradients — the simplest and most widely used form of distributed training.

> 🚧 Agent-generated content pending. See [track index](index.md) for context.

## What it is
Data parallelism replicates the model across N GPUs. Each GPU processes a mini-batch shard, computes gradients, and then synchronizes gradients across all GPUs via AllReduce before updating. This scales throughput linearly (ideally) without changing the optimization dynamics. For models too large to fit on one GPU, ZeRO and FSDP extend data parallelism by sharding optimizer states, gradients, and parameters.

## Core concepts
- **AllReduce** — collective communication: sum gradients across all GPUs, broadcast result
- **DDP** — PyTorch DistributedDataParallel; ring-AllReduce, overlaps compute and communication
- **ZeRO Stage 1/2/3** — shard optimizer states / gradients / parameters across GPUs
- **FSDP** — PyTorch Fully Sharded Data Parallel; production ZeRO Stage 3
- **Gradient accumulation** — accumulate gradients over multiple steps before sync; increases effective batch size
- **Batch size scaling** — linear scaling rule: scale LR ∝ batch size (Goyal et al.)

## Essential reading
| Paper | Year | Authors | Why essential |
|---|---|---|---|
| [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/abs/1910.02054) | 2019 | Rajbhandari et al. | ZeRO — how to actually train very large models |
| [Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour](https://arxiv.org/abs/1706.02677) | 2017 | Goyal et al. | Linear scaling rule for batch sizes |

## Current SotA
> *Updated: 2026-05-25*
DeepSpeed + FSDP are the standard libraries. ZeRO++ reduces communication overhead. For very large training runs (1000+ GPUs), 3D parallelism (data + tensor + pipeline) is required. Most frontier models use all three forms of parallelism simultaneously.

## Connected topics
- [[model-parallelism]] — when model doesn't fit on one GPU even with ZeRO
- [[mixed-precision]] — FP16/BF16 reduces memory and communication bandwidth
- [[gradient-checkpointing]] — trades compute for memory within each device

## Further reading
- [Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism](https://arxiv.org/abs/1909.08053) — Shoeybi et al. 2019
