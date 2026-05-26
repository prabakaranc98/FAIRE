```yaml
---
title: Quantization
track: 09-algorithms-systems-ai
tags: [quantization, model compression, deep learning, inference optimization]
depth: applied
prereqs: [deep-learning, neural-networks]
updated: 2024-10-26
has_mvb: true
---
# Quantization
> **TL;DR:** Quantization reduces the memory footprint and computational cost of neural networks by representing weights and activations with lower precision, enabling efficient deployment on resource-constrained devices.

---

## For your reader type

| I am... | Start here | Goal |
|---|---|---|
| MS/applied practitioner | [Key algorithms](#key-algorithms--techniques) → [MVB](#minimum-valuable-build) | Build something that works |
| Curious generalist | [What it is](#what-it-is) → [Why it matters](#why-it-matters-at-the-frontier) | Build intuition |
| Math/theory student | [Core concepts](#core-concepts) → [Mathematical foundations](#mathematical-foundations) | Understand the mechanics |
| Researcher / frontier | [Current SotA](#current-sota) → [Seminal papers](#seminal-papers--test-of-time) | Know where the open problems are |

---

## What it is
Imagine you're trying to run a cutting-edge AI model on your laptop, but it's too big to fit in memory. Or, picture a massive language model that's costing a fortune to run in the cloud. Quantization is the key to unlocking these problems, allowing you to shrink the size of these models without sacrificing too much performance. This technique is like compressing a high-resolution image to make it easier to share online.

Quantization is a model compression technique that reduces the precision of the weights and activations in a neural network. Instead of using 32-bit floating-point numbers (FP32), quantization uses lower-bit representations like 8-bit integers (INT8) or even 4-bit integers (INT4). This reduction in bit-width directly translates to smaller model sizes and faster inference times, as the computations require less memory bandwidth and fewer operations.

The core idea is to map a continuous range of floating-point values to a discrete set of integer values. This mapping introduces some approximation error, but by carefully choosing the quantization scheme and calibration data, the impact on model accuracy can be minimized. Quantization is essential for deploying large models on edge devices, mobile phones, and other resource-constrained environments.

## Why it matters at the frontier
Quantization is a critical technique for pushing the boundaries of AI deployment. As models grow larger and more complex, their computational and memory demands increase exponentially. Quantization offers a practical solution to this problem, enabling researchers and engineers to deploy state-of-the-art models on a wider range of hardware platforms.

The frontier research focuses on developing novel quantization methods that minimize the accuracy loss associated with low-bit representations. For example, how can we develop quantization methods that maintain agentic capabilities in compressed LLMs, specifically focusing on workflow generation, tool use, and real-world application accuracy, while achieving higher compression ratios? Addressing this question is crucial for enabling the widespread adoption of AI in real-world applications.

## Core concepts
- **Precision** — The number of bits used to represent a floating-point or integer value; lower precision reduces memory usage but can impact accuracy.
- **Quantization scheme** — The method used to map floating-point values to integer values, such as uniform quantization, non-uniform quantization, or quantization-aware training.
- **Zero point** — The value that maps to zero in the quantized representation, used to ensure that zero is exactly representable.
- **Scale factor** — A scaling factor used to map the range of floating-point values to the range of integer values.
- **Calibration** — The process of selecting a representative dataset to determine the optimal quantization parameters (zero point and scale factor) for each layer.
- **Post-training quantization (PTQ)** — A quantization technique applied after the model has been trained, without requiring retraining.
- **Quantization-aware training (QAT)** — A quantization technique where the model is trained with quantization in mind, allowing the model to adapt to the lower-precision representation.

## Mathematical foundations
The quantization process maps a floating-point value to a discrete integer value. The general equation for quantization is:
\[
\text{Quantized}(x) = \text{round}\left(\frac{x - \text{zero\_point}}{\text{scale}}\right)
\]
where \(x\) is the original floating-point value, \(\text{zero\_point}\) is the zero-point value, and \(\text{scale}\) is the scaling factor. This equation represents the core quantization process, where a floating-point value is mapped to a discrete integer value.

Additive quantization approximates the original weight matrix by the sum of multiple codebooks:
\[
\text{AQLM}(W) = \sum_{i=1}^{C} W_i
\]
where \(W\) is the original weight matrix, \(C\) is the number of codebooks, and \(W_i\) is the weight matrix from each codebook. This equation represents the additive quantization process, where the original weight matrix is approximated by the sum of multiple codebooks.

Delta sparsification decomposes the weight matrix using SVD:
\[
\text{ImPart}(S) = \text{SVD}(W)
\]
where \(S\) is the singular value matrix, \(W\) is the weight matrix. This equation represents the delta sparsification process, where the weight matrix is decomposed using SVD.

## Key algorithms / techniques
- **Post-Training Quantization (PTQ)** — Quantizes a pre-trained model without further training, offering a quick and easy way to reduce model size and improve inference speed.
- **Quantization-Aware Training (QAT)** — Trains the model with simulated quantization during training, allowing the model to adapt to the lower-precision representation and minimize accuracy loss.
- **Additive Quantization (AQLM)** — Approximates the original weight matrix by the sum of multiple codebooks, achieving state-of-the-art results at very low bit counts (Li et al., 2024).
- **Dynamic Quantization** — Determines the quantization parameters (scale and zero point) dynamically for each layer or activation tensor, adapting to the specific range of values.

## Essential reading

| Paper | Year | Authors | Why essential |
|---|---|---|---|
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | Dao et al. | Demonstrates how to optimize attention mechanisms for quantized models, improving performance. |
| Extreme Compression of Large Language Models via Additive Quantization | 2024 | Li et al. | Introduces AQLM, a novel quantization method that pushes the boundaries of low-bit quantization, achieving state-of-the-art results in extreme compression regimes. |
| GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance | 2025 | Tim et al. | Introduces a novel quantization approach that integrates gradient information from the end loss into the quantization objective while preserving cross-weight dependencies within output channels. |

## Seminal papers & test-of-time

| Paper | Year | Key contribution |
|---|---|---|
| FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning | 2023 | Introduces FlashAttention-2, an optimized attention algorithm that leverages memory access patterns to achieve faster performance. |
| Extreme Compression of Large Language Models via Additive Quantization | 2024 | Introduces AQLM, a novel additive quantization method for extreme LLM compression, achieving state-of-the-art results at very low bit counts. |

## Current SotA
AQLM achieves state-of-the-art compression ratios while maintaining high accuracy on various language modeling benchmarks (Li et al., 2024). FlashAttention-3 achieves faster and more accurate attention computations through asynchrony and low-precision techniques (Shah et al., 2024). GuidedQuant integrates gradient information from the end loss into the quantization objective, preserving cross-weight dependencies within output channels (Tim et al., 2025).

## What's happening now
Research is actively exploring dynamic quantization techniques that adapt the bit-width per parameter based on its importance, as determined by loss gradients and activation statistics. This aims to achieve optimal performance-memory trade-offs across diverse model architectures and datasets.

Engineering efforts are focused on developing efficient quantization toolkits and libraries that seamlessly integrate with existing deep learning frameworks. These tools aim to simplify the quantization process and enable developers to easily deploy quantized models on various hardware platforms.

Open problems include maintaining agentic capabilities in compressed LLMs, specifically focusing on workflow generation, tool use, and real-world application accuracy, while achieving higher compression ratios. This requires developing quantization methods that preserve the essential knowledge and reasoning abilities of large language models.

## In production
- Amazon SageMaker AI — Post-training quantization (PTQ) of large language models (LLMs) — Enables scalable, cost-effective production deployment on Amazon SageMaker AI without retraining — [https://aws.amazon.com/blogs/machine-learning/accelerating-llm-inference-with-post-training-weight-and-activation-using-awq-and-gptq-on-amazon-sagemaker-ai/]
- Databricks — Serving quantized Llama2-70B-Chat models on NVIDIA H100 GPUs using the TensorRT-LLM stack — Real numbers for production deployment at scale — [https://www.databricks.com/blog/serving-quantized-llms-nvidia-h100-tensor-core-gpus]
- Databricks — Production-ready, scalable PEFT (LoRA) serving stack — Covers the full production lifecycle: auto-scaling, load balancing, multi-region deployment, health monitoring — [https://www.databricks.com/blog/fast-peft-serving-scale]
- NVIDIA — NVIDIA’s QAT Toolkit for TensorFlow 2 — Enables quantization-aware training to prepare models for TensorRT deployment on NVIDIA GPUs, focused on production-scale inference acceleration — [https://developer.nvidia.com/blog/accelerating-quantized-networks-with-qat-toolkit-and-tensorrt/]
- Google Cloud — Using quantization to optimize Vertex AI models — Quantization reduces the size of your model, which can improve the performance of your model, especially on edge devices — [https://cloud.google.com/vertex-ai/docs/general/quantization]
- Microsoft Azure — Quantization-aware training with Azure Machine Learning — Quantization-aware training (QAT) is a technique that improves the inference speed and reduces the model size of deep learning models — [https://learn.microsoft.com/en-us/azure/machine-learning/how-to-quantize-aware-training]
- OctoAI — OctoAI Text Generation Inference — Optimizes and deploys quantized LLMs, including Llama 2, Falcon, and more, with up to 10x throughput improvement — [https://octo.ai/blog/accelerate-llm-inference-with-quantization/]
- Intel — Intel® Neural Compressor — An open-source Python library that delivers unified interfaces to neural network compression technologies, such as quantization, pruning, and distillation, across different deep learning frameworks — [https://www.intel.com/content/www/us/en/developer/tools/neural-compressor/overview.html]
- Qualcomm — AI Model Efficiency Toolkit (AIMET) — A library that provides advanced quantization and compression techniques for on-device AI inference — [https://qaihub.qualcomm.com/aimet]

## Minimum Valuable Build
**What you're building:** You will quantize a pre-trained MistralAI model to INT4 using `bitsandbytes`.
**Why this build:** This demonstrates how to reduce the memory footprint of a large language model using quantization, enabling it to run on resource-constrained devices.
**Stack:** `transformers==4.37.0`, `torch==2.2.0`, `bitsandbytes==0.43.0`, `accelerate`, HuggingFace model ID: `meghanamakkapati/MistralAI_INT4_quantization`
**Estimated time:** 30 minutes

### The recipe

1. **Install the required libraries:**
```bash
pip install transformers==4.37.0 torch==2.2.0 bitsandbytes==0.43.0 accelerate
```

2. **Import necessary modules:**
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
```

3. **Load the pre-trained model and tokenizer:**
```python
model_name = "meghanamakkapati/MistralAI_INT4_quantization"
model = AutoModelForCausalLM.from_pretrained(model_name,
                                             device_map='auto',
                                             torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

4. **Create a simple inference function:**
```python
def generate_text(prompt, model, tokenizer):
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    outputs = model.generate(**inputs, max_length=50)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)
```

5. **Run inference with the quantized model:**
```python
prompt = "The capital of France is"
generated_text = generate_text(prompt, model, tokenizer)
print(generated_text)
```

### Expected output
The code should load the quantized MistralAI model and generate text based on the given prompt. The output will be a continuation of the prompt, such as "The capital of France is Paris." The model should run without encountering out-of-memory errors on a consumer GPU with ≤16GB of VRAM.

### Common failure modes
- **CUDA out of memory error:** → Reduce the `max_length` parameter in the `generate_text` function or try moving the model to the CPU.
- **Incorrect output:** → Verify that the correct model name and tokenizer are being used.

---

> *If this build worked for you — a ⭐ on [GitHub](https://github.com/prabakaranc98/FAIRE) is the only signal we collect.*

---

## Code & implementations
- **bitsandbytes:** [https://github.com/TimDettmers/bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- **transformers:** [https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)

## What comes next
- [[Pruning]] — another model compression technique that removes less important weights from the network.
- [[Knowledge Distillation]] — transfers knowledge from a large, complex model to a smaller, more efficient model.
- [[Sparsity]] — a technique that reduces the number of parameters in a model by setting some of them to zero, often used in conjunction with quantization.
- [[FlashAttention]] — Quantization can be combined with FlashAttention to further improve the efficiency of attention mechanisms, as demonstrated by FlashAttention-3 (Shah et al., 2024).
- [[Additive Quantization]] — AQLM, an additive quantization method, achieves state-of-the-art results at very low bit counts (Li et al., 2024).

## Connected topics
- [Data Parallelism](./data-parallelism.md) — Quantization can be used to reduce memory footprint in data-parallel training.
- [KV Cache](./kv-cache.md) — Quantization is often used to reduce the memory footprint of KV caches.
- [Optimization](../04-neural-networks-dl/optimization.md) — Quantization can affect the optimization process and requires careful tuning.
- [Backpropagation](../04-neural-networks-dl/backpropagation.md) — Quantization can impact the gradients calculated during backpropagation.
- [Transformer Architecture](../07-attention-memory-reasoning/transformer.md) — Quantization is commonly used to optimize Transformer models for efficiency.
- [Neural Tangent Kernel (NTK)](../15-ml-theory-foundations/ntk.md) — Quantization can affect the behavior and analysis of neural networks, including NTK.
- [[FlashAttention]] — Quantization can be combined with FlashAttention to further improve the efficiency of attention mechanisms.
- [[Sparsification]] — Importance-aware delta sparsification (ImPart) improves model compression and merging in LLMs (Li et al., 2025).

## Further reading
- Dao et al. (2023) — "FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning" — [https://arxiv.org/pdf/2307.08691.pdf] — This paper provides a detailed explanation of the FlashAttention-2 algorithm and its performance benefits.
- Li et al. (2024) — "Extreme Compression of Large Language Models via Additive Quantization" — [https://arxiv.org/abs/2401.06118v3] — This paper introduces AQLM, a novel additive quantization method for extreme LLM compression.
- Tim et al. (2025) — "GuidedQuant: Large Language Model Quantization via Exploiting End Loss Guidance" — [https://arxiv.org/abs/2505.07004v4] — This paper introduces a novel quantization approach that integrates gradient information from the end loss into the quantization objective.
- Bikshandi & Shah (2023) — "A Case Study in CUDA Kernel Fusion: Implementing FlashAttention-2 on NVIDIA Hopper Architecture using the CUTLASS Library" — [https://arxiv.org/pdf/2312.11918] — This paper provides an optimized implementation of FlashAttention-2, demonstrating its efficiency on NVIDIA Hopper architecture.
- Shah et al. (2024) — "FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision" — [https://arxiv.org/html/2407.08608v2] — FlashAttention-3 builds upon its predecessors by incorporating asynchrony and low-precision techniques to further accelerate attention computations.
- Li et al. (2025) — "ImPart: Importance-Aware Delta-Sparsification for Improved Model Compression and Merging in LLMs" — [https://arxiv.org/abs/2504.13237v1] — This paper introduces ImPart, a novel importance-aware delta sparsification approach for improved model compression and merging in LLMs.
- Zadouri et al. (2024) — "FlashAttention-4: Algorithm and Kernel Pipelining Co-Design for Asymmetric Hardware Scaling" — [https://arxiv.org/html/2603.05451] — This paper explores the co-design of algorithms and kernel pipelining for FlashAttention, focusing on asymmetric hardware scaling.
```