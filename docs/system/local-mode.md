---
title: Local LLM Mode
description: Run FAIRE entirely on an Apple Silicon Mac (M4 Pro recommended) with no cloud spend.
---

# Local LLM Mode

FAIRE talks to LLMs through `langchain_openai.ChatOpenAI`, which speaks the
standard OpenAI Chat Completions protocol. By default we POST to OpenRouter
(cloud). To run locally, **point `OPENAI_API_BASE` at any OpenAI-compatible
endpoint** — MLX, Ollama, vLLM, LMStudio, anything. That's the entire
integration. No code branches, no parallel pipelines.

## Quick start (Apple Silicon, MLX)

```bash
# One-time: install mlx-lm, download ~24GB of pre-quantized Qwen models,
# launch a server on :8080. Keep this terminal open.
./scripts/local-setup.sh
```

Then in `agents/.env`:

```bash
OPENAI_API_BASE=http://127.0.0.1:8080/v1

# These names must match what your local server exposes
WRITER_MODEL=mlx-community/Qwen2.5-Coder-32B-Instruct-4bit
MVB_MODEL=mlx-community/Qwen2.5-Coder-32B-Instruct-4bit
REVIEWER_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
CRITIC_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
RESEARCH_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
FALLBACK_MODEL=mlx-community/Qwen2.5-7B-Instruct-4bit
```

Restart the agent normally — everything else works as-is:

```bash
./start.sh --interval 1 --run-now
```

## Verify

```bash
# Server is up and serving the model
curl http://127.0.0.1:8080/v1/models

# A live page generation hits the local server (watch the MLX terminal)
curl -X POST "http://127.0.0.1:8765/generate?topic=foo&track=01-ai&page_type=core-concept&depth_emphasis=applied"
```

In `agents/logs/server.log` you'll see the usual cycle output. The MLX server's
own log shows the inference calls — that's how you confirm cloud isn't being
hit.

## Model recommendations (M4 Pro 48GB)

| Role | Model | Disk | Speed |
|---|---|---|---|
| writer / mvb | `Qwen2.5-Coder-32B-Instruct` (4-bit) | ~19GB | ~15–18 tok/s |
| reviewer / critic / research / fallback | `Qwen2.5-7B-Instruct` (4-bit) | ~5GB | ~40–60 tok/s |

Both fit in unified memory simultaneously (~24GB total) with headroom for KV
cache. Expect ~8–14 minutes per v2 page locally vs ~2–3 minutes on cloud.

On M4 base (16GB): swap the writer to `Qwen2.5-Coder-14B-Instruct-4bit`
(~9GB). On M4 Max (64GB+): bump the writer to a 5-bit quant or try
`Llama-3.3-70B-Instruct-4bit`.

## Going back to cloud

Comment out `OPENAI_API_BASE` and restore the cloud `WRITER_MODEL` etc.
in `agents/.env`. Restart the agent. Everything works again.

## Other local servers

The same env-var pattern works for any OpenAI-compatible server:

| Server | `OPENAI_API_BASE` | Notes |
|---|---|---|
| MLX (Apple Silicon) | `http://127.0.0.1:8080/v1` | Native, fastest on M-series |
| Ollama | `http://127.0.0.1:11434/v1` | Cross-platform, easiest install |
| vLLM | `http://127.0.0.1:8000/v1` | Best throughput on NVIDIA |
| LMStudio | `http://127.0.0.1:1234/v1` | GUI, good for testing |

For all of these the `WRITER_MODEL` etc. values must be whatever name the
server exposes (`ollama list`, `mlx_lm.server --help`, etc.).

## When local quality isn't enough

If a particular page consistently fails review on the local writer, just flip
back to cloud for that page:

```bash
# Temporarily unset, generate the hard page, set back
unset OPENAI_API_BASE
curl -X POST "http://127.0.0.1:8765/generate?topic=hard-topic&track=04-neural-networks-deep-learning&page_type=core-concept&depth_emphasis=applied"
export OPENAI_API_BASE=http://127.0.0.1:8080/v1
```

No code changes. No branching. Same agent, same prompts, different inference
endpoint.
