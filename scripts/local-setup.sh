#!/usr/bin/env bash
# Local-LLM setup for FAIRE on Apple Silicon (M4 Pro recommended, 24-48GB RAM).
#
# What this does:
#   1. Installs mlx-lm into the agents venv (one-time, ~200MB)
#   2. Downloads two pre-quantized Qwen 2.5 models from mlx-community (~24GB total)
#   3. Starts an OpenAI-compatible HTTP server on :8080
#
# After this runs, set OPENAI_API_BASE=http://127.0.0.1:8080/v1 in agents/.env
# and the existing FAIRE agent (start.sh) routes there instead of OpenRouter.
#
# See docs/system/local-mode.md for the full guide.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
AGENTS_DIR="$ROOT_DIR/agents"
MODELS_DIR="$ROOT_DIR/mlx_models"

WRITER_MODEL="mlx-community/Qwen2.5-Coder-32B-Instruct-4bit"
REVIEWER_MODEL="mlx-community/Qwen2.5-7B-Instruct-4bit"

echo "════════════════════════════════════════════"
echo "  FAIRE — Local LLM Setup (MLX, Apple Silicon)"
echo "════════════════════════════════════════════"
echo "  agents venv:  $AGENTS_DIR/.venv"
echo "  models dir:   $MODELS_DIR"
echo "  writer model: $WRITER_MODEL"
echo "  reviewer:     $REVIEWER_MODEL"
echo "════════════════════════════════════════════"

# Sanity check: must be on Apple Silicon
if [[ "$(uname -sm)" != "Darwin arm64" ]]; then
    echo "❌ This script requires Apple Silicon (Darwin arm64). Got: $(uname -sm)"
    echo "   For non-Mac, use Ollama or vLLM and point OPENAI_API_BASE accordingly."
    exit 1
fi

# Step 1: install mlx-lm into the agents venv
echo ""
echo "[1/3] Installing mlx-lm into agents venv..."
cd "$AGENTS_DIR"
uv pip install --upgrade mlx-lm

# Step 2: pre-download models via huggingface CLI (mlx-community pre-quantized)
echo ""
echo "[2/3] Downloading models (~24GB total, this takes a while)..."
mkdir -p "$MODELS_DIR"

# hf_hub_download caches under ~/.cache/huggingface — mlx_lm.server will read from there.
"$AGENTS_DIR/.venv/bin/python3" -c "
from huggingface_hub import snapshot_download
print('  → downloading $WRITER_MODEL ...')
snapshot_download('$WRITER_MODEL')
print('  → downloading $REVIEWER_MODEL ...')
snapshot_download('$REVIEWER_MODEL')
print('  ✓ done')
"

# Step 3: launch the MLX server
echo ""
echo "[3/3] Starting MLX server on :8080 (writer model loaded by default)..."
echo ""
echo "  Next: in another terminal, set in agents/.env:"
echo "    OPENAI_API_BASE=http://127.0.0.1:8080/v1"
echo "    WRITER_MODEL=$WRITER_MODEL"
echo "    REVIEWER_MODEL=$REVIEWER_MODEL"
echo "    CRITIC_MODEL=$REVIEWER_MODEL"
echo "    RESEARCH_MODEL=$REVIEWER_MODEL"
echo "    MVB_MODEL=$WRITER_MODEL"
echo "    FALLBACK_MODEL=$REVIEWER_MODEL"
echo ""
echo "  Then restart the agent normally: ./start.sh --interval 1 --run-now"
echo "  Verify: curl http://127.0.0.1:8080/v1/models"
echo "  Stop:   Ctrl+C in this window"
echo ""

exec "$AGENTS_DIR/.venv/bin/python3" -m mlx_lm.server \
    --model "$WRITER_MODEL" \
    --port 8080 \
    --host 127.0.0.1
