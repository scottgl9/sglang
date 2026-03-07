#!/usr/bin/env bash
# Launch SGLang server on AMD Strix Halo (gfx1151 / RDNA 3.5)
#
# Usage:
#   bash launch_sglang_gfx1151.sh awq          # Qwen3-Coder-Next AWQ-4bit (default)
#   bash launch_sglang_gfx1151.sh mtp           # Qwen3.5-122B-A10B with MTP/NextN speculation
#   bash launch_sglang_gfx1151.sh tool-use      # Qwen3.5-122B-A10B with tool calling
#   bash launch_sglang_gfx1151.sh custom <args> # Pass arbitrary sglang args

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv_gfx1151"

# ── Runtime environment ──────────────────────────────────────
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_HOME=/opt/rocm
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libhsa-runtime64.so.1
export SGLANG_USE_AITER=0
export GPU_MAX_HW_QUEUES=2
export HSA_XNACK=0

# ── Activate venv ────────────────────────────────────────────
if [ -f "$VENV_DIR/bin/activate" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "Error: venv not found at $VENV_DIR. Run build_sglang_gfx1151.sh first."
    exit 1
fi

# ── Strix Halo common flags ─────────────────────────────────
# RDNA 3.5 notes:
#   --dtype float16: better fp16 than bf16 throughput on RDNA 3.5
#   --attention-backend triton: required (no FlashAttention for gfx1100)
#   --mem-fraction-static: conservative for shared CPU/GPU LPDDR5X memory
#   LPDDR5X (~256 GB/s) is the bottleneck; INT4 AWQ minimizes memory reads

MODE="${1:-awq}"
shift || true

case "$MODE" in
    awq)
        # Qwen3-Coder-Next AWQ-4bit (MoE, compressed-tensors, ~46GB)
        echo "=== Launching: Qwen3-Coder-Next AWQ-4bit ==="
        exec python -m sglang.launch_server \
            --model-path cyankiwi/Qwen3-Coder-Next-AWQ-4bit \
            --quantization awq \
            --attention-backend triton \
            --dtype float16 \
            --mem-fraction-static 0.70 \
            --context-length 8192 \
            --host 0.0.0.0 --port 30000 \
            "$@"
        ;;

    mtp)
        # Qwen3.5-122B-A10B with MTP/NextN speculative decoding
        echo "=== Launching: Qwen3.5-122B-A10B with MTP ==="
        exec python -m sglang.launch_server \
            --model-path Qwen/Qwen3.5-122B-A10B \
            --attention-backend triton \
            --dtype float16 \
            --port 8000 \
            --mem-fraction-static 0.8 \
            --context-length 262144 \
            --reasoning-parser qwen3 \
            --speculative-algo NEXTN \
            --speculative-num-steps 3 \
            --speculative-eagle-topk 1 \
            --speculative-num-draft-tokens 4 \
            "$@"
        ;;

    tool-use)
        # Qwen3.5-122B-A10B with tool calling support
        echo "=== Launching: Qwen3.5-122B-A10B with tool use ==="
        exec python -m sglang.launch_server \
            --model-path Qwen/Qwen3.5-122B-A10B \
            --attention-backend triton \
            --dtype float16 \
            --port 8000 \
            --mem-fraction-static 0.8 \
            --context-length 262144 \
            --reasoning-parser qwen3 \
            --tool-call-parser qwen3_coder \
            "$@"
        ;;

    custom)
        # Pass through arbitrary args to sglang
        echo "=== Launching: custom configuration ==="
        exec python -m sglang.launch_server "$@"
        ;;

    *)
        echo "Unknown mode: $MODE"
        echo "Usage: $0 {awq|mtp|tool-use|custom} [extra args...]"
        exit 1
        ;;
esac
