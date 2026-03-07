#!/usr/bin/env bash
# Build SGLang for AMD Strix Halo (gfx1151 / RDNA 3.5)
# gfx1151 maps to gfx1100 via HSA_OVERRIDE_GFX_VERSION=11.0.0
#
# Prerequisites:
#   - ROCm 7.2.0 installed at /opt/rocm-7.2.0
#   - PyTorch ROCm available (system or from ~/sandbox/vllm/venv)
#   - Kernel param: amdgpu.cwsr_enable=0 in GRUB
#
# Usage: bash build_sglang_gfx1151.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv_gfx1151"

# ── Environment ──────────────────────────────────────────────
export HSA_OVERRIDE_GFX_VERSION=11.0.0
export ROCM_HOME=/opt/rocm
export ROCM_PATH=/opt/rocm
export HIP_PLATFORM=amd
export PYTORCH_ROCM_ARCH=gfx1100
export AMDGPU_TARGET=gfx1100
export SGLANG_USE_AITER=0
export LD_PRELOAD=/opt/rocm-7.2.0/lib/libhsa-runtime64.so.1

echo "=== Building SGLang for Strix Halo (gfx1151 -> gfx1100) ==="

# ── Create venv ──────────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating venv at $VENV_DIR (inheriting system site-packages)..."
    python3 -m venv --system-site-packages "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# Ensure pip/setuptools are up to date
pip install --upgrade pip setuptools wheel setuptools_scm

# ── Build sgl-kernel ─────────────────────────────────────────
echo "=== Building sgl-kernel ==="
cd "$SCRIPT_DIR/sgl-kernel"
pip install -e . --no-build-isolation 2>&1 | tail -5
echo "sgl-kernel installed."

# ── Install SGLang ───────────────────────────────────────────
echo "=== Installing SGLang ==="
cd "$SCRIPT_DIR/python"
pip install -e ".[all_hip]" 2>&1 | tail -10
echo "SGLang installed."

# ── Install AITER (disabled at runtime, available as fallback) ──
echo "=== Installing AITER (no prebuilt kernels) ==="
PREBUILD_KERNELS=0 GPU_ARCHS=gfx1100 pip install --no-deps \
    "git+https://github.com/ROCm/aiter.git@main" 2>&1 | tail -5 || \
    echo "Warning: AITER install failed (optional, SGLANG_USE_AITER=0 disables it)"

# ── Verify ───────────────────────────────────────────────────
echo ""
echo "=== Verification ==="
python -c "import sgl_kernel; print('sgl_kernel: OK')"
python -c "import sglang; print('sglang: OK')"

echo ""
echo "=== Build complete ==="
echo "Activate with: source $VENV_DIR/bin/activate"
echo "See launch_sglang_gfx1151.sh for model serving examples."
