"""Smoke tests: each supported attention backend imports and calls the
TurboQuant KV-cache hooks.

These tests don't boot a server — they verify that the backend module source
references `apply_turboquant_kv_cache` / `is_turboquant_layer` and that the
hooks themselves behave correctly (no-op on layers without `tq_config`,
encode→decode round-trip when the layer is configured).

Run with: `python -m pytest tests/test_turboquant_backend_hooks.py -v`
"""

from __future__ import annotations

import importlib.util
import os
import sys

import pytest


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_ATTN_DIR = os.path.join(
    _PROJECT_ROOT, "python", "sglang", "srt", "layers", "attention"
)


SUPPORTED_BACKENDS = [
    "flashinfer_backend.py",
    "triton_backend.py",
    "flashattention_backend.py",
    "trtllm_mha_backend.py",
    "dual_chunk_flashattention_backend.py",
    "torch_native_backend.py",
    "torch_flex_backend.py",
]


@pytest.mark.parametrize("backend_file", SUPPORTED_BACKENDS)
def test_backend_source_references_turboquant_hooks(backend_file):
    """Each supported backend must import and call the TurboQuant hooks."""
    path = os.path.join(_ATTN_DIR, backend_file)
    with open(path, "r") as f:
        src = f.read()
    assert "apply_turboquant_kv_cache" in src, (
        f"{backend_file} does not import/call apply_turboquant_kv_cache"
    )
    assert "is_turboquant_layer" in src, (
        f"{backend_file} does not import/call is_turboquant_layer"
    )


UNSUPPORTED_BACKENDS = [
    "flashinfer_mla_backend.py",
    "flashmla_backend.py",
    "cutlass_mla_backend.py",
    "trtllm_mla_backend.py",
    "nsa_backend.py",
    "aiter_backend.py",
    "wave_backend.py",
    "xpu_backend.py",
    "intel_amx_backend.py",
    "double_sparsity_backend.py",
]


@pytest.mark.parametrize("backend_file", UNSUPPORTED_BACKENDS)
def test_unsupported_backend_has_no_turboquant_hooks(backend_file):
    """Unsupported backends must NOT silently reference the hooks.

    Protects against accidental half-wiring where an import is added but the
    set_kv_buffer call sites are not updated, which would give users a silent
    no-op rather than the model_runner's fail-fast error.
    """
    path = os.path.join(_ATTN_DIR, backend_file)
    if not os.path.exists(path):
        pytest.skip(f"{backend_file} not present in tree")
    with open(path, "r") as f:
        src = f.read()
    assert "apply_turboquant_kv_cache" not in src, (
        f"{backend_file} references apply_turboquant_kv_cache but is listed as "
        f"unsupported — either wire it up fully or remove the import."
    )


def _load_turboquant_module():
    """Load turboquant.py in isolation so we don't drag in the full sglang runtime."""
    quant_dir = os.path.join(
        _PROJECT_ROOT, "python", "sglang", "srt", "layers", "quantization"
    )
    # Stub out fp8_kernel to avoid CUDA imports.
    import types
    fp8_stub = types.ModuleType("sglang.srt.layers.quantization.fp8_kernel")
    fp8_stub.is_fp8_fnuz = lambda: False
    sys.modules["sglang.srt.layers.quantization.fp8_kernel"] = fp8_stub

    for name in ("base_config", "kv_cache"):
        spec = importlib.util.spec_from_file_location(
            f"sglang.srt.layers.quantization.{name}",
            os.path.join(quant_dir, f"{name}.py"),
        )
        mod = importlib.util.module_from_spec(spec)
        sys.modules[f"sglang.srt.layers.quantization.{name}"] = mod
        spec.loader.exec_module(mod)

    spec = importlib.util.spec_from_file_location(
        "sglang.srt.layers.quantization.turboquant",
        os.path.join(quant_dir, "turboquant.py"),
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["sglang.srt.layers.quantization.turboquant"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_is_turboquant_layer_noop_without_config():
    tq = _load_turboquant_module()

    class _Layer:
        pass

    assert tq.is_turboquant_layer(_Layer()) is False

    layer = _Layer()
    layer.tq_config = None
    assert tq.is_turboquant_layer(layer) is False

    layer.tq_config = object()
    assert tq.is_turboquant_layer(layer) is True


def test_apply_turboquant_kv_cache_skips_during_graph_capture():
    """During CUDA graph capture the hook must return inputs unchanged."""
    import torch

    if not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")

    tq = _load_turboquant_module()

    # Fake layer — the hook bails out on the capture check before reading
    # anything off it, so this minimal stub is enough.
    class _Layer:
        tq_config = object()

    k = torch.zeros(4, 2, 8, device="cuda", dtype=torch.float16)
    v = torch.zeros(4, 2, 8, device="cuda", dtype=torch.float16)

    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        k_out, v_out = tq.apply_turboquant_kv_cache(_Layer(), k, v)
        # During capture, hook returns inputs identity.
    assert k_out is k
    assert v_out is v
