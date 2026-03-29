"""TurboQuant KV cache quantization (PolarQuant + QJL residual correction).

Based on the TurboQuant paper (ICLR 2026): https://arxiv.org/abs/2504.19874
Combines PolarQuant (polar-coordinate quantization) with QJL (1-bit
Johnson-Lindenstrauss residual correction) for efficient KV cache compression.

Pure PyTorch implementation — no custom CUDA/Triton kernels.
"""

import logging
from typing import Any, Dict, List, Optional

import torch

from sglang.srt.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from sglang.srt.layers.quantization.kv_cache import BaseKVCacheMethod

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core math functions
# ---------------------------------------------------------------------------

_SEED = 42


def _get_rotation_matrix(
    head_dim: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Generate a fixed random orthogonal rotation matrix."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SEED)
    R, _ = torch.linalg.qr(
        torch.randn(head_dim, head_dim, generator=gen, dtype=torch.float32)
    )
    return R.to(device=device, dtype=dtype)


def polar_quantize(
    x: torch.Tensor, bits: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """PolarQuant: quantize KV vectors via per-vector absmax symmetric quant.

    Separates magnitude (scale) from direction and uniformly quantizes each
    component using the per-vector absolute-max as the symmetric range.

    Args:
        x: [..., head_dim] tensor.
        bits: number of quantization bits per component.

    Returns:
        codes: [..., head_dim] uint8 quantized components.
        scale: [..., 1] float16 per-vector absmax (replaces separate radius).
    """
    scale = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    n_levels = 2**bits
    # Map [-scale, scale] → [0, n_levels-1]
    codes = ((x / scale + 1) / 2 * (n_levels - 1)).round().clamp(0, n_levels - 1).to(
        torch.uint8
    )
    return codes, scale.to(torch.float16)


def polar_dequantize(
    codes: torch.Tensor, scale: torch.Tensor, bits: int
) -> torch.Tensor:
    """Inverse of :func:`polar_quantize`."""
    n_levels = 2**bits
    x = codes.float() / (n_levels - 1) * 2 - 1
    return x * scale.float()


def _rademacher_matrix(
    d: int, device: torch.device, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Deterministic Rademacher (±1) matrix for QJL projection."""
    gen = torch.Generator(device="cpu")
    gen.manual_seed(_SEED)
    return (
        torch.randint(0, 2, (d, d), generator=gen, dtype=dtype) * 2 - 1
    ).to(device=device)


def qjl_encode_residual(
    residual: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """1-bit QJL: project residual and store sign bits + residual norm.

    Args:
        residual: [..., head_dim] residual tensor.

    Returns:
        sign_bits: [..., head_dim] uint8 (0 or 1).
        res_norm: [..., 1] float16 residual L2 norm (needed for decode scaling).
    """
    d = residual.shape[-1]
    jl_matrix = _rademacher_matrix(d, residual.device)
    projected = residual.float() @ jl_matrix
    sign_bits = (projected > 0).to(torch.uint8)
    res_norm = residual.float().norm(dim=-1, keepdim=True)
    return sign_bits, res_norm.to(torch.float16)


def qjl_decode_residual(
    sign_bits: torch.Tensor,
    res_norm: torch.Tensor,
    head_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Reconstruct residual estimate from QJL sign bits.

    Unbiased estimator: (||r|| · √(π/2) / d) · J^T @ signs
    """
    jl_matrix = _rademacher_matrix(head_dim, device)
    signs = sign_bits.float() * 2 - 1
    # J^T @ signs gives direction; scale by norm * sqrt(pi/2) / d
    direction = signs @ jl_matrix.T
    return direction * (res_norm.float() * (torch.pi / 2) ** 0.5 / head_dim)


# ---------------------------------------------------------------------------
# High-level encode / decode combining PolarQuant + QJL
# ---------------------------------------------------------------------------


def turboquant_encode(
    x: torch.Tensor,
    bits: int = 3,
    use_polar: bool = True,
    use_qjl: bool = True,
) -> dict[str, torch.Tensor]:
    """Encode a KV tensor with TurboQuant.

    Returns a dict of compressed components.
    """
    result: dict[str, torch.Tensor] = {}

    if use_polar:
        codes, radius = polar_quantize(x, bits)
        result["codes"] = codes
        result["radius"] = radius

        if use_qjl:
            recon = polar_dequantize(codes, radius, bits)
            residual = x.float() - recon
            qjl_bits, res_norm = qjl_encode_residual(residual)
            result["qjl_bits"] = qjl_bits
            result["qjl_norm"] = res_norm
    else:
        # Fallback: store raw fp16
        result["raw"] = x.to(torch.float16)

    return result


def turboquant_decode(
    encoded: dict[str, torch.Tensor],
    bits: int = 3,
    use_polar: bool = True,
    use_qjl: bool = True,
) -> torch.Tensor:
    """Decode a TurboQuant-compressed KV tensor."""
    if not use_polar:
        return encoded["raw"].float()

    recon = polar_dequantize(encoded["codes"], encoded["radius"], bits)

    if use_qjl and "qjl_bits" in encoded:
        head_dim = encoded["codes"].shape[-1]
        device = encoded["codes"].device
        residual_est = qjl_decode_residual(
            encoded["qjl_bits"], encoded["qjl_norm"], head_dim, device
        )
        recon = recon + residual_est

    return recon


# ---------------------------------------------------------------------------
# SGLang quantization config & KV cache method
# ---------------------------------------------------------------------------


class TurboQuantConfig(QuantizationConfig):
    """TurboQuant KV cache quantization (PolarQuant + QJL residual)."""

    def __init__(
        self,
        bits: float = 3.5,
        use_polar: bool = True,
        use_qjl: bool = True,
    ):
        super().__init__()
        self.bits = bits  # effective bits per dim (e.g. 3 + 0.5 for QJL overhead)
        self.polar_bits = int(bits)  # integer bits used for PolarQuant
        self.use_polar = use_polar
        self.use_qjl = use_qjl

    @classmethod
    def get_name(cls) -> str:
        return "turboquant"

    @classmethod
    def get_supported_act_dtypes(cls) -> List[torch.dtype]:
        return [torch.float16, torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        # Pure PyTorch — works on any GPU
        return 70

    @classmethod
    def get_config_filenames(cls) -> List[str]:
        return ["turboquant_config.json"]

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> "TurboQuantConfig":
        bits = cls.get_from_keys_or(config, ["bits"], 3.5)
        use_polar = cls.get_from_keys_or(config, ["use_polar"], True)
        use_qjl = cls.get_from_keys_or(config, ["use_qjl"], True)
        return cls(bits=bits, use_polar=use_polar, use_qjl=use_qjl)

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> Optional["QuantizeMethodBase"]:
        from sglang.srt.layers.radix_attention import RadixAttention

        if isinstance(layer, RadixAttention):
            return TurboQuantKVCacheMethod(self)
        return None

    def get_scaled_act_names(self) -> List[str]:
        return []


class TurboQuantKVCacheMethod(BaseKVCacheMethod):
    """KV cache quant method using TurboQuant (PolarQuant + QJL)."""

    def __init__(self, quant_config: TurboQuantConfig):
        super().__init__(quant_config)
        self.quant_config = quant_config

    def create_weights(self, layer: torch.nn.Module):
        """Store config on the layer; rotation matrix is created lazily."""
        # Call parent to set up k_scale / v_scale parameters
        super().create_weights(layer)
        layer.tq_initialized = False
        layer.tq_config = self.quant_config

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        super().process_weights_after_loading(layer)
        layer.tq_initialized = True
