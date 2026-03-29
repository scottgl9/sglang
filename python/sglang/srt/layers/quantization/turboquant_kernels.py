"""Triton kernels for TurboQuant KV cache quantization.

Provides fused encode (rotate + quantize) and decode (dequantize + unrotate) kernels.
Falls back to the PyTorch implementation in turboquant.py when Triton is unavailable.
"""

import logging
from typing import Dict, Optional

import torch

logger = logging.getLogger(__name__)

try:
    import triton
    import triton.language as tl

    # Verify Triton has an active GPU driver (import succeeds even without GPU)
    try:
        triton.runtime.driver.active  # noqa: B018
        HAS_TRITON = True
    except (RuntimeError, AttributeError):
        HAS_TRITON = False
except ImportError:
    HAS_TRITON = False

if HAS_TRITON:

    @triton.jit
    def _turboquant_encode_kernel(
        # Inputs
        input_ptr,
        R_ptr,
        codebook_ptr,
        outlier_mask_ptr,
        # Outputs
        codes_ptr,
        scale_ptr,
        outlier_ptr,
        # Strides
        stride_input_row,
        stride_R_row,
        stride_codes_row,
        stride_outlier_row,
        # Params
        D: tl.constexpr,  # head_dim
        n_normal: tl.constexpr,  # number of non-outlier channels
        n_outlier: tl.constexpr,  # number of outlier channels
        N_LEVELS: tl.constexpr,  # codebook size
        BLOCK_D: tl.constexpr,
    ):
        """Fused: matrix multiply by R, compute per-vector scale, quantize to uint8.

        Each program handles one (batch*head) row.
        """
        row_idx = tl.program_id(0)

        # Load input vector [D]
        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D
        x = tl.load(input_ptr + row_idx * stride_input_row + offs_d, mask=mask_d, other=0.0)

        # Rotate: x_rot = x @ R  (row-vector times matrix)
        # We compute each output element as dot(x, R[:, j])
        # For simplicity with Triton, we iterate over output dimensions
        # and accumulate. But since D is typically 128, we can do a
        # blocked matmul approach.
        # Store rotated values in registers
        x_rot = tl.zeros([BLOCK_D], dtype=tl.float32)
        for j in range(D):
            # R[offs_d, j] — column j of R
            r_col = tl.load(R_ptr + offs_d * stride_R_row + j, mask=mask_d, other=0.0)
            dot_val = tl.sum(x * r_col)
            # We need x_rot[j] = dot(x, R[:, j])
            # But we can't easily do scatter in Triton like this.
            # Instead, let's compute x_rot as full matmul differently.
            pass

        # Alternative approach: compute x_rot[j] = sum_i x[i] * R[i, j] for each j
        # We process each output element
        for j in range(D):
            r_col = tl.load(R_ptr + offs_d * stride_R_row + j, mask=mask_d, other=0.0)
            val = tl.sum(x * r_col)

            # Check if channel j is an outlier
            is_outlier = tl.load(outlier_mask_ptr + j)

            if is_outlier:
                # Store as float16 in outlier buffer
                # Need to figure out outlier index — count outliers before j
                # For simplicity, we store at a pre-computed position
                pass
            else:
                pass

        # Due to the complexity of scatter-based indexing in Triton,
        # we use a simpler two-pass approach:

        # Pass 1: Compute all rotated values and store to a temp buffer
        # Actually, let's compute the full rotated vector by loading R row by row
        # x_rot[j] = sum_i x[i] * R[i][j]
        # This is equivalent to: for each j, dot product of x with column j of R

        # We'll store results to codes/outlier directly

        # Compute rotated vector into local array
        # We need a different approach — compute per output element

        # Actually the clean Triton way: each program computes one row's worth
        # of output. We accumulate x_rot as a [BLOCK_D] vector.

        # x_rot = x @ R where x is [D] and R is [D, D]
        # x_rot[j] = sum_i x[i] * R[i, j]
        # Equivalent to loading row i of R^T: R^T[j, :] and dotting with x

        # Let's just do it element by element for the output
        # and split into normal/outlier channels

        normal_idx = 0
        outlier_idx = 0

        # Compute absmax scale over normal channels (two-pass: first compute rotated normals)
        # We'll store rotated values temporarily
        abs_max = 0.0

        # First pass: compute rotated values, find absmax of normal channels
        for j in range(D):
            # Load R^T row j = R column j
            r_col = tl.load(R_ptr + offs_d * stride_R_row + j, mask=mask_d, other=0.0)
            val = tl.sum(x * r_col)
            is_outlier = tl.load(outlier_mask_ptr + j)

            if not is_outlier:
                abs_val = tl.abs(val)
                abs_max = tl.maximum(abs_max, abs_val)

        # Clamp scale
        scale = tl.maximum(abs_max, 1e-8)
        tl.store(scale_ptr + row_idx, scale.to(tl.float16))

        # Second pass: quantize normal channels, store outliers
        normal_idx_counter = 0
        outlier_idx_counter = 0
        for j in range(D):
            r_col = tl.load(R_ptr + offs_d * stride_R_row + j, mask=mask_d, other=0.0)
            val = tl.sum(x * r_col)
            is_outlier = tl.load(outlier_mask_ptr + j)

            if is_outlier:
                tl.store(
                    outlier_ptr + row_idx * stride_outlier_row + outlier_idx_counter,
                    val.to(tl.float16),
                )
                outlier_idx_counter += 1
            else:
                # Normalize and quantize via nearest codebook entry
                normalized = val / scale
                # Find nearest codebook entry
                best_idx = 0
                best_dist = tl.abs(normalized - tl.load(codebook_ptr + 0))
                for c in range(1, N_LEVELS):
                    cb_val = tl.load(codebook_ptr + c)
                    dist = tl.abs(normalized - cb_val)
                    if dist < best_dist:
                        best_dist = dist
                        best_idx = c
                tl.store(
                    codes_ptr + row_idx * stride_codes_row + normal_idx_counter,
                    best_idx.to(tl.uint8),
                )
                normal_idx_counter += 1

    @triton.jit
    def _turboquant_decode_kernel(
        # Inputs
        codes_ptr,
        scale_ptr,
        outlier_ptr,
        codebook_ptr,
        outlier_mask_ptr,
        R_T_ptr,
        # Output
        output_ptr,
        # Strides
        stride_codes_row,
        stride_outlier_row,
        stride_RT_row,
        stride_output_row,
        # Params
        D: tl.constexpr,
        n_normal: tl.constexpr,
        n_outlier: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        """Dequantize + merge outliers + unrotate.

        Each program handles one (batch*head) row.
        """
        row_idx = tl.program_id(0)

        # Load scale for this row
        scale = tl.load(scale_ptr + row_idx).to(tl.float32)

        # Reconstruct the rotated vector [D] by merging normal (dequantized) and outlier channels
        # We'll compute the unrotated output element by element:
        # output[i] = sum_j rotated[j] * R_T[j, i]
        # where rotated[j] is either dequantized or outlier depending on mask

        offs_d = tl.arange(0, BLOCK_D)
        mask_d = offs_d < D

        # For each output element i, compute dot product of rotated vector with R_T column i
        for i in range(D):
            acc = 0.0
            normal_idx = 0
            outlier_idx = 0

            for j in range(D):
                is_outlier = tl.load(outlier_mask_ptr + j)
                rt_val = tl.load(R_T_ptr + j * stride_RT_row + i)

                if is_outlier:
                    val = tl.load(
                        outlier_ptr + row_idx * stride_outlier_row + outlier_idx
                    ).to(tl.float32)
                    outlier_idx += 1
                else:
                    code = tl.load(
                        codes_ptr + row_idx * stride_codes_row + normal_idx
                    ).to(tl.int32)
                    cb_val = tl.load(codebook_ptr + code)
                    val = cb_val * scale
                    normal_idx += 1

                acc += val * rt_val

            tl.store(output_ptr + row_idx * stride_output_row + i, acc.to(tl.float16))


def turboquant_encode_triton(
    k: torch.Tensor,
    v: torch.Tensor,
    R: torch.Tensor,
    codebook_k: torch.Tensor,
    codebook_v: torch.Tensor,
    outlier_mask: torch.Tensor,
    bits: int = 4,
    use_qjl: bool = True,
) -> dict:
    """Encode KV tensors using Triton kernels. Falls back to PyTorch if Triton unavailable."""
    if not HAS_TRITON:
        from sglang.srt.layers.quantization.turboquant import turboquant_encode_v2

        return turboquant_encode_v2(
            k, v, R, codebook_k, codebook_v, outlier_mask, bits=bits, use_qjl=use_qjl
        )

    # For now, the Triton kernel handles the core rotation + quantization.
    # QJL residual correction is still done in PyTorch since it's not the bottleneck.
    # The Triton kernel operates on flattened (batch*heads, head_dim) tensors.

    batch_shape = k.shape[:-1]  # everything except head_dim
    D = k.shape[-1]
    n_outlier = outlier_mask.sum().item()
    n_normal = D - n_outlier
    n_levels = 2**bits

    result: dict = {}
    result["outlier_mask"] = outlier_mask
    result["codebook_k"] = codebook_k
    result["codebook_v"] = codebook_v

    for name, x, codebook in [("k", k, codebook_k), ("v", v, codebook_v)]:
        x_flat = x.reshape(-1, D).float().contiguous()
        n_rows = x_flat.shape[0]

        codes = torch.empty(n_rows, n_normal, dtype=torch.uint8, device=x.device)
        scale = torch.empty(n_rows, dtype=torch.float16, device=x.device)
        outliers = torch.empty(n_rows, n_outlier, dtype=torch.float16, device=x.device)

        R_cont = R.float().contiguous()
        cb_cont = codebook.float().contiguous().to(x.device)
        om_cont = outlier_mask.to(x.device).contiguous()

        BLOCK_D = max(D, 128)  # Must be >= D
        # Ensure BLOCK_D is power of 2 for Triton
        BLOCK_D = 1
        while BLOCK_D < D:
            BLOCK_D *= 2

        grid = (n_rows,)
        _turboquant_encode_kernel[grid](
            x_flat,
            R_cont,
            cb_cont,
            om_cont,
            codes,
            scale,
            outliers,
            # Strides
            x_flat.stride(0),
            R_cont.stride(0),
            codes.stride(0),
            outliers.stride(0),
            # Params
            D=D,
            n_normal=n_normal,
            n_outlier=n_outlier,
            N_LEVELS=n_levels,
            BLOCK_D=BLOCK_D,
        )

        result[f"{name}_codes"] = codes.reshape(*batch_shape, n_normal)
        result[f"{name}_scale"] = scale.reshape(*batch_shape, 1)
        result[f"{name}_outliers"] = outliers.reshape(*batch_shape, n_outlier)

    # QJL residual correction (still in PyTorch — not perf-critical)
    if use_qjl:
        from sglang.srt.layers.quantization.turboquant import (
            _dequantize_from_codebook,
            qjl_encode_residual,
        )

        normal_mask = ~outlier_mask
        for name, x, codebook in [("k", k, codebook_k), ("v", v, codebook_v)]:
            x_rot = (x.float() @ R.float()).to(x.dtype)
            x_normal = x_rot[..., normal_mask]
            recon_normal = (
                _dequantize_from_codebook(result[f"{name}_codes"], codebook)
                * result[f"{name}_scale"].float()
            )
            residual = x_normal.float() - recon_normal
            qjl_bits, qjl_norm = qjl_encode_residual(residual)
            result[f"{name}_qjl_bits"] = qjl_bits
            result[f"{name}_qjl_norm"] = qjl_norm

    return result


def turboquant_decode_triton(
    encoded: dict,
    R_T: torch.Tensor,
    use_qjl: bool = True,
) -> tuple:
    """Decode TurboQuant-encoded KV tensors using Triton. Falls back to PyTorch."""
    if not HAS_TRITON:
        from sglang.srt.layers.quantization.turboquant import turboquant_decode_v2

        return turboquant_decode_v2(encoded, R_T, use_qjl=use_qjl)

    outlier_mask = encoded["outlier_mask"]
    normal_mask = ~outlier_mask
    D = outlier_mask.shape[0]
    n_outlier = outlier_mask.sum().item()
    n_normal = D - n_outlier

    results = []
    for name in ["k", "v"]:
        codes = encoded[f"{name}_codes"]
        batch_shape = codes.shape[:-1]
        scale = encoded[f"{name}_scale"]
        outliers = encoded[f"{name}_outliers"]
        codebook = encoded[f"codebook_{name}"]

        # Apply QJL correction before unrotation
        if use_qjl and f"{name}_qjl_bits" in encoded:
            from sglang.srt.layers.quantization.turboquant import (
                _dequantize_from_codebook,
                qjl_decode_residual,
            )

            recon_normal = (
                _dequantize_from_codebook(codes, codebook) * scale.float()
            )
            res = qjl_decode_residual(
                encoded[f"{name}_qjl_bits"],
                encoded[f"{name}_qjl_norm"],
                n_normal,
                codes.device,
            )
            recon_normal = recon_normal + res

            # Rebuild full rotated vector and unrotate in PyTorch
            # (QJL path modifies the normal channels, so we can't use pure Triton decode)
            full = torch.zeros(*batch_shape, D, dtype=torch.float32, device=codes.device)
            full[..., normal_mask] = recon_normal.float()
            full[..., outlier_mask] = outliers.float()
            out = (full @ R_T.float()).to(torch.float16)
            results.append(out)
        else:
            # Pure Triton decode path (no QJL)
            codes_flat = codes.reshape(-1, n_normal).contiguous()
            n_rows = codes_flat.shape[0]
            scale_flat = scale.reshape(-1).contiguous()
            outliers_flat = outliers.reshape(-1, n_outlier).contiguous()

            output = torch.empty(n_rows, D, dtype=torch.float16, device=codes.device)

            R_T_cont = R_T.float().contiguous()
            cb_cont = codebook.float().contiguous().to(codes.device)
            om_cont = outlier_mask.to(codes.device).contiguous()

            BLOCK_D = 1
            while BLOCK_D < D:
                BLOCK_D *= 2

            grid = (n_rows,)
            _turboquant_decode_kernel[grid](
                codes_flat,
                scale_flat,
                outliers_flat,
                cb_cont,
                om_cont,
                R_T_cont,
                output,
                # Strides
                codes_flat.stride(0),
                outliers_flat.stride(0),
                R_T_cont.stride(0),
                output.stride(0),
                # Params
                D=D,
                n_normal=n_normal,
                n_outlier=n_outlier,
                BLOCK_D=BLOCK_D,
            )

            results.append(output.reshape(*batch_shape, D))

    return results[0], results[1]
