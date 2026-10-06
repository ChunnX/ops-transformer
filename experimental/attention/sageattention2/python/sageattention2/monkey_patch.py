"""
Industrial-grade hot monkey-patching tool for PyTorch scaled_dot_product_attention (SDPA).
Provides replace_sdpa(), restore_sdpa(), and context manager sdpa_patch().
Supports automatic fallback to original SDPA for non-inference or unsupported configurations.
"""

from __future__ import annotations

import contextlib
import functools
from typing import Any, Callable, Optional

import torch
import torch.nn.functional as F

from .dispatch import sageattn

_ORIGINAL_SDPA: Callable = F.scaled_dot_product_attention
_PATCH_STACK: list[Callable] = []


def is_sdpa_patched() -> bool:
    """Return True if F.scaled_dot_product_attention has been replaced by SageAttention2."""
    return getattr(F.scaled_dot_product_attention, "_is_sageattn_patch", False)


def _can_use_sageattn(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attn_mask: Optional[torch.Tensor] = None,
    dropout_p: float = 0.0,
    is_causal: bool = False,
) -> tuple[bool, str]:
    """Check whether query, key, value and arguments meet SageAttention2 constraints."""
    if attn_mask is not None:
        return False, "attn_mask is not supported"
    if dropout_p is not None and dropout_p > 0.0:
        return False, "dropout_p > 0 is not supported in inference-only SageAttention2"
    if query.dim() != 4 or key.dim() != 4 or value.dim() != 4:
        return False, "Tensors must be 4D (B, H, S, D)"
    if query.dtype not in (torch.float16, torch.bfloat16):
        return False, f"dtype {query.dtype} is not float16 or bfloat16"
    if query.dtype != key.dtype or query.dtype != value.dtype:
        return False, "Tensors must share identical dtype"
    if query.device != key.device or query.device != value.device:
        return False, "Tensors must share identical device"
    head_dim = query.shape[-1]
    if head_dim <= 0 or head_dim > 128:
        return False, f"head_dim {head_dim} must be in range [1, 128]"
    if key.shape[-1] != head_dim or value.shape[-1] != head_dim:
        return False, "head_dim mismatch between Q, K, V"
    hq = query.shape[1]
    hkv = key.shape[1]
    if hq % hkv != 0:
        return False, f"GQA constraint violated: Hq ({hq}) must be divisible by Hkv ({hkv})"
    sq = query.shape[2]
    sk = key.shape[2]
    if is_causal and sq != sk:
        return False, f"is_causal requires sq == sk (got {sq} != {sk})"
    if query.requires_grad or key.requires_grad or value.requires_grad:
        return False, "requires_grad=True is not supported in forward-only SageAttention2"
    if query.stride(-1) != 1 or key.stride(-1) != 1 or value.stride(-1) != 1:
        return False, "Last dimension of tensors must be contiguous (stride(-1) == 1)"
    return True, ""


def replace_sdpa(
    pv_accum_dtype: Optional[str] = None,
    qk_quant_gran: str = "per_thread",
    smooth_k: bool = True,
    smooth_v: bool = False,
    fallback_to_original: bool = True,
    **patch_kwargs: Any,
) -> Callable:
    """
    Replace torch.nn.functional.scaled_dot_product_attention with SageAttention2.

    Args:
        pv_accum_dtype: Optional override for PV accumulation precision ('fp32', 'fp32+fp32', 'fp32+fp16').
        qk_quant_gran: Quantization granularity ('per_thread' or 'per_warp').
        smooth_k: Whether to enable smooth_k mean centering.
        smooth_v: Whether to enable smooth_v (effective only in 'fp32' accum mode).
        fallback_to_original: If True, falls back to original PyTorch SDPA when inputs
                              do not meet SageAttention2 constraints (e.g. non-zero dropout, custom attn_mask).
                              If False, raises ValueError upon unsupported configurations.
        **patch_kwargs: Additional keyword arguments forwarded to sageattn.

    Returns:
        The previously installed SDPA function (can be passed to restore_sdpa).
    """
    prev_sdpa = F.scaled_dot_product_attention
    _PATCH_STACK.append(prev_sdpa)

    @functools.wraps(_ORIGINAL_SDPA)
    def _sageattn_sdpa_wrapper(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        dropout_p: float = 0.0,
        is_causal: bool = False,
        scale: Optional[float] = None,
        enable_gqa: bool = False,
        **sdpa_kwargs: Any,
    ) -> torch.Tensor:
        can_use, reason = _can_use_sageattn(
            query=query,
            key=key,
            value=value,
            attn_mask=attn_mask,
            dropout_p=dropout_p,
            is_causal=is_causal,
        )

        if can_use:
            out = sageattn(
                q=query,
                k=key,
                v=value,
                tensor_layout="HND",
                is_causal=is_causal,
                sm_scale=scale,
                pv_accum_dtype=pv_accum_dtype,
                qk_quant_gran=qk_quant_gran,
                smooth_k=smooth_k,
                smooth_v=smooth_v,
                return_lse=False,
                **patch_kwargs,
            )
            return out
        else:
            if fallback_to_original:
                # Forward to original PyTorch SDPA implementation
                return _ORIGINAL_SDPA(
                    query,
                    key,
                    value,
                    attn_mask=attn_mask,
                    dropout_p=dropout_p,
                    is_causal=is_causal,
                    scale=scale,
                    enable_gqa=enable_gqa,
                    **sdpa_kwargs,
                )
            else:
                raise ValueError(
                    f"SageAttention2 cannot execute SDPA configuration: {reason}. "
                    f"Set fallback_to_original=True in replace_sdpa() to enable automatic fallback."
                )

    _sageattn_sdpa_wrapper._is_sageattn_patch = True  # type: ignore[attr-defined]
    _sageattn_sdpa_wrapper._prev_sdpa = prev_sdpa      # type: ignore[attr-defined]

    F.scaled_dot_product_attention = _sageattn_sdpa_wrapper
    return prev_sdpa


def restore_sdpa(target: Optional[Callable] = None) -> None:
    """
    Restore torch.nn.functional.scaled_dot_product_attention.

    Args:
        target: Optional specific function to restore to (e.g. the return value of replace_sdpa()).
                If None, pops from the patch stack, or defaults to the original PyTorch SDPA.
    """
    global _PATCH_STACK
    if target is not None:
        F.scaled_dot_product_attention = target
        if _PATCH_STACK and _PATCH_STACK[-1] == target:
            _PATCH_STACK.pop()
    elif _PATCH_STACK:
        F.scaled_dot_product_attention = _PATCH_STACK.pop()
    else:
        F.scaled_dot_product_attention = _ORIGINAL_SDPA


@contextlib.contextmanager
def sdpa_patch(
    pv_accum_dtype: Optional[str] = None,
    qk_quant_gran: str = "per_thread",
    smooth_k: bool = True,
    smooth_v: bool = False,
    fallback_to_original: bool = True,
    **kwargs: Any,
):
    """
    Context manager for temporary scoped monkey-patching of F.scaled_dot_product_attention.

    Example:
        with sdpa_patch(pv_accum_dtype="fp32"):
            output = F.scaled_dot_product_attention(q, k, v)
    """
    prev = replace_sdpa(
        pv_accum_dtype=pv_accum_dtype,
        qk_quant_gran=qk_quant_gran,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        fallback_to_original=fallback_to_original,
        **kwargs,
    )
    try:
        yield
    finally:
        restore_sdpa(prev)
