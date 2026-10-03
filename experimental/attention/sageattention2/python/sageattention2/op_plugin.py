"""
NPU availability detection and kernel launch bridge for SageAttention2 on Ascend NPU.
Aligned with ops-transformer custom ops and torch_npu specifications.
Custom operator loading and registration are managed by _extension.py and _ops.py;
this module provides the unified NPU launch interface.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    import torch_npu
except ImportError:
    torch_npu = None

from .error_check import (
    DEFAULT_PV_ACCUM_DTYPE,
    DEFAULT_QK_QUANT_GRAN,
    DEFAULT_TENSOR_LAYOUT,
)


def is_npu_available() -> bool:
    """Return True if torch has NPU support and NPU device is available."""
    if torch is None or not hasattr(torch, "npu"):
        return False
    try:
        return torch.npu.is_available()
    except Exception:
        return False


def is_npu_kernel_loaded() -> bool:
    """Return True if compiled C++/aclnn NPU custom operator is available."""
    if torch is None:
        return False
    return hasattr(getattr(torch.ops, "sageattention2", None), "forward")


def run_sageattn_npu(
    q: Any,
    k: Any,
    v: Any,
    tensor_layout: str = DEFAULT_TENSOR_LAYOUT,
    is_causal: bool = False,
    qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any]]:
    """
    Execute SageAttention2 via Ascend NPU aclnn / C++ kernel.
    Single unified entrypoint: torch.ops.sageattention2.forward.
    """
    if not is_npu_available():
        raise RuntimeError("torch_npu / Ascend NPU is not available in current environment.")

    # Device binding for distributed scenarios (xDiT/torchrun)
    if hasattr(v, "device") and hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
        if v.device.type == "npu":
            torch.npu.set_device(v.device)

    op_fn = getattr(getattr(torch.ops, "sageattention2", None), "forward", None)
    if op_fn is None:
        raise NotImplementedError(
            "Ascend 950PR NPU kernel is not loaded. Build ops-transformer or enable SAGEATTN_ALLOW_CPU_STUB=1."
        )

    res = op_fn(
        q,
        k,
        v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
        **kwargs,
    )
    if isinstance(res, tuple):
        return res[0], res[1] if len(res) > 1 else None
    return res, None
