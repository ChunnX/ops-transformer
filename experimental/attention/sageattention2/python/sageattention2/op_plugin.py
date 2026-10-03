"""
NPU availability detection and kernel launch bridge for SageAttention2 on Ascend NPU.
Aligned with ops-transformer custom ops and torch_npu specifications.
(扩展库的加载与注册统一由 _extension.py / _ops.py 负责，本模块仅做探测与转发。)
"""

from __future__ import annotations

from typing import Any, Callable, Optional, Tuple

try:
    import torch
except ImportError:
    torch = None

try:
    import torch_npu
except ImportError:
    torch_npu = None


_NPU_SAGEATTN_OP_NAME = "npu_sage_attention2"


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
    for ns in ("cann_ops_transformer", "sageattention2", "custom"):
        if hasattr(torch.ops, ns):
            lib = getattr(torch.ops, ns)
            if hasattr(lib, _NPU_SAGEATTN_OP_NAME) or hasattr(lib, "sageattn_qk_int8_pv_fp8_asc") or hasattr(lib, "forward"):
                return True
    if torch_npu is not None and hasattr(torch_npu, _NPU_SAGEATTN_OP_NAME):
        return True
    return False


def run_sageattn_npu(
    q: Any,
    k: Any,
    v: Any,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = "per_thread",
    sm_scale: Optional[float] = None,
    pv_accum_dtype: str = "fp32+fp16",
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> Tuple[Any, Optional[Any]]:
    """
    Execute SageAttention2 via Ascend NPU aclnn / C++ kernel.
    """
    if not is_npu_available():
        raise RuntimeError("torch_npu / Ascend NPU is not available in current environment.")

    # Device binding for distributed scenarios (xDiT/torchrun workaround)
    if hasattr(v, "device") and hasattr(torch, "npu") and hasattr(torch.npu, "set_device"):
        if v.device.type == "npu":
            torch.npu.set_device(v.device)

    op_fn: Optional[Callable[..., Any]] = None
    if hasattr(torch.ops, "sageattention2") and hasattr(torch.ops.sageattention2, "forward"):
        op_fn = getattr(torch.ops.sageattention2, "forward")
    elif hasattr(torch.ops, "cann_ops_transformer") and hasattr(torch.ops.cann_ops_transformer, _NPU_SAGEATTN_OP_NAME):
        op_fn = getattr(torch.ops.cann_ops_transformer, _NPU_SAGEATTN_OP_NAME)
    elif hasattr(torch.ops, "sageattention2") and hasattr(torch.ops.sageattention2, "sageattn_qk_int8_pv_fp8_asc"):
        op_fn = getattr(torch.ops.sageattention2, "sageattn_qk_int8_pv_fp8_asc")
    elif torch_npu is not None and hasattr(torch_npu, _NPU_SAGEATTN_OP_NAME):
        op_fn = getattr(torch_npu, _NPU_SAGEATTN_OP_NAME)

    if op_fn is None:
        raise NotImplementedError(
            "Ascend 950PR NPU kernel is not loaded. Build ops-transformer or enable SAGEATTN_ALLOW_CPU_STUB=1."
        )

    if hasattr(torch.ops, "sageattention2") and op_fn is getattr(torch.ops.sageattention2, "forward", None):
        res = op_fn(
            q,
            k,
            v,
            tensor_layout=tensor_layout,
            is_causal=is_causal,
            sm_scale=sm_scale,
            smooth_k=smooth_k,
            return_lse=return_lse,
        )
    else:
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
