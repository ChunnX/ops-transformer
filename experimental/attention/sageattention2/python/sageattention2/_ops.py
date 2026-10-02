# SageAttention2 PyTorch 适配层：torch.library dispatcher（参数校验 / 自动分发 / head_dim pad / LSE 透传）
import contextlib
import math
import os
import warnings
from typing import Any

import torch

VALID_TENSOR_LAYOUTS = ("HND", "NHD")
VALID_QK_QUANT_GRANS = ("per_warp", "per_thread")
VALID_PV_ACCUM_DTYPES = ("fp32", "fp32+fp32", "fp32+fp16")
DEFAULT_QK_QUANT_GRAN = "per_thread"
DEFAULT_PV_ACCUM_DTYPE = "fp32+fp16"
FP32_MAX = torch.finfo(torch.float32).max
# head_dim pad 档位：D<64 pad 到 64；(64,128) 内 pad 到 128；D>128 报错
PAD_BUCKETS = (64, 128)
# plan cache 容量上限（防止长时运行内存无界增长）
_PLAN_CACHE_LIMIT = 64
_ENV_PV_ACCUM_DTYPE = "SAGEATTN_PV_ACCUM_DTYPE"
_ENV_QK_QUANT_GRAN = "SAGEATTN_QK_QUANT_GRAN"

# plan cache：键含全部语义参数与 shape/dtype，值为 dispatch 预解析结果（pad 档位、默认 scale 等）
_PLAN_CACHE = {}


def _check_tensor_layout(tensor_layout: str) -> None:
    if tensor_layout not in VALID_TENSOR_LAYOUTS:
        raise ValueError(
            f"tensor_layout must be one of {list(VALID_TENSOR_LAYOUTS)}, but got {tensor_layout!r}."
        )


def _check_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError(
                f"{name} must be torch.float16 or torch.bfloat16, but got {tensor.dtype}."
            )
    if not (q.dtype == k.dtype == v.dtype):
        raise ValueError(
            f"q/k/v must share the same dtype, but got q={q.dtype}, k={k.dtype}, v={v.dtype}."
        )
    if not (q.device == k.device == v.device):
        raise ValueError(
            f"q/k/v must reside on the same device, but got q={q.device}, k={k.device}, v={v.device}."
        )


def _check_shape(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str, is_causal: bool) -> None:
    if q.dim() != 4 or k.dim() != 4 or v.dim() != 4:
        raise ValueError(
            f"q/k/v must be 4D tensors, but got dims q={q.dim()}, k={k.dim()}, v={v.dim()}."
        )
    head = 2 if tensor_layout == "NHD" else 1
    seq = 1 if tensor_layout == "NHD" else 2
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError(
            f"q/k/v must share the same batch size, but got q={q.shape[0]}, k={k.shape[0]}, v={v.shape[0]}."
        )
    if q.shape[3] != k.shape[3] or k.shape[3] != v.shape[3]:
        raise ValueError(
            f"q/k/v must share the same head_dim, but got q={q.shape[3]}, k={k.shape[3]}, v={v.shape[3]}."
        )
    if k.shape != v.shape:
        raise ValueError(
            f"k and v must share the same shape, but got k={tuple(k.shape)} and v={tuple(v.shape)}."
        )
    if q.shape[head] % k.shape[head] != 0:
        raise ValueError(
            f"num_qo_heads must be divisible by num_kv_heads, but got Hq={q.shape[head]} and Hk={k.shape[head]}."
        )
    if is_causal and q.shape[seq] != k.shape[seq]:
        raise ValueError(
            f"is_causal is valid only when qo_len equals kv_len, but got Sq={q.shape[seq]} and Sk={k.shape[seq]}."
        )


def _check_stride(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must have stride(-1) == 1 (contiguous last dim), but got stride {tuple(tensor.stride())}."
            )


def _check_params(
    qk_quant_gran: str,
    pv_accum_dtype: str,
    sm_scale: float | None,
    head_dim: int,
) -> None:
    if qk_quant_gran not in VALID_QK_QUANT_GRANS:
        raise ValueError(
            f"qk_quant_gran must be one of {list(VALID_QK_QUANT_GRANS)}, but got {qk_quant_gran!r}."
        )
    if pv_accum_dtype not in VALID_PV_ACCUM_DTYPES:
        raise ValueError(
            f"pv_accum_dtype must be one of {list(VALID_PV_ACCUM_DTYPES)}, but got {pv_accum_dtype!r}."
        )
    if sm_scale is not None and (not math.isfinite(sm_scale) or abs(sm_scale) > FP32_MAX):
        raise ValueError(
            f"sm_scale must be finite and within the fp32 representable range "
            f"({-FP32_MAX}, {FP32_MAX}), but got {sm_scale}."
        )
    if head_dim > PAD_BUCKETS[-1]:
        raise ValueError(
            f"head_dim must not be greater than {PAD_BUCKETS[-1]}, but got {head_dim}."
        )


def _get_env_or(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def _padded_head_dim(head_dim: int) -> int:
    for bucket in PAD_BUCKETS:
        if head_dim <= bucket:
            return bucket
    return head_dim


def _pad_to(value: int, bucket: int) -> int:
    return bucket - value if value < bucket else 0


def _resolve_plan(
    head_dim_og: int,
    sm_scale: float | None,
) -> dict:
    key = (head_dim_og, sm_scale)
    plan = _PLAN_CACHE.get(key)
    if plan is not None:
        return plan

    plan = {
        "head_dim_og": head_dim_og,
        "padded_head_dim": _padded_head_dim(head_dim_og),
        "sm_scale": sm_scale if sm_scale is not None else 1.0 / math.sqrt(head_dim_og),
    }
    if len(_PLAN_CACHE) >= _PLAN_CACHE_LIMIT:
        _PLAN_CACHE.pop(next(iter(_PLAN_CACHE)))
    _PLAN_CACHE[key] = plan
    return plan


def _forward_meta_shape(
    q: torch.Tensor, tensor_layout: str, return_lse: bool
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Meta 推导：o 与 q 同布局同 shape（dtype 同 q）；lse 为 [B, Hq, Sq] 的 FP32。"""
    out = torch.empty(q.shape, dtype=q.dtype, device=q.device)
    if return_lse:
        head = 2 if tensor_layout == "NHD" else 1
        seq = 1 if tensor_layout == "NHD" else 2
        lse = torch.empty(
            (q.shape[0], q.shape[head], q.shape[seq]), dtype=torch.float32, device=q.device
        )
    else:
        lse = None
    return out, lse


def _forward_impl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
    sm_scale: float | None = None,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """NPU 实现体：校验 + plan 解析 + head_dim pad + csrc 扩展调用 + 输出裁剪。"""
    _check_tensor_layout(tensor_layout)
    _check_dtype(q, k, v)
    _check_shape(q, k, v, tensor_layout, is_causal)
    _check_stride(q, k, v)
    _check_params(qk_quant_gran, pv_accum_dtype, sm_scale, q.shape[3])

    # smooth_v 仅 fp32 全程累加档生效；其余档 warning 后忽略（对齐开源）
    if smooth_v and pv_accum_dtype != "fp32":
        warnings.warn(
            "smooth_v is only supported with pv_accum_dtype='fp32' and will be ignored for "
            f"pv_accum_dtype={pv_accum_dtype!r}.",
            UserWarning,
            stacklevel=3,
        )
        smooth_v = False

    plan = _resolve_plan(q.shape[3], sm_scale)
    head_dim_og = plan["head_dim_og"]
    padded_head_dim = plan["padded_head_dim"]
    effective_scale = plan["sm_scale"]

    pad = _pad_to(head_dim_og, padded_head_dim)
    if pad > 0:
        q = torch.nn.functional.pad(q, (0, pad))
        k = torch.nn.functional.pad(k, (0, pad))
        v = torch.nn.functional.pad(v, (0, pad))

    from ._extension import sa2_attention

    out_padded, lse = sa2_attention(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=effective_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )
    # kernel 已返回最终 LSE（核内含平滑校正），此处直接透传，不做二次校正
    out = out_padded[..., :head_dim_og]
    if pad > 0:
        out = out.contiguous()
    return out, (lse if return_lse else None)


def _forward_meta(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
    sm_scale: float | None = None,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Meta 实现：复用全部校验，仅做 shape/dtype 推导。"""
    _check_tensor_layout(tensor_layout)
    _check_dtype(q, k, v)
    _check_shape(q, k, v, tensor_layout, is_causal)
    _check_stride(q, k, v)
    _check_params(qk_quant_gran, pv_accum_dtype, sm_scale, q.shape[3])
    return _forward_meta_shape(q, tensor_layout, return_lse)


def _forward_cpu(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
    sm_scale: float | None = None,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """CPU 参考实现分发：复用全部参数校验与 warning，转调 reference.sageattn_reference。"""
    _check_tensor_layout(tensor_layout)
    _check_dtype(q, k, v)
    _check_shape(q, k, v, tensor_layout, is_causal)
    _check_stride(q, k, v)
    _check_params(qk_quant_gran, pv_accum_dtype, sm_scale, q.shape[3])

    if smooth_v and pv_accum_dtype != "fp32":
        warnings.warn(
            "smooth_v is only supported with pv_accum_dtype='fp32' and will be ignored for "
            f"pv_accum_dtype={pv_accum_dtype!r}.",
            UserWarning,
            stacklevel=3,
        )
        smooth_v = False

    plan = _resolve_plan(q.shape[-1], sm_scale)
    effective_scale = plan["sm_scale"]

    from .reference import sageattn_reference

    res = sageattn_reference(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=effective_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )
    if return_lse:
        out, lse = res
        return out, lse
    return res, None


_SCHEMA_OP = "sageattention2::forward"
_SCHEMA = (
    "(Tensor q, Tensor k, Tensor v, *, str tensor_layout=\"HND\", bool is_causal=False, "
    "str qk_quant_gran=\"per_thread\", float? sm_scale=None, str pv_accum_dtype=\"fp32+fp16\", "
    "bool smooth_k=True, bool smooth_v=False, bool return_lse=False) -> (Tensor, Tensor?)"
)

torch.library.define(_SCHEMA_OP, _SCHEMA)
torch.library.impl(_SCHEMA_OP, "CPU")(_forward_cpu)
torch.library.impl(_SCHEMA_OP, "PrivateUse1")(_forward_impl)
try:
    torch.library.impl(_SCHEMA_OP, "AutogradPrivateUse1")(_forward_impl)
except Exception:  # noqa: BLE001, S110
    pass
torch.library.register_fake(_SCHEMA_OP)(_forward_meta)


def sageattn_qk_int8_pv_fp8_asc(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    qk_quant_gran: str = DEFAULT_QK_QUANT_GRAN,
    sm_scale: float | None = None,
    pv_accum_dtype: str = DEFAULT_PV_ACCUM_DTYPE,
    smooth_k: bool = True,
    smooth_v: bool = False,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """INT8 QK^T + FP8 PV 注意力（NPU，Ascend 950）；参数语义与开源 sageattn_qk_int8_pv_fp8_cuda 逐项对齐。"""
    if "attn_mask" in kwargs and kwargs.pop("attn_mask") is not None:
        raise ValueError("sageattn_qk_int8_pv_fp8_asc does not support attn_mask.")
    if kwargs:
        unexpected = sorted(kwargs.keys())
        raise TypeError(f"sageattn_qk_int8_pv_fp8_asc() got unexpected keyword argument(s): {unexpected}")
    out, lse = torch.ops.sageattention2.forward(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )
    return (out, lse) if return_lse else out


def sageattn(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str = "HND",
    is_causal: bool = False,
    sm_scale: float | None = None,
    return_lse: bool = False,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """自动分发入口：NPU 上固定分发到 FP8 PV 实现（sageattn_qk_int8_pv_fp8_asc）。"""
    if "attn_mask" in kwargs and kwargs.pop("attn_mask") is not None:
        raise ValueError("sageattn does not support attn_mask.")
    pv_accum_dtype = kwargs.pop("pv_accum_dtype", None) or _get_env_or(
        _ENV_PV_ACCUM_DTYPE, DEFAULT_PV_ACCUM_DTYPE
    )
    qk_quant_gran = kwargs.pop("qk_quant_gran", None) or _get_env_or(
        _ENV_QK_QUANT_GRAN, DEFAULT_QK_QUANT_GRAN
    )
    smooth_k = kwargs.pop("smooth_k", True)
    smooth_v = kwargs.pop("smooth_v", False)
    if kwargs:
        unexpected = sorted(kwargs.keys())
        raise TypeError(f"sageattn() got unexpected keyword argument(s): {unexpected}")
    return sageattn_qk_int8_pv_fp8_asc(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=is_causal,
        qk_quant_gran=qk_quant_gran,
        sm_scale=sm_scale,
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=smooth_k,
        smooth_v=smooth_v,
        return_lse=return_lse,
    )


@contextlib.contextmanager
def replace_sdpa(**kwargs: Any):
    """上下文管理器：作用域内将 F.scaled_dot_product_attention 替换为 sageattn（plug-and-play 冒烟）。"""
    functional = torch.nn.functional
    original_functional = functional.scaled_dot_product_attention
    original_torch = getattr(torch, "scaled_dot_product_attention", None)

    def _sage_sdpa(query, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, scale=None, **sdpa_kwargs):
        if attn_mask is not None:
            raise ValueError("sageattn replacement of scaled_dot_product_attention does not support attn_mask.")
        if dropout_p != 0.0:
            raise ValueError("sageattn replacement of scaled_dot_product_attention does not support dropout.")

        merged_kwargs = {**kwargs, **sdpa_kwargs}
        merged_kwargs.pop("enable_gqa", None)

        effective_scale = scale
        if effective_scale is None:
            effective_scale = merged_kwargs.pop("sm_scale", merged_kwargs.pop("scale", None))
        else:
            merged_kwargs.pop("sm_scale", None)
            merged_kwargs.pop("scale", None)

        ctx_causal = merged_kwargs.pop("is_causal", False)
        effective_is_causal = is_causal or ctx_causal

        return sageattn(
            query, key, value,
            is_causal=effective_is_causal,
            sm_scale=effective_scale,
            **merged_kwargs,
        )

    functional.scaled_dot_product_attention = _sage_sdpa
    if original_torch is not None:
        torch.scaled_dot_product_attention = _sage_sdpa
    try:
        yield
    finally:
        functional.scaled_dot_product_attention = original_functional
        if original_torch is not None:
            torch.scaled_dot_product_attention = original_torch
