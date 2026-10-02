"""SageAttention2 Reusable Input Validation and Helper Utilities."""
import math
import os
import warnings
from typing import Any, NamedTuple

import torch

from ._constants import (
    DEFAULT_PV_ACCUM_DTYPE,
    DEFAULT_QK_QUANT_GRAN,
    FP32_MAX,
    MAX_HEAD_DIM,
    PAD_BUCKETS,
    VALID_PV_ACCUM_DTYPES,
    VALID_QK_QUANT_GRANS,
    VALID_TENSOR_LAYOUTS,
)


def check_tensor_layout(tensor_layout: str) -> None:
    if tensor_layout not in VALID_TENSOR_LAYOUTS:
        raise ValueError(
            f"tensor_layout must be one of {list(VALID_TENSOR_LAYOUTS)}, but got {tensor_layout!r}."
        )


def check_dtype(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
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


def check_shape(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, tensor_layout: str, is_causal: bool) -> None:
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


def check_stride(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.stride(-1) != 1:
            raise ValueError(
                f"{name} must have stride(-1) == 1 (contiguous last dim), but got stride {tuple(tensor.stride())}."
            )


def check_params(
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
    if head_dim <= 0 or head_dim > MAX_HEAD_DIM:
        raise ValueError(
            f"head_dim must be in (0, {MAX_HEAD_DIM}], but got {head_dim}."
        )


def validate_all_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    pv_accum_dtype: str,
    sm_scale: float | None,
) -> None:
    """统一聚合校验输入张量维度、设备、类型与超参数。"""
    check_tensor_layout(tensor_layout)
    check_dtype(q, k, v)
    check_shape(q, k, v, tensor_layout, is_causal)
    check_stride(q, k, v)
    check_params(qk_quant_gran, pv_accum_dtype, sm_scale, q.shape[3])


def sanitize_smooth_v(smooth_v: bool, pv_accum_dtype: str) -> bool:
    """smooth_v 仅 fp32 全程累加档生效；其余档 warning 后忽略（对齐开源）。"""
    if smooth_v and pv_accum_dtype != "fp32":
        warnings.warn(
            "smooth_v is only supported with pv_accum_dtype='fp32' and will be ignored for "
            f"pv_accum_dtype={pv_accum_dtype!r}.",
            UserWarning,
            stacklevel=3,
        )
        return False
    return smooth_v


def get_env_or_default(name: str, default: str) -> str:
    value = os.environ.get(name)
    return value if value else default


def padded_head_dim(head_dim: int) -> int:
    for bucket in PAD_BUCKETS:
        if head_dim <= bucket:
            return bucket
    return head_dim


def pad_to_bucket(value: int, bucket: int) -> int:
    return bucket - value if value < bucket else 0


class QKV(NamedTuple):
    """Container for attention Q, K, V tensors.

    Supports:
    - Standard tuple unpacking: `q, k, v = create_qkv(...)`
    - Attribute access: `tensors.q`, `tensors.k`, `tensors.v`
    - Dict conversion: `tensors.as_dict()` -> `{"q": ..., "k": ..., "v": ...}`
    """

    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor

    def as_dict(self) -> dict[str, torch.Tensor]:
        return {"q": self.q, "k": self.k, "v": self.v}


def create_qkv(
    b: int = 1,
    hq: int = 32,
    sq: int = 4096,
    d: int = 128,
    hk: int | None = None,
    sk: int | None = None,
    *,
    dtype: torch.dtype | str = torch.float16,
    device: str | torch.device = "cpu",
    layout: str = "HND",
    generator: torch.Generator | None = None,
    requires_grad: bool = False,
    scale_factor: int = 1,
    **kwargs: Any,
) -> QKV:
    """统一注意力 Q/K/V 输入张量生成工厂。

    消除测试、基准压测与调用中重复的 `torch.randn(...)` 模版代码。
    支持字段别名兼容（batch / n_heads / seq_len / head_dim 等）与字典解包（**spec）。

    参数：
        b / batch: 批大小，默认为 1。
        hq / n_heads / num_heads: Query 头数，默认为 32。
        sq / seq_len / qo_len: Query 序列长，默认为 4096。
        d / head_dim: Head Dim 维度，默认为 128。
        hk / num_kv_heads: Key/Value 头数，若为 None 则默认同 hq（支持 GQA/MQA）。
        sk / kv_len: Key/Value 序列长，若为 None 则默认同 sq（支持非对称/Decode）。
        dtype: 数据类型（torch.float16 / torch.bfloat16 或字符串 "float16"/"bfloat16"）。
        device: 设备（"cpu" / "npu" / "cuda" 或 torch.device）。
        layout / tensor_layout: 内存布局（"HND" 或 "NHD"），默认为 "HND"。
        scale_factor: 序列缩减因子（用于 quick smoke 压测）。
    """
    batch = kwargs.pop("batch", b)
    num_q_heads = kwargs.pop("n_heads", kwargs.pop("num_heads", hq))
    seq_q = kwargs.pop("seq_len", kwargs.pop("qo_len", sq))
    head_dim = kwargs.pop("head_dim", d)
    num_kv_heads = kwargs.pop("num_kv_heads", hk if hk is not None else num_q_heads)
    seq_k = kwargs.pop("kv_len", sk if sk is not None else seq_q)
    tensor_layout = kwargs.pop("tensor_layout", layout)

    if scale_factor > 1:
        seq_q = max(1, seq_q // scale_factor)
        seq_k = max(1, seq_k // scale_factor)

    if isinstance(dtype, str):
        dtype = torch.bfloat16 if dtype in ("bfloat16", "bf16") else torch.float16

    if tensor_layout == "HND":
        shape_q = (batch, num_q_heads, seq_q, head_dim)
        shape_k = (batch, num_kv_heads, seq_k, head_dim)
        shape_v = (batch, num_kv_heads, seq_k, head_dim)
    else:  # NHD
        shape_q = (batch, seq_q, num_q_heads, head_dim)
        shape_k = (batch, seq_k, num_kv_heads, head_dim)
        shape_v = (batch, seq_k, num_kv_heads, head_dim)

    q = torch.randn(shape_q, dtype=dtype, device=device, generator=generator, requires_grad=requires_grad)
    k = torch.randn(shape_k, dtype=dtype, device=device, generator=generator, requires_grad=requires_grad)
    v = torch.randn(shape_v, dtype=dtype, device=device, generator=generator, requires_grad=requires_grad)
    return QKV(q=q, k=k, v=v)


def create_attention_inputs(
    spec: dict[str, Any] | None = None,
    *,
    device: str | torch.device = "cpu",
    scale_factor: int = 1,
    **kwargs: Any,
) -> tuple[QKV, dict[str, Any]]:
    """将配置字典一分为二：
    1. 构造 QKV 张量集合 (qkv: QKV)
    2. 提取注意力执行参数字典 (op_kwargs: dict[str, Any])

    示例：
        qkv, op_kwargs = create_attention_inputs(BENCHMARK_SPECS["P-01"], device="npu")
        out = sageattn(qkv.q, qkv.k, qkv.v, **op_kwargs)
        # 或直接字典展开
        out = sageattn(**qkv.as_dict(), **op_kwargs)
    """
    merged = {**(spec or {}), **kwargs}
    qkv = create_qkv(device=device, scale_factor=scale_factor, **merged)
    op_keys = (
        "tensor_layout",
        "is_causal",
        "qk_quant_gran",
        "pv_accum_dtype",
        "smooth_k",
        "smooth_v",
        "return_lse",
        "sm_scale",
    )
    op_kwargs = {k: merged[k] for k in op_keys if k in merged}
    return qkv, op_kwargs


def compute_attention_flops(
    b: int, hq: int, sq: int, sk: int, d: int, is_causal: bool = False
) -> float:
    """计算标准注意力计算量 (FLOPs)：
    非因果：4 * B * H * Sq * Sk * D (QK^T + PV 各 2*B*H*Sq*Sk*D)
    因果：2 * B * H * Sq * Sk * D (下三角掩码下计算量减半)
    """
    factor = 2.0 if is_causal else 4.0
    return factor * float(b) * float(hq) * float(sq) * float(sk) * float(d)


def compute_memory_traffic_bytes(
    b: int, hq: int, hk: int, sq: int, sk: int, d: int, dtype: torch.dtype | str
) -> int:
    """计算注意力理论最小显存访问字节数 (Bytes)：Q 读 + K 读 + V 读 + Out 写"""
    elem_bytes = 2 if dtype in (torch.float16, torch.bfloat16, "float16", "bfloat16") else 4
    q_bytes = b * hq * sq * d * elem_bytes
    k_bytes = b * hk * sk * d * elem_bytes
    v_bytes = b * hk * sk * d * elem_bytes
    out_bytes = b * hq * sq * d * elem_bytes
    return q_bytes + k_bytes + v_bytes + out_bytes


def compute_geomean(values: list[float]) -> float:
    """计算加速比几何平均数"""
    valid = [v for v in values if v > 0]
    if not valid:
        return 0.0
    return math.exp(sum(math.log(v) for v in valid) / len(valid))

