# SageAttention2 torch 扩展加载器：JIT 构建 csrc/attention.cpp 并提供 NPU 前向入口与流句柄解析
import os

import torch

_EXT = None
_EXT_NAME = "sageattention2_ext"
_EXT_VERBOSE = os.environ.get("SAGEATTN_EXT_VERBOSE", "0") == "1"
# CANN toolkit 头文件（acl/acl_base.h 等）候选路径
_CANN_INCLUDE_CANDIDATES = (
    os.path.join(os.environ.get("ASCEND_HOME_PATH", ""), "include"),
    "/usr/local/Ascend/ascend-toolkit/latest/include",
    os.path.expanduser("~/Ascend/ascend-toolkit/latest/include"),
)


def _cann_include_dirs() -> list:
    dirs = []
    for candidate in _CANN_INCLUDE_CANDIDATES:
        if candidate and os.path.isdir(candidate):
            dirs.append(candidate)
    return dirs


def _load_extension():
    """懒加载并缓存 torch 扩展；构建产物由 torch cpp_extension 缓存机制复用。"""
    global _EXT
    if _EXT is not None:
        return _EXT
    try:
        import torch_npu  # noqa: F401  私有后端（PrivateUse1）初始化
    except ImportError as exc:
        raise ImportError("sageattention2 NPU path requires torch_npu to be installed.") from exc

    from torch.utils import cpp_extension

    csrc = os.path.join(os.path.dirname(os.path.abspath(__file__)), "csrc", "attention.cpp")
    _EXT = cpp_extension.load(
        name=_EXT_NAME,
        sources=[csrc],
        extra_include_paths=_cann_include_dirs(),
        extra_cflags=["-O2"],
        verbose=_EXT_VERBOSE,
    )
    return _EXT


def _resolve_stream_handle(device: torch.device) -> int:
    """解析当前 NPU 流句柄（aclrtStream），供 csrc 侧两段式第二段调用使用。"""
    import torch_npu  # noqa: F401

    index = device.index if device.index is not None else torch.npu.current_device()
    with torch.npu.device(index):
        return int(torch.npu.current_stream().npu_stream)


def sa2_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    tensor_layout: str,
    is_causal: bool,
    qk_quant_gran: str,
    sm_scale: float,
    pv_accum_dtype: str,
    smooth_k: bool,
    smooth_v: bool,
    return_lse: bool,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """调用 csrc 扩展执行 INT8 QK^T + FP8 PV 前向；return_lse 为 False 时 lse 返回 None。"""
    extension = _load_extension()
    stream_handle = _resolve_stream_handle(q.device)
    out, lse = extension.sa2_attention(
        q, k, v,
        tensor_layout=tensor_layout,
        is_causal=bool(is_causal),
        qk_quant_gran=qk_quant_gran,
        sm_scale=float(sm_scale),
        pv_accum_dtype=pv_accum_dtype,
        smooth_k=bool(smooth_k),
        smooth_v=bool(smooth_v),
        return_lse=bool(return_lse),
        stream_handle=stream_handle,
    )
    return out, (lse if return_lse else None)
