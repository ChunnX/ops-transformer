# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Torch dispatcher entry for the packaged FlashKDA metadata AICPU executable."""
import torch
from torch.library import impl
from cannbotdsl.aicpu.toolchain import CompiledAicpuKernel
from cann_ops_transformer.op_builder import get_as_library
from .op_kernel import flash_kda_metadata as _source

_flash_kda_metadata = _source.flash_kda_metadata
metadata_capacity_upper_bound = _source.metadata_capacity_upper_bound

def _load_native_metadata():
    from pathlib import Path
    binary = Path(__file__).parent / "_native/flash_kda_metadata.so"
    if not binary.is_file():
        raise FileNotFoundError(binary)
    kernel = _source.flash_kda_metadata_kernel
    compiled = CompiledAicpuKernel(
        str(binary), kernel.spec, name=kernel.fn.__name__, launch_mode="interface"
    )
    # Keep the arena source byte-for-byte unchanged. The product wrapper owns
    # binding its packaged executable to the arena module's documented cache.
    with _source._AICPU_LOCK:
        _source._AICPU_COMPILED = compiled

_load_native_metadata()
_library = get_as_library()
_library.define(
    "flash_kda_metadata(Tensor q, Tensor v, Tensor initial_state, "
    "str layout_qkv, Tensor? cu_seqlens=None) -> Tensor"
)


@impl(_library, "flash_kda_metadata", "PrivateUse1")
def _npu(q, v, initial_state, layout_qkv, cu_seqlens=None):
    return _flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens)


@impl(_library, "flash_kda_metadata", "Meta")
def _meta(q, v, initial_state, layout_qkv, cu_seqlens=None):
    batch = initial_state.shape[0]
    length = q.shape[2] if layout_qkv == "BNSD" else q.shape[1] if layout_qkv == "BSND" else q.shape[0]
    capacity = metadata_capacity_upper_bound(batch, length, is_packed=layout_qkv == "TND")
    return q.new_empty((capacity,), dtype=torch.int32)


def flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens=None):
    """Build int32 scheduling metadata on the NPU; TND requires int32 cu_seqlens."""
    return torch.ops.cann_ops_transformer.flash_kda_metadata(q, v, initial_state, layout_qkv, cu_seqlens)
