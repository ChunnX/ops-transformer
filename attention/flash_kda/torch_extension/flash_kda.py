# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
"""Torch dispatcher entry for the packaged FlashKDA Native executable."""
import torch
from torch.library import impl
from cannbotdsl.package.native import register_resources
from cann_ops_transformer.op_builder import get_as_library
from .op_kernel.flash_kda import flash_kda as _flash_kda

register_resources(__package__)
_library = get_as_library()
_library.define(
    "flash_kda(Tensor q, Tensor k, Tensor v, Tensor g, Tensor beta, float scale, "
    "Tensor initial_state, Tensor A_log, Tensor dt_bias, float lower_bound, "
    "str layout_qkv, Tensor metadata) -> (Tensor, Tensor)"
)
impl(_library, "flash_kda", "PrivateUse1")(_flash_kda)


@impl(_library, "flash_kda", "Meta")
def _meta(q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata):
    return torch.empty_like(v), torch.empty_like(initial_state)


def flash_kda(q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata):
    """Return (output, final_state), using metadata from flash_kda_metadata.

    q/k/v/g and beta are BF16. initial_state, A_log and dt_bias are FP32.
    layout_qkv is BNSD, BSND or TND; both head dimensions are 128.
    """
    return torch.ops.cann_ops_transformer.flash_kda(
        q, k, v, g, beta, scale, initial_state, A_log, dt_bias, lower_bound, layout_qkv, metadata)
