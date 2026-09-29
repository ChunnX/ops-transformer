# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""AttnPrologue 的 PyTorch schema、Meta 推导和 NPU 分派。"""

import math

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class AttnPrologueOpBuilder(OpBuilder):
    def __init__(self):
        super().__init__("ds41.attn_prologue", category="attention")

    def sources(self):
        return []

    def schema(self):
        # Cache 是可变输入；三个返回张量均为新分配的输出。
        return [
            "ds41.attn_prologue(Tensor x, Tensor wqa, Tensor wqb, Tensor wkv, "
            "Tensor descale_x, Tensor descale_wqa, Tensor descale_wqb, Tensor descale_wkv, "
            "Tensor norm_weight_qr, Tensor norm_weight_kv, Tensor rope_sin, Tensor rope_cos, "
            "Tensor cache_index, Tensor(a!) kv_cache, *, float norm_eps=1e-6) "
            "-> (Tensor, Tensor, Tensor)"
        ]

    def register_meta(self):
        impl(get_as_library(), self.name, "Meta")(_attn_prologue_meta)


def _attn_prologue_meta(
    x, wqa, wqb, wkv, descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv, rope_sin, rope_cos, cache_index, kv_cache,
    *, norm_eps=1e-6,
):
    """只访问逻辑元数据，不加载 DSL 或查询 NPU 物理布局。"""
    if not math.isfinite(norm_eps) or norm_eps <= 0:
        raise ValueError("Parameter norm_eps must be finite and positive")
    for tensor in (x, wqa, wqb, wkv):
        torch._check(tensor.ndim == 2, lambda: "Projection inputs must be matrices")
        torch._check(tensor.dtype == torch.float8_e4m3fn, lambda: "Payloads must use E4M3FN")
        for size in tensor.shape:
            torch._check(size > 0, lambda: "Projection dimensions must be positive")
    tokens, hidden = x.shape
    rank, head = wqa.shape[0], wkv.shape[0]
    for width in (hidden, rank, head):
        torch._check(width % 64 == 0, lambda: "H, R and D must be divisible by 64")
    torch._check(wqa.shape[1] == hidden, lambda: "Wqa hidden dimension mismatch")
    torch._check(wkv.shape[1] == hidden, lambda: "Wkv hidden dimension mismatch")
    torch._check(wqb.shape[1] == rank, lambda: "Wqb rank dimension mismatch")
    torch._check(wqb.shape[0] % head == 0, lambda: "Wqb output width must be divisible by D")
    return (
        x.new_empty((tokens, wqb.shape[0] // head, head), dtype=torch.bfloat16),
        x.new_empty((tokens, rank), dtype=torch.float8_e4m3fn),
        x.new_empty((tokens, rank // 64, 2), dtype=torch.int8),
    )


attn_prologue_op_builder = AttnPrologueOpBuilder()
attn_prologue_op_builder._ensure_initialized()


@impl(get_as_library(), attn_prologue_op_builder.name, "PrivateUse1")
def attn_prologue(
    x, wqa, wqb, wkv, descale_x, descale_wqa, descale_wqb, descale_wkv,
    norm_weight_qr, norm_weight_kv, rope_sin, rope_cos, cache_index, kv_cache,
    *, norm_eps=1e-6,
):
    """调用随 wheel 安装的当前 DSL 实现，由内核入口执行完整输入校验。"""
    from ._kernel import attn_prologue as dsl_attn_prologue

    return dsl_attn_prologue(
        x, wqa, wqb, wkv, descale_x, descale_wqa, descale_wqb, descale_wkv,
        norm_weight_qr, norm_weight_kv, rope_sin, rope_cos, cache_index, kv_cache,
        norm_eps=norm_eps,
    )
