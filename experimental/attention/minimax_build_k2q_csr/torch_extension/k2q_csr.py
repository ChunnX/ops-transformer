# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Optional, Tuple

import torch
from torch.library import impl

from cann_ops_transformer.op_builder import OpBuilder, get_as_library


class K2qCsrOpBuilder(OpBuilder):
    def __init__(self):
        super(K2qCsrOpBuilder, self).__init__("k2q_csr", category="attention")

    def sources(self):
        return ["csrc/attention/k2q_csr.cpp"]

    def schema(self) -> str:
        return (
            "k2q_csr(Tensor q2k, Tensor cu_seqlens, Tensor cu_block_lens, *, "
            "int order_method=0, int total_rows=-1, int max_kv=-1, "
            "int use_simt=0, int q_global_offset=0) -> (Tensor, Tensor, Tensor)"
        )

    def register_meta(self):
        @impl(get_as_library(), self.name, "Meta")
        def k2q_csr_meta(
            q2k,
            cu_seqlens,
            cu_block_lens,
            order_method=0,
            total_rows=-1,
            max_kv=-1,
            use_simt=0,
            q_global_offset=0,
        ):
            torch._check(
                q2k.dim() == 3,
                lambda: f"q2k must be 3-D [H, T, topk], got dim={q2k.dim()}",
            )
            num_heads = q2k.size(0)
            num_tokens = q2k.size(1)
            topk = q2k.size(2)
            tr = int(total_rows)
            if tr < 0:
                tr = 0
            row_ptr = torch.empty(num_heads, tr + 1, dtype=torch.int32, device="meta")
            q_ind = torch.empty(
                num_heads, num_tokens * topk, dtype=torch.int32, device="meta"
            )
            slot = torch.empty(
                num_heads, num_tokens * topk, dtype=torch.int32, device="meta"
            )
            return row_ptr, q_ind, slot


k2q_csr_op_builder = K2qCsrOpBuilder()


@impl(get_as_library(), k2q_csr_op_builder.name, "PrivateUse1")
def _k2q_csr(
    q2k,
    cu_seqlens,
    cu_block_lens,
    order_method=0,
    total_rows=-1,
    max_kv=-1,
    use_simt=0,
    q_global_offset=0,
):
    op_module = k2q_csr_op_builder.load()
    return op_module.k2q_csr(
        q2k,
        cu_seqlens,
        cu_block_lens,
        int(order_method),
        int(total_rows),
        int(max_kv),
        int(use_simt),
        int(q_global_offset),
    )


def k2q_csr(
    q2k: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_block_lens: torch.Tensor,
    *,
    order_method: int = 0,
    total_rows: Optional[int] = None,
    max_kv: Optional[int] = None,
    use_simt: bool = False,
    q_global_offset: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """将 q2k 索引转为 k2q CSR，编排五阶段 aclnn 算子。

    Args:
        q2k (Tensor): 每条 query 的 KV block 索引，shape ``[H, T, topk]``，dtype int32，NPU。
        cu_seqlens (Tensor): query 侧前缀和，shape ``[B+1]``，dtype int32，NPU。
        cu_block_lens (Tensor): KV block 前缀和，shape ``[B+1]``，dtype int32，NPU。
            末元素即 ``total_rows``。
        order_method (int): ``0`` 按 batch 内 block 顺序建 row_map；``1`` 为 round-robin 交错。
            默认 ``0``。
        total_rows (int, optional): KV 总行数。传入非负整数则 Host 直用；``None`` 时从
            ``cu_block_lens`` Device 回读末元素。建议已知 shape 时显式传入。
        max_kv (int, optional): 单 batch 最大 KV block 数。``None`` 时从 ``cu_block_lens``
            差分最大值推导。
        use_simt (bool): ``True`` 时 Hist/Scatter 走 SIMT，仅 Ascend 950 生效；
            910b / 910_93 由 tiling 强制为 0。默认 ``False``。
        q_global_offset (bool): ``True`` 时 ``q_ind`` 为全局 Q token 下标；
            ``False`` 为 batch-local。默认 ``False``。

    Returns:
        Tuple[Tensor, Tensor, Tensor]:
            ``row_ptr`` shape ``[H, total_rows+1]``；
            ``q_ind`` / ``slot`` shape ``[H, T*topk]``，未写入位为 ``-1``。
    """
    if order_method not in (0, 1):
        raise ValueError(f"order_method must be 0 or 1, got {order_method}")
    if total_rows is not None and total_rows < 0:
        raise ValueError(f"total_rows must be >= 0 when provided, got {total_rows}")
    if max_kv is not None and max_kv < 0:
        raise ValueError(f"max_kv must be >= 0 when provided, got {max_kv}")
    tr = -1 if total_rows is None else int(total_rows)
    mk = -1 if max_kv is None else int(max_kv)
    return _k2q_csr(
        q2k,
        cu_seqlens,
        cu_block_lens,
        int(order_method),
        tr,
        mk,
        int(bool(use_simt)),
        int(bool(q_global_offset)),
    )
