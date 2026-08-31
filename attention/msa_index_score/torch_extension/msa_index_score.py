# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
from typing import Callable, Optional

import torch
from torch.library import impl
from cann_ops_transformer.op_builder import OpBuilder, get_as_library

SCORE_STRIDE_ALIGN = 16
DEFAULT_SPARSE_MODE = 3
DEFAULT_INIT_BLOCKS = 0
DEFAULT_LOCAL_BLOCKS = 1
DEFAULT_LAYOUT_KEY = "BBND"
SUPPORTED_LAYOUT_KEYS = ("TND", "BBND", "BNBD")
MSA_BLOCK_SIZE = 128


def _round_up(value: int, align: int) -> int:
    return (value + align - 1) // align * align


def _normalize_layout_key(layout_key: Optional[str]) -> str:
    layout = (
        DEFAULT_LAYOUT_KEY
        if layout_key is None or layout_key == ""
        else str(layout_key).upper()
    )
    if layout not in SUPPORTED_LAYOUT_KEYS:
        raise ValueError(f"layout_key must be TND, BBND or BNBD, got {layout_key}")
    return layout


def _infer_max_blocks(
    layout: str,
    block_table: Optional[torch.Tensor],
    actual_seq_klen: Optional[torch.Tensor],
) -> int:
    if layout != "TND":
        torch._check(
            block_table is not None,
            lambda: f"layout_key={layout} requires block_table",
        )
        torch._check(block_table.dim() == 2, lambda: "block_table must be 2D [B, MB]")
        return int(block_table.size(1))
    torch._check(
        actual_seq_klen is not None and actual_seq_klen.dim() == 1,
        lambda: "layout_key=TND requires actual_seq_klen [B+1]",
    )
    pref = [int(v) for v in actual_seq_klen.tolist()]
    max_blocks = 0
    for i in range(len(pref) - 1):
        kv = pref[i + 1] - pref[i]
        blocks = 0 if kv <= 0 else (kv + MSA_BLOCK_SIZE - 1) // MSA_BLOCK_SIZE
        if blocks > max_blocks:
            max_blocks = blocks
    return 1 if max_blocks <= 0 else max_blocks


def _empty_score(query: torch.Tensor, max_blocks: int, device) -> torch.Tensor:
    score_stride = _round_up(int(max_blocks), SCORE_STRIDE_ALIGN)
    return torch.empty(
        (query.size(1), query.size(0), score_stride),
        dtype=torch.float32,
        device=device,
    )


def _register_msa_index_score_impl(
    op_name: str, dispatch_key: str, handler: Callable[..., torch.Tensor]
) -> None:
    @impl(get_as_library(), op_name, dispatch_key)
    def msa_index_score(
        query: torch.Tensor,
        key: torch.Tensor,
        start_loc: torch.Tensor,
        block_table: Optional[torch.Tensor],
        scale: Optional[torch.Tensor],
        atten_mask: Optional[torch.Tensor],
        actual_seq_qlen: Optional[torch.Tensor],
        actual_seq_klen: Optional[torch.Tensor],
        *,
        layout_key: str = DEFAULT_LAYOUT_KEY,
        sparse_mode: int = DEFAULT_SPARSE_MODE,
        init_blocks: int = DEFAULT_INIT_BLOCKS,
        local_blocks: int = DEFAULT_LOCAL_BLOCKS,
    ):
        return handler(
            query=query,
            key=key,
            start_loc=start_loc,
            block_table=block_table,
            scale=scale,
            atten_mask=atten_mask,
            actual_seq_qlen=actual_seq_qlen,
            actual_seq_klen=actual_seq_klen,
            layout_key=layout_key,
            sparse_mode=sparse_mode,
            init_blocks=init_blocks,
            local_blocks=local_blocks,
        )


def _msa_index_score_meta(**kwargs) -> torch.Tensor:
    query = kwargs["query"]
    torch._check(query.dim() == 3, lambda: "query must be 3D [T, Hq, D]")
    layout = _normalize_layout_key(kwargs["layout_key"])
    max_blocks = _infer_max_blocks(
        layout, kwargs["block_table"], kwargs["actual_seq_klen"]
    )
    return _empty_score(query, max_blocks, device="meta")


def _msa_index_score_npu(**kwargs) -> torch.Tensor:
    key = kwargs["key"]
    layout = _normalize_layout_key(kwargs["layout_key"])
    if layout == "BBND" and key.dim() == 3 and int(key.size(1)) == MSA_BLOCK_SIZE:
        key = key.unsqueeze(2)
    return _get_op_module().msa_index_score(
        kwargs["query"],
        key,
        kwargs["block_table"],
        kwargs["scale"],
        kwargs["atten_mask"],
        kwargs["actual_seq_qlen"],
        kwargs["actual_seq_klen"],
        kwargs["start_loc"],
        layout,
        kwargs["sparse_mode"],
        kwargs["init_blocks"],
        kwargs["local_blocks"],
    )


class MsaIndexScoreOpBuilder(OpBuilder):
    def __init__(self):
        super(MsaIndexScoreOpBuilder, self).__init__(
            "msa_index_score", category="attention"
        )

    def sources(self):
        """Path to C++ source code."""
        return ["csrc/attention/msa_index_score.cpp"]

    def schema(self) -> str:
        # Positional order must match the public Python API: query, key, start_loc.
        # cann_ops_transformer.__init__ re-exports torch.ops once the schema exists.
        return (
            "msa_index_score(Tensor query, Tensor key, Tensor start_loc, "
            "Tensor? block_table, Tensor? scale, Tensor? atten_mask, "
            'Tensor? actual_seq_qlen, Tensor? actual_seq_klen, *, str layout_key="BBND", '
            "int sparse_mode=3, int init_blocks=0, int local_blocks=1) -> Tensor"
        )

    def register_meta(self):
        _register_msa_index_score_impl(self.name, "Meta", _msa_index_score_meta)


_msa_index_score_op_builder = MsaIndexScoreOpBuilder()
_msa_index_score_op_builder._ensure_initialized()
_op_module = None


def _get_op_module():
    global _op_module
    if _op_module is None:
        _op_module = _msa_index_score_op_builder.load()
    return _op_module


_register_msa_index_score_impl(
    _msa_index_score_op_builder.name, "PrivateUse1", _msa_index_score_npu
)


def msa_index_score(
    query: torch.Tensor,
    key: torch.Tensor,
    start_loc: torch.Tensor,
    *,
    block_table: Optional[torch.Tensor] = None,
    scale: Optional[torch.Tensor] = None,
    atten_mask: Optional[torch.Tensor] = None,
    actual_seq_qlen: Optional[torch.Tensor] = None,
    actual_seq_klen: Optional[torch.Tensor] = None,
    layout_key: str = DEFAULT_LAYOUT_KEY,
    sparse_mode: int = DEFAULT_SPARSE_MODE,
    init_blocks: int = DEFAULT_INIT_BLOCKS,
    local_blocks: int = DEFAULT_LOCAL_BLOCKS,
) -> torch.Tensor:
    """MSA Index Branch block score，封装 aclnnMsaIndexScore。

    950 额外支持 query/key 同型
    ``torch_npu.hifloat8`` / ``torch.float8_e5m2`` / ``torch.float8_e4m3fn``（无 scale）。
    key 布局由 ``layout_key`` 指定：PA BBND ``[NP, P, N2, D]``、BNBD ``[NP, N2, P, D]``、TND ``[T2, N2, D]``。
    允许 ``q_len`` / ``kv_len`` 为 0（含整 batch）：跳过该请求的 QK；空 KV 的 score 为 ``-inf``。

    Args:
        query: ``[T1, N1, D]`` float16 / bfloat16；950 还可为三种 FP8
        key: 与 ``layout_key`` 对应；非量化须与 query 同 dtype；int8 仅 fp16 query + scale
        start_loc: ``[B]``，当前 query 所在逻辑 block 索引（local_mask）
        block_table: ``[B, MB]``，BBND/BNBD 必选；TND 不传
        scale: int8 反量化。PA 为 ``[NP, N2, P]``；TND 为 ``[T2, N2]``；非量化 / FP8 为 None
        atten_mask: sparse_mode=3 时 ``[2048, 2048]`` int8
        actual_seq_qlen: query 前缀和 ``[B+1]``
        actual_seq_klen: PA 为各请求 S2 ``[B]``；TND 为 key 前缀和 ``[B+1]``
        layout_key: ``TND`` / ``BBND`` / ``BNBD``，默认 ``BBND``
        sparse_mode: 0 或 3
        init_blocks / local_blocks: local_mask 强制选块；与 Triton raw score 对齐时置 0

    Returns:
        ``[N1, T1, RoundUp(MB, 16)]`` float32
    """
    return torch.ops.cann_ops_transformer.msa_index_score(
        query,
        key,
        start_loc,
        block_table,
        scale,
        atten_mask,
        actual_seq_qlen,
        actual_seq_klen,
        layout_key=_normalize_layout_key(layout_key),
        sparse_mode=sparse_mode,
        init_blocks=init_blocks,
        local_blocks=local_blocks,
    )
