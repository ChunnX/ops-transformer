#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Schema validation and invariant checks for ``input_data`` dicts.

These functions verify that an ``input_data`` dictionary (as produced by
``sparse_flash_mla_golden.gen_data``) is structurally consistent before and
after a batch-consistency transform.
"""

from typing import Any, Dict, List

import torch


# Fields that must always be present at the top level.
_REQUIRED_TOP_KEYS = ["B", "layout_q", "layout_kv", "params", "metadata_input", "input"]

# Fields that must be present inside ``input``.
_REQUIRED_INPUT_KEYS = [
    "q",
    "ori_kv",
    "cu_seqlens_q",
    "seqused_ori_kv",
    "seqused_cmp_kv",
    "cmp_residual_kv",
    "sinks",
    "softmax_scale",
    "ori_mask_mode",
    "cmp_mask_mode",
    "ori_win_left",
    "ori_win_right",
    "layout_q",
    "layout_kv",
]

# Fields that must be present inside ``metadata_input``.
_REQUIRED_METADATA_KEYS = [
    "N1",
    "N2",
    "D",
    "B",
    "max_seqlen_q",
    "max_seqlen_ori_kv",
    "max_seqlen_cmp_kv",
    "layout_q",
    "layout_kv",
]


class SchemaError(ValueError):
    """Raised when ``input_data`` fails schema validation."""


def _require_keys(data: Dict[str, Any], required: List[str], context: str) -> None:
    missing = [k for k in required if k not in data]
    if missing:
        raise SchemaError(f"{context}: missing keys {missing}")


def validate_schema(input_data: Dict[str, Any]) -> None:
    """Check that *input_data* has all required keys and consistent layout strings."""
    _require_keys(input_data, _REQUIRED_TOP_KEYS, "top-level")

    tensor_input = input_data["input"]
    _require_keys(tensor_input, _REQUIRED_INPUT_KEYS, "input")

    metadata_input = input_data["metadata_input"]
    _require_keys(metadata_input, _REQUIRED_METADATA_KEYS, "metadata_input")

    layout_q = tensor_input["layout_q"]
    layout_kv = tensor_input["layout_kv"]
    if isinstance(layout_q, (list, tuple)):
        layout_q = layout_q[0]
    if layout_q not in ("BSND", "TND"):
        raise SchemaError(f"Unsupported layout_q: {layout_q}")
    if layout_kv not in ("BSND", "TND", "PA_BBND"):
        raise SchemaError(f"Unsupported layout_kv: {layout_kv}")
    if input_data["layout_q"] != layout_q or input_data["layout_kv"] != layout_kv:
        raise SchemaError("top-level and input layout values do not match")

    B = int(input_data["B"])
    if B < 1:
        raise SchemaError(f"B must be >= 1, got {B}")

    if input_data.get("cpu_output") is None:
        raise SchemaError("cpu_output is None")

    if tensor_input["sinks"] is None:
        raise SchemaError("sinks is None")

    q = tensor_input["q"]
    if layout_q == "BSND" and q.dim() != 4:
        raise SchemaError(f"BSND Q must be rank 4, got rank {q.dim()}")
    if layout_q == "TND":
        if q.dim() != 3:
            raise SchemaError(f"TND Q must be rank 3, got rank {q.dim()}")
        if tensor_input.get("cu_seqlens_q") is None:
            raise SchemaError("TND Q requires cu_seqlens_q")

    if layout_kv == "TND":
        if tensor_input.get("ori_kv") is not None and tensor_input.get("cu_seqlens_ori_kv") is None:
            raise SchemaError("TND ori_kv requires cu_seqlens_ori_kv")
        if tensor_input.get("cmp_kv") is not None and tensor_input.get("cu_seqlens_cmp_kv") is None:
            raise SchemaError("TND cmp_kv requires cu_seqlens_cmp_kv")
    if layout_kv == "PA_BBND":
        if tensor_input.get("ori_kv") is not None and tensor_input.get("seqused_ori_kv") is None:
            raise SchemaError("PA_BBND ori_kv requires seqused_ori_kv")
        if tensor_input.get("cmp_kv") is not None and tensor_input.get("seqused_cmp_kv") is None:
            raise SchemaError("PA_BBND cmp_kv requires seqused_cmp_kv")


def get_q_lengths(input_data: Dict[str, Any]) -> List[int]:
    """Extract per-batch **physical** Q output lengths as a Python list.

    Priority (physical Q shape first, not seqused_q):

    * **BSND** – each batch has ``q.shape[1]`` tokens (the physical S1 dim).
    * **TND**  – each batch has ``cu_seqlens_q[i+1] - cu_seqlens_q[i]`` tokens.
    * Fallback – ``seqused_q`` if the above is unavailable.
    * Fallback – ``S1`` from top-level for BSND.

    ``seqused_q`` is *not* prioritised because the real golden commonly
    sets it to ``None``; even when present it represents valid (padded)
    lengths, not the physical output tensor dimension.
    """
    ti = input_data["input"]
    layout_q = ti["layout_q"]
    if isinstance(layout_q, (list, tuple)):
        layout_q = layout_q[0]
    B = int(input_data["B"])
    q = ti["q"]

    if layout_q == "BSND":
        S1 = q.shape[1]
        return [S1] * B
    elif layout_q == "TND":
        cu_seqlens_q = ti.get("cu_seqlens_q")
        if cu_seqlens_q is not None:
            cu_list = cu_seqlens_q.to("cpu", torch.int64).tolist()
            return [cu_list[i + 1] - cu_list[i] for i in range(len(cu_list) - 1)]
    # Last-resort fallbacks
    seqused_q = ti.get("seqused_q")
    if seqused_q is not None:
        return seqused_q.to("cpu", torch.int64).tolist()
    if layout_q == "BSND":
        S1 = int(input_data.get("S1", q.shape[1] if q.dim() >= 2 else 0))
        return [S1] * B
    raise SchemaError("Cannot determine per-batch Q lengths")


# Backward-compatible alias
_get_seqused_q = get_q_lengths


def check_invariants(input_data: Dict[str, Any]) -> None:
    """Verify internal consistency after a transform.

    Checks performed:

    * ``B`` matches the length of every per-batch vector.
    * ``cu_seqlens_*`` prefix sums match actual tensor shapes.
    * ``seqused_*`` lengths do not exceed tensor dim-1.
    * ``max_seqlen_*`` in ``metadata_input`` are at least the max of ``seqused_*``.
    * PA block table ids are within physical cache range (when applicable).
    """
    ti = input_data["input"]
    mi = input_data["metadata_input"]
    B = int(input_data["B"])
    layout_q = ti["layout_q"]
    if isinstance(layout_q, (list, tuple)):
        layout_q = layout_q[0]
    layout_kv = ti["layout_kv"]

    # -- per-batch vector lengths must match B --
    for field_name in ("seqused_q", "seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv"):
        val = ti.get(field_name)
        if val is not None and val.shape[0] != B:
            raise SchemaError(
                f"{field_name} length {val.shape[0]} != B={B}"
            )

    # -- Q tensor consistency --
    q = ti["q"]
    q_lengths = get_q_lengths(input_data)
    if layout_q == "BSND":
        if q.shape[0] != B:
            raise SchemaError(f"Q dim-0 {q.shape[0]} != B={B}")
        cu_q = ti.get("cu_seqlens_q")
        if cu_q is not None:
            cu_list = cu_q.to("cpu", torch.int64).tolist()
            expected = [i * q.shape[1] for i in range(B + 1)]
            if cu_list != expected:
                raise SchemaError(f"BSND cu_seqlens_q {cu_list} != expected {expected}")
    elif layout_q == "TND":
        cu_q = ti.get("cu_seqlens_q")
        if cu_q is not None:
            cu_list = cu_q.to("cpu", torch.int64).tolist()
            if len(cu_list) != B + 1:
                raise SchemaError(f"cu_seqlens_q length {len(cu_list)} != B+1={B + 1}")
            if cu_list[-1] != q.shape[0]:
                raise SchemaError(
                    f"cu_seqlens_q[-1]={cu_list[-1]} != Q dim-0 {q.shape[0]}"
                )
            derived = [cu_list[i + 1] - cu_list[i] for i in range(B)]
            if derived != q_lengths:
                raise SchemaError(
                    f"cu_seqlens_q-derived lengths {derived} != q_lengths {q_lengths}"
                )

    seqused_q = ti.get("seqused_q")
    if seqused_q is not None:
        valid_q_lengths = seqused_q.to("cpu", torch.int64).tolist()
        if any(length < 0 or length > physical for length, physical in zip(valid_q_lengths, q_lengths)):
            raise SchemaError(
                f"seqused_q values {valid_q_lengths} exceed physical Q lengths {q_lengths}"
            )

    # -- KV tensor consistency --
    ori_kv = ti["ori_kv"]
    seqused_ori_kv = ti.get("seqused_ori_kv")
    if layout_kv == "BSND":
        if ori_kv is not None and ori_kv.shape[0] != B:
            raise SchemaError(f"ori_kv dim-0 {ori_kv.shape[0]} != B={B}")
        if seqused_ori_kv is not None:
            max_s = int(seqused_ori_kv.max().item())
            if ori_kv is not None and max_s > ori_kv.shape[1]:
                raise SchemaError(
                    f"max seqused_ori_kv {max_s} > ori_kv dim-1 {ori_kv.shape[1]}"
                )
    elif layout_kv == "TND":
        cu_ori = ti.get("cu_seqlens_ori_kv")
        if cu_ori is not None:
            cu_list = cu_ori.to("cpu", torch.int64).tolist()
            if len(cu_list) != B + 1:
                raise SchemaError(f"cu_seqlens_ori_kv length {len(cu_list)} != B+1={B + 1}")
            if ori_kv is not None and cu_list[-1] != ori_kv.shape[0]:
                raise SchemaError(
                    f"cu_seqlens_ori_kv[-1]={cu_list[-1]} != ori_kv dim-0 {ori_kv.shape[0]}"
                )
    elif layout_kv == "PA_BBND":
        ori_bt = ti.get("ori_block_table")
        if ori_bt is not None and ori_bt.shape[0] != B:
            raise SchemaError(f"ori_block_table dim-0 {ori_bt.shape[0]} != B={B}")
        cmp_bt = ti.get("cmp_block_table")
        if cmp_bt is not None and cmp_bt.shape[0] != B:
            raise SchemaError(f"cmp_block_table dim-0 {cmp_bt.shape[0]} != B={B}")

    # -- cmp_kv consistency (TND) --
    cmp_kv = ti.get("cmp_kv")
    cu_cmp = ti.get("cu_seqlens_cmp_kv")
    if layout_kv == "TND" and cmp_kv is not None and cu_cmp is not None:
        cu_list = cu_cmp.to("cpu", torch.int64).tolist()
        if len(cu_list) != B + 1:
            raise SchemaError(f"cu_seqlens_cmp_kv length {len(cu_list)} != B+1={B + 1}")
        if cu_list[-1] != cmp_kv.shape[0]:
            raise SchemaError(
                f"cu_seqlens_cmp_kv[-1]={cu_list[-1]} != cmp_kv dim-0 {cmp_kv.shape[0]}"
            )

    # -- metadata_input derived fields --
    max_seqlen_q = mi.get("max_seqlen_q")
    if max_seqlen_q is not None and q_lengths:
        actual_max = max(q_lengths)
        if int(max_seqlen_q) < actual_max:
            raise SchemaError(
                f"metadata max_seqlen_q {max_seqlen_q} < actual max Q length {actual_max}"
            )

    seqused_ori_kv_mi = ti.get("seqused_ori_kv")
    max_seqlen_ori_kv = mi.get("max_seqlen_ori_kv")
    if seqused_ori_kv_mi is not None and max_seqlen_ori_kv is not None:
        actual_max = int(seqused_ori_kv_mi.max().item())
        if int(max_seqlen_ori_kv) < actual_max:
            raise SchemaError(
                f"metadata max_seqlen_ori_kv {max_seqlen_ori_kv} < actual max {actual_max}"
            )

    seqused_cmp_kv_mi = ti.get("seqused_cmp_kv")
    max_seqlen_cmp_kv = mi.get("max_seqlen_cmp_kv")
    if seqused_cmp_kv_mi is not None and max_seqlen_cmp_kv is not None:
        actual_max = int(seqused_cmp_kv_mi.max().item())
        if int(max_seqlen_cmp_kv) < actual_max:
            raise SchemaError(
                f"metadata max_seqlen_cmp_kv {max_seqlen_cmp_kv} < actual max {actual_max}"
            )

    # -- PA block id range check --
    if layout_kv == "PA_BBND" and ori_kv is not None:
        from .layouts import validate_block_ids
        cache_block_num = ori_kv.shape[0]
        validate_block_ids(ti.get("ori_block_table"), cache_block_num)
        cmp_bt = ti.get("cmp_block_table")
        if cmp_kv is not None:
            cmp_cache_block_num = cmp_kv.shape[0]
            validate_block_ids(cmp_bt, cmp_cache_block_num)


def build_baseline_origins(input_data: Dict[str, Any]) -> List[Any]:
    """Build the token-origin list for a baseline (untransformed) case.

    Returns a list of :class:`TokenOrigin` in output order.
    For BSND output the order is ``(b, s)`` lexicographic.
    For TND output the order matches the flattened T1 dimension.
    """
    from .model import TokenOrigin

    q_lengths = get_q_lengths(input_data)
    origins = []
    for b, length in enumerate(q_lengths):
        for s in range(length):
            origins.append(TokenOrigin(source_batch=b, source_token=s))
    return origins
