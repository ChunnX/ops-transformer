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

"""Batch-consistency transforms: Mode 1 (reorder) and Mode 2 (split).

Both modes share the same core ``_select_batches`` primitive which, given a
list of *batch_ids*, produces a new ``input_data`` dict containing only the
selected (and possibly reordered) batches.  The difference is:

* **Mode 1** – *batch_ids* is a full permutation of ``[0 .. B-1]``.
* **Mode 2** – *batch_ids* is a strict subset, and multiple sub-cases are
  produced from a partition of ``[0 .. B-1]``.

All transforms are pure: the original ``input_data`` is never mutated.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch

from .model import ConsistencyCase, TokenOrigin
from . import layouts
from . import schema


# ------------------------------------------------------------------ #
#  Core: select / reorder batches
# ------------------------------------------------------------------ #

def _clone_input_data(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """Shallow-copy the dict structure so nested dicts are independent.

    Tensors are shared by reference until a transform explicitly replaces
    them (``index_select`` / ``torch.cat`` always return new tensors).
    """
    new_data = {}
    for k, v in input_data.items():
        if isinstance(v, dict):
            new_data[k] = dict(v)
        elif isinstance(v, list):
            new_data[k] = list(v)
        else:
            new_data[k] = v
    new_data["input"] = dict(input_data["input"])
    new_data["metadata_input"] = dict(input_data["metadata_input"])
    new_data["params"] = dict(input_data["params"])
    return new_data


def _get_layout_q(input_data: Dict[str, Any]) -> str:
    layout_q = input_data["input"]["layout_q"]
    if isinstance(layout_q, (list, tuple)):
        layout_q = layout_q[0]
    return layout_q


def _select_q(input_data: Dict[str, Any], batch_ids: List[int]) -> Tuple[torch.Tensor, Optional[List[int]]]:
    """Select / reorder Q. Returns (new_q, new_cu_seqlens_q_or_None)."""
    ti = input_data["input"]
    layout_q = _get_layout_q(input_data)
    q = ti["q"]
    if layout_q == "BSND":
        new_q = layouts.select_bsnd(q, batch_ids)
        return new_q, None
    else:  # TND
        cu_q = ti["cu_seqlens_q"]
        if cu_q is None:
            raise schema.SchemaError("TND Q requires cu_seqlens_q")
        new_q, new_cu = layouts.select_tnd(q, cu_q, batch_ids)
        return new_q, new_cu


def _select_kv(
    kv: Optional[torch.Tensor],
    block_table: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    layout_kv: str,
    batch_ids: List[int],
) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[List[int]]]:
    """Select / reorder KV for one of ori/cmp.

    Returns (new_kv, new_block_table, new_cu_seqlens).
    For PA_BBND, kv and block_table are handled: kv unchanged, table rows selected.
    For TND, kv segments are selected/reordered, cu_seqlens recomputed.
    For BSND, kv batch dim is selected.
    """
    if layout_kv == "PA_BBND":
        new_kv = kv  # physical cache: unchanged (shared read-only)
        new_bt = layouts.select_pa_block_table(block_table, batch_ids)
        return new_kv, new_bt, None
    elif layout_kv == "TND":
        if kv is None:
            return None, None, None
        if cu_seqlens is None:
            raise schema.SchemaError("TND KV tensor requires matching cu_seqlens")
        new_kv, new_cu = layouts.select_tnd(kv, cu_seqlens, batch_ids)
        return new_kv, None, new_cu
    else:  # BSND
        new_kv = layouts.select_bsnd(kv, batch_ids)
        return new_kv, None, None


def _select_sparse_indices(
    sparse_indices: Optional[torch.Tensor],
    layout_q: str,
    orig_cu_seqlens_q: Optional[torch.Tensor],
    batch_ids: List[int],
) -> Tuple[Optional[torch.Tensor], Optional[List[int]]]:
    """Select / reorder sparse indices (laid out like Q).

    Uses *orig_cu_seqlens_q* (the pre-transform prefix sum) to segment the
    original tensor.

    Returns (new_sparse_indices, new_cu_seqlens_q_or_None).
    """
    if sparse_indices is None:
        return None, None
    if layout_q == "BSND":
        return layouts.select_bsnd(sparse_indices, batch_ids), None
    else:  # TND
        new_si, new_cu = layouts.select_tnd(sparse_indices, orig_cu_seqlens_q, batch_ids)
        return new_si, new_cu


def _cu_tensor(values: List[int], reference: Optional[torch.Tensor]) -> torch.Tensor:
    """Build a prefix-sum tensor while preserving the original dtype/device."""
    if reference is None:
        return torch.tensor(values, dtype=torch.int32)
    return torch.tensor(values, dtype=reference.dtype, device=reference.device)


def _max_kv_length(
    kv: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    layout_kv: str,
) -> int:
    """Return the metadata max length, matching golden generation semantics."""
    if seqused is not None and seqused.numel() > 0:
        return int(seqused.max().item())
    if kv is None:
        return 0
    if layout_kv == "BSND":
        return int(kv.shape[1])
    if layout_kv == "TND":
        if cu_seqlens is None:
            raise schema.SchemaError("Cannot derive TND KV max length without cu_seqlens")
        lengths = layouts.lengths_from_cu_seqlens(cu_seqlens)
        return max(lengths, default=0)
    raise schema.SchemaError("PA_BBND requires seqused KV lengths")


def _physical_kv_max_length(
    kv: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    layout_kv: str,
) -> int:
    """Return the physical/logical S2 capacity represented by a transformed case."""
    if kv is None:
        return 0
    if layout_kv == "BSND":
        return int(kv.shape[1])
    if layout_kv == "TND":
        if cu_seqlens is None:
            raise schema.SchemaError("Cannot derive TND S2 without cu_seqlens")
        return max(layouts.lengths_from_cu_seqlens(cu_seqlens), default=0)
    if seqused is None:
        raise schema.SchemaError("PA_BBND requires seqused KV lengths")
    return int(seqused.max().item()) if seqused.numel() else 0


def _logical_token_count(
    kv: Optional[torch.Tensor],
    seqused: Optional[torch.Tensor],
    layout_kv: str,
) -> Optional[int]:
    """Compute T2/T3 for reporting without copying PA physical cache storage."""
    if kv is None:
        return None
    if layout_kv == "TND":
        return int(kv.shape[0])
    if layout_kv == "BSND":
        return int(kv.shape[0] * kv.shape[1])
    if seqused is None:
        raise schema.SchemaError("PA_BBND requires seqused KV lengths")
    return int(seqused.to(torch.int64).sum().item())


def _select_batches(input_data: Dict[str, Any], batch_ids: List[int]) -> Dict[str, Any]:
    """Produce a new ``input_data`` with only *batch_ids* selected/reordered.

    This is the shared core for Mode 1 and Mode 2.
    """
    schema.validate_schema(input_data)

    new_data = _clone_input_data(input_data)
    ti = new_data["input"]
    mi = new_data["metadata_input"]
    params = new_data["params"]
    layout_q = _get_layout_q(input_data)
    layout_kv = ti["layout_kv"]
    new_B = len(batch_ids)

    # Save ORIGINAL cu_seqlens_q before any modification (for sparse indices / topk)
    orig_cu_q = input_data["input"].get("cu_seqlens_q")

    # -- Q --
    new_q, new_cu_q = _select_q(input_data, batch_ids)
    ti["q"] = new_q

    # -- ori_kv --
    ori_kv = ti.get("ori_kv")
    ori_bt = ti.get("ori_block_table")
    cu_ori = ti.get("cu_seqlens_ori_kv")
    new_ori_kv, new_ori_bt, new_cu_ori = _select_kv(ori_kv, ori_bt, cu_ori, layout_kv, batch_ids)
    ti["ori_kv"] = new_ori_kv
    if new_ori_bt is not None:
        ti["ori_block_table"] = new_ori_bt
    if new_cu_ori is not None:
        new_cu_ori_tensor = _cu_tensor(new_cu_ori, cu_ori)
        ti["cu_seqlens_ori_kv"] = new_cu_ori_tensor
        mi["cu_seqlens_ori_kv"] = new_cu_ori_tensor
        params["cu_seqlens_ori_kv"] = new_cu_ori

    # -- cmp_kv --
    cmp_kv = ti.get("cmp_kv")
    cmp_bt = ti.get("cmp_block_table")
    cu_cmp = ti.get("cu_seqlens_cmp_kv")
    new_cmp_kv, new_cmp_bt, new_cu_cmp = _select_kv(cmp_kv, cmp_bt, cu_cmp, layout_kv, batch_ids)
    if new_cmp_kv is not None:
        ti["cmp_kv"] = new_cmp_kv
    if new_cmp_bt is not None:
        ti["cmp_block_table"] = new_cmp_bt
    if new_cu_cmp is not None:
        new_cu_cmp_tensor = _cu_tensor(new_cu_cmp, cu_cmp)
        ti["cu_seqlens_cmp_kv"] = new_cu_cmp_tensor
        mi["cu_seqlens_cmp_kv"] = new_cu_cmp_tensor
        params["cu_seqlens_cmp_kv"] = new_cu_cmp

    # -- sparse indices (laid out like Q) -- must use ORIGINAL cu_seqlens_q
    cmp_sparse = ti.get("cmp_sparse_indices")
    if cmp_sparse is not None:
        new_cs, _ = _select_sparse_indices(cmp_sparse, layout_q, orig_cu_q, batch_ids)
        ti["cmp_sparse_indices"] = new_cs

    ori_sparse = ti.get("ori_sparse_indices")
    if ori_sparse is not None:
        new_os, _ = _select_sparse_indices(ori_sparse, layout_q, orig_cu_q, batch_ids)
        ti["ori_sparse_indices"] = new_os

    # -- cu_seqlens_q --
    if layout_q == "TND":
        # TND: cu_seqlens_q was recomputed by _select_q
        if new_cu_q is not None:
            new_cu_q_tensor = _cu_tensor(new_cu_q, orig_cu_q)
            ti["cu_seqlens_q"] = new_cu_q_tensor
            mi["cu_seqlens_q"] = new_cu_q_tensor
            params["cu_seqlens_q"] = new_cu_q
    else:
        # BSND: recompute cu_seqlens_q if it was originally set (not None)
        if orig_cu_q is not None:
            S1 = new_q.shape[1]
            new_cu_q_bsnd = list(range(0, (new_B + 1) * S1, S1))
            new_cu_q_tensor = _cu_tensor(new_cu_q_bsnd, orig_cu_q)
            ti["cu_seqlens_q"] = new_cu_q_tensor
            mi["cu_seqlens_q"] = new_cu_q_tensor
            params["cu_seqlens_q"] = new_cu_q_bsnd
        # If originally None, leave as None (don't insert)

    # -- per-batch vectors --
    for field_name in ("seqused_q", "seqused_ori_kv", "seqused_cmp_kv", "cmp_residual_kv"):
        val = ti.get(field_name)
        if val is not None:
            new_val = layouts.select_per_batch_vec(val, batch_ids)
            ti[field_name] = new_val
            mi[field_name] = new_val
            params[field_name] = new_val.to("cpu", torch.int64).tolist()

    # -- topk_length (if present, laid out like Q) -- must use ORIGINAL cu_seqlens_q for TND
    for tk_field in ("ori_topk_length", "cmp_topk_length"):
        tk = ti.get(tk_field)
        if tk is not None:
            if layout_q == "BSND":
                ti[tk_field] = layouts.select_bsnd(tk, batch_ids)
            else:  # TND: use ORIGINAL cu_seqlens_q to segment the ORIGINAL tensor
                new_tk, _ = layouts.select_tnd(tk, orig_cu_q, batch_ids)
                ti[tk_field] = new_tk
            mi[tk_field] = ti[tk_field]
            params[tk_field] = ti[tk_field].detach().cpu().tolist()

    # -- derived scalar fields --
    new_data["B"] = new_B
    mi["B"] = new_B
    params["B"] = new_B
    if "B" in ti:
        ti["B"] = new_B

    # Compute physical Q lengths for the new case
    new_q_lengths = schema.get_q_lengths(new_data)

    # T1 / T2 / T3 are reporting fields, but keeping them coherent makes
    # transformed PT cases independently inspectable and reusable.
    new_T1 = int(new_q.shape[0]) if layout_q == "TND" else int(new_q.shape[0] * new_q.shape[1])
    new_T2 = _logical_token_count(new_ori_kv, ti.get("seqused_ori_kv"), layout_kv)
    new_T3 = _logical_token_count(new_cmp_kv, ti.get("seqused_cmp_kv"), layout_kv)
    for field_name, value in (("T1", new_T1), ("T2", new_T2), ("T3", new_T3)):
        if value is None:
            continue
        new_data[field_name] = value
        params[field_name] = value
        if field_name in ti:
            ti[field_name] = value

    # S1 / S2 (top-level and params)
    new_S1 = max(new_q_lengths) if new_q_lengths else 1
    new_data["S1"] = new_S1
    params["S1"] = new_S1

    new_S2 = _physical_kv_max_length(
        new_ori_kv, ti.get("seqused_ori_kv"), ti.get("cu_seqlens_ori_kv"), layout_kv
    )
    new_data["S2"] = new_S2
    params["S2"] = new_S2

    # max_seqlen_*  (based on physical Q lengths, not seqused_q)
    mi["max_seqlen_q"] = max(new_q_lengths, default=1)
    mi["max_seqlen_ori_kv"] = _max_kv_length(
        new_ori_kv, ti.get("seqused_ori_kv"), ti.get("cu_seqlens_ori_kv"), layout_kv
    )
    mi["max_seqlen_cmp_kv"] = _max_kv_length(
        new_cmp_kv, ti.get("seqused_cmp_kv"), ti.get("cu_seqlens_cmp_kv"), layout_kv
    )

    # cpu_output and softmax_lse are stale after transform; clear them
    new_data["cpu_output"] = None
    new_data["softmax_lse"] = None

    # Validate the result
    schema.check_invariants(new_data)

    return new_data


def _build_origins_from_batch_ids(batch_ids: List[int], q_lengths: List[int]) -> List[TokenOrigin]:
    """Build output_origins for a sub-case given the batch_ids mapping.

    New batch *i* corresponds to old batch *batch_ids[i]*.
    """
    origins = []
    for new_b, old_b in enumerate(batch_ids):
        for s in range(q_lengths[new_b]):
            origins.append(TokenOrigin(source_batch=old_b, source_token=s))
    return origins


def _get_q_lengths(input_data: Dict[str, Any]) -> List[int]:
    """Extract per-batch Q lengths from input_data."""
    return schema.get_q_lengths(input_data)


# ------------------------------------------------------------------ #
#  Mode 1: batch reorder
# ------------------------------------------------------------------ #

def generate_non_identity_permutation(B: int, seed: Optional[int] = None) -> List[int]:
    """Generate a non-identity permutation of ``[0 .. B-1]``."""
    if B < 2:
        raise ValueError(f"Mode 1 requires B >= 2, got B={B}")
    rng = random.Random(seed) if seed is not None else random.Random()
    while True:
        order = list(range(B))
        rng.shuffle(order)
        if order != list(range(B)):
            return order


def transform_mode1_reorder(
    input_data: Dict[str, Any],
    order: Optional[List[int]] = None,
    seed: Optional[int] = None,
) -> ConsistencyCase:
    """Mode 1: reorder batches.

    *order* is a permutation of ``[0 .. B-1]``. If ``None``, a random
    non-identity permutation is generated using *seed*.
    """
    B = int(input_data["B"])
    if B < 2:
        raise ValueError(f"Mode 1 requires B >= 2, got B={B}")

    if order is None:
        order = generate_non_identity_permutation(B, seed)
    else:
        if sorted(order) != list(range(B)):
            raise ValueError(f"order must be a permutation of [0..{B-1}], got {order}")

    new_data = _select_batches(input_data, order)
    q_lengths = _get_q_lengths(new_data)
    origins = _build_origins_from_batch_ids(order, q_lengths)

    meta = {"mode": 1, "order": order, "seed": seed, "layout_q": _get_layout_q(input_data),
            "layout_kv": input_data["input"]["layout_kv"]}

    return ConsistencyCase(
        name=f"mode1_reorder_B{B}",
        input_data=new_data,
        output_origins=origins,
        transform_meta=meta,
    )


# ------------------------------------------------------------------ #
#  Mode 2: batch split
# ------------------------------------------------------------------ #

def generate_split_groups(B: int, num_groups: Optional[int] = None,
                          seed: Optional[int] = None) -> List[List[int]]:
    """Partition ``[0 .. B-1]`` into 2..N non-empty groups.

    ``num_groups`` must satisfy ``2 <= num_groups <= B``.
    """
    if B < 2:
        raise ValueError(f"Mode 2 requires B >= 2, got B={B}")
    rng = random.Random(seed) if seed is not None else random.Random()

    if num_groups is None:
        num_groups = rng.randint(2, B)

    if num_groups < 2 or num_groups > B:
        raise ValueError(
            f"num_groups must satisfy 2 <= num_groups <= B ({B}), got {num_groups}"
        )

    indices = list(range(B))
    rng.shuffle(indices)

    # Distribute indices into num_groups non-empty groups
    groups = [[] for _ in range(num_groups)]
    for i, idx in enumerate(indices):
        groups[i % num_groups].append(idx)
    # Sort within each group for determinism
    for g in groups:
        g.sort()
    # Sort groups by their first element for determinism
    groups.sort(key=lambda g: g[0])
    return groups


def transform_mode2_split(
    input_data: Dict[str, Any],
    groups: Optional[List[List[int]]] = None,
    seed: Optional[int] = None,
) -> List[ConsistencyCase]:
    """Mode 2: split batches into multiple sub-cases.

    *groups* is a partition of ``[0 .. B-1]``. If ``None``, a random
    partition is generated using *seed*.

    Returns one :class:`ConsistencyCase` per group.
    """
    B = int(input_data["B"])
    if B < 2:
        raise ValueError(f"Mode 2 requires B >= 2, got B={B}")

    if groups is None:
        groups = generate_split_groups(B, seed=seed)
    else:
        # Validate: no empty groups
        for i, g in enumerate(groups):
            if len(g) == 0:
                raise ValueError(f"groups[{i}] is empty; empty groups are not allowed")
        all_ids = sorted(idx for g in groups for idx in g)
        if all_ids != list(range(B)):
            raise ValueError(f"groups must partition [0..{B-1}], got {groups}")

    cases = []
    for gi, group in enumerate(groups):
        new_data = _select_batches(input_data, group)
        q_lengths = _get_q_lengths(new_data)
        origins = _build_origins_from_batch_ids(group, q_lengths)

        meta = {"mode": 2, "group_index": gi, "batch_ids": group, "seed": seed,
                "layout_q": _get_layout_q(input_data),
                "layout_kv": input_data["input"]["layout_kv"]}

        cases.append(ConsistencyCase(
            name=f"mode2_split_g{gi}_B{len(group)}",
            input_data=new_data,
            output_origins=origins,
            transform_meta=meta,
        ))
    return cases


# ------------------------------------------------------------------ #
#  Mode 3 & 4: explicit unsupported
# ------------------------------------------------------------------ #

class UnsupportedTransformError(Exception):
    """Raised when a transform mode is not yet implemented."""


def transform_mode3_token_split(*args, **kwargs):
    """Mode 3: batch-internal token split.

    NOT implemented in Phase 1.

    Rationale: Mode 3 splits a single batch's query tokens into multiple
    logical batches.  This requires guaranteeing that each derived token
    sees the same attention context as in the baseline, which depends on
    query position, causal mask, and sliding-window semantics.  A CPU
    golden equivalence pre-check must be built first.  See development
    plan §10 for the full prerequisite list.
    """
    raise UnsupportedTransformError(
        "Mode 3 (batch-internal token split) is not implemented in Phase 1. "
        "Reason: requires CPU golden equivalence pre-check and query-position / "
        "mask semantic validation. See plan §10."
    )


def transform_mode4_prepend_delete(*args, **kwargs):
    """Mode 4: prepend / delete tokens.

    NOT implemented in Phase 1.

    Rationale: Mode 4 changes query positions, causal/window masks, TND
    prefix sums, and query-token sparse indices.  It cannot be unconditionally
    enabled for all SMLA cases.  See development plan §11 for the full
    prerequisite list.
    """
    raise UnsupportedTransformError(
        "Mode 4 (prepend/delete tokens) is not implemented in Phase 1. "
        "Reason: changes query position, causal/window mask, TND prefix sums, "
        "and sparse indices. Requires mask-semantic pre-check and stateful "
        "prefill/decode test design. See plan §11."
    )
