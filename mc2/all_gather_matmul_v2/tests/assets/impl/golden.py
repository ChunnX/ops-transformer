#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Golden for aclnnAllGatherMatmulV2.

AllGatherMatmulV2 extends V1 with quant (fp8/mxfp/per_block) support.

Two golden channels share the same per-rank input/output contract (2.1 §2.1):

* TTK/ST channel (carded env): delegates to the shared all_gather golden logic
  which handles V1+V2 uniformly. The shared layer has a known defect (G-T1) on
  the non-quant transB path: the (n, k) non-contiguous transpose view passed
  to the aclnn API is fed to ``torch.matmul`` as-is, i.e. the logical weight
  ``W = x2_view.T`` is never restored. This adapter fixes it by restoring each
  rank's x2 to the logical (k, n) layout BEFORE calling the shared layer
  (fix lives here on purpose: the shared layer mc2/common is common code
  shared with v1 & friends and must not be modified from the operator side).

* Simulation channel (channel A, no NPU): a self-contained CPU golden
  (``agm_v2_sim_golden``) with no TTK dependency. It reads the raw dump bytes
  of the API actual args (x2 dumped as passed, together with shape/strides)
  and restores transB itself, so it is not affected by G-T1.

Per-rank input/output contract (frozen, 2.1 §2.1):
  rank r holds x1_r (m, k); x2_r is the local weight with logical shape (k, n)
  (transB: the aclnn caller passes a (k, n) view with swapped strides (1, k)
  over an (n, k) row-major storage — W.t() contiguous — so that op_api
  CheckShape (k-axis = view dim0) passes and IsTransposeLastTwoDims detects
  the transpose; the inner/tiling-level desc and the TTK flatten_tensors
  context carry the (n, k) form, where W = x2.T — see
  ``logical_weight_from_view``); bias_r (n,) is optional and may differ per
  rank.
  gathered = cat([x1_0, ..., x1_{R-1}], dim=0)  # M-axis concat, rank ascending
  output_r = gathered @ W_r + bias_r            # per-rank independent
  gatherOut_r = gathered                        # same on all ranks, bitwise
Dtype chain: inputs are read in their original dtype, upcast to fp32 for the
matmul, and the golden output is rounded back to the declared output dtype
(follows x1: FP16/BF16) before comparison.
"""

import importlib.util
from pathlib import Path

# ---------------------------------------------------------------------------
# Simulation-channel golden (channel A) — no TTK dependency below this line
# ---------------------------------------------------------------------------

SIM_DTYPES = ("fp16", "bf16")

_TORCH_DTYPES = {
    "fp16": "float16",
    "bf16": "bfloat16",
}


def _torch():
    import torch

    return torch


def _to_torch_dtype(dtype):
    torch = _torch()
    name = _TORCH_DTYPES[dtype]
    return getattr(torch, name)


def logical_weight_from_view(x2_view, k):
    """Restore the logical (k, n) weight from an x2 API-view dump.

    两种合法实参形态（按视图 shape 与 k 的匹配关系判别）：
    * (k, n) view + strides (1, k)，storage 为 (n, k) 行主序（W.t() 连续）
      —— 950 CCU/ai_cpu 外层 aclnn 实参形态（op_api CheckShape 按 dim0 取
      k 轴，IsTransposeLastTwoDims 按 stride 判定，TransX2Tensor 重建为
      inner (n,k) 连续 desc）：视图本身即逻辑 W，直接使用；
    * (n, k) 形态（inner/tiling desc 或 TTK flatten_tensors 上下文）
      —— W = x2.T（G-T1 修复语义）。
    """
    if int(x2_view.shape[0]) == k:
        return x2_view
    if int(x2_view.shape[1]) == k:
        return x2_view.t()
    raise ValueError(
        f"x2 view shape {tuple(x2_view.shape)} matches neither (k={k}, n) "
        f"nor (n, k={k})")


def agm_v2_sim_golden(case, rank_inputs):
    """CPU golden for the simulation channel (channel A).

    Args:
        case: dict with keys ``dtype`` (fp16|bf16), ``m``, ``k``, ``n``,
            ``rank_size``, ``bias`` (bool), ``trans_b`` (bool),
            ``gather_out`` (bool).
        rank_inputs: list of per-rank dicts with torch tensors already in the
            API-view layout:
              ``x1`` (m, k) in original dtype;
              ``x2`` API view（transB 时为转置 view，形态见
              ``logical_weight_from_view``）;
              ``bias`` (n,) in original dtype or None.

    Returns:
        dict with per-rank goldens:
          ``output``: list of (m*R, n) tensors in original dtype (rounded);
          ``gather_out``: list of (m*R, k) tensors in original dtype
            (bit-preserving concat of the raw x1 inputs) or None when
            ``gather_out`` is False.
    """
    torch = _torch()
    dtype = _to_torch_dtype(case["dtype"])
    rank_size = case["rank_size"]
    if len(rank_inputs) != rank_size:
        raise ValueError(
            f"rank_inputs length {len(rank_inputs)} != rank_size {rank_size}"
        )

    # Bit-preserving gatherOut golden: concat raw x1 (original dtype, no
    # float conversion) so NaN payloads and signed zeros survive verbatim.
    gather_golden = torch.cat(
        [inp["x1"].reshape(-1, inp["x1"].shape[-1]) for inp in rank_inputs],
        dim=0,
    )

    # FP32 compute chain: read in original dtype -> float() -> matmul.
    gathered_f32 = torch.cat(
        [inp["x1"].to(torch.float32) for inp in rank_inputs], dim=0
    )

    outputs = []
    for inp in rank_inputs:
        x2 = inp["x2"]
        if case.get("trans_b", False):
            x2 = logical_weight_from_view(x2, case["k"])
        w_f32 = x2.to(torch.float32)
        golden = torch.matmul(gathered_f32, w_f32)
        bias = inp.get("bias")
        if bias is not None and getattr(bias, "numel", lambda: 0)() > 0:
            golden = golden + bias.to(torch.float32)
        # Round back to the declared output dtype (follows x1).
        outputs.append(golden.to(dtype))

    gather_out = None
    if case.get("gather_out", True):
        gather_out = [gather_golden.to(dtype).contiguous()] * rank_size

    return {"output": outputs, "gather_out": gather_out}


# ---------------------------------------------------------------------------
# TTK/ST channel — shared golden with the operator-side transB adaptation
# ---------------------------------------------------------------------------


def _load_shared_utils():
    """Load golden_utils from mc2/common/tests/assets."""
    utils_path = (
        Path(__file__).resolve().parents[4]
        / "common"
        / "tests"
        / "assets"
        / "golden_utils.py"
    )
    spec = importlib.util.spec_from_file_location(
        f"mc2_common_golden_utils_{abs(hash(utils_path))}", utils_path
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _extract_is_trans_b(first_ctx):
    """Extract is_trans_b from the first context (attrs first, remark second).

    Mirrors the remark grammar of the shared golden (``is_trans_b=1``).
    """
    attrs = getattr(first_ctx, "attributes", None) or {}
    tx2 = attrs.get(
        "transposeX2", attrs.get("isTransB", attrs.get("is_trans_b", None))
    )
    if tx2 is not None:
        return bool(tx2)
    remark = getattr(first_ctx, "remark", None) or ""
    for part in remark.split(","):
        kv = part.split("=", 1)
        if len(kv) == 2 and kv[0].strip() == "is_trans_b":
            try:
                return kv[1].strip() == "1"
            except ValueError:
                pass
    return False


def _is_non_quant_ctx(first_ctx):
    """True when x1 dtype is FP16/BF16 (non-quant scope of this golden fix)."""
    dtypes = getattr(first_ctx, "flat_tensor_dtypes", None) or []
    if not dtypes:
        return True  # cannot judge; behave like the shared layer
    return str(dtypes[0]).lower() in ("float16", "fp16", "bfloat16", "bf16")


def all_gather_matmul_v2_multi_device_golden(
    thread_contexts, device_ids, all_precision
):
    """Use shared golden_all_gather_compare with the operator-side transB fix.

    G-T1 (2.1 §2.1): the shared ``golden_all_gather_compare`` non-quant path
    feeds the x2 API actual argument straight into ``torch.matmul`` without
    restoring the transB view, so transB cases produce a wrong main output.
    Fix (2.1 §7 T1): restore each rank's x2 to the logical (k, n) layout here,
    in the adapter, before delegating; the shared layer is NOT modified.
    """
    utils = _load_shared_utils()
    world_size = len(device_ids)
    first_ctx = next(iter(thread_contexts.values()))
    is_trans_b = _extract_is_trans_b(first_ctx)
    restored = False
    if is_trans_b and _is_non_quant_ctx(first_ctx):
        for did in device_ids:
            tc = thread_contexts[did]
            flatten = getattr(tc, "flatten_tensors", None)
            if flatten is None or len(flatten) < 2 or flatten[1] is None:
                continue
            x2 = flatten[1]
            if getattr(x2, "dim", lambda: 0)() != 2:
                continue
            # x2 arrives as the API actual arg in TTK context: an (n, k)
            # transpose-form tensor（inner/TTK 形态）. W = x2.T restores the
            # logical (k, n) weight (G-T1 fix semantics, 2.1 §7 T1).
            flatten[1] = x2.t().contiguous()
            restored = True
    try:
        utils.golden_all_gather_compare(
            thread_contexts, device_ids, all_precision, world_size
        )
    finally:
        # Keep contexts pristine for downstream consumers (e.g. cascade):
        # undo the restoration by re-creating the API-view layout.
        if restored:
            for did in device_ids:
                tc = thread_contexts[did]
                flatten = getattr(tc, "flatten_tensors", None)
                if flatten is None or len(flatten) < 2:
                    continue
                x2 = flatten[1]
                if x2 is not None and getattr(x2, "dim", lambda: 0)() == 2:
                    flatten[1] = x2.t().contiguous()


def all_gather_matmul_v2_e2e_golden(
    cpu_inputs_per_rank, attrs, rank=None, ws=None, dist_avail=True
):
    import torch

    world_size = len(cpu_inputs_per_rank)
    all_x1 = [
        inp[0].float() if len(inp) > 0 and inp[0] is not None else None
        for inp in cpu_inputs_per_rank
    ]
    gathered = torch.cat([x for x in all_x1 if x is not None], dim=0)
    del all_x1
    goldens = []
    for r in range(world_size):
        inp = cpu_inputs_per_rank[r]
        x2 = inp[1].float() if len(inp) > 1 and inp[1] is not None else None
        if x2 is None:
            goldens.append(None)
            continue
        g = torch.matmul(gathered, x2)
        bias = inp[2] if len(inp) > 2 and inp[2] is not None else None
        if bias is not None and hasattr(bias, "numel") and bias.numel() > 0:
            g = g + bias.float()
        goldens.append(g.contiguous())
    del gathered
    return goldens
