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

"""CPU golden for GenericLightningIndexer: matmul → act → head reduce → mask → TopK → post."""

import numpy as np
import torch


def _to_np(x):
    if x is None:
        return None
    if torch.is_tensor(x):
        return x.detach().float().cpu().numpy()
    return np.asarray(x)


def _visible_len(q_idx, s1, l_orig, cmp_ratio, sparse_mode):
    """Compressed-axis visible count for query index q_idx (0-based)."""
    r = int(cmp_ratio)
    if sparse_mode == 0:
        return l_orig // r
    if sparse_mode == 4:
        return (q_idx + 1) // r
    # sparse_mode 3 rightDownCausal
    return max((l_orig - s1 + q_idx + 1) // r, 0)


def generic_lightning_indexer_golden(
    query,
    key,
    weights=None,
    actual_seq_q=None,
    actual_seq_k=None,
    cmp_residual_k=None,
    *,
    topk=2048,
    cmp_ratio=1,
    layout_q="BSND",
    sparse_mode=0,
    act_type=0,
    has_weighted_sum=True,
    expand_indices=False,
    append_tail=False,
    return_value=False,
    score_scale=1.0,
    key_reduce_mode=0,
    select_mode=0,
    start_loc=None,
    init_blocks=0,
    local_blocks=0,
    **kwargs,
):
    """NumPy golden"""
    del kwargs
    q = _to_np(query)
    k = _to_np(key)
    w = _to_np(weights) if has_weighted_sum else None
    r = int(cmp_ratio)
    k_sel = int(topk)

    if layout_q == "TND":
        t1, n1, d = q.shape
        t2 = k.shape[0]
        q4 = q.reshape(1, t1, n1, d)
        k4 = k.reshape(1, t2, 1, d)
        if w is not None:
            w = w.reshape(1, t1, n1)
        b, s1 = 1, t1
        s2 = t2
    else:
        b, s1, n1, d = q.shape
        s2 = k.shape[1]
        q4, k4 = q, k

    residual = np.zeros((b,), dtype=np.int32)
    if cmp_residual_k is not None:
        residual = np.asarray(_to_np(cmp_residual_k), dtype=np.int32).reshape(-1)
    sloc = np.zeros((b,), dtype=np.int32)
    if start_loc is not None:
        sloc = np.asarray(_to_np(start_loc), dtype=np.int32).reshape(-1)

    if int(key_reduce_mode) == 1 or int(select_mode) == 1:
        pooled_b = []
        for bi in range(b):
            act_s2 = s2
            if actual_seq_k is not None:
                act_s2 = int(np.asarray(_to_np(actual_seq_k)).reshape(-1)[bi])
            l_orig = act_s2
            scores = np.einsum("snd,tnd->snt", q4[bi], k4[bi, :act_s2]) * float(
                score_scale
            )
            n_blocks = (act_s2 + r - 1) // r
            n_out = ((max(n_blocks, 1) + 15) // 16) * 16
            pooled = np.full((s1, n1, n_out), -np.inf, dtype=np.float32)
            for qi in range(s1):
                if sparse_mode == 0:
                    vis_tokens = act_s2
                elif sparse_mode == 4:
                    vis_tokens = qi + 1
                else:
                    vis_tokens = max(l_orig - s1 + qi + 1, 0)
                vis_tokens = min(vis_tokens, act_s2)
                for h in range(n1):
                    for blk in range(n_blocks):
                        b0 = blk * r
                        b1 = min((blk + 1) * r, act_s2)
                        vis_end = min(vis_tokens, b1)
                        if vis_end > b0:
                            pooled[qi, h, blk] = scores[qi, h, b0:vis_end].max()
                sl = int(sloc[bi]) if start_loc is not None else (s1 - 1) // max(r, 1)
                if init_blocks > 0:
                    pooled[qi, :, : min(int(init_blocks), n_out)] += 1e30
                if local_blocks > 0:
                    lo = max(0, sl + 1 - int(local_blocks))
                    hi = min(sl + 1, n_out)
                    pooled[qi, :, lo:hi] += 1e29
            pooled_b.append(pooled)
        values = np.stack(pooled_b, axis=0)
        if layout_q == "TND":
            values = values[0]
        indices = np.zeros((0,), dtype=np.int32)
        return torch.from_numpy(indices), torch.from_numpy(values.astype(np.float32))

    indices_rows = []
    value_rows = []
    for bi in range(b):
        act_s2 = s2
        if actual_seq_k is not None:
            act_s2 = int(np.asarray(_to_np(actual_seq_k)).reshape(-1)[bi])
        l_orig = act_s2 * r + int(residual[bi])
        scores = np.einsum("snd,tnd->snt", q4[bi], k4[bi, :act_s2]) * float(score_scale)
        if act_type == 0:
            scores = np.maximum(scores, 0.0)
        if has_weighted_sum:
            head = np.einsum("snt,sn->st", scores, w[bi])
        else:
            head = scores.sum(axis=1)
        out_idx = []
        out_val = []
        for qi in range(s1):
            vis = min(_visible_len(qi, s1, l_orig, r, sparse_mode), act_s2)
            row = head[qi].copy()
            if vis < row.shape[0]:
                row[vis:] = -np.inf
            order = np.argsort(-row, kind="stable")
            picked = order[:k_sel]
            vals = row[picked]
            if vis < k_sel:
                picked[vis:] = -1
                vals[vis:] = -np.inf
            if expand_indices:
                expanded = []
                for p in picked:
                    if p < 0:
                        expanded.extend([-1] * r)
                    else:
                        expanded.extend([int(p) * r + o for o in range(r)])
                if append_tail:
                    if sparse_mode == 4:
                        gq = l_orig - s1 + qi
                        mod = (gq + 1) % r
                        start = (gq + 1) - mod
                        extra = list(range(start, start + mod))
                    else:
                        tail = int(residual[bi])
                        start = l_orig - tail
                        extra = list(range(start, l_orig)) if tail > 0 else []
                    extra += [-1] * (r - 1 - len(extra))
                    expanded.extend(extra[: r - 1])
                out_idx.append(expanded)
            else:
                out_idx.append(picked.tolist())
            out_val.append(vals.tolist())
        indices_rows.append(out_idx)
        value_rows.append(out_val)

    indices = np.asarray(indices_rows, dtype=np.int32)
    values = np.asarray(value_rows, dtype=np.float32)
    if layout_q == "TND":
        indices = indices[0]
        values = values[0]
    if not return_value:
        values = np.zeros((0,), dtype=np.float32)
    return torch.from_numpy(indices), torch.from_numpy(values)


__golden__ = {
    "e2e": {
        "torch_npu.npu_generic_lightning_indexer": "generic_lightning_indexer_golden"
    },
    "aclnn": {"aclnnGenericLightningIndexer": "generic_lightning_indexer_golden"},
}
