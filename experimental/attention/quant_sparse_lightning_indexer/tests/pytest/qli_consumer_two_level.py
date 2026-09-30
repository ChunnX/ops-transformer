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
# QuantSparseLightningIndexer（两级TopK 第二级 consumer）精度测试
#
# 用途: 验证第二级算子 quant_sparse_lightning_indexer —— 吃第一级输出的块级候选索引，
#       在候选块集合内做 token 级 topk。两条路径分别与 torch 参考实现比对:
#   ① 吃 golden 候选：隔离 consumer 自身精度（候选来自参考实现，不经第一级 NPU 输出）
#   ② 吃第一级 NPU 候选：生产链路端到端（第一级 candidate 入口 → consumer）
#
# 来源与裁剪（2026-09-30）:
#   本文件由 /home/tjj/scripts/qliv2_debug/test_qli_v2_two_level_precision.py 裁剪而来：
#   - 去掉「现网基础入口」与「第一级自身精度」两组断言，只保留上述 consumer 两条路径
#     （第一级仍会在路径②中调用，用于生成端到端候选输入，不对其结果做精度断言）；
#     第一级与现网路径的精度用例见 ../quant_lightning_indexer_v2/tests/pytest/qli_cand_orig.py
#   - 用例表沿用原精度矩阵（ratio1/2、mask0/3、g32/64、TND、B=4 变长、pin 边界、
#     单行长上下文 decode_long_m1 等）
#   参考实现（score 公式、块内 amax+pin、候选掩蔽 topk、官方边界容忍比对）拷贝自
#   /home/tjj/ds41experimental/attention/quant_lightning_indexer_v2/tests/pytest/test_qli_cand_device.py
#   （与仓库 tests/pytest/quant_lightning_indexer_v2_golden.py::cal_atten_per_batch_int8 对齐）。
#
# 运行:
#   source <CANN>/set_env.sh        # 需能 import 到含本算子的 cann_ops_transformer 包
#   ASCEND_RT_VISIBLE_DEVICES=<空闲卡> pytest qli_consumer_two_level.py [-k <用例 id>]
import math
import os
import sys

import numpy as np
import pytest
import torch
import torch_npu  # noqa: F401

# 本分支三个入口（consumer 为被测；candidate 仅用于路径②生成候选输入）
from cann_ops_transformer.ops.attention.quant_lightning_indexer_v2 import (  # noqa: E402
    quant_lightning_indexer_candidate,
    quant_lightning_indexer_metadata,
)
from cann_ops_transformer.ops.attention.quant_sparse_lightning_indexer import (  # noqa: E402
    quant_sparse_lightning_indexer,
)

# 官方比对规则（对齐仓库 tests/pytest/result_compare_method.py，取自主算子测试目录）
REPO_PYTEST_DIR = os.path.normpath(
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..",
        "..",
        "..",
        "quant_lightning_indexer_v2",
        "tests",
        "pytest",
    )
)
if REPO_PYTEST_DIR not in sys.path:
    sys.path.insert(0, REPO_PYTEST_DIR)
from result_compare_method import compare_topk_valid  # noqa: E402

NEG_INF = float("-inf")
CAND_BLOCKS = 2048
CAND_BS = 8
HEAD_DIM = 128
NEG_HUGE = -1.0e30


def select_candidate_blocks_ref(score_row, compress_len, topk_blocks, bs):
    """参考实现: pad -> 块内 amax -> pin 尾块 -> topk -> -1 槽"""
    width = score_row.shape[-1]
    pad = (-width) % bs
    padded = torch.nn.functional.pad(score_row, (0, pad), value=NEG_INF)
    nb = padded.shape[-1] // bs
    blocks = padded.unflatten(-1, (-1, bs)).amax(dim=-1)
    last = (compress_len - 1) // bs
    blocks[last] = float("inf")
    k = min(topk_blocks, nb)
    topv, topi = blocks.topk(k)
    out = torch.full((topk_blocks,), -1, dtype=torch.int32)
    keep = topv > NEG_INF
    out[:k][keep] = topi[keep].to(torch.int32)
    return out, blocks


def topk_golden(score_row, topk):
    """有效位置降序取前 topk，尾部补 -1；排序键 (-score, index) 稳定排序（tie 取小索引）"""
    n = score_row.shape[-1]
    indices = torch.arange(n)
    valid = score_row != NEG_INF
    v_idx = indices[valid]
    if v_idx.numel() == 0:
        return torch.full((topk,), -1, dtype=torch.int32)
    stacked = torch.stack([-score_row[v_idx], v_idx.to(torch.float32)], dim=1)
    _, sorted_idx = torch.sort(stacked, dim=0, stable=True)
    order = sorted_idx[:, 0]
    out = torch.full((topk,), -1, dtype=torch.int32)
    actual = min(order.numel(), topk)
    out[:actual] = v_idx[order[:actual]].to(torch.int32)
    return out


def gen_case_inputs(case):
    """生成 NPU 输入 (int8/PA) 与 golden 所需稠密表示; 支持 layout=TND 与 pa_gap (key 0 轴非连续)"""
    b = case["batch"]
    s1 = case["q_seq"]
    g = case["g"]
    seqused_k = case["seqused_k"]
    seqs_q = case.get("seqs_q") or [s1] * b
    ratio = case["cmp_ratio"]
    res = case.get("cmp_res")
    topk = case["topk"]
    block_size = case.get("pa_block_size", 128)
    layout = case.get("layout", "BSND")
    pa_gap = case.get("pa_gap", False)
    gen = torch.Generator().manual_seed(case.get("seed", 7))
    np.random.seed(case.get("seed", 7) + 500)

    if layout == "TND":
        s2max = max(seqused_k)
        q_flat = torch.cat(
            [
                torch.randint(
                    -20, 21, (sq, g, HEAD_DIM), dtype=torch.int32, generator=gen
                )
                for sq in seqs_q
            ]
        ).to(torch.int8)
        q = q_flat.npu()
        cu_q = [0]
        for sq in seqs_q:
            cu_q.append(cu_q[-1] + sq)
        w = (
            torch.cat(
                [torch.empty(sq, g).uniform_(-1, 1, generator=gen) for sq in seqs_q]
            )
            .to(torch.float16)
            .npu()
        )
        q_scale = (
            torch.cat(
                [torch.empty(sq, g).uniform_(0.5, 1.0, generator=gen) for sq in seqs_q]
            )
            .to(torch.float16)
            .npu()
        )
        k_dense = torch.randint(
            -20, 21, (b, 1, s2max, HEAD_DIM), dtype=torch.int32, generator=gen
        ).to(torch.int8)
        k_scale_dense = (
            torch.empty(b, 1, s2max).uniform_(0.5, 1.0, generator=gen).to(torch.float16)
        )
        max_blk = math.ceil(s2max / block_size)
        blk_ids = np.random.permutation(np.arange(max_blk * b + 4)).astype(np.int32)
        block_table = np.full((b, max_blk), -1, dtype=np.int32)
        cur = 0
        for bi in range(b):
            for ib in range(math.ceil(seqused_k[bi] / block_size)):
                block_table[bi][ib] = blk_ids[cur]
                cur += 1
        block_table_t = torch.from_numpy(block_table).npu()
        k_phys = torch.zeros(max_blk * b + 4, block_size, 1, HEAD_DIM, dtype=torch.int8)
        k_scale_phys = torch.zeros(max_blk * b + 4, block_size, 1, dtype=torch.float16)
        for bi in range(b):
            for ib in range(max_blk):
                bid = block_table[bi][ib]
                if bid == -1:
                    continue
                start = ib * block_size
                end = min(start + block_size, s2max)
                k_phys[bid, : end - start, 0, :] = k_dense[bi, 0, start:end, :]
                k_scale_phys[bid, : end - start, 0] = k_scale_dense[bi, 0, start:end]
        cu_q_t = torch.tensor(cu_q, dtype=torch.int32).npu()
        seqused_k_t = torch.tensor(seqused_k, dtype=torch.int32).npu()
        cmp_res_t = (
            torch.tensor(res, dtype=torch.int32).npu() if res is not None else None
        )
        metadata = quant_lightning_indexer_metadata(
            num_heads_q=g,
            num_heads_k=1,
            head_dim=HEAD_DIM,
            topk=topk,
            quant_mode=2,
            cu_seqlens_q=cu_q_t,
            seqused_k=seqused_k_t,
            cmp_residual_k=cmp_res_t,
            batch_size=b,
            max_seqlen_q=max(seqs_q),
            max_seqlen_k=s2max,
            layout_q="TND",
            layout_k="PA_BBND",
            mask_mode=case["mask_mode"],
            cmp_ratio=ratio,
        )
        return {
            "q": q,
            "w": w,
            "q_scale": q_scale,
            "k_dense": k_dense,
            "k_scale_dense": k_scale_dense,
            "k_phys": k_phys.npu(),
            "k_scale_phys": k_scale_phys.npu(),
            "block_table": block_table_t,
            "seqused_k_t": seqused_k_t,
            "cu_q_t": cu_q_t,
            "cmp_res_t": cmp_res_t,
            "metadata": metadata,
            "output_idx_offset_t": None,
            "output_idx_offsets": None,
            "s2max": s2max,
            "seqs_q": seqs_q,
            "cu_q": cu_q,
        }

    q = (
        torch.randint(-20, 21, (b, s1, g, HEAD_DIM), dtype=torch.int32, generator=gen)
        .to(torch.int8)
        .npu()
    )
    w = torch.empty(b, s1, g).uniform_(-1, 1, generator=gen).to(torch.float16).npu()
    q_scale = (
        torch.empty(b, s1, g).uniform_(0.5, 1.0, generator=gen).to(torch.float16).npu()
    )

    s2max = max(seqused_k)
    k_dense = torch.randint(
        -20, 21, (b, 1, s2max, HEAD_DIM), dtype=torch.int32, generator=gen
    ).to(torch.int8)
    k_scale_dense = (
        torch.empty(b, 1, s2max).uniform_(0.5, 1.0, generator=gen).to(torch.float16)
    )

    max_blk_per_batch = math.ceil(s2max / block_size)
    gap = 2 if pa_gap else 1
    pool = case.get("pool") or ((max_blk_per_batch * b + 4) * gap)
    table_w = case.get("table_w") or max_blk_per_batch
    total_blocks = pool
    inputs_cu_k = [0]
    for s2 in seqused_k:
        inputs_cu_k.append(inputs_cu_k[-1] + s2)

    if pa_gap:
        n_blk = total_blocks // gap
        blk_ids = np.random.permutation(np.arange(n_blk)).astype(np.int32)
        block_table = np.full((b, table_w), -1, dtype=np.int32)
        cur = 0
        for bi in range(b):
            for ib in range(math.ceil(seqused_k[bi] / block_size)):
                block_table[bi][ib] = blk_ids[cur]
                cur += 1
        block_table_t = torch.from_numpy(block_table).npu()
        blk_elems = block_size * HEAD_DIM
        buf = torch.zeros(n_blk * gap * blk_elems, dtype=torch.int8, device="npu")
        k_phys = torch.as_strided(
            buf,
            (n_blk, block_size, 1, HEAD_DIM),
            (gap * blk_elems, HEAD_DIM, HEAD_DIM, 1),
        )
        sbuf = torch.zeros(n_blk * gap * block_size, dtype=torch.float16, device="npu")
        k_scale_phys = torch.as_strided(
            sbuf, (n_blk, block_size, 1), (gap * block_size, 1, 1)
        )
        kp = torch.zeros(n_blk, block_size, 1, HEAD_DIM, dtype=torch.int8)
        ksp = torch.zeros(n_blk, block_size, 1, dtype=torch.float16)
        for bi in range(b):
            for ib in range(math.ceil(seqused_k[bi] / block_size)):
                bid = block_table[bi][ib]
                if bid == -1:
                    continue
                start = ib * block_size
                end = min(start + block_size, s2max)
                kp[bid, : end - start, 0, :] = k_dense[bi, 0, start:end, :]
                ksp[bid, : end - start, 0] = k_scale_dense[bi, 0, start:end]
        k_phys.copy_(kp.npu())
        k_scale_phys.copy_(ksp.npu())
    else:
        blk_ids = np.random.permutation(np.arange(total_blocks)).astype(np.int32)
        block_table = np.full((b, table_w), -1, dtype=np.int32)
        cur = 0
        for bi in range(b):
            for ib in range(math.ceil(seqused_k[bi] / block_size)):
                block_table[bi][ib] = blk_ids[cur]
                cur += 1
        block_table_t = torch.from_numpy(block_table).npu()
        k_phys = torch.zeros(total_blocks, block_size, 1, HEAD_DIM, dtype=torch.int8)
        k_scale_phys = torch.zeros(total_blocks, block_size, 1, dtype=torch.float16)
        for bi in range(b):
            for ib in range(math.ceil(seqused_k[bi] / block_size)):
                bid = block_table[bi][ib]
                if bid == -1:
                    continue
                start = ib * block_size
                end = min(start + block_size, s2max)
                k_phys[bid, : end - start, 0, :] = k_dense[bi, 0, start:end, :]
                k_scale_phys[bid, : end - start, 0] = k_scale_dense[bi, 0, start:end]
        k_phys = k_phys.npu()
        k_scale_phys = k_scale_phys.npu()

    seqused_k_t = torch.tensor(seqused_k, dtype=torch.int32).npu()
    cmp_res_t = torch.tensor(res, dtype=torch.int32).npu() if res is not None else None

    metadata = quant_lightning_indexer_metadata(
        num_heads_q=g,
        num_heads_k=1,
        head_dim=HEAD_DIM,
        topk=topk,
        quant_mode=2,
        seqused_k=seqused_k_t,
        cmp_residual_k=cmp_res_t,
        batch_size=b,
        max_seqlen_q=max(seqs_q),
        max_seqlen_k=s2max,
        layout_q="BSND",
        layout_k="PA_BBND",
        mask_mode=case["mask_mode"],
        cmp_ratio=ratio,
    )
    return {
        "q": q,
        "w": w,
        "q_scale": q_scale,
        "k_dense": k_dense,
        "k_scale_dense": k_scale_dense,
        "k_phys": k_phys,
        "k_scale_phys": k_scale_phys,
        "block_table": block_table_t,
        "seqused_k_t": seqused_k_t,
        "cu_q_t": None,
        "cmp_res_t": cmp_res_t,
        "metadata": metadata,
        "output_idx_offset_t": None,
        "output_idx_offsets": None,
        "s2max": s2max,
        "max_blk_per_batch": max_blk_per_batch,
        "seqs_q": seqs_q,
        "k_pool_blocks": total_blocks,
    }


def _common_kwargs(inputs, case):
    return dict(
        cu_seqlens_q=inputs.get("cu_q_t"),
        cu_seqlens_k=None,
        seqused_k=inputs["seqused_k_t"],
        cmp_residual_k=inputs.get("cmp_res_t"),
        block_table=inputs["block_table"],
        output_idx_offset=None,
        metadata=inputs["metadata"],
        max_seqlen_q=max(case.get("seqs_q", [case["q_seq"]])),
        layout_q=case.get("layout", "BSND"),
        layout_k="PA_BBND",
        mask_mode=case["mask_mode"],
        cmp_ratio=case["cmp_ratio"],
    )


def npu_run_source(inputs, case):
    """② 主算子第一级: 开启候选（candidate_topk_blocks=2048 / candidate_block_size=8）"""
    cb = case.get("cand_blocks", CAND_BLOCKS)
    assert cb == CAND_BLOCKS, (
        f"本分支仅支持 candidate_topk_blocks=2048，用例给的 {cb} 会被 tiling 拒绝"
    )
    ret = quant_lightning_indexer_candidate(
        inputs["q"],
        inputs["k_phys"],
        inputs["w"],
        inputs["q_scale"],
        inputs["k_scale_phys"],
        topk=case["topk"],
        quant_mode=2,
        candidate_topk_blocks=cb,
        candidate_block_size=CAND_BS,
        **_common_kwargs(inputs, case),
    )
    idx, _vals, cand = ret[0], ret[1], ret[2]
    if len(ret) > 3:  # 第 4 个是预留空输出 candidate_block_length（恒空）
        assert ret[3].numel() == 0, "预留输出 candidate_block_length 应为空 tensor"
    return idx, cand


def npu_run_consumer(inputs, case, cand_in):
    """③ 第二级 consumer: 吃候选索引，在候选块内选 topk"""
    assert quant_sparse_lightning_indexer is not None, (
        "当前环境没有拆出来的 consumer 算子"
    )
    idx, _vals = quant_sparse_lightning_indexer(
        inputs["q"],
        inputs["k_phys"],
        inputs["w"],
        inputs["q_scale"],
        inputs["k_scale_phys"],
        topk=case["topk"],
        quant_mode=2,
        candidate_topk_index=cand_in,
        candidate_block_length=None,
        candidate_block_size=CAND_BS,
        **_common_kwargs(inputs, case),
    )
    return idx


def _golden_row_score(q_b, i, w_b, qs_b, k_b, ks_b):
    """单行 golden score: q 行 [G,D] x k [S2,D]"""
    q_row = q_b[i].to(torch.float32)
    qk = torch.matmul(q_row, k_b.to(torch.float32).t())
    qk_relu = (qk / 1024.0).clamp_min(0.0).to(torch.float16)
    cur_w = (w_b[i] * qs_b[i]).to(torch.float16)
    score = torch.matmul(
        cur_w.to(torch.float32).unsqueeze(0), qk_relu.to(torch.float32)
    ).reshape(-1)
    return score * ks_b.to(torch.float32)


def golden_run_ext(inputs, case, kind, cand_blocks_in=None):
    """torch 参考实现。
    kind="base"     : 未掩蔽 topk（对齐现网路径）
    kind="source"   : 未掩蔽 topk + 块级候选（amax/pin/topk），对齐第一级
    kind="consumer" : 候选掩蔽后的 topk（候选外可达位置 NEG_HUGE 降级但保留入选资格），对齐第二级
    cand_blocks_in  : kind="consumer" 时的候选块列表（喂给算子与 golden 的必须是同一份）
    返回 (idx_golden, cand_golden_rows, score_rows, blk_rows)
    """
    b = case["batch"]
    ratio = case["cmp_ratio"]
    res = case.get("cmp_res") or [0] * b
    mask_mode = case["mask_mode"]
    seqs_q = case.get("seqs_q") or [case["q_seq"]] * b
    q_all = inputs["q"].cpu().to(torch.int32)
    w_all = inputs["w"].cpu()
    qs_all = inputs["q_scale"].cpu()
    k_all = inputs["k_dense"].cpu().to(torch.int32)
    ks_all = inputs["k_scale_dense"].cpu()
    seqused_k = case["seqused_k"]
    if case.get("layout") == "TND":
        cu_q = inputs["cu_q"]
        q_l = [q_all[cu_q[bi] : cu_q[bi + 1]] for bi in range(b)]
        w_l = [w_all[cu_q[bi] : cu_q[bi + 1]] for bi in range(b)]
        qs_l = [qs_all[cu_q[bi] : cu_q[bi + 1]] for bi in range(b)]
    else:
        q_l = [q_all[bi] for bi in range(b)]
        w_l = [w_all[bi] for bi in range(b)]
        qs_l = [qs_all[bi] for bi in range(b)]
    k_l = [k_all[bi, 0, : seqused_k[bi], :] for bi in range(b)]
    ks_l = [ks_all[bi, 0, : seqused_k[bi]] for bi in range(b)]
    if cand_blocks_in is not None:
        cand_blocks_in = cand_blocks_in.reshape(-1, cand_blocks_in.shape[-1])

    idx_out_rows, cand_out_rows, score_rows, blk_rows = [], [], [], []
    s2max = max(seqused_k)
    nb_max = (s2max + CAND_BS - 1) // CAND_BS
    row_global = 0
    for bi in range(b):
        s1 = seqs_q[bi]
        s2 = seqused_k[bi]
        act_k = seqused_k[bi] * ratio + res[bi]
        k_b = k_l[bi]
        valid_lens = []
        for i in range(s1):
            if mask_mode == 3:
                base = act_k - s1 + i + 1
                if base < 0:
                    valid_lens.append(-1)
                else:
                    valid_lens.append(int(base // ratio))
            else:
                valid_lens.append(s2)
        q_b = q_l[bi].permute(1, 0, 2)  # [G, S1, D]
        qk = torch.matmul(q_b, k_b.t())  # [G, S1, S2]
        qk_relu = (qk.to(torch.float32) / 1024.0).clamp_min(0.0).to(torch.float16)
        cur_w = (w_l[bi] * qs_l[bi]).to(torch.float16)  # [S1, G]
        score = torch.einsum(
            "in,nij->ij", cur_w.to(torch.float32), qk_relu.to(torch.float32)
        )
        score = score * ks_l[bi].to(torch.float32)  # [S1, S2]
        for i in range(s1):
            if mask_mode == 3:
                if valid_lens[i] < 0:
                    score[i, :] = NEG_INF
                elif valid_lens[i] < s2:
                    score[i, valid_lens[i] :] = NEG_INF
        for i in range(s1):
            row = score[i]
            if kind in ("base", "source"):
                idx_row = topk_golden(row, case["topk"])
            else:
                assert cand_blocks_in is not None
                blk_list = cand_blocks_in[row_global + i]
                nb = math.ceil(s2 / CAND_BS)
                blk_mask = torch.zeros(nb, dtype=torch.bool)
                valid_blk = blk_list[(blk_list >= 0) & (blk_list < nb)]
                blk_mask[valid_blk.to(torch.long)] = True
                pos_mask = blk_mask.repeat_interleave(CAND_BS)[:s2]
                masked = torch.where(
                    pos_mask | (row == NEG_INF), row, torch.full_like(row, NEG_HUGE)
                )
                idx_row = topk_golden(masked, case["topk"])
                row = masked
            idx_out_rows.append(idx_row)
            if kind == "source":
                cand_row, blk_score_row = select_candidate_blocks_ref(
                    score[i],
                    valid_lens[i],
                    case.get("cand_blocks", CAND_BLOCKS),
                    CAND_BS,
                )
                cand_out_rows.append(cand_row)
                blk_pad = torch.full((nb_max,), NEG_INF)
                blk_pad[: blk_score_row.numel()] = blk_score_row
                blk_rows.append(blk_pad)
            row_pad = torch.full((s2max,), NEG_INF)
            if valid_lens[i] >= 0:
                row_pad[:s2] = row
            score_rows.append(row_pad)
        row_global += s1
    idx_golden = torch.stack(idx_out_rows).reshape(-1, 1, case["topk"])
    return idx_golden, cand_out_rows, score_rows, blk_rows


def cmp_indices(npu_rows, gold_rows, name, values4d=None, s1=1):
    """官方比对规则（对齐 tests/pytest/result_compare_method.py::check_result）:
    1) 整行排序后多重集合相等 -> PASS
    2) 回退: compare_topk_valid，差异元素按边界值相对误差 <= 0.001 容忍"""
    npu_m = (
        npu_rows.cpu().numpy() if torch.is_tensor(npu_rows) else np.asarray(npu_rows)
    )
    gold_m = (
        gold_rows.cpu().numpy() if torch.is_tensor(gold_rows) else np.asarray(gold_rows)
    )
    assert npu_m.shape == gold_m.shape, f"{name} shape {npu_m.shape} vs {gold_m.shape}"
    rows = npu_m.shape[0]
    n_multiset = n_boundary = 0
    fails = []
    for r in range(rows):
        g = np.asarray(gold_m[r], dtype=np.int64)
        n = np.asarray(npu_m[r], dtype=np.int64)
        if np.array_equal(np.sort(g), np.sort(n)):
            n_multiset += 1
            continue
        valid_len = int((g != -1).sum())
        npu_valid_len = int((n != -1).sum())
        if valid_len == 0:
            fails.append((r, "gold 全 -1 但 npu 有有效值"))
            continue
        if npu_valid_len != valid_len:
            fails.append(
                (r, f"-1 槽计数不匹配 gold_valid={valid_len} npu_valid={npu_valid_len}")
            )
            continue
        oob = bool((n < -1).any())
        if values4d is not None:
            oob = oob or bool((n >= values4d.shape[-1]).any())
        if oob:
            fails.append((r, "npu 越界索引"))
            continue
        b_idx, i_idx = r // s1, r % s1
        diff_npu, diff_cpu = [], []
        ok_row, max_re = compare_topk_valid(
            g[:valid_len],
            n[:valid_len],
            values4d,
            (b_idx, i_idx, 0),
            diff_npu,
            diff_cpu,
            None,
            None,
            thres=0.001,
            return_value_flag=False,
        )
        if ok_row:
            n_boundary += 1
        else:
            fails.append(
                (
                    r,
                    f"boundary max_re={max_re:.6f} npu_only={diff_npu[:4]} gold_only={diff_cpu[:4]}",
                )
            )
    ok = not fails
    msg = f"{name}: rows={rows} multiset_eq={n_multiset} boundary_ok={n_boundary} fail={len(fails)}"
    if fails:
        msg += " | " + "; ".join(f"r{r}:{info}" for r, info in fails[:3])
    return ok, msg


# name, batch, q_seq, seqused_k, cmp_ratio, cmp_res, mask, g, topk, cand_blocks(可选)
CASES = [
    ("r1_decode", 1, 1, [2048], 1, None, 3, 32, 512),
    ("r1_decode_m0", 1, 1, [2048], 1, None, 0, 32, 512),
    ("r1_prefill_m3", 2, 8, [1024, 2048], 1, None, 3, 32, 512),
    ("r1_tail", 2, 4, [1000, 2000], 1, None, 3, 32, 512),
    ("r2_decode", 1, 1, [1024], 2, [1], 3, 32, 512),
    ("r2_prefill", 1, 16, [1024], 2, [1], 3, 32, 512),
    ("g64_decode", 1, 1, [2048], 1, None, 0, 64, 512),
    ("sparse2048", 1, 1, [4096], 1, None, 0, 32, 2048),
    # 候选容量 2048 < 位置块数 2049 (s2=16392, CAND_BS=8): 触发部分覆盖 + pin 尾块
    ("pin_overflow", 1, 1, [16392], 1, None, 0, 32, 512),
    ("b4_varlen", 4, 4, [512, 1024, 1536, 2048], 1, None, 3, 32, 512),
    ("s1tail7", 2, 7, [1024, 2048], 1, None, 3, 32, 512),
    ("tiny", 1, 1, [1], 1, None, 0, 32, 512),
    ("tnd_decode", 1, 1, [2048], 1, None, 3, 32, 512),
    # 部分覆盖 + 因果 mask 的 prefill（s1=8, s2=16392）
    ("partial_m3_prefill", 1, 8, [16392], 1, None, 3, 32, 512),
    # 2026-09-30 修复回归锚点: 单行 query + 长上下文 (nb=16384 >> 候选容量 2048)，
    # metadata 会把该行的 s2 切给多核 —— 修复前候选输出从不写入。生产 pool/table 规格。
    {
        "name": "decode_long_m1",
        "batch": 1,
        "q_seq": 1,
        "seqused_k": [131072],
        "cmp_ratio": 1,
        "mask_mode": 3,
        "g": 32,
        "topk": 512,
        "pool": 7936,
        "table_w": 1055,
    },
]


def case_dict(t):
    if isinstance(t, dict):  # 直接给 dict 的用例 (可带 pool/table_w 等生产规格)
        return t
    name, b, s1, sk, ratio, res, mask, g, topk = t[:9]
    d = {
        "name": name,
        "batch": b,
        "q_seq": s1,
        "seqused_k": sk,
        "cmp_ratio": ratio,
        "cmp_res": res,
        "mask_mode": mask,
        "g": g,
        "topk": topk,
    }
    if len(t) > 9:
        d["cand_blocks"] = t[9]
    if name.startswith("tnd"):
        d["layout"] = "TND"
        if b > 1:
            d["seqs_q"] = [max(1, s1 // b)] * b
    return d


def run_case(case):
    """跑 consumer 两条路径并断言；返回结果摘要字符串。异常即用例失败。"""
    seqs_q = case.get("seqs_q") or [case["q_seq"]] * case["batch"]
    inputs = gen_case_inputs(case)
    s2max = max(case["seqused_k"])
    s1_for_rows = case["q_seq"]

    def check(idx_npu, idx_gold, score_rows, tag):
        if case.get("layout") == "TND":
            fails = []
            for r in range(sum(seqs_q)):
                v4 = score_rows[r].reshape(1, 1, 1, -1).numpy()
                ok_r, msg_r = cmp_indices(
                    idx_npu.reshape(-1, case["topk"])[r].cpu().unsqueeze(0),
                    idx_gold.reshape(-1, case["topk"])[r].unsqueeze(0),
                    "idx",
                    v4,
                    1,
                )
                if not ok_r:
                    fails.append(msg_r)
            assert not fails, f"{tag}: TND 逐行比对失败 {fails[:2]}"
            return f"{tag}: 逐行 {sum(seqs_q)} 行全过"
        v4 = (
            torch.stack(score_rows)
            .reshape(case["batch"], s1_for_rows, s2max)
            .unsqueeze(1)
            .numpy()
        )
        ok, msg = cmp_indices(
            idx_npu.reshape(-1, case["topk"]),
            idx_gold.reshape(-1, case["topk"]),
            f"sparse_indices({tag})",
            v4,
            s1_for_rows,
        )
        assert ok, f"{tag}: {msg}"
        return f"{tag}: {msg}"

    # 先跑第一级取候选（路径②的输入；其结果不在本文件断言范围内）
    _idx_src, cand_src = npu_run_source(inputs, case)
    torch.npu.synchronize()

    # ① 吃 golden 候选：隔离 consumer 自身精度
    _idx_g, cand_gold_rows, _score_src, _blk = golden_run_ext(inputs, case, "source")
    cand_gold_npu = torch.stack(cand_gold_rows).reshape(cand_src.shape).npu()
    idx_cons_g = npu_run_consumer(inputs, case, cand_gold_npu)
    torch.npu.synchronize()
    idx_gold_cg, _, score_rows_cg, _ = golden_run_ext(
        inputs, case, "consumer", torch.stack(cand_gold_rows)
    )
    msg_g = check(idx_cons_g, idx_gold_cg, score_rows_cg, "consumer,golden候选")

    # ② 吃第一级 NPU 候选：生产链路端到端
    idx_cons_n = npu_run_consumer(inputs, case, cand_src)
    torch.npu.synchronize()
    idx_gold_cn, _, score_rows_cn, _ = golden_run_ext(
        inputs, case, "consumer", cand_src.cpu()
    )
    msg_n = check(idx_cons_n, idx_gold_cn, score_rows_cn, "consumer,NPU候选链路")

    return f"{msg_g} | {msg_n}"


_ALL_CASES = [case_dict(t) for t in CASES]


@pytest.mark.parametrize("case", _ALL_CASES, ids=[c["name"] for c in _ALL_CASES])
def test_consumer_two_level(case):
    torch.npu.empty_cache()
    summary = run_case(case)
    print(f"[{case['name']}] PASS {summary}")


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
