# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
已知缺陷用例的子进程 runner: 单用例独立进程执行, 崩溃(EE9999/aicore
exception)只污染本进程, 由父进程按输出/退出码判定。

用法:
    python _defect_case_runner.py <case_id> [--device 0]

输出协议(父进程解析):
    DEFECT_CASE_RESULT: PASS <case_id>
    DEFECT_CASE_RESULT: FAIL <case_id> <简述>     # 精度/比对失败(进程存活)
    DEFECT_CASE_RESULT: CRASH <case_id> <简述>    # NPU 运行时异常(进程存活, 已捕获)
退出码: 0=PASS, 1=FAIL, 2=CRASH; 进程被信号杀死(如 SIGABRT)由父进程按
负返回码识别为 CRASH。
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import torch
import torch_npu

DEVICE_ID = 0
HEAD_DIM = 128
N2 = 1
SEED_BASE = 20260826


def _pack_pa_key(pool_key_rows, k_pool_lens, block_size):
    """逻辑 (sum_pools, N2, D) -> PA 物理块布局 (blockNum, blockSize, N2, D)。"""
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    block_num = sum(blocks_per_batch)
    n2, d = pool_key_rows.shape[1], pool_key_rows.shape[2]
    blocks = torch.zeros(block_num, block_size, n2, d, dtype=pool_key_rows.dtype)
    pool_start = blk_start = 0
    for pools, nb in zip(k_pool_lens, blocks_per_batch):
        if nb == 0:
            continue
        seg = pool_key_rows[pool_start:pool_start + pools]
        pad = nb * block_size - pools
        if pad:
            seg = torch.cat([seg, torch.zeros(pad, n2, d, dtype=seg.dtype)], dim=0)
        blocks[blk_start:blk_start + nb] = seg.view(nb, block_size, n2, d)
        pool_start += pools
        blk_start += nb
    perm = torch.randperm(block_num)
    return blocks[perm].contiguous(), block_size, block_num, perm


def _build_block_table(batch, k_pool_lens, block_size, perm):
    blocks_per_batch = [(pl + block_size - 1) // block_size for pl in k_pool_lens]
    max_blocks = max(blocks_per_batch)
    bt = torch.full((batch, max_blocks), -1, dtype=torch.int32)
    cur = 0
    for b, nb in enumerate(blocks_per_batch):
        for i in range(nb):
            bt[b, i] = perm[cur]
            cur += 1
    return bt


def build_case_inputs(cfg):
    """根据用例配置构造 CPU 输入。返回 dict(含 golden/NPU 共用的全部张量)。"""
    seed = SEED_BASE + (hash(cfg["id"]) & 0xFFFF)
    torch.manual_seed(seed)
    dtype = cfg["dtype"]
    ps = cfg["pool_size"]
    q_lens = list(cfg["q_lens"])
    k_pools = list(cfg["k_pools"])
    n1 = cfg["n1"]
    batch = len(q_lens)

    actual_seq_q = None
    actual_seq_k = None
    block_table = None

    # ---- query / weights ----
    if cfg["layout_q"] == "BSND":
        assert all(q == q_lens[0] for q in q_lens), "BSND 要求各 batch S1 相同"
        query = torch.randn(batch, q_lens[0], n1, HEAD_DIM, dtype=dtype)
        weights = torch.randn(batch, q_lens[0], n1, dtype=dtype)
    else:
        t1 = sum(q_lens)
        query = torch.randn(t1, n1, HEAD_DIM, dtype=dtype)
        weights = torch.randn(t1, n1, dtype=dtype)
        actual_seq_q = torch.tensor(q_lens, dtype=torch.int64).cumsum(0)

    # ---- pool_key ----
    if cfg["layout_k"] == "PA_BBND":
        logical = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype)
        pool_key, block_size, block_num, perm = _pack_pa_key(logical, k_pools, cfg["pa_block_size"])
        block_table = _build_block_table(batch, k_pools, cfg["pa_block_size"], perm)
        actual_seq_k = torch.tensor(k_pools, dtype=torch.int64)  # 非前缀和
    elif cfg["layout_q"] == "TND":
        pool_key = torch.randn(sum(k_pools), N2, HEAD_DIM, dtype=dtype)
        actual_seq_k = torch.tensor(k_pools, dtype=torch.int64).cumsum(0)  # 前缀和
    else:
        assert all(p == k_pools[0] for p in k_pools), "BSND 要求各 batch 池数相同"
        pool_key = torch.randn(batch, k_pools[0], N2, HEAD_DIM, dtype=dtype)

    # ---- pool_tail_k ----
    tail = cfg.get("tail")
    if tail is None:
        pool_tail_k = torch.randint(0, ps, (batch,), dtype=torch.int64)
    else:
        pool_tail_k = torch.tensor(list(tail), dtype=torch.int64)
        assert bool((pool_tail_k >= 0).all()) and bool((pool_tail_k < ps).all())

    return dict(query=query, pool_key=pool_key, weights=weights, pool_tail_k=pool_tail_k,
                actual_seq_q=actual_seq_q, actual_seq_k=actual_seq_k, block_table=block_table)


def run_case(cfg, device_id):
    from pool_key_indexer_reference import pool_key_indexer_reference
    from result_compare import compare_indices, compare_values
    from cann_ops_transformer import pool_key_indexer

    inputs = build_case_inputs(cfg)
    ps = cfg["pool_size"]
    topk = cfg["topk"]

    # ---- CPU golden ----
    g_indices, g_values = pool_key_indexer_reference(
        inputs["query"], inputs["pool_key"], inputs["weights"], inputs["pool_tail_k"],
        actual_seq_q=inputs["actual_seq_q"], actual_seq_k=inputs["actual_seq_k"],
        block_table=inputs["block_table"],
        layout_q=cfg["layout_q"], layout_k=cfg["layout_k"],
        topk=topk, pool_size=ps, mask_mode=cfg["mask_mode"],
        return_value=cfg["return_value"])

    # ---- NPU ----
    dev = f"npu:{device_id}"
    npu_indices, npu_values = pool_key_indexer(
        inputs["query"].to(dev), inputs["pool_key"].to(dev),
        inputs["weights"].to(dev), inputs["pool_tail_k"].to(dev),
        actual_seq_q=(inputs["actual_seq_q"].to(dev) if inputs["actual_seq_q"] is not None else None),
        actual_seq_k=(inputs["actual_seq_k"].to(dev) if inputs["actual_seq_k"] is not None else None),
        block_table=(inputs["block_table"].to(dev) if inputs["block_table"] is not None else None),
        layout_q=cfg["layout_q"], layout_k=cfg["layout_k"],
        topk=topk, pool_size=ps, mask_mode=cfg["mask_mode"],
        quant_mode=-1, return_value=cfg["return_value"])
    torch.npu.synchronize()
    npu_indices = npu_indices.cpu()
    npu_values = npu_values.cpu()

    # ---- 比对(indices 集合门禁 + values 参考日志) ----
    ok_idx = compare_indices(g_indices, npu_indices)
    if not ok_idx:
        return False, "indices set comparison failed"
    if cfg["return_value"]:
        assert npu_values.shape == g_values.shape, (
            f"values shape mismatch: golden {g_values.shape} vs npu {npu_values.shape}")
        compare_values(g_values, npu_values, cfg["dtype"], label="values")
    else:
        assert npu_values.numel() == 0 or npu_values.shape[-1] == 0, (
            f"expected empty values, got shape {npu_values.shape}")
    return True, ""


def main():
    global DEVICE_ID
    args = sys.argv[1:]
    if not args:
        print("usage: _defect_case_runner.py <case_id> [--device N]")
        sys.exit(3)
    case_id = args[0]
    if "--device" in args:
        DEVICE_ID = int(args[args.index("--device") + 1])

    from known_defect_cases import CASE_BY_ID
    cfg = CASE_BY_ID[case_id]

    torch_npu.npu.set_device(DEVICE_ID)
    print(f"[runner] case={case_id} group={cfg['group']} layout=({cfg['layout_q']},{cfg['layout_k']}) "
          f"dtype={cfg['dtype']} q_lens={cfg['q_lens']} k_pools={cfg['k_pools']} n1={cfg['n1']} "
          f"topk={cfg['topk']} ps={cfg['pool_size']} mask={cfg['mask_mode']} "
          f"val={cfg['return_value']} tail={cfg.get('tail')} note={cfg.get('note')}", flush=True)

    try:
        ok, detail = run_case(cfg, DEVICE_ID)
    except RuntimeError as e:
        # NPU 运行时异常(aicore exception / EE9999 等)
        msg = str(e).replace("\n", " ")[:500]
        print(f"DEFECT_CASE_RESULT: CRASH {case_id} RuntimeError: {msg}", flush=True)
        sys.exit(2)
    except AssertionError as e:
        msg = str(e).replace("\n", " ")[:500]
        print(f"DEFECT_CASE_RESULT: FAIL {case_id} assert: {msg}", flush=True)
        sys.exit(1)
    except Exception:
        msg = traceback.format_exc().replace("\n", " ")[-500:]
        print(f"DEFECT_CASE_RESULT: CRASH {case_id} unexpected: {msg}", flush=True)
        sys.exit(2)

    if ok:
        print(f"DEFECT_CASE_RESULT: PASS {case_id}", flush=True)
        sys.exit(0)
    print(f"DEFECT_CASE_RESULT: FAIL {case_id} {detail}", flush=True)
    sys.exit(1)


if __name__ == "__main__":
    main()
