#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# the CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""PoolKeyIndexer 性能测试套件(套件4): 自定义算子 vs NPU 组合标杆性能对比。

对比对象:
  - 被测: torch.ops.cann_ops_transformer.pool_key_indexer(自定义融合算子)
  - 标杆: pool_key_indexer_reference 输入在 NPU 时的组合标杆实现
    (纯 torch 标准算子向量化, 跑在同一 NPU 上; 详见统一标杆模块说明)

流程(每场景):
  1. 正确性门禁(缩小规模, 默认 S1=S2=512): 算子/NPU标杆 分别与 CPU 金标
     (pool_key_indexer_reference, CPU 输入) 做 indices 集合比对(100% 行通过
     为门禁, 与精度套件同标准), 任一失败则该场景跳过性能测试(对比无意义);
  2. 性能测试(全规模): warmup 后 N 次迭代, torch.npu.Event 计时,
     报告 median/min, 计算加速比(标杆 median / 算子 median)。

默认场景矩阵: pool_size in {1, 16} x 布局 in {BSND/BSND, TND/TND,
BSND/PA_BBND, TND/PA_BBND}; 固定配置 B=5, S1=4096, 池化前 S2=4096,
headDim=128, N1=32, topk=2048, mask_mode=3, fp16。

用法:
  python perf_compare.py                          # 全场景
  python perf_compare.py --pool-sizes 16          # 指定 pool_size(逗号分隔)
  python perf_compare.py --layouts BSND/BSND,BSND/PA_BBND
  python perf_compare.py --iters 30 --warmup 5
  python perf_compare.py --mask-mode 0            # 无掩码场景
  python perf_compare.py --gate-only              # 只跑正确性门禁
  python perf_compare.py --gate-ref npu           # 门禁改 op vs NPU标杆(两方)
  python perf_compare.py --perf-only              # 跳过门禁(不推荐)
  python perf_compare.py --csv result.csv         # 追加写 CSV

环境变量: PKI_DEVICE_ID=N 指定 NPU 设备(默认取 --device)。

说明: NPU 组合标杆的分数中间矩阵 (B,S1,N1,P) 为 fp32, ps=1 全规模峰值
~21GB(与融合算子无需物化该矩阵形成对照, 亦为融合价值的一部分)。
"""

import argparse
import csv
import os
import statistics
import sys

import torch
import torch_npu

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pool_key_indexer_reference import pool_key_indexer_reference
from pki_common import pack_pa_key
from result_compare import compare_indices
from cann_ops_transformer import pool_key_indexer

N2 = 1

SCENARIO_LAYOUTS = [
    ("BSND/BSND", "BSND", "BSND"),
    ("TND/TND", "TND", "TND"),
    ("BSND/PA_BBND", "BSND", "PA_BBND"),
    ("TND/PA_BBND", "TND", "PA_BBND"),
]


# ---------------------------------------------------------------------------
# 输入构造(各 batch 等长; PA 打包保证 block_table 与物理布局一致)
# ---------------------------------------------------------------------------


def build_inputs(
    layout_q,
    layout_k,
    *,
    batch,
    s1,
    s2_tokens,
    n1,
    head_dim,
    pool_size,
    dtype,
    mask_mode,
    pa_block_size,
    seed=42,
):
    """构造指定布局的 CPU 输入(等长 batch), 返回 dict。"""
    torch.manual_seed(seed)
    pools = s2_tokens // pool_size
    query = torch.randn(batch, s1, n1, head_dim, dtype=dtype)
    weights = torch.randn(batch, s1, n1, dtype=dtype)
    pool_tail_k = torch.randint(0, pool_size, (batch,), dtype=torch.int64)
    # 因果场景保证尾块可非零: 固定部分 batch 为 pool_size-1
    if mask_mode == 3 and pool_size > 1:
        pool_tail_k[-1] = pool_size - 1
    logical_key = torch.randn(batch * pools, N2, head_dim, dtype=dtype)

    actual_seq_q = actual_seq_k = block_table = None
    if layout_q == "TND":
        query = query.reshape(batch * s1, n1, head_dim)
        weights = weights.reshape(batch * s1, n1)
        actual_seq_q = torch.full((batch,), s1, dtype=torch.int64).cumsum(0)
    if layout_k == "TND":
        pool_key = logical_key  # (T2, N2, D)
        actual_seq_k = torch.full((batch,), pools, dtype=torch.int64).cumsum(0)
    elif layout_k == "PA_BBND":
        pool_key, block_table, _ = pack_pa_key(
            logical_key, [pools] * batch, pa_block_size
        )
        actual_seq_k = torch.full((batch,), pools, dtype=torch.int64)  # 非前缀和
    else:
        pool_key = logical_key.reshape(batch, pools, N2, head_dim)
    return dict(
        query=query,
        pool_key=pool_key,
        weights=weights,
        pool_tail_k=pool_tail_k,
        actual_seq_q=actual_seq_q,
        actual_seq_k=actual_seq_k,
        block_table=block_table,
    )


def _to_dev(inputs, dev):
    return {k: (v.to(dev) if torch.is_tensor(v) else v) for k, v in inputs.items()}


def _call_op(inp, *, layout_q, layout_k, topk, pool_size, mask_mode):
    return pool_key_indexer(
        inp["query"],
        inp["pool_key"],
        inp["weights"],
        inp["pool_tail_k"],
        actual_seq_q=inp["actual_seq_q"],
        actual_seq_k=inp["actual_seq_k"],
        block_table=inp["block_table"],
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        quant_mode=-1,
        return_value=True,
    )


def _call_ref(inp, *, layout_q, layout_k, topk, pool_size, mask_mode):
    """NPU 组合标杆(统一标杆入口, NPU 输入自动派发向量化实现)。"""
    return pool_key_indexer_reference(
        inp["query"],
        inp["pool_key"],
        inp["weights"],
        inp["pool_tail_k"],
        actual_seq_q=inp["actual_seq_q"],
        actual_seq_k=inp["actual_seq_k"],
        block_table=inp["block_table"],
        layout_q=layout_q,
        layout_k=layout_k,
        topk=topk,
        pool_size=pool_size,
        mask_mode=mask_mode,
        return_value=True,
    )


# ---------------------------------------------------------------------------
# 正确性门禁(缩小规模) 与 性能测试
# ---------------------------------------------------------------------------


def _gate_inputs(cfg, layout_q, layout_k):
    """门禁规模输入(全规模配置缩到 gate_s1/gate_s2, 其余不变)。"""
    return build_inputs(
        layout_q,
        layout_k,
        mask_mode=cfg["mask_mode"],
        pa_block_size=cfg["pa_block_size"],
        batch=cfg["batch"],
        s1=cfg["gate_s1"],
        s2_tokens=cfg["gate_s2"],
        n1=cfg["n1"],
        head_dim=cfg["head_dim"],
        pool_size=cfg["pool_size"],
        dtype=cfg["dtype"],
    )


def correctness_gate(cfg, layout_q, layout_k, verbose=True):
    """缩小规模比对: 算子/NPU标杆 vs CPU金标(三方, 默认) 或 算子 vs NPU标杆(两方)。

    indices 集合比对 100% 行通过为门禁。"""
    inp = _gate_inputs(cfg, layout_q, layout_k)
    dev = f"npu:{cfg['device']}"
    inp_dev = _to_dev(inp, dev)
    op_idx, _ = _call_op(
        inp_dev,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=cfg["topk"],
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
    )
    ref_idx, _ = _call_ref(
        inp_dev,
        layout_q=layout_q,
        layout_k=layout_k,
        topk=cfg["topk"],
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
    )
    op_idx, ref_idx = op_idx.cpu(), ref_idx.cpu()

    if cfg["gate_ref"] == "npu":
        # 两方门禁: 算子 vs NPU 组合标杆(不经过 CPU 金标)
        ok_op = compare_indices(ref_idx, op_idx)
        ok_ref = True
        if verbose:
            tag = f"ps={cfg['pool_size']:<2} {layout_q}/{layout_k}"
            print(f"[gate] {tag}: op_vs_npu_ref={'PASS' if ok_op else 'FAIL'}")
        return ok_op

    # 三方门禁(默认): 算子/NPU标杆 分别 vs CPU 金标
    ref_cpu = pool_key_indexer_reference(
        inp["query"],
        inp["pool_key"],
        inp["weights"],
        inp["pool_tail_k"],
        actual_seq_q=inp["actual_seq_q"],
        actual_seq_k=inp["actual_seq_k"],
        block_table=inp["block_table"],
        layout_q=layout_q,
        layout_k=layout_k,
        topk=cfg["topk"],
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
        return_value=True,
    )
    ok_op = compare_indices(ref_cpu[0], op_idx)
    ok_ref = compare_indices(ref_cpu[0], ref_idx)
    if verbose:
        tag = f"ps={cfg['pool_size']:<2} {layout_q}/{layout_k}"
        print(
            f"[gate] {tag}: op_vs_cpu={'PASS' if ok_op else 'FAIL'} "
            f"ref_vs_cpu={'PASS' if ok_ref else 'FAIL'}"
        )
    return ok_op and ok_ref


def bench(fn, iters, warmup):
    """NPU 事件计时, 返回 (median_ms, min_ms)。"""
    for _ in range(warmup):
        fn()
    torch.npu.synchronize()
    times = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.npu.synchronize()
        times.append(start.elapsed_time(end))
    return statistics.median(times), min(times)


def run_scenario(cfg, name, layout_q, layout_k):
    """一个场景: 门禁 -> 全规模性能对比, 返回结果行 dict(或 None)。"""
    tag = f"ps={cfg['pool_size']:<2} {name}"
    if not cfg["perf_only"]:
        if not correctness_gate(cfg, layout_q, layout_k):
            print(f"[skip] {tag}: 正确性门禁未通过, 跳过性能对比")
            return None
    if cfg["gate_only"]:
        return None

    inp = build_inputs(
        layout_q,
        layout_k,
        mask_mode=cfg["mask_mode"],
        pa_block_size=cfg["pa_block_size"],
        batch=cfg["batch"],
        s1=cfg["s1"],
        s2_tokens=cfg["s2_tokens"],
        n1=cfg["n1"],
        head_dim=cfg["head_dim"],
        pool_size=cfg["pool_size"],
        dtype=cfg["dtype"],
    )
    dev = f"npu:{cfg['device']}"
    inp_dev = _to_dev(inp, dev)
    op_kwargs = dict(
        layout_q=layout_q,
        layout_k=layout_k,
        topk=cfg["topk"],
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
    )

    op_ms, op_min = bench(
        lambda: _call_op(inp_dev, **op_kwargs), cfg["iters"], cfg["warmup"]
    )
    ref_ms, ref_min = bench(
        lambda: _call_ref(inp_dev, **op_kwargs), cfg["iters"], cfg["warmup"]
    )
    speedup = ref_ms / op_ms if op_ms > 0 else float("inf")
    row = dict(
        scene=name,
        pool_size=cfg["pool_size"],
        mask_mode=cfg["mask_mode"],
        op_median_ms=op_ms,
        op_min_ms=op_min,
        ref_median_ms=ref_ms,
        ref_min_ms=ref_min,
        speedup=speedup,
    )
    print(
        f"[perf] {tag}: op={op_ms:.3f}ms(min {op_min:.3f}) "
        f"ref={ref_ms:.3f}ms(min {ref_min:.3f}) speedup={speedup:.2f}x "
        f"(n={cfg['iters']})"
    )
    # 释放全规模输入引用后清空缓存(HBM 峰值 ~21GB, ps=1)
    inp = inp_dev = None
    torch.npu.empty_cache()
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--pool-sizes", default="1,16", help="逗号分隔, 默认 1,16")
    parser.add_argument(
        "--layouts",
        default=None,
        help="逗号分隔的场景名, 默认全部(BSND/BSND,TND/TND,BSND/PA_BBND,TND/PA_BBND)",
    )
    parser.add_argument("--batch", type=int, default=5)
    parser.add_argument("--s1", type=int, default=4096, help="query token 数")
    parser.add_argument(
        "--s2-tokens", type=int, default=4096, help="池化前 key token 数"
    )
    parser.add_argument("--head-dim", type=int, default=128)
    parser.add_argument("--n1", type=int, default=32)
    parser.add_argument("--topk", type=int, default=2048)
    parser.add_argument("--mask-mode", type=int, default=3, choices=(0, 3))
    parser.add_argument("--dtype", default="float16", choices=("float16", "bfloat16"))
    parser.add_argument("--pa-block-size", type=int, default=16)
    parser.add_argument("--gate-s1", type=int, default=512, help="门禁规模 S1")
    parser.add_argument("--gate-s2", type=int, default=512, help="门禁规模 S2")
    parser.add_argument("--iters", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument(
        "--device", type=int, default=int(os.environ.get("PKI_DEVICE_ID", "0"))
    )
    parser.add_argument(
        "--gate-ref",
        default="cpu",
        choices=("cpu", "npu"),
        help="门禁标杆: cpu=算子/NPU标杆 vs CPU金标(三方, 默认); "
        "npu=算子 vs NPU标杆(两方)",
    )
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--perf-only", action="store_true")
    parser.add_argument("--csv", default=None, help="结果追加写 CSV 路径")
    args = parser.parse_args()

    layouts = SCENARIO_LAYOUTS
    if args.layouts:
        wanted = {s.strip() for s in args.layouts.split(",")}
        layouts = [s for s in SCENARIO_LAYOUTS if s[0] in wanted]
        missing = wanted - {s[0] for s in layouts}
        if missing:
            parser.error(
                f"未知场景: {missing}, 可选 {[s[0] for s in SCENARIO_LAYOUTS]}"
            )

    torch_npu.npu.set_device(args.device)
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16

    print("=" * 78)
    print("PoolKeyIndexer 性能对比: 自定义算子 vs NPU 组合标杆")
    print(
        f"配置: B={args.batch} S1={args.s1} S2(池前)={args.s2_tokens} "
        f"D={args.head_dim} N1={args.n1} topk={args.topk} "
        f"mask={args.mask_mode} dtype={args.dtype} "
        f"iters={args.iters} warmup={args.warmup} device={args.device} "
        f"gate_ref={args.gate_ref}"
    )
    print("=" * 78)

    rows = []
    for ps in [int(x) for x in args.pool_sizes.split(",")]:
        for name, lq, lk in layouts:
            cfg = dict(
                batch=args.batch,
                s1=args.s1,
                s2_tokens=args.s2_tokens,
                n1=args.n1,
                head_dim=args.head_dim,
                topk=args.topk,
                pool_size=ps,
                dtype=dtype,
                mask_mode=args.mask_mode,
                pa_block_size=args.pa_block_size,
                gate_s1=args.gate_s1,
                gate_s2=args.gate_s2,
                iters=args.iters,
                warmup=args.warmup,
                device=args.device,
                gate_only=args.gate_only,
                perf_only=args.perf_only,
                gate_ref=args.gate_ref,
            )
            row = run_scenario(cfg, name, lq, lk)
            if row is not None:
                rows.append(row)

    if rows:
        print("-" * 78)
        print(
            f"{'场景':<14} {'ps':>3} {'mask':>4} {'算子(ms)':>10} {'标杆(ms)':>10} "
            f"{'加速比':>8}"
        )
        for r in rows:
            print(
                f"{r['scene']:<14} {r['pool_size']:>3} {r['mask_mode']:>4} "
                f"{r['op_median_ms']:>10.3f} {r['ref_median_ms']:>10.3f} "
                f"{r['speedup']:>7.2f}x"
            )
        print("-" * 78)
        med_op = statistics.median([r["op_median_ms"] for r in rows])
        med_ref = statistics.median([r["ref_median_ms"] for r in rows])
        print(
            f"汇总: {len(rows)} 个场景, 算子中位 {med_op:.3f}ms, "
            f"标杆中位 {med_ref:.3f}ms, 中位加速比 {med_ref / med_op:.2f}x"
        )

    if args.csv and rows:
        exists = os.path.exists(args.csv)
        with open(args.csv, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
        print(f"结果已追加写入 {args.csv}")


if __name__ == "__main__":
    main()
