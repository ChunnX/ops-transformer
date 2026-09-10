#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PoolKeyIndexer 模型场景基线复现(A5_kernel.csv)。

CSV 场景:
  Input : 1,4096,32,128(BF16 query BSND); 1,1024,1,128(BF16 pool_key BSND);
          1,4096,32(BF16 weights); 1(INT64 pool_tail_k)
  Output: 1,4096,2051(INT32 sparse_indices) + FLOAT sparse_values
  基线  : Duration 1188.686us, aiv_scalar_ratio 92.4%

由 2051 = topk + pool_size - 1 且 topk 因子约束 => topk=2048, pool_size=4。
mask_mode / pool_tail_k 值未知, 默认跑因果(mask=3)+tail=3, 并支持参数扫描。
"""

import argparse
import statistics
import sys

import torch
import torch_npu  # noqa: F401

from cann_ops_transformer import pool_key_indexer

DEV = None


def build_inputs(s1, s2_pools, n1, pool_size, tail, dtype, seed=42):
    torch.manual_seed(seed)
    b = 1
    query = torch.randn(b, s1, n1, 128, dtype=dtype)
    weights = torch.randn(b, s1, n1, dtype=dtype)
    pool_key = torch.randn(b, s2_pools, 1, 128, dtype=dtype)
    pool_tail_k = torch.tensor([tail], dtype=torch.int64)
    return query, pool_key, weights, pool_tail_k


def bench(mask_mode, tail, s1, s2_pools, n1, topk, pool_size, dtype,
          iters=50, warmup=10, return_value=False, seed=42):
    query, pool_key, weights, pool_tail_k = build_inputs(s1, s2_pools, n1, pool_size, tail, dtype, seed)
    query, pool_key, weights, pool_tail_k = (t.to(DEV) for t in (query, pool_key, weights, pool_tail_k))

    def call():
        return pool_key_indexer(
            query, pool_key, weights, pool_tail_k,
            layout_q="BSND", layout_k="BSND",
            topk=topk, pool_size=pool_size, mask_mode=mask_mode,
            quant_mode=-1, return_value=return_value,
        )

    out = call()
    torch.npu.synchronize()
    out_len = out[0].shape[-1]

    for _ in range(warmup):
        call()
    torch.npu.synchronize()

    times = []
    for _ in range(iters):
        start = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.npu.synchronize()
        times.append(start.sacred_event_time_ms if False else start.elapsed_time(end) * 1000.0)  # us
    return statistics.median(times), min(times), out_len


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--scan", action="store_true", help="扫描 mask_mode x tail 组合")
    args = p.parse_args()

    global DEV
    torch.npu.set_device(args.device)
    DEV = f"npu:{args.device}"

    # CSV 模型场景
    s1, s2_pools, n1, topk, pool_size = 4096, 1024, 32, 2048, 4
    dtype = torch.bfloat16

    print(f"场景: B=1 S1={s1} S2_pools={s2_pools} N1={n1} D=128 BF16 BSND "
          f"topk={topk} pool_size={pool_size}")
    print(f"基线(CSV): 1188.686us, aiv_scalar 92.4%")
    print("-" * 72)

    if args.scan:
        combos = [(mm, tk) for mm in (3, 0) for tk in (3, 0)]
    else:
        combos = [(3, 3)]

    for mask_mode, tail in combos:
        med, mn, out_len = bench(mask_mode, tail, s1, s2_pools, n1, topk, pool_size,
                                 dtype, args.iters, args.warmup)
        print(f"mask_mode={mask_mode} tail={tail}: median={med:9.1f}us  min={mn:9.1f}us  "
              f"out_last_dim={out_len} (期望 2051)")


if __name__ == "__main__":
    sys.exit(main())
