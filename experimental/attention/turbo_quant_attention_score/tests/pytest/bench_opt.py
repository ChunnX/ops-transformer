#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""Optimized TurboQuant score: NPU-friendly unpack + batched GEMM.

Bottleneck from msprof (kv4k): LeftShift/RightShift on AI_CPU ~76%.
Replace bit-shift loops with vectorized byte math staying on NPU.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

from golden import BLOCK_SIZE, CENTROIDS, HEAD_DIM

BASELINE_US = {
    "gqa_decode_kv4k": 1501.634,
    "gqa_decode_kv32k": 9278.507,
    "gqa_decode_kv128k": 35681.690,
}


def unpack_primary_v(packed: torch.Tensor, bits: int) -> torch.Tensor:
    """Unpack via few wide tensor shifts (avoid AI_CPU micro-loops)."""
    values = packed.to(torch.int32)
    if bits == 3:
        triplets = values.reshape(*values.shape[:-1], 16, 3)
        words = triplets[..., 0] | (triplets[..., 1] << 8) | (triplets[..., 2] << 16)
        parts = [((words >> s) & 7) for s in range(0, 24, 3)]
        return torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], HEAD_DIM)

    mask = (1 << bits) - 1
    parts = [((values >> s) & mask) for s in range(0, 8, bits)]
    return torch.stack(parts, dim=-1).reshape(*packed.shape[:-1], HEAD_DIM)


def unpack_signs_v(packed: torch.Tensor) -> torch.Tensor:
    values = packed.to(torch.int32)
    parts = [((values >> s) & 1) for s in range(8)]
    bits = torch.stack(parts, dim=-1)
    return bits.reshape(*packed.shape[:-1], HEAD_DIM).to(torch.float32).mul_(2).sub_(1)


def make_inputs(kv_len: int, q_heads=32, kv_heads=8, mse_bits=3, seed=0):
    rng = np.random.default_rng(seed)
    num_blocks = (kv_len + BLOCK_SIZE - 1) // BLOCK_SIZE
    idx_bytes = (HEAD_DIM * mse_bits + 7) // 8
    qjl_bytes = (HEAD_DIM + 7) // 8
    q = torch.from_numpy(
        rng.uniform(-1, 1, (1, q_heads, HEAD_DIM)).astype(np.float32)
    ).npu()
    qjl = torch.from_numpy(
        rng.uniform(-1, 1, (1, q_heads, HEAD_DIM)).astype(np.float32)
    ).npu()
    idx = torch.from_numpy(
        rng.integers(
            0, 256, (num_blocks, BLOCK_SIZE, kv_heads, idx_bytes), dtype=np.uint8
        )
    ).npu()
    qjl_p = torch.from_numpy(
        rng.integers(
            0, 256, (num_blocks, BLOCK_SIZE, kv_heads, qjl_bytes), dtype=np.uint8
        )
    ).npu()
    norm = torch.from_numpy(
        rng.uniform(0.1, 2.0, (num_blocks, BLOCK_SIZE, kv_heads)).astype(np.float32)
    ).npu()
    gamma = torch.from_numpy(
        rng.uniform(0.1, 2.0, (num_blocks, BLOCK_SIZE, kv_heads)).astype(np.float32)
    ).npu()
    table = torch.arange(num_blocks, device="npu", dtype=torch.int32).view(1, -1)
    lengths = torch.tensor([kv_len], device="npu", dtype=torch.int32)
    return q, qjl, idx, qjl_p, norm, gamma, table, lengths, mse_bits


@torch.no_grad()
def compute_opt(q, qjl, idx, qjl_p, norm, gamma, table, lengths, mse_bits, chunk=1024):
    bits = int(mse_bits)
    centroids = torch.tensor(CENTROIDS[bits], dtype=torch.float32, device=q.device)
    coeff = math.sqrt(math.pi / 2.0) / HEAD_DIM
    num_tokens, q_heads, _ = q.shape
    kv_heads = idx.shape[2]
    groups = q_heads // kv_heads
    max_kv = int(lengths.max().item())
    q_main = q.reshape(num_tokens, kv_heads, groups, HEAD_DIM)
    q_res = qjl.reshape(num_tokens, kv_heads, groups, HEAD_DIM)
    out = torch.empty(
        (num_tokens, q_heads, max_kv), dtype=torch.float32, device=q.device
    )

    for start in range(0, max_kv, chunk):
        end = min(start + chunk, max_kv)
        pos = torch.arange(start, end, device=q.device, dtype=torch.long)
        logical = pos // BLOCK_SIZE
        physical = table[0, logical]
        offsets = pos.remainder(BLOCK_SIZE)
        # [L, KvH, ...] gather without expand_as batch dim
        primary = centroids[unpack_primary_v(idx[physical, offsets], bits).long()]
        signs = unpack_signs_v(qjl_p[physical, offsets])
        n = norm[physical, offsets]
        g = gamma[physical, offsets]
        # primary: [L, KvH, D]
        main = torch.einsum("thgd,lhd->thgl", q_main, primary)
        corr = torch.einsum("thgd,lhd->thgl", q_res, signs)
        main = main * n.transpose(0, 1).unsqueeze(0).unsqueeze(2)
        corr = corr * g.transpose(0, 1).unsqueeze(0).unsqueeze(2)
        score = (main + coeff * corr).reshape(num_tokens, q_heads, end - start)
        valid = pos < lengths[0]
        out[..., start:end] = score.masked_fill(~valid, -torch.inf)
    torch.npu.synchronize()
    return out


def bench(fn, warmup=10, samples=60, drop_max=10):
    for _ in range(warmup):
        fn()
        torch.npu.synchronize()
    times = []
    for _ in range(samples):
        torch.npu.synchronize()
        t0 = time.perf_counter()
        fn()
        torch.npu.synchronize()
        times.append((time.perf_counter() - t0) * 1e6)
    times.sort()
    return float(np.mean(times[: samples - drop_max]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--kv-len", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=1024)
    parser.add_argument("--out", default="/root/docs/perf_opt_kv4k.json")
    args = parser.parse_args()
    inputs = make_inputs(args.kv_len)

    def once():
        return compute_opt(*inputs, chunk=args.chunk)

    once()
    us = bench(once)
    case = f"gqa_decode_kv{args.kv_len // 1024}k"
    base = BASELINE_US.get(case)
    result = {
        "case": case,
        "impl": "opt_vector_unpack_gemm",
        "chunk": args.chunk,
        "latency_us": round(us, 3),
        "baseline_us": base,
        "ratio": round(us / base, 3) if base else None,
        "target_1_2x_us": round(base * 1.2, 3) if base else None,
    }
    print(json.dumps(result, ensure_ascii=False), flush=True)
    Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
