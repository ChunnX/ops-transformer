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

"""msprof-friendly microbench for TurboQuant batched score path."""

import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch_npu  # noqa: F401

from golden import BLOCK_SIZE, CENTROIDS, HEAD_DIM, _unpack_primary, _unpack_signs


def make_inputs(kv_len=4096, q_heads=32, kv_heads=8, mse_bits=3, seed=0):
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
def compute(q, qjl, idx, qjl_p, norm, gamma, table, lengths, mse_bits, chunk=1024):
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
        physical = table[:, logical]
        offsets = pos.remainder(BLOCK_SIZE).expand_as(physical)
        primary = centroids[_unpack_primary(idx[physical, offsets], bits).long()]
        signs = _unpack_signs(qjl_p[physical, offsets])
        n = norm[physical, offsets]
        g = gamma[physical, offsets]
        main = torch.einsum("thgd,lhd->thgl", q_main, primary[0])
        corr = torch.einsum("thgd,lhd->thgl", q_res, signs[0])
        main = main * n[0].transpose(0, 1).unsqueeze(0).unsqueeze(2)
        corr = corr * g[0].transpose(0, 1).unsqueeze(0).unsqueeze(2)
        score = (main + coeff * corr).reshape(num_tokens, q_heads, end - start)
        valid = pos < lengths[0]
        out[..., start:end] = score.masked_fill(~valid, -torch.inf)
    torch.npu.synchronize()
    return out


def main():
    iters = int(os.environ.get("TQAS_ITERS", "30"))
    warmup = int(os.environ.get("TQAS_WARMUP", "5"))
    inputs = make_inputs(4096)
    for _ in range(warmup):
        compute(*inputs)
    torch.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        compute(*inputs)
    torch.npu.synchronize()
    us = (time.perf_counter() - t0) * 1e6 / iters
    print(f"mean_latency_us={us:.3f}")
    Path("/tmp/tqas_msprof_done").write_text(str(us))


if __name__ == "__main__":
    main()
