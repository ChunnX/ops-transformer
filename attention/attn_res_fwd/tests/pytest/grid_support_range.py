# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""AttnResFwd 支持域泛化测试，不依赖历史生产 shape。"""

from __future__ import annotations

import gc
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch_npu

TEST_DIR = "/data/kdy/ops-transformer/attention/attn_res_fwd/tests/pytest"
sys.path.insert(0, TEST_DIR)
import attn_res_fwd_torch_runner as torch_runner

DTYPES = {
    "bf16": (torch.bfloat16, 1.0 / 128.0),
    "fp16": (torch.float16, 1.0 / 1024.0),
    "fp32": (torch.float32, 1.0 / 1024.0),
}
FP32_THD = 1.0 / 1024.0
UB_BYTES = 192 * 1024
OVERHEAD = 32 * 1024


@dataclass(frozen=True)
class Case:
    tag: str
    t: int
    n: int
    h: int
    dtype: str = "bf16"
    need_bwd: bool = False
    seed: int = 0
    eps: float = 1e-5
    proj_rank: int = 2
    noncontig: bool = False


def align(v: int, a: int) -> int:
    return (v + a - 1) // a * a


def expected_branch(c: Case) -> str:
    elem = 4 if c.dtype == "fp32" else 2
    b = c.n + 1
    work = 4 * c.h * 4 + 2 * c.h * elem + OVERHEAD
    max_rows = 0 if work >= UB_BYTES else (UB_BYTES - work) // (c.h * elem + 4)
    if b <= max_rows:
        return "RESIDENT"
    meta = align(b, 8)
    reload_ub = (
        3 * align(c.h, 16) * elem
        + 3 * align(c.h, 8) * 4
        + 10 * meta * 4
        + 8 * 4
        + OVERHEAD
    )
    if c.need_bwd:
        reload_ub += 8 * 4 + meta * 4
    return "RELOAD" if reload_ub <= UB_BYTES else "HSLICE"


def make_tensor(shape, dtype, seed, noncontig=False):
    gen = torch.Generator(device="cpu")
    gen.manual_seed(seed)
    if not noncontig or not shape or shape[-1] == 0:
        return torch.randn(shape, dtype=dtype, generator=gen)
    wide = torch.randn((*shape[:-1], shape[-1] * 2), dtype=dtype, generator=gen)
    return wide[..., ::2]


def make_inputs(c: Case):
    dtype = DTYPES[c.dtype][0]
    ps = make_tensor((c.t, c.h), dtype, c.seed + 1, c.noncontig)
    br = make_tensor((c.t, c.n, c.h), dtype, c.seed + 2, c.noncontig)
    proj_shape = (c.h,) if c.proj_rank == 1 else (1, c.h)
    pw = make_tensor(proj_shape, dtype, c.seed + 3, c.noncontig)
    nw = make_tensor((c.h,), dtype, c.seed + 4, c.noncontig)
    return ps, br, pw, nw


def golden(inputs, eps):
    ps, br, pw, nw = inputs
    v = torch.cat((br, ps.unsqueeze(1)), dim=1).float()
    inv = torch.rsqrt(v.square().mean(-1) + eps)
    score_weight = nw.float() * pw.reshape(-1, ps.shape[-1])[0].float()
    probs = (v * inv.unsqueeze(-1) * score_weight).sum(-1).softmax(-1)
    hidden = (probs.unsqueeze(-1) * v).sum(1).to(ps.dtype)
    return hidden, inv, probs


def compare(actual, expected, thd):
    a = actual.detach().float().cpu().numpy().reshape(-1)
    e = expected.detach().float().cpu().numpy().reshape(-1)
    if a.shape != e.shape or not np.isfinite(a).all():
        return False, float("inf"), float("inf"), int(a.size)
    if a.size == 0:
        return True, 0.0, 0.0, 0
    abs_err = np.abs(a - e)
    scale = np.maximum(np.abs(a), np.abs(e))
    floor = (1.0 / (1 << 14)) / thd
    rule = np.where(abs_err < thd, abs_err, abs_err / (np.maximum(scale, floor) + 1e-9))
    bad = int(np.count_nonzero(rule > thd))
    return bad == 0, float(rule.max()), float(abs_err.max()), bad


def build_cases():
    cases = []
    h_values = (
        1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129,
        255, 256, 257, 511, 512, 513, 1023, 1024, 1025, 2047, 2048, 4095,
        4096, 7168, 8191, 8192, 8193, 12288, 16384, 32768, 65536,
    )
    n_edges = (1, 7, 8, 9, 31, 32, 33, 63, 64, 65, 99, 100)
    for i, h in enumerate(h_values):
        cases.append(Case("h_sweep", 1 + i % 3, n_edges[i % len(n_edges)], h,
                          need_bwd=(i % 4 == 0), seed=1000 + i))

    branch_h = (64, 7168, 16384)
    for n in range(1, 101):
        h = branch_h[(n - 1) % len(branch_h)]
        cases.append(Case("n_full", 1, n, h, need_bwd=(n % 10 == 0), seed=2000 + n))

    for dtype in DTYPES:
        for h, n in ((64, 8), (7168, 100), (16384, 100)):
            for need_bwd in (False, True):
                cases.append(Case("dtype_branch", 2, n, h, dtype, need_bwd, 3000 + h + n))
        for need_bwd in (False, True):
            cases.append(Case("max_cross", 1, 100, 65536, dtype, need_bwd, 3900))

    for i, t in enumerate((0, 1, 2, 39, 40, 41, 127, 1024, 4096)):
        cases.append(Case("t_sweep", t, 100, 64, need_bwd=(i % 2 == 0), seed=4000 + t))

    for rank in (1, 2):
        for noncontig in (False, True):
            for h, n in ((257, 13), (8193, 100)):
                cases.append(Case("layout", 3, n, h, need_bwd=True, seed=5000 + h,
                                  proj_rank=rank, noncontig=noncontig))

    for i, eps in enumerate((1e-12, 1e-8, 1e-5, 1e-2, 1.0, 100.0)):
        cases.append(Case("eps", 2, (i * 17) % 100 + 1, (65, 7168, 16384)[i % 3],
                          need_bwd=True, seed=6000 + i, eps=eps))

    rng = random.Random(7000)
    random_h = (1, 5, 19, 67, 130, 259, 777, 1537, 3073, 6145, 9001, 15001, 30001)
    for i in range(30):
        h = rng.choice(random_h)
        n = rng.randint(1, 100)
        max_t = max(1, min(64, 8_000_000 // max(1, n * h)))
        cases.append(Case("random", rng.randint(1, max_t), n, h, rng.choice(tuple(DTYPES)),
                          rng.choice((False, True)), 7000 + i, 10 ** rng.randint(-8, -2),
                          rng.choice((1, 2)), rng.choice((False, True))))
    return cases


def check_invalid_rejected(fn, device):
    dtype = torch.bfloat16
    base = (
        torch.randn(2, 64, dtype=dtype),
        torch.randn(2, 3, 64, dtype=dtype),
        torch.randn(1, 64, dtype=dtype),
        torch.randn(64, dtype=dtype),
    )
    invalid = {
        "h0_fwd": (torch.randn(2, 0, dtype=dtype), torch.randn(2, 3, 0, dtype=dtype),
                   torch.randn(1, 0, dtype=dtype), torch.randn(0, dtype=dtype), False),
        "h0_bwd": (torch.randn(2, 0, dtype=dtype), torch.randn(2, 3, 0, dtype=dtype),
                   torch.randn(1, 0, dtype=dtype), torch.randn(0, dtype=dtype), True),
        "n0": (base[0], torch.randn(2, 0, 64, dtype=dtype), base[2], base[3], False),
        "n101": (base[0], torch.randn(2, 101, 64, dtype=dtype), base[2], base[3], False),
        "proj_2h": (base[0], base[1], torch.randn(2, 64, dtype=dtype), base[3], False),
    }
    accepted = []
    for name, (*inputs, need_bwd) in invalid.items():
        npu_inputs = tuple(x.to(device) for x in inputs)
        try:
            fn(*npu_inputs, 1e-5, need_bwd)
            torch.npu.synchronize()
            accepted.append(name)
        except RuntimeError:
            pass
    print(f"INVALID_SUMMARY total={len(invalid)} rejected={len(invalid)-len(accepted)} accepted={accepted}")
    return not accepted


def main():
    device_id = int(os.environ.get("DEVICE_ID", "7"))
    torch.npu.set_device(device_id)
    fn = torch_runner.get_attn_res_fwd_fn()
    cases = build_cases()
    branch_counts = {"RESIDENT": 0, "RELOAD": 0, "HSLICE": 0}
    dtype_counts = {k: 0 for k in DTYPES}
    failures = []
    started = time.time()
    for idx, c in enumerate(cases, 1):
        branch = expected_branch(c)
        branch_counts[branch] += 1
        dtype_counts[c.dtype] += 1
        cpu_inputs = make_inputs(c)
        expected = golden(cpu_inputs, c.eps)
        npu_inputs = tuple(x.to(f"npu:{device_id}") for x in cpu_inputs)
        try:
            actual = fn(*npu_inputs, c.eps, c.need_bwd)
            torch.npu.synchronize()
        except Exception as exc:
            print(f"ERROR idx={idx} case={c} branch={branch}: {type(exc).__name__}: {exc}", flush=True)
            return 2
        actual_tuple = actual if c.need_bwd else (actual,)
        names = ("hidden", "inv", "probs") if c.need_bwd else ("hidden",)
        thresholds = (DTYPES[c.dtype][1], FP32_THD, FP32_THD) if c.need_bwd else (DTYPES[c.dtype][1],)
        details = []
        ok = True
        for name, lhs, rhs, thd in zip(names, actual_tuple, expected, thresholds):
            tensor_ok, max_rule, max_abs, bad = compare(lhs, rhs, thd)
            ok &= tensor_ok
            details.append(f"{name}:bad={bad},rule={max_rule:.6g},abs={max_abs:.6g}")
        if not ok:
            failures.append((c, branch, details))
            print(f"FAIL idx={idx} case={c} branch={branch} " + " | ".join(details), flush=True)
        elif idx % 20 == 0 or idx == len(cases):
            print(f"PASS {idx}/{len(cases)} branch={branch} dtype={c.dtype} "
                  f"shape=({c.t},{c.n},{c.h}) elapsed={time.time()-started:.1f}s", flush=True)
        del cpu_inputs, npu_inputs, expected, actual, actual_tuple
        gc.collect()
    print(f"SUMMARY total={len(cases)} failures={len(failures)} branches={branch_counts} "
          f"dtypes={dtype_counts} elapsed={time.time()-started:.1f}s")
    for c, branch, details in failures:
        print(f"FAILED_CASE case={c} branch={branch} " + " | ".join(details))
    invalid_ok = check_invalid_rejected(fn, f"npu:{device_id}")
    return 1 if failures or not invalid_ok else 0


if __name__ == "__main__":
    raise SystemExit(main())
