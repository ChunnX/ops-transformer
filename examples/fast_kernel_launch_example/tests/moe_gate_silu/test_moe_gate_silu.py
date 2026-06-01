#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import argparse
from dataclasses import dataclass
from typing import Iterable

import pytest
import torch
import torch.nn.functional as F
import torch_npu

import ascend_ops


@dataclass(frozen=True)
class CompareMetrics:
    max_abs: float
    mean_abs: float
    max_rel: float
    allclose: bool


TEST_CASES = [
    (1, 32),
    (7, 64),
    (40, 128),
    (128, 256),
    (2, 8192),
]

DTYPES = [
    torch.bfloat16,
]

TOLERANCE = {
    torch.bfloat16: (1e-2, 1e-2),
}


def _npu_available() -> bool:
    return hasattr(torch, "npu") and torch.npu.is_available()


def _cpu_reference(x: torch.Tensor) -> torch.Tensor:
    hidden = x.shape[-1] // 2
    x_fp32 = x.to(torch.float32)
    out = F.silu(x_fp32[:, :hidden]) * x_fp32[:, hidden:]
    return out.to(x.dtype)


def _compare(actual: torch.Tensor, expected: torch.Tensor, dtype: torch.dtype) -> CompareMetrics:
    actual_fp32 = actual.to(torch.float32)
    expected_fp32 = expected.to(torch.float32)
    abs_diff = (actual_fp32 - expected_fp32).abs()
    rel_diff = abs_diff / expected_fp32.abs().clamp_min(1e-6)
    rtol, atol = TOLERANCE[dtype]
    return CompareMetrics(
        max_abs=abs_diff.max().item(),
        mean_abs=abs_diff.mean().item(),
        max_rel=rel_diff.max().item(),
        allclose=torch.allclose(actual_fp32, expected_fp32, rtol=rtol, atol=atol),
    )


def _run_case(rows: int, hidden: int, dtype: torch.dtype, seed: int = 2026) -> CompareMetrics:
    torch.manual_seed(seed)
    assert hidden % 2 == 0
    x_cpu = torch.randn(rows, hidden, dtype=dtype)
    total_rows_cpu = torch.tensor([rows], dtype=torch.int64)

    expected = _cpu_reference(x_cpu)
    x_npu = x_cpu.npu()
    total_rows_npu = total_rows_cpu.npu()

    actual_npu = torch.ops.ascend_ops.moe_gate_silu(x_npu, total_rows_npu)
    torch.npu.synchronize()
    actual = actual_npu.cpu()

    assert actual.shape == expected.shape
    return _compare(actual, expected, dtype)


def _dtype_from_name(name: str) -> torch.dtype:
    if name == "bfloat16":
        return torch.bfloat16
    raise ValueError(f"Unsupported dtype: {name}")


def _iter_cli_cases(rows: int | None, hidden: int | None) -> Iterable[tuple[int, int]]:
    if rows is not None and hidden is not None:
        yield rows, hidden
        return
    yield from TEST_CASES


def test_moe_gate_silu_interface_exist():
    assert hasattr(torch.ops.ascend_ops, "moe_gate_silu")


@pytest.mark.skipif(not _npu_available(), reason="NPU device not found")
@pytest.mark.parametrize("rows, hidden", TEST_CASES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_moe_gate_silu_precision(rows: int, hidden: int, dtype: torch.dtype):
    metrics = _run_case(rows, hidden, dtype)
    rtol, atol = TOLERANCE[dtype]
    assert metrics.allclose, (
        f"moe_gate_silu precision failed: rows={rows}, hidden={hidden}, dtype={dtype}, "
        f"max_abs={metrics.max_abs:.6e}, mean_abs={metrics.mean_abs:.6e}, "
        f"max_rel={metrics.max_rel:.6e}, rtol={rtol}, atol={atol}"
    )


def main():
    parser = argparse.ArgumentParser(description="Run moe_gate_silu precision comparison.")
    parser.add_argument("--rows", type=int, default=None)
    parser.add_argument("--hidden", type=int, default=None)
    parser.add_argument("--dtype", choices=["bfloat16"], default="bfloat16")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if not _npu_available():
        raise RuntimeError("NPU device not found")
    if (args.rows is None) ^ (args.hidden is None):
        raise ValueError("--rows and --hidden must be specified together")
    if args.hidden is not None and args.hidden % 2 != 0:
        raise ValueError("--hidden must be even")

    dtype = _dtype_from_name(args.dtype)
    rtol, atol = TOLERANCE[dtype]
    print("rows,hidden,dtype,max_abs,mean_abs,max_rel,allclose,rtol,atol")
    failed = False
    for rows, hidden in _iter_cli_cases(args.rows, args.hidden):
        metrics = _run_case(rows, hidden, dtype, args.seed)
        failed = failed or not metrics.allclose
        print(
            f"{rows},{hidden},{args.dtype},{metrics.max_abs:.6e},"
            f"{metrics.mean_abs:.6e},{metrics.max_rel:.6e},{metrics.allclose},{rtol},{atol}"
        )
    if failed:
        raise AssertionError("Precision comparison failed")


if __name__ == "__main__":
    main()
