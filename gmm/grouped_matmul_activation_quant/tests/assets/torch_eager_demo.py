#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
"""
Torch eager smoke demo for GroupedMatmulActivationQuant WeightNz.

This script checks two things:
1. The torch eager API can call into aclnn and finish device execution, which
   means the request has passed through aclnn, tiling, and kernel.
2. The fused operator result matches the Python golden implementation.

Run after installing/building the torch_extension package in an Ascend 950
environment, for example:

    cd /path/to/ops-transformer
    python gmm/grouped_matmul_activation_quant/tests/assets/torch_eager_demo.py
"""

import argparse
import os
import sys
import traceback

import numpy as np
import torch
import torch_npu


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
TORCH_EXTENSION_DIR = os.path.join(REPO_ROOT, "torch_extension")
if TORCH_EXTENSION_DIR not in sys.path:
    sys.path.insert(0, TORCH_EXTENSION_DIR)

from cann_ops_transformer.ops import npu_grouped_matmul_activation_quant  # noqa: E402
from golden import grouped_matmul_activation_quant_golden  # noqa: E402


GE_DTYPE_FLOAT8_E5M2 = 35
GE_DTYPE_FLOAT8_E4M3FN = 36
E8M0_ONE_BYTE = 127


def _log(message):
    print(f"[GMMAQ demo] {message}", flush=True)


def _ceil_div(value, factor):
    return (value + factor - 1) // factor


def _pack_weight_to_nz(weight_kn):
    # Non-transposed WeightNz storage shape:
    # (E, ceil(N / 32), ceil(K / 16), 16, 32)
    e, k, n = weight_kn.shape
    nz = np.zeros((e, _ceil_div(n, 32), _ceil_div(k, 16), 16, 32), dtype=np.float32)
    for group_idx in range(e):
        for k_idx in range(k):
            for n_idx in range(n):
                nz[group_idx, n_idx // 32, k_idx // 16, k_idx % 16, n_idx % 32] = weight_kn[group_idx, k_idx, n_idx]
    return nz


def _make_inputs(seed=20260617):
    # Shape comes from test_grouped_matmul_activation_quant_tiling.csv:
    # valid_no_trans_e4m3.
    e, m, k, n = 1, 64, 128, 128
    _log(f"prepare inputs: E={e}, M={m}, K={k}, N={n}, seed={seed}")
    rng = np.random.default_rng(seed)

    # Use small values to keep the reference stable while still exercising math.
    x_f32 = rng.uniform(-0.5, 0.5, size=(m, k)).astype(np.float32)
    weight_f32 = rng.uniform(-0.25, 0.25, size=(e, k, n)).astype(np.float32)

    x_cpu = torch.from_numpy(x_f32).to(torch.float8_e4m3fn)
    weight_cpu = torch.from_numpy(weight_f32).to(torch.float8_e4m3fn)

    # Golden consumes decoded FP8 values arranged as WeightNz.
    x_golden = x_cpu.to(torch.float32).numpy()
    weight_golden = _pack_weight_to_nz(weight_cpu.to(torch.float32).numpy())

    x_scale_golden = np.ones((m, _ceil_div(k, 64), 2), dtype=np.float32)
    weight_scale_golden = np.ones((e, _ceil_div(k, 64), n, 2), dtype=np.float32)
    group_list_golden = np.array([m], dtype=np.int64)

    # Device tensors. For E8M0 scale, byte 127 represents 1.0.
    x_npu = x_cpu.npu()
    weight_npu = torch.from_numpy(weight_golden).to(torch.float8_e4m3fn).npu()
    _log(f"format cast weight to FRACTAL_NZ: before shape={tuple(weight_npu.shape)}, dim={weight_npu.dim()}")
    weight_npu = torch_npu.npu_format_cast(weight_npu, 29, customize_dtype=torch.float8_e4m3fn)
    _log(f"format cast weight done: after shape={tuple(weight_npu.shape)}, dim={weight_npu.dim()}")
    if weight_npu.dim() != 5:
        raise RuntimeError(f"weight must keep 5D WeightNz storage after npu_format_cast, but got {weight_npu.dim()}D.")
    weight_scale_npu = torch.full((e, _ceil_div(k, 64), n, 2), E8M0_ONE_BYTE, dtype=torch.uint8).npu()
    x_scale_npu = torch.full((m, _ceil_div(k, 64), 2), E8M0_ONE_BYTE, dtype=torch.uint8).npu()
    group_list_npu = torch.from_numpy(group_list_golden).npu()

    _log("input tensors are ready on NPU")
    return {
        "x_npu": x_npu,
        "weight_npu": weight_npu,
        "weight_scale_npu": weight_scale_npu,
        "x_scale_npu": x_scale_npu,
        "group_list_npu": group_list_npu,
        "x_golden": x_golden,
        "weight_golden": weight_golden,
        "weight_scale_golden": weight_scale_golden,
        "x_scale_golden": x_scale_golden,
        "group_list_golden": group_list_golden,
    }


def _to_numpy_float32(tensor):
    return tensor.detach().cpu().to(torch.float32).numpy()


def _run(args):
    _log("start torch eager demo")
    _log(f"python={sys.version.split()[0]}, torch={torch.__version__}")

    data = _make_inputs()

    _log("call npu_grouped_matmul_activation_quant")
    y_npu, y_scale_npu = npu_grouped_matmul_activation_quant(
        data["x_npu"],
        [data["weight_npu"]],
        [data["weight_scale_npu"]],
        data["group_list_npu"],
        x_scale=data["x_scale_npu"],
        bias=None,
        activation_type="gelu_tanh",
        dequant_dtype=torch.float32,
        group_list_type=0,
        tuning_config=None,
        quant_mode="MX",
        y_dtype=torch.float8_e4m3fn,
        round_mode="rint",
        scale_alg=0,
        dst_type_max=0.0,
        weight_scale_dtype=torch_npu.float8_e8m0fnu,
        x_scale_dtype=torch_npu.float8_e8m0fnu,
    )
    _log("torch op returned, synchronize NPU to force tiling/kernel execution")
    torch_npu.npu.synchronize()
    _log(f"NPU execution finished: y shape={tuple(y_npu.shape)}, y_scale shape={tuple(y_scale_npu.shape)}")

    _log("calculate golden output")
    y_ref, y_scale_ref = grouped_matmul_activation_quant_golden(
        data["x_golden"],
        data["group_list_golden"],
        [data["weight_golden"]],
        [data["weight_scale_golden"]],
        bias=None,
        x_scale=data["x_scale_golden"],
        activation_type="gelu_tanh",
        dequant_dtype=0,
        transpose_weight=False,
        group_list_type=0,
        tuning_config=None,
        quant_mode="MX",
        y_dtype=GE_DTYPE_FLOAT8_E4M3FN,
        round_mode="rint",
        scale_alg=0,
        dst_type_max=0.0,
        output_dtypes=["float8_e4m3fn", "float8_e8m0"],
    )

    y_actual = _to_numpy_float32(y_npu)
    y_scale_actual = _to_numpy_float32(y_scale_npu)
    y_expected = np.asarray(y_ref).astype(np.float32)
    y_scale_expected = np.asarray(y_scale_ref).astype(np.float32)

    _log(f"compare y with golden: atol={args.atol}, rtol={args.rtol}")
    np.testing.assert_allclose(y_actual, y_expected, atol=args.atol, rtol=args.rtol)
    _log(f"compare y_scale with golden: atol={args.atol}, rtol={args.rtol}")
    np.testing.assert_allclose(y_scale_actual, y_scale_expected, atol=args.atol, rtol=args.rtol)

    _log("GroupedMatmulActivationQuant torch eager demo passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()
    try:
        _run(args)
    except Exception as err:
        _log(f"FAILED: {type(err).__name__}: {err}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
