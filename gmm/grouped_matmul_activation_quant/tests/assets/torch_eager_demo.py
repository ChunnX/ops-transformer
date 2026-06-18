#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

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

try:
    from golden import grouped_matmul_activation_quant_golden  # noqa: E402
except ModuleNotFoundError:
    grouped_matmul_activation_quant_golden = None


GE_DTYPE_FLOAT8_E4M3FN = 36
E8M0_ONE_BYTE = 127
ACL_FORMAT_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29


def log(msg):
    print(f"[GMMAQ demo] {msg}", flush=True)


def ceil_div(a, b):
    return (a + b - 1) // b


def get_format(tensor):
    try:
        return torch_npu.get_npu_format(tensor)
    except Exception as err:
        return f"get format failed: {err}"


def pack_weight_to_nz(weight_kn):
    # non-transposed WeightNz storage shape:
    # (E, ceil(N / 32), ceil(K / 16), 16, 32)
    e, k, n = weight_kn.shape
    nz = np.zeros((e, ceil_div(n, 32), ceil_div(k, 16), 16, 32), dtype=np.float32)
    for group_idx in range(e):
        for k_idx in range(k):
            for n_idx in range(n):
                nz[group_idx, n_idx // 32, k_idx // 16, k_idx % 16, n_idx % 32] = \
                    weight_kn[group_idx, k_idx, n_idx]
    return nz


def make_inputs(seed=20260617):
    e, m, k, n = 1, 64, 128, 128
    log(f"prepare inputs: E={e}, M={m}, K={k}, N={n}, seed={seed}")
    rng = np.random.default_rng(seed)

    x_f32 = rng.uniform(-0.5, 0.5, size=(m, k)).astype(np.float32)
    weight_f32 = rng.uniform(-0.25, 0.25, size=(e, k, n)).astype(np.float32)

    x_cpu = torch.from_numpy(x_f32).to(torch.float8_e4m3fn)
    weight_cpu = torch.from_numpy(weight_f32).to(torch.float8_e4m3fn)

    x_golden = x_cpu.to(torch.float32).numpy()
    weight_golden = pack_weight_to_nz(weight_cpu.to(torch.float32).numpy())

    x_scale_golden = np.ones((m, ceil_div(k, 64), 2), dtype=np.float32)
    weight_scale_golden = np.ones((e, ceil_div(k, 64), n, 2), dtype=np.float32)
    group_list_golden = np.array([m], dtype=np.int64)

    # x: must be ND
    x_npu = x_cpu.npu()
    x_npu = torch_npu.npu_format_cast(x_npu, ACL_FORMAT_ND)

    # weight: must be FRACTAL_NZ
    weight_npu = torch.from_numpy(weight_golden).to(torch.float8_e4m3fn).npu()
    log(f"format cast weight to FRACTAL_NZ: before shape={tuple(weight_npu.shape)}, dim={weight_npu.dim()}")
    weight_npu = torch_npu.npu_format_cast(
        weight_npu,
        ACL_FORMAT_FRACTAL_NZ,
        customize_dtype=torch.float8_e4m3fn,
    )
    log(f"format cast weight done: after shape={tuple(weight_npu.shape)}, dim={weight_npu.dim()}")
    if weight_npu.dim() != 5:
        raise RuntimeError(f"weight must keep 5D WeightNz storage, but got {weight_npu.dim()}D.")

    # scale and group_list: must be ND
    weight_scale_npu = torch.full(
        (e, ceil_div(k, 64), n, 2),
        E8M0_ONE_BYTE,
        dtype=torch.uint8,
    ).npu()
    weight_scale_npu = torch_npu.npu_format_cast(weight_scale_npu, ACL_FORMAT_ND)

    x_scale_npu = torch.full(
        (m, ceil_div(k, 64), 2),
        E8M0_ONE_BYTE,
        dtype=torch.uint8,
    ).npu()
    x_scale_npu = torch_npu.npu_format_cast(x_scale_npu, ACL_FORMAT_ND)

    group_list_npu = torch.from_numpy(group_list_golden).npu()
    group_list_npu = torch_npu.npu_format_cast(group_list_npu, ACL_FORMAT_ND)

    log(f"x format={get_format(x_npu)}, shape={tuple(x_npu.shape)}")
    log(f"weight format={get_format(weight_npu)}, shape={tuple(weight_npu.shape)}")
    log(f"weight_scale format={get_format(weight_scale_npu)}, shape={tuple(weight_scale_npu.shape)}")
    log(f"x_scale format={get_format(x_scale_npu)}, shape={tuple(x_scale_npu.shape)}")
    log(f"group_list format={get_format(group_list_npu)}, shape={tuple(group_list_npu.shape)}")

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


def to_numpy_float32(tensor):
    return tensor.detach().cpu().to(torch.float32).numpy()


def run(args):
    log("start torch eager demo")
    log(f"python={sys.version.split()[0]}, torch={torch.__version__}")

    data = make_inputs()

    log("call npu_grouped_matmul_activation_quant")
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

    log("torch op returned, synchronize NPU")
    torch_npu.npu.synchronize()
    log(f"NPU execution finished: y shape={tuple(y_npu.shape)}, y_scale shape={tuple(y_scale_npu.shape)}")
    log(f"y format={get_format(y_npu)}")
    log(f"y_scale format={get_format(y_scale_npu)}")

    y_actual = to_numpy_float32(y_npu)
    y_scale_actual = to_numpy_float32(y_scale_npu)

    log(f"first 16 y values: {y_actual.reshape(-1)[:16]}")
    log(f"first 16 y_scale values: {y_scale_actual.reshape(-1)[:16]}")

    if grouped_matmul_activation_quant_golden is None:
        log("golden.py not found, skip golden compare.")
        log("GroupedMatmulActivationQuant torch eager smoke demo passed.")
        return

    log("calculate golden output")
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

    y_expected = np.asarray(y_ref).astype(np.float32)
    y_scale_expected = np.asarray(y_scale_ref).astype(np.float32)

    log(f"compare y with golden: atol={args.atol}, rtol={args.rtol}")
    np.testing.assert_allclose(y_actual, y_expected, atol=args.atol, rtol=args.rtol)

    log(f"compare y_scale with golden: atol={args.atol}, rtol={args.rtol}")
    np.testing.assert_allclose(y_scale_actual, y_scale_expected, atol=args.atol, rtol=args.rtol)

    log("GroupedMatmulActivationQuant torch eager demo passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    args = parser.parse_args()

    try:
        run(args)
    except Exception as err:
        log(f"FAILED: {type(err).__name__}: {err}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()