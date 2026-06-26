#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import argparse
import ast
import csv
import importlib.util
import os
import sys
import types
import traceback

import numpy as np
import torch
import torch_npu


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", "..", "..", ".."))
TORCH_EXTENSION_DIR = os.path.join(REPO_ROOT, "torch_extension")
if TORCH_EXTENSION_DIR not in sys.path:
    sys.path.insert(0, TORCH_EXTENSION_DIR)


def install_lightweight_package_stub():
    package_name = "cann_ops_transformer"
    if package_name in sys.modules:
        return
    package_dir = os.path.join(TORCH_EXTENSION_DIR, "cann_ops_transformer")
    package = types.ModuleType(package_name)
    package.__file__ = os.path.join(package_dir, "__init__.py")
    package.__path__ = [package_dir]
    sys.modules[package_name] = package


def load_grouped_matmul_activation_quant():
    install_lightweight_package_stub()
    op_module_path = os.path.join(
        TORCH_EXTENSION_DIR,
        "cann_ops_transformer",
        "ops",
        "grouped_matmul_activation_quant.py",
    )
    spec = importlib.util.spec_from_file_location("gmmaq_torch_extension_op", op_module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load module spec from {op_module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.grouped_matmul_activation_quant


grouped_matmul_activation_quant = load_grouped_matmul_activation_quant()

try:
    from golden import grouped_matmul_activation_quant_golden  # noqa: E402
except ModuleNotFoundError:
    grouped_matmul_activation_quant_golden = None


GE_DTYPE_FLOAT8_E4M3FN = 36
GE_DTYPE_FLOAT8_E5M2 = 35
E8M0_ONE_BYTE = 127
ACL_FORMAT_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29
DEFAULT_CASE_CSV = os.path.join(REPO_ROOT, "group_matmul_activation_quant_mxfp8_aclnn_converted.csv")


def log(msg):
    print(f"[GMMAQ demo] {msg}", flush=True)


def ceil_div(a, b):
    return (a + b - 1) // b


def get_format(tensor):
    try:
        return torch_npu.get_npu_format(tensor)
    except Exception as err:
        return f"get format failed: {err}"


def dtype_name_to_torch(dtype_name):
    if dtype_name == "float8_e5m2":
        return torch.float8_e5m2
    if dtype_name == "float8_e4m3fn":
        return torch.float8_e4m3fn
    raise ValueError(f"unsupported dtype name: {dtype_name}")


def ge_dtype_to_output_name(ge_dtype):
    if ge_dtype == GE_DTYPE_FLOAT8_E5M2:
        return "float8_e5m2"
    if ge_dtype == GE_DTYPE_FLOAT8_E4M3FN:
        return "float8_e4m3fn"
    raise ValueError(f"unsupported y_dtype: {ge_dtype}")


def ge_dtype_to_torch(ge_dtype):
    return dtype_name_to_torch(ge_dtype_to_output_name(ge_dtype))


def load_case_from_csv(csv_path, case_index):
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"case csv not found: {csv_path}")
    with open(csv_path, newline="", encoding="utf-8") as csv_file:
        rows = list(csv.DictReader(csv_file))
    if not rows:
        raise ValueError(f"case csv is empty: {csv_path}")
    if case_index < 0 or case_index >= len(rows):
        raise IndexError(f"case_index {case_index} out of range, total {len(rows)}")

    row = rows[case_index]
    input_ori_shapes = ast.literal_eval(row["input_ori_shapes"])
    input_dtypes = ast.literal_eval(row["input_dtypes"])
    output_dtypes = ast.literal_eval(row["output_dtypes"])
    attrs = ast.literal_eval(row["attributes"])

    x_shape = input_ori_shapes[0]
    group_shape = input_ori_shapes[1]
    weight_shape = input_ori_shapes[2][0]
    e = int(group_shape[0])
    m = int(x_shape[0])
    k = int(x_shape[1])
    transpose_weight = bool(attrs["transpose_weight"])
    n = int(weight_shape[1] if transpose_weight else weight_shape[2])

    return {
        "case_name": row["testcase_name"],
        "e": e,
        "m": m,
        "k": k,
        "n": n,
        "x_dtype_name": input_dtypes[0],
        "y_dtype": int(attrs["y_dtype"]),
        "y_output_name": output_dtypes[0],
        "group_list": np.array(attrs.get("group_list_expect", [m]), dtype=np.int64),
        "group_list_type": int(attrs["group_list_type"]),
        "transpose_weight": transpose_weight,
        "scale_alg": int(attrs["scale_alg"]),
        "quant_mode": attrs.get("quant_mode", None),
    }


def make_inputs(case, seed=20260617):
    e, m, k, n = case["e"], case["m"], case["k"], case["n"]
    log(f"prepare inputs from {case['case_name']}: E={e}, M={m}, K={k}, N={n}, seed={seed}")
    rng = np.random.default_rng(seed)

    x_f32 = rng.uniform(-0.5, 0.5, size=(m, k)).astype(np.float32)
    if case["transpose_weight"]:
        weight_f32 = rng.uniform(-0.25, 0.25, size=(e, n, k)).astype(np.float32)
    else:
        weight_f32 = rng.uniform(-0.25, 0.25, size=(e, k, n)).astype(np.float32)

    x_cpu = torch.from_numpy(x_f32).to(dtype_name_to_torch(case["x_dtype_name"]))
    weight_cpu = torch.from_numpy(weight_f32).to(torch.float8_e4m3fn)

    x_golden = x_cpu.to(torch.float32).numpy()
    weight_golden = weight_cpu.to(torch.float32).numpy()

    x_scale_golden = np.ones((m, ceil_div(k, 64), 2), dtype=np.float32)
    if case["transpose_weight"]:
        weight_scale_golden = np.ones((e, n, ceil_div(k, 64), 2), dtype=np.float32)
    else:
        weight_scale_golden = np.ones((e, ceil_div(k, 64), n, 2), dtype=np.float32)
    group_list_golden = case["group_list"]

    # x: must be ND
    x_npu = x_cpu.npu()
    x_npu = torch_npu.npu_format_cast(x_npu, ACL_FORMAT_ND)

    # weight: must be FRACTAL_NZ
    weight_npu = weight_cpu.npu()
    log(f"format cast weight to FRACTAL_NZ: before shape={tuple(weight_npu.shape)}, dim={weight_npu.dim()}")
    weight_npu = torch_npu.npu_format_cast(
        weight_npu,
        ACL_FORMAT_FRACTAL_NZ,
        customize_dtype=torch.float8_e4m3fn,
    )
    log(f"format cast weight done: after shape={tuple(weight_npu.shape)}, dim={weight_npu.dim()}")
    if weight_npu.dim() != 3:
        raise RuntimeError(f"weight should keep 3D logical shape after npu_format_cast, but got {weight_npu.dim()}D.")

    # scale and group_list: must be ND
    weight_scale_npu = torch.full(
        tuple(weight_scale_golden.shape),
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
        "case": case,
    }


def to_numpy_float32(tensor):
    return tensor.detach().cpu().to(torch.float32).numpy()


def run(args):
    log("start torch eager demo")
    log(f"python={sys.version.split()[0]}, torch={torch.__version__}")

    case = load_case_from_csv(args.case_csv, args.case_index)
    data = make_inputs(case, args.seed)

    log("call grouped_matmul_activation_quant")
    y_npu, y_scale_npu = grouped_matmul_activation_quant(
        data["x_npu"],
        data["group_list_npu"],
        [data["weight_npu"]],
        [data["weight_scale_npu"]],
        "gelu_tanh",
        bias=None,
        x_scale=data["x_scale_npu"],
        group_list_type=case["group_list_type"],
        tuning_config=None,
        quant_mode=case["quant_mode"],
        y_dtype=ge_dtype_to_torch(case["y_dtype"]),
        round_mode="rint",
        scale_alg=case["scale_alg"],
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
        [],
        "gelu_tanh",
        x_scale=data["x_scale_golden"],
        transpose_weight=case["transpose_weight"],
        group_list_type=case["group_list_type"],
        tuning_config=None,
        quant_mode=case["quant_mode"],
        y_dtype=case["y_dtype"],
        round_mode="rint",
        scale_alg=case["scale_alg"],
        dst_type_max=0.0,
        output_dtypes=[case["y_output_name"], "float8_e8m0"],
    )

    y_expected = np.asarray(y_ref).astype(np.float32)
    y_scale_expected = np.asarray(y_scale_ref).astype(np.float32)
    log(f"first 10 y_ref values: {np.asarray(y_ref).reshape(-1)[:10]}")
    log(f"first 10 y_scale_ref values: {np.asarray(y_scale_ref).reshape(-1)[:10]}")
    log(f"first 10 y_expected values: {y_expected.reshape(-1)[:10]}")
    log(f"first 10 y_scale_expected values: {y_scale_expected.reshape(-1)[:10]}")

    log(f"compare y with golden: atol={args.atol}, rtol={args.rtol}")
    np.testing.assert_allclose(y_actual, y_expected, atol=args.atol, rtol=args.rtol)

    log(f"compare y_scale with golden: atol={args.atol}, rtol={args.rtol}")
    np.testing.assert_allclose(y_scale_actual, y_scale_expected, atol=args.atol, rtol=args.rtol)

    log("GroupedMatmulActivationQuant torch eager demo passed.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--atol", type=float, default=0.0)
    parser.add_argument("--rtol", type=float, default=0.0)
    parser.add_argument("--case-csv", default=DEFAULT_CASE_CSV)
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=20260617)
    args = parser.parse_args()

    try:
        run(args)
    except Exception as err:
        log(f"FAILED: {type(err).__name__}: {err}")
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()
