#!/usr/bin/env python3
# -*- coding: UTF-8 -*-

import math
import traceback

import torch
import torch_npu


ACL_FORMAT_ND = 2
ACL_FORMAT_FRACTAL_NZ = 29


def log(msg):
    print(f"[format_cast demo] {msg}", flush=True)


def get_storage_sizes(tensor):
    candidates = [
        (torch_npu, "get_npu_storage_sizes"),
        (getattr(torch_npu, "npu", None), "get_npu_storage_sizes"),
        (getattr(torch_npu, "_C", None), "_npu_get_npu_storage_sizes"),
        (getattr(torch_npu, "_C", None), "_npu_get_storage_sizes"),
        (getattr(torch_npu, "_C", None), "_npu_getNpuStorageSizes"),
        (getattr(torch_npu, "_C", None), "_npu_get_storage_size"),
    ]
    errors = []
    for module, func_name in candidates:
        if module is None or not hasattr(module, func_name):
            continue
        try:
            func = getattr(module, func_name)
            return f"{func(tensor)} (from {module.__name__}.{func_name})"
        except Exception as err:
            errors.append(f"{func_name}: {type(err).__name__}: {err}")
    return "unavailable" if not errors else "unavailable, tried " + "; ".join(errors)


def print_storage_related_apis():
    for module_name, module in (
        ("torch_npu", torch_npu),
        ("torch_npu.npu", getattr(torch_npu, "npu", None)),
        ("torch_npu._C", getattr(torch_npu, "_C", None)),
    ):
        if module is None:
            continue
        names = [
            name for name in dir(module)
            if "storage" in name.lower() or ("format" in name.lower() and "npu" in name.lower())
        ]
        log(f"{module_name} storage/format related APIs: {names}")


def get_format(tensor):
    try:
        return torch_npu.get_npu_format(tensor)
    except Exception as err:
        return f"get_npu_format failed: {err}"


def print_tensor(name, tensor):
    log(
        f"{name}: shape={tuple(tensor.shape)}, dim={tensor.dim()}, "
        f"stride={tuple(tensor.stride())}, dtype={tensor.dtype}, "
        f"device={tensor.device}, format={get_format(tensor)}, "
        f"numel={tensor.numel()}, storage_nbytes={tensor.untyped_storage().nbytes()}, "
        f"npu_storage_sizes={get_storage_sizes(tensor)}"
    )


def make_fp8_tensor(shape):
    data = torch.randint(-8, 8, shape, dtype=torch.int8).to(torch.float8_e4m3fn)
    return data.npu()


def try_format_cast(name, tensor, target_format, customize_dtype=None, expected_storage_shape=None):
    print_tensor(f"{name} before", tensor)
    try:
        if customize_dtype is None:
            out = torch_npu.npu_format_cast(tensor, target_format)
        else:
            out = torch_npu.npu_format_cast(tensor, target_format, customize_dtype=customize_dtype)
        print_tensor(f"{name} after", out)
        if expected_storage_shape is not None:
            log(f"{name}: expected FRACTAL_NZ storage shape = {expected_storage_shape}")
        log(f"{name}: same python shape after cast = {tuple(tensor.shape) == tuple(out.shape)}")
        return out
    except Exception as err:
        log(f"{name}: npu_format_cast failed: {type(err).__name__}: {err}")
        traceback.print_exc()
        return None


def main():
    e, k, n = 1, 128, 128
    log(f"torch={torch.__version__}")
    log(f"test shape params: E={e}, K={k}, N={n}")
    print_storage_related_apis()

    log("case 1: 2D logical matrix (K, N) -> FRACTAL_NZ")
    weight_2d = make_fp8_tensor((k, n))
    try_format_cast(
        "weight_2d_to_nz",
        weight_2d,
        ACL_FORMAT_FRACTAL_NZ,
        torch.float8_e4m3fn,
        expected_storage_shape=(math.ceil(n / 32), math.ceil(k / 16), 16, 32),
    )

    log("case 2: 3D grouped logical weight (E, K, N) -> FRACTAL_NZ")
    weight_3d = make_fp8_tensor((e, k, n))
    try_format_cast(
        "weight_3d_to_nz",
        weight_3d,
        ACL_FORMAT_FRACTAL_NZ,
        torch.float8_e4m3fn,
        expected_storage_shape=(e, math.ceil(n / 32), math.ceil(k / 16), 16, 32),
    )

    log("case 3: 5D WeightNz view shape -> FRACTAL_NZ")
    weight_5d_shape = (e, math.ceil(n / 32), math.ceil(k / 16), 16, 32)
    weight_5d = make_fp8_tensor(weight_5d_shape)
    try_format_cast(
        "weight_5d_to_nz",
        weight_5d,
        ACL_FORMAT_FRACTAL_NZ,
        torch.float8_e4m3fn,
        expected_storage_shape=weight_5d_shape,
    )

    log("case 4: 3D scale-like tensor -> ND")
    scale_3d = torch.ones((64, math.ceil(k / 64), 2), dtype=torch.uint8).npu()
    try_format_cast("scale_3d_to_nd", scale_3d, ACL_FORMAT_ND)


if __name__ == "__main__":
    main()
