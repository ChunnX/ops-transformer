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
        f"numel={tensor.numel()}, storage_nbytes={tensor.untyped_storage().nbytes()}"
    )


def make_fp8_tensor(shape):
    data = torch.randint(-8, 8, shape, dtype=torch.int8).to(torch.float8_e4m3fn)
    return data.npu()


def try_format_cast(name, tensor, target_format, customize_dtype=None):
    print_tensor(f"{name} before", tensor)
    try:
        if customize_dtype is None:
            out = torch_npu.npu_format_cast(tensor, target_format)
        else:
            out = torch_npu.npu_format_cast(tensor, target_format, customize_dtype=customize_dtype)
        print_tensor(f"{name} after", out)
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

    log("case 1: 2D logical matrix (K, N) -> FRACTAL_NZ")
    weight_2d = make_fp8_tensor((k, n))
    try_format_cast("weight_2d_to_nz", weight_2d, ACL_FORMAT_FRACTAL_NZ, torch.float8_e4m3fn)

    log("case 2: 3D grouped logical weight (E, K, N) -> FRACTAL_NZ")
    weight_3d = make_fp8_tensor((e, k, n))
    try_format_cast("weight_3d_to_nz", weight_3d, ACL_FORMAT_FRACTAL_NZ, torch.float8_e4m3fn)

    log("case 3: 5D WeightNz view shape -> FRACTAL_NZ")
    weight_5d_shape = (e, math.ceil(n / 32), math.ceil(k / 16), 16, 32)
    weight_5d = make_fp8_tensor(weight_5d_shape)
    try_format_cast("weight_5d_to_nz", weight_5d, ACL_FORMAT_FRACTAL_NZ, torch.float8_e4m3fn)

    log("case 4: 3D scale-like tensor -> ND")
    scale_3d = torch.ones((64, math.ceil(k / 64), 2), dtype=torch.uint8).npu()
    try_format_cast("scale_3d_to_nd", scale_3d, ACL_FORMAT_ND)


if __name__ == "__main__":
    main()
