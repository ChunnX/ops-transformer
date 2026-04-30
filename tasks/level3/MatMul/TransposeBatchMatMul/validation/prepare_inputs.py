import torch
import numpy as np


DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).lower() in ("true", "1")


def get_init_inputs(row, device="cpu"):
    return ()


def get_inputs(row, device="cpu"):
    batch = int(row["batch"])
    m = int(row["m"])
    k = int(row["k"])
    n = int(row["n"])
    dtype_str = str(row.get("dtype", "fp16"))
    dtype = DTYPE_MAP.get(dtype_str, torch.float16)
    has_bias = _parse_bool(row.get("has_bias", False))

    perm_x1_str = str(row.get("perm_x1", "0,1,2"))
    perm_x2_str = str(row.get("perm_x2", "0,1,2"))
    perm_y_str = str(row.get("perm_y", "1,0,2"))

    perm_x1 = [int(x) for x in perm_x1_str.split(",")]
    perm_x2 = [int(x) for x in perm_x2_str.split(",")]
    perm_y = [int(x) for x in perm_y_str.split(",")]

    # Build x1 shape according to perm_x1 inverse
    # perm_x1 tells us how to interpret x1 dims
    # e.g. perm_x1=[0,1,2] means x1[i,j,k] -> (batch, m, k)
    # e.g. perm_x1=[1,0,2] means x1[i,j,k] -> (m, batch, k)
    x1_shape = [0, 0, 0]
    x1_shape[perm_x1[0]] = batch
    x1_shape[perm_x1[1]] = m
    x1_shape[perm_x1[2]] = k

    x2_shape = [0, 0, 0]
    x2_shape[perm_x2[0]] = batch
    x2_shape[perm_x2[1]] = k
    x2_shape[perm_x2[2]] = n

    x1 = torch.randn(*x1_shape, dtype=dtype, device=device)
    x2 = torch.randn(*x2_shape, dtype=dtype, device=device)

    bias = None
    if has_bias:
        bias = torch.randn(batch, 1, n, dtype=dtype, device=device)

    return x1, x2, bias, None, perm_x1, perm_x2, perm_y, 1
