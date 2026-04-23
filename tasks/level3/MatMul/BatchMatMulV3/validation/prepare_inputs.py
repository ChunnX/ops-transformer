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
    adj_x1 = _parse_bool(row.get("adj_x1", False))
    adj_x2 = _parse_bool(row.get("adj_x2", False))
    has_bias = _parse_bool(row.get("has_bias", False))

    x1 = torch.randn(batch, m, k, dtype=dtype, device=device)
    x2_raw = torch.randn(batch, k, n, dtype=dtype, device=device)
    if adj_x2:
        x2 = x2_raw.transpose(-2, -1).contiguous()
    else:
        x2 = x2_raw

    bias = None
    if has_bias:
        bias = torch.randn(batch, 1, n, dtype=dtype, device=device)

    return x1, x2, bias, adj_x1, adj_x2
