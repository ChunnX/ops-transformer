import torch


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
    m = int(row["m"])
    k = int(row["k"])
    n = int(row["n"])
    dtype_str = str(row.get("dtype", "fp16"))
    dtype = DTYPE_MAP.get(dtype_str, torch.float16)
    transpose_x1 = _parse_bool(row.get("transpose_x1", False))
    transpose_x2 = _parse_bool(row.get("transpose_x2", False))
    has_bias = _parse_bool(row.get("has_bias", False))

    scale = 0.2
    x1_shape = (k, m) if transpose_x1 else (m, k)
    x2_shape = (n, k) if transpose_x2 else (k, n)
    x1 = (torch.randn(x1_shape, dtype=torch.float32, device=device) * scale).to(dtype)
    x2 = (torch.randn(x2_shape, dtype=torch.float32, device=device) * scale).to(dtype)

    bias = None
    if has_bias:
        bias = (torch.randn(n, dtype=torch.float32, device=device) * scale).to(dtype)

    return x1, x2, bias, transpose_x1, transpose_x2
