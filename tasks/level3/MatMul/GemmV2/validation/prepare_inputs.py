import torch


DTYPE_MAP = {"fp16": torch.float16, "bf16": torch.bfloat16}


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
    dtype = DTYPE_MAP.get(str(row.get("dtype", "fp16")), torch.float16)
    transpose_a = _parse_bool(row.get("transpose_a", False))
    transpose_b = _parse_bool(row.get("transpose_b", False))

    scale = 0.2
    a_shape = (k, m) if transpose_a else (m, k)
    b_shape = (n, k) if transpose_b else (k, n)
    a = (torch.randn(a_shape, dtype=torch.float32, device=device) * scale).to(dtype)
    b = (torch.randn(b_shape, dtype=torch.float32, device=device) * scale).to(dtype)

    alpha = torch.ones((m,), dtype=dtype, device=device)
    beta = torch.zeros((m,), dtype=dtype, device=device)
    c = torch.zeros((m, n), dtype=torch.float32, device=device)
    return a, b, alpha, beta, c, transpose_a, transpose_b


def custom_check_precision(row, outputs, outputs_new):
    abs_diff = torch.abs(outputs.float() - outputs_new.float()).reshape(-1)
    rel_diff = abs_diff / (torch.abs(outputs.float()).reshape(-1) + 1e-7)
    dtype = str(row.get("dtype", "fp16"))
    rtol = 0.03 if dtype == "bf16" else 0.015
    atol = 0.03 if dtype == "bf16" else 0.015
    is_pass = int(not torch.any((abs_diff > atol) & (rel_diff > rtol)))
    return is_pass, abs_diff, rel_diff
