import torch
import numpy as np


DTYPE_MAP = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}


def _parse_bool(val):
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return str(val).lower() in ("true", "1")


def _parse_list(val, default_val):
    if val is None or val == "" or (isinstance(val, float) and np.isnan(val)):
        return default_val
    if isinstance(val, str):
        return [int(x) for x in val.split(",")]
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    return default_val


def get_init_inputs(row, device="cpu"):
    stride = _parse_list(row.get("stride"), [1, 1, 1])
    padding = _parse_list(row.get("padding"), [0, 0, 0])
    dilation = _parse_list(row.get("dilation"), [1, 1, 1])
    groups = int(row.get("groups", 1))
    return (stride, padding, dilation, groups)


def get_inputs(row, device="cpu"):
    batch = int(row["batch"])
    cin = int(row["cin"])
    cout = int(row["cout"])
    di = int(row["di"])
    hi = int(row["hi"])
    wi = int(row["wi"])
    dk = int(row.get("dk", 3))
    hk = int(row.get("hk", 3))
    wk = int(row.get("wk", 3))
    stride = _parse_list(row.get("stride"), [1, 1, 1])
    padding = _parse_list(row.get("padding"), [0, 0, 0])
    dilation = _parse_list(row.get("dilation"), [1, 1, 1])
    groups = int(row.get("groups", 1))
    dtype_str = str(row.get("dtype", "fp16"))
    dtype = DTYPE_MAP.get(dtype_str, torch.float16)

    torch.manual_seed(42)
    x = torch.randn(batch, di, hi, wi, cin, dtype=dtype, device=device)
    
    filter_size = torch.tensor([dk, hk, wk, cin, cout], dtype=torch.int32, device=device)
    
    torch.manual_seed(43)
    filter_tensor = torch.randn(cout, cin // groups, dk, hk, wk, dtype=dtype, device=device)
    
    dout = (di + 2 * padding[0] - dilation[0] * (dk - 1) - 1) // stride[0] + 1
    ho = (hi + 2 * padding[1] - dilation[1] * (hk - 1) - 1) // stride[1] + 1
    wo = (wi + 2 * padding[2] - dilation[2] * (wk - 1) - 1) // stride[2] + 1
    
    torch.manual_seed(44)
    out_backprop = torch.randn(batch, dout, ho, wo, cout, dtype=dtype, device=device)

    return x, filter_size, out_backprop, filter_tensor


def custom_check_precision(row, ref_output, test_output):
    import torch
    
    outputs = [ref_output] if not isinstance(ref_output, (list, tuple)) else list(ref_output)
    outputs_new = [test_output] if not isinstance(test_output, (list, tuple)) else list(test_output)

    dtype_str = str(row.get("dtype", "fp16"))
    if dtype_str == "fp16":
        max_abs_error = 0.15
        max_rel_error = 0.15
    elif dtype_str == "bf16":
        max_abs_error = 0.15
        max_rel_error = 0.15
    else:
        max_abs_error = 0.05
        max_rel_error = 0.05

    all_abs, all_rel = [], []
    accurate = True

    for out, out_new in zip(outputs, outputs_new):
        if out_new.dtype == torch.bool:
            out = out.to(torch.int)
            out_new = out_new.to(torch.int)

        abs_diff = torch.abs(out - out_new)
        ref_abs = torch.abs(out)
        rel_diff = abs_diff / (ref_abs + 1e-5)

        all_abs.append(abs_diff.reshape(-1))
        all_rel.append(rel_diff.reshape(-1))

        if (torch.any(~torch.isfinite(out)) or
            torch.any(~torch.isfinite(out_new)) or
            torch.any(~torch.isfinite(abs_diff)) or
            torch.any(~torch.isfinite(rel_diff))):
            accurate = False

        abs_error_exceeded = abs_diff > max_abs_error
        rel_error_exceeded = rel_diff > max_rel_error
        
        if (abs_error_exceeded & rel_error_exceeded).any():
            accurate = False

    all_abs_cat = torch.cat(all_abs)
    all_rel_cat = torch.cat(all_rel)

    return (1 if accurate else 0), all_abs_cat, all_rel_cat
