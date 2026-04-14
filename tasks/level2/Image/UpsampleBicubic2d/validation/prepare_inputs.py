import torch
import numpy as np


def get_inputs(param, device=None):
    """Generate input tensors based on test_cases.csv parameters."""
    if device is None:
        device = torch.device("cpu")

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(param["dtype"], torch.float32)

    n = int(param["n"])
    c = int(param["c"])
    h_in = int(param["h_in"])
    w_in = int(param["w_in"])
    h_out = int(param["h_out"])
    w_out = int(param["w_out"])
    align_corners = param["align_corners"] == "True"

    input_tensor = torch.randn(n, c, h_in, w_in, dtype=dtype, device=device)

    output_size = [h_out, w_out]

    return (input_tensor, output_size, align_corners)


def get_init_inputs(param, device=None):
    """Return empty list."""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """Custom precision check. Returns (is_pass, all_abs_diff, all_rel_diff)."""
    dtype = param["dtype"]

    if dtype == "float16":
        RTOL = 1.0 / 512
    elif dtype == "bfloat16":
        RTOL = 0.02
    elif dtype == "float32":
        RTOL = 1.0 / 2048 + 1.0 / 16384
    else:
        RTOL = 0

    is_pass = True
    abs_diffs = []
    rel_diffs = []

    for out_ref, out_new in zip(outputs, outputs_new):
        if out_ref is None:
            continue
        out_ref_f = out_ref.float()
        out_new_f = out_new.float()

        abs_diff = (out_ref_f - out_new_f).abs()
        abs_diffs.append(abs_diff.flatten())

        nonzero_mask = out_ref_f.abs() > 1e-6
        if nonzero_mask.any():
            rel_diff = abs_diff[nonzero_mask] / out_ref_f[nonzero_mask].abs()
            rel_diffs.append(rel_diff.flatten())
        else:
            rel_diffs.append(torch.tensor([0.0]))

        max_abs = abs_diff.max().item()
        max_rel = rel_diffs[-1].max().item() if rel_diffs[-1].numel() > 0 else 0.0

        if max_abs > RTOL and max_rel > RTOL:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
