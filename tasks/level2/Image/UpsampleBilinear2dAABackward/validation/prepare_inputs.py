import torch
import math


def get_inputs(param, device=None):
    """Generate input tensors based on test_cases.csv row."""
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(param['dtype'], torch.float32)

    N = int(param['N'])
    C = int(param['C'])
    H_in = int(param['H_in'])
    W_in = int(param['W_in'])
    H_out = int(param['H_out'])
    W_out = int(param['W_out'])
    ac = param['align_corners']
    align_corners = bool(ac) if not isinstance(ac, str) else (ac.lower() == 'true')

    grad_output = torch.randn(N, C, H_out, W_out, dtype=dtype, device=device)
    output_size = [H_out, W_out]
    input_size = [N, C, H_in, W_in]

    return (grad_output, output_size, input_size, align_corners)


def get_init_inputs(param, device=None):
    """Return empty list."""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """Custom precision check for upsample backward."""
    dtype = param['dtype']
    if dtype == 'float32':
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0
    elif dtype == 'float16':
        RTOL = 1.0 / 512.0
    elif dtype == 'bfloat16':
        RTOL = 1.0 / 256.0 + 1.0 / 16384.0
    else:
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0

    is_pass = True
    abs_diffs = []
    rel_diffs = []

    for out, out_new in zip(outputs, outputs_new):
        if out is None or out_new is None:
            continue
        out_f = out.float()
        out_new_f = out_new.float()
        abs_diff = (out_f - out_new_f).abs()
        abs_diffs.append(abs_diff.flatten())

        denom = torch.max(out_f.abs(), out_new_f.abs())
        denom = torch.clamp(denom, min=1e-8)
        rel_diff = abs_diff / denom
        rel_diffs.append(rel_diff.flatten())

        if not torch.all(abs_diff <= RTOL * denom + 1e-6):
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
