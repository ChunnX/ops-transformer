import torch


def get_inputs(param, device=None):
    """Generate input tensors and params matching Model.forward() signature."""
    N = int(param['N'])
    C = int(param['C'])
    H_in = int(param['H_in'])
    W_in = int(param['W_in'])
    H_out = int(param['H_out'])
    W_out = int(param['W_out'])

    ac = param['align_corners']
    align_corners = bool(ac) if not isinstance(ac, str) else (ac.lower() == 'true')

    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(param['dtype'], torch.float32)

    grad_output = torch.randn(N, C, H_out, W_out, dtype=dtype, device=device)
    output_size = [N, C, H_out, W_out]
    input_size = [N, C, H_in, W_in]

    return (grad_output, output_size, input_size, align_corners)


def get_init_inputs(param, device=None):
    """Return empty list."""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """Custom precision check for upsample_bilinear2d_grad."""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(param['dtype'], torch.float32)

    if dtype == torch.float16:
        RTOL = 1.0 / 256
    elif dtype == torch.bfloat16:
        RTOL = 1.0 / 32
    elif dtype == torch.float32:
        RTOL = 1.0 / 2048 + 1.0 / 16384
    else:
        RTOL = 0

    abs_diffs = []
    rel_diffs = []

    for golden, custom in zip(outputs, outputs_new):
        golden_f = golden.float()
        custom_f = custom.float()
        abs_diff = (golden_f - custom_f).abs()
        abs_diffs.append(abs_diff.flatten())

        max_val = torch.max(golden_f.abs(), custom_f.abs())
        max_val = torch.clamp(max_val, min=1e-8)
        rel_diff = abs_diff / max_val
        rel_diffs.append(rel_diff.flatten())

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])

    max_abs = all_abs_diff.float().max().item()
    max_rel = all_rel_diff.float().max().item()
    is_pass = (max_abs <= RTOL) or (max_rel <= RTOL)

    return is_pass, all_abs_diff, all_rel_diff
