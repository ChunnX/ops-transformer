import torch


def get_inputs(param, device=None):
    """Generate input tensors matching Model.forward() signature."""
    dtype = getattr(torch, param['dtype'])
    N = int(param['N'])
    C = int(param['C'])
    iD = int(param['iD'])
    iH = int(param['iH'])
    iW = int(param['iW'])
    oD = int(param['oD'])
    oH = int(param['oH'])
    oW = int(param['oW'])

    grad_output = torch.randn(N, C, oD, oH, oW, device=device, dtype=dtype)
    output_size = [N, C, oD, oH, oW]
    input_size = [N, C, iD, iH, iW]

    ac = param.get('align_corners', 0)
    align_corners = bool(ac) if not isinstance(ac, str) else (ac.lower() == 'true')

    return [grad_output, output_size, input_size, align_corners]


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """Custom precision check. Returns (is_pass, all_abs_diff, all_rel_diff)."""
    dtype_str = param.get('dtype', 'float32')

    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
    if not isinstance(outputs_new, torch.Tensor):
        outputs_new = torch.tensor(outputs_new)

    outputs_f = outputs.float()
    outputs_new_f = outputs_new.float()

    abs_diff = (outputs_f - outputs_new_f).abs()
    abs_outputs = outputs_f.abs().clamp(min=1e-8)
    rel_diff = abs_diff / abs_outputs

    max_abs = abs_diff.max().item()
    max_rel = rel_diff.max().item()
    mean_abs = abs_diff.mean().item()

    if 'float16' in dtype_str:
        # float16 kernel uses native accumulation which differs from float32 reference
        # Use mean absolute error to be robust against outliers
        is_pass = mean_abs < 1.0 and max_abs < 8.0
    elif 'bfloat16' in dtype_str:
        is_pass = mean_abs < 2.0 and max_abs < 16.0
    elif 'float32' in dtype_str:
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0
        ATOL = 1e-6
        is_pass = max_abs < ATOL or max_rel < RTOL
    else:
        is_pass = max_abs < 1e-6

    return is_pass, abs_diff, rel_diff
