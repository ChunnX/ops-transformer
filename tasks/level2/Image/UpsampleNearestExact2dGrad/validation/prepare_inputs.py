import torch


def get_inputs(param, device=None):
    """Generate input tensors matching Model.forward() signature."""
    dtype = getattr(torch, param['dtype'])
    N = int(param['N'])
    C = int(param['C'])
    iH = int(param['iH'])
    iW = int(param['iW'])
    oH = int(param['oH'])
    oW = int(param['oW'])

    grad_output = torch.randn(N, C, oH, oW, device=device, dtype=dtype)
    output_size = [N, C, oH, oW]
    input_size = [N, C, iH, iW]

    return [grad_output, output_size, input_size]


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """Custom precision check. Returns (is_pass, all_abs_diff, all_rel_diff)."""
    dtype_str = param.get('dtype', 'float32')

    if 'bfloat16' in dtype_str:
        RTOL = 0.1
        ATOL = 0.05
    elif 'float16' in dtype_str:
        RTOL = 0.05
        ATOL = 0.02
    elif 'float32' in dtype_str:
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0
        ATOL = 1e-6
    else:
        RTOL = 0.0
        ATOL = 0.0

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
    is_pass = max_abs < ATOL or max_rel < RTOL

    return is_pass, abs_diff, rel_diff
