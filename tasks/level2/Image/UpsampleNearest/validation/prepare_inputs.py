import torch


def get_inputs(param, device=None):
    """Generate input tensors matching Model.forward() signature."""
    dtype = getattr(torch, param['dtype'])
    N = int(param['N'])
    C = int(param['C'])
    H = int(param['H'])
    W = int(param['W'])
    output_h = int(param['output_h'])
    output_w = int(param['output_w'])

    em = param.get('exact_mode', 1)
    exact_mode = bool(em) if not isinstance(em, str) else (em.lower() == 'true')

    x = torch.randn(N, C, H, W, device=device, dtype=dtype)
    output_size = [output_h, output_w]

    return [x, output_size, exact_mode]


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """Custom precision check. Returns (is_pass, all_abs_diff, all_rel_diff)."""
    dtype_str = param.get('dtype', 'float32')

    if 'float16' in dtype_str:
        RTOL = 1.0 / 512.0
        ATOL = 1e-3
    elif 'bfloat16' in dtype_str:
        RTOL = 1.0 / 256.0 + 1.0 / 16384.0
        ATOL = 1e-2
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
