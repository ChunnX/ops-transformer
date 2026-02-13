import torch


def get_inputs(param, device=None):
    """
    Generate input tensors for MaskedSoftmaxWithRelPosBias.
    x: 4D (B*W, N, S1, S2) or 5D (B, W, N, S1, S2)
    atten_mask (optional): 3D (W, S1, S2), 4D (W, 1, S1, S2) or 5D (1, W, 1, S1, S2)
    relative_pos_bias: 3D (N, S1, S2), 4D (1, N, S1, S2) or 5D (1, 1, N, S1, S2)
    """
    x_shape = eval(param.get("x_shape", "[1, 2, 4, 8, 16]"))
    bias_shape = eval(param.get("bias_shape", "[1, 1, 4, 8, 16]"))
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    scale_value = float(param.get("scale_value", "1.0"))
    has_attn = int(param.get("has_attn", "0"))

    x = torch.rand(x_shape, device=device, dtype=dtype)
    bias = torch.rand(bias_shape, device=device, dtype=dtype)
    if has_attn:
        # atten_mask: same S1,S2 as x; W from x
        if len(x_shape) == 5:
            att_shape = [x_shape[1], x_shape[3], x_shape[4]]  # W, S1, S2
        else:
            att_shape = [x_shape[0] // bias_shape[0] if bias_shape[0] > 0 else 1, x_shape[2], x_shape[3]]
        atten_mask = torch.rand(att_shape, device=device, dtype=dtype)
        return (x, atten_mask, bias, scale_value)
    return (x, None, bias, scale_value)


def get_init_inputs(param, device=None):
    """No init params for this op."""
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    if dtype == torch.float16:
        RTOL_GENERAL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL_GENERAL = 1 / 256 + 1 / 16384
    else:
        RTOL_GENERAL = 1 / 2048 + 1 / 16384

    rtol = RTOL_GENERAL
    outputs = [outputs] if not isinstance(outputs, list) else outputs
    outputs_new = [outputs_new] if not isinstance(outputs_new, list) else outputs_new

    all_abs_diff, all_rel_diff = [], []
    is_pass = 1
    for out, out_new in zip(outputs, outputs_new):
        abs_diff = torch.abs(out - out_new)
        rel_diff = abs_diff / (torch.abs(out) + 1e-7)
        all_abs_diff.append(abs_diff.view(-1))
        all_rel_diff.append(rel_diff.view(-1))
        tolerance = rtol * torch.maximum(torch.tensor(1.0, device=out.device), torch.abs(out))
        error_mask = abs_diff > tolerance
        if torch.any(error_mask):
            is_pass = 0

    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff
