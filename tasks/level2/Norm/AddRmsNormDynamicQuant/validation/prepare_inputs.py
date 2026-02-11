import torch
from framework.utils import check_precision


def get_inputs(param, device=None):
    """x1, x2: same dtype and shape (2~8 dims). Constraint aligned with ATK generate_add_rms_norm_dynamic_quant."""
    shape = eval(param.get("input_shape", "[2, 4]"))
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    x1 = (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)
    x2 = (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)
    return (x1, x2)


def get_init_inputs(param, device=None):
    """gamma.shape = [x1.shape[-1]]; smooth1/smooth2 when use_smooth: shape = gamma.shape, dtype = gamma.dtype."""
    input_shape = eval(param.get("input_shape", "[2, 4]"))
    normalized_shape = eval(param.get("normalized_shape", "[4]"))
    # Constraint: gamma last dim = x last dim (ATK: gamma.shape[0] = shape[-1])
    if isinstance(normalized_shape, list) and len(normalized_shape) >= 1 and len(input_shape) >= 1:
        normalized_shape = [input_shape[-1]]
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    gamma = torch.rand(normalized_shape, device=device, dtype=dtype)
    epsilon = float(param.get("epsilon", 1e-6))
    use_smooth = int(param.get("use_smooth", 0)) != 0
    if use_smooth:
        smooth_scale1 = torch.rand(normalized_shape, device=device, dtype=dtype)
        smooth_scale2 = torch.rand(normalized_shape, device=device, dtype=dtype)
        return [gamma, epsilon, smooth_scale1, smooth_scale2]
    return [gamma, epsilon, None, None]


def custom_check_precision(param, outputs, outputs_new):
    # When use_smooth=0, kernel returns y2/scale2 with shape [1]; expand to reference shape for comparison.
    use_smooth = int(param.get("use_smooth", 0)) != 0
    outputs_new_list = list(outputs_new)
    if not use_smooth:
        if outputs_new_list[1].numel() == 1 and outputs_new_list[1].shape != outputs[1].shape:
            outputs_new_list[1] = outputs_new_list[1].expand_as(outputs[1]).clone()
        if outputs_new_list[4].numel() == 1 and outputs_new_list[4].shape != outputs[4].shape:
            outputs_new_list[4] = outputs_new_list[4].expand_as(outputs[4]).clone()

    is_pass = 1
    atol = 1e-2
    rtol = 1e-1
    all_abs_diff = []
    all_rel_diff = []
    for idx, (out, out_new) in enumerate(zip(outputs, outputs_new_list)):
        out_f = out.float() if out.dtype != torch.float32 else out
        out_new_f = out_new.float() if out_new.dtype != torch.float32 else out_new
        if out.dtype == torch.int8:
            if (out != out_new).sum() > out.numel() * 0.1:
                is_pass = 0
            all_abs_diff.append(torch.abs(out_f - out_new_f))
            deno = torch.maximum(torch.abs(out_f), torch.abs(out_new_f)) + 1e-7
            all_rel_diff.append(torch.abs(out_f - out_new_f) / deno)
        else:
            # Flatten so [2] vs [2,1] compare element-wise
            out_flat = out_f.flatten()
            out_new_flat = out_new_f.flatten()
            abs_d = torch.abs(out_flat - out_new_flat)
            deno = torch.maximum(torch.abs(out_flat), torch.abs(out_new_flat)) + 1e-7
            rel_d = abs_d / deno
            all_abs_diff.append(abs_d)
            all_rel_diff.append(rel_d)
            if not torch.allclose(out_flat, out_new_flat, atol=atol, rtol=rtol):
                is_pass = 0
    abs_diff = torch.cat([d.flatten() for d in all_abs_diff]) if all_abs_diff else torch.tensor([0.0])
    rel_diff = torch.cat([d.flatten() for d in all_rel_diff]) if all_rel_diff else torch.tensor([0.0])
    return is_pass, abs_diff, rel_diff
