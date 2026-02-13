import torch
from framework.utils import check_precision


def get_inputs(param, device=None):
    shape = eval(param.get("input_shape", "[2, 4]"))
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    x1 = (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)
    x2 = (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)
    # Pass (x1, x2) only; smooth_scale1/2 are optional and passed as None from module
    return (x1, x2)


def get_init_inputs(param, device=None):
    shape = eval(param.get("normalized_shape", "[4]"))
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    gamma = torch.rand(shape, device=device, dtype=dtype)
    epsilon = float(param.get("epsilon", 1e-6))
    return [gamma, epsilon]


def custom_check_precision(param, outputs, outputs_new):
    # When smooth_scale1/2 are not used, kernel does not write y2 and scale2 (they are undefined).
    # Reference returns y2=0, scale2=1. Skip comparing those two so we only validate y1,y3,y4,x,scale1.
    skip_indices = {1, 6}  # y2, scale2
    is_pass = 1
    atol = 1e-2
    rtol = 1e-1
    all_abs_diff = []
    all_rel_diff = []
    for idx, (out, out_new) in enumerate(zip(outputs, outputs_new)):
        if idx in skip_indices:
            continue
        out_f = out.float() if out.dtype != torch.float32 else out
        out_new_f = out_new.float() if out_new.dtype != torch.float32 else out_new
        if out.dtype == torch.int8:
            if (out != out_new).sum() > out.numel() * 0.1:
                is_pass = 0
            all_abs_diff.append(torch.abs(out_f - out_new_f))
            deno = torch.maximum(torch.abs(out_f), torch.abs(out_new_f)) + 1e-7
            all_rel_diff.append(torch.abs(out_f - out_new_f) / deno)
        else:
            abs_d = torch.abs(out_f - out_new_f)
            deno = torch.maximum(torch.abs(out_f), torch.abs(out_new_f)) + 1e-7
            rel_d = abs_d / deno
            all_abs_diff.append(abs_d)
            all_rel_diff.append(rel_d)
            if not torch.allclose(out_f, out_new_f, atol=atol, rtol=rtol):
                is_pass = 0
    abs_diff = torch.cat([d.flatten() for d in all_abs_diff]) if all_abs_diff else torch.tensor([0.0])
    rel_diff = torch.cat([d.flatten() for d in all_rel_diff]) if all_rel_diff else torch.tensor([0.0])
    return is_pass, abs_diff, rel_diff
