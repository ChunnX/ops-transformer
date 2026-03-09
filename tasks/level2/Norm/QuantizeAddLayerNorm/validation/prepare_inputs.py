import torch
from typing import Optional


def get_inputs(param, device=None):
    shape = eval(param.get("input_shape", "[2, 256]"))
    dtype_str = param.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)

    x1 = torch.rand(shape, device=device, dtype=dtype)
    x2 = torch.rand(shape, device=device, dtype=dtype)
    return (x1, x2)


def get_init_inputs(param, device=None):
    shape = eval(param.get("normalized_shape", "[256]"))
    dtype_str = param.get("dtype", "bfloat16")
    dtype = getattr(torch, dtype_str)
    epsilon = float(param.get("epsilon", 1e-5))
    axis = int(param.get("axis", -1))
    dtype_attr = int(param.get("dtype_attr", 0))
    additional_output = bool(param.get("additional_output", True))

    gamma = torch.rand(shape, device=device, dtype=dtype)
    beta = torch.rand(shape, device=device, dtype=dtype)
    bias = torch.rand(shape, device=device, dtype=dtype)
    # 算子定义：FP16 主输入时 scales/zero_points 需为 DT_FLOAT；BF16/FP32 时与主输入一致
    scale_zp_dtype = torch.float32 if dtype == torch.float16 else dtype
    scale_mode = str(param.get("scale_mode", "per_channel")).strip()
    if scale_mode == "per_tensor":
        axis = 65535  # AXIS_VALUE_FOR_PER_TENSOR，使 tiling 走 Normal(1001/1101)，否则 axis=-1 走 Single Row 导致 234 失败
        val = float(torch.rand(1, device=device).item() * 0.01 + 0.01)
        scales = torch.full(shape, val, device=device, dtype=scale_zp_dtype)
        zero_points = torch.zeros(shape, device=device, dtype=scale_zp_dtype)
    else:
        scales = torch.rand(shape, device=device, dtype=scale_zp_dtype) * 0.01 + 0.01
        zero_points = torch.zeros(shape, device=device, dtype=scale_zp_dtype)

    return [gamma, beta, bias, scales, zero_points, dtype_attr, axis, epsilon, additional_output]


def custom_check_precision(param, outputs, outputs_new):
    # y: int8，整型比较 atol=1；x: round(sum) 的 bf16/fp16，允许表示误差 abs<=1
    is_pass = 1
    all_abs_diff, all_rel_diff = [], []

    out_y = outputs[0].to(torch.int32)
    out_y_new = outputs_new[0].to(torch.int32)
    abs_diff = torch.abs(out_y - out_y_new)
    all_abs_diff.append(abs_diff.view(-1))
    deno = torch.maximum(torch.abs(out_y), torch.abs(out_y_new)).float() + 1e-7
    rel = (abs_diff.float() / deno).clamp(max=1.0)
    all_rel_diff.append(rel.view(-1))
    if torch.any(abs_diff > 1):
        is_pass = 0

    if outputs[1].numel() > 0 and outputs_new[1].numel() > 0:
        out_x, out_x_new = outputs[1], outputs_new[1]
        abs_x = torch.abs(out_x.float() - out_x_new.float())
        deno_x = torch.abs(out_x.float()) + 1.0
        rel_x = (abs_x / deno_x).clamp(max=1.0)
        all_abs_diff.append(abs_x.view(-1))
        all_rel_diff.append(rel_x.view(-1))
        if torch.any(abs_x > 1.0):
            is_pass = 0

    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff
