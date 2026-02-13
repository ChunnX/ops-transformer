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
    additional_output = bool(param.get("additional_output", False))

    gamma = torch.rand(shape, device=device, dtype=dtype)
    beta = torch.rand(shape, device=device, dtype=dtype)
    bias = torch.rand(shape, device=device, dtype=dtype)
    scales1 = torch.rand(shape, device=device, dtype=dtype) * 0.01 + 0.01
    scales2 = torch.rand(shape, device=device, dtype=dtype) * 0.01 + 0.01
    zero_points1 = torch.zeros(shape, device=device, dtype=dtype)
    zero_points2 = torch.zeros(shape, device=device, dtype=dtype)

    return [gamma, beta, bias, scales1, scales2, zero_points1, zero_points2, dtype_attr, axis, epsilon, additional_output]


def custom_check_precision(param, outputs, outputs_new):
    # y1/y2: int8，整型比较 atol=1；x: round(sum) 的 bf16/fp16，允许表示误差 abs<=1
    is_pass = 1
    all_abs_diff, all_rel_diff = [], []
    # y1, y2: int8，舍入允许 ±1；rel 截断避免分母接近 0 时爆炸
    for i in range(2):
        out = outputs[i].to(torch.int32)
        out_new = outputs_new[i].to(torch.int32)
        abs_diff = torch.abs(out - out_new)
        all_abs_diff.append(abs_diff.view(-1))
        deno = torch.maximum(torch.abs(out), torch.abs(out_new)).float() + 1e-7
        rel = (abs_diff.float() / deno).clamp(max=1.0)
        all_rel_diff.append(rel.view(-1))
        if torch.any(abs_diff > 1):
            is_pass = 0
    # x: fp，kernel 输出 round(sum) 的 bf16/fp16，与 Python 可能有表示差异，只做 abs 容差 1.0
    out_x, out_x_new = outputs[2], outputs_new[2]
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
