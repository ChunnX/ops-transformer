import torch
import numpy as np


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(param["dtype"], torch.float32)

    batch = int(param["batch"])
    channels = int(param["channels"])
    grad_h = int(param["grad_h"])
    grad_w = int(param["grad_w"])
    input_h = int(param["input_h"])
    input_w = int(param["input_w"])
    align_corners_val = param["align_corners"]
    if isinstance(align_corners_val, str):
        align_corners = align_corners_val.strip().lower() == "true"
    else:
        align_corners = bool(align_corners_val)

    grad_output = torch.randn(batch, channels, grad_h, grad_w, dtype=dtype, device=device)
    input_size = [input_h, input_w]

    return (grad_output, input_size, align_corners)


def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(param["dtype"], torch.float32)

    if dtype == torch.float32:
        RTOL = 1 / 2048 + 1 / 16384
    elif dtype == torch.float16:
        RTOL = 1 / 512
    elif dtype == torch.bfloat16:
        RTOL = 2e-2
    else:
        RTOL = 1 / 512

    ATOL = 1e-3

    is_pass = True
    abs_diffs = []
    rel_diffs = []

    for golden, custom in zip(outputs, outputs_new):
        golden_f = golden.float()
        custom_f = custom.float()

        abs_diff = torch.abs(golden_f - custom_f)
        max_val = torch.max(torch.abs(golden_f))
        rel_diff = abs_diff / (max_val + 1e-8)

        max_abs = abs_diff.max().item()
        max_rel = rel_diff.max().item()

        abs_diffs.append(abs_diff.flatten())
        rel_diffs.append(rel_diff.flatten())

        if max_rel > RTOL and max_abs > ATOL:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
