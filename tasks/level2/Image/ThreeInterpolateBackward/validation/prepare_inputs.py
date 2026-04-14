import torch
import numpy as np


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    B = int(param["B"])
    C = int(param["C"])
    N = int(param["N"])
    M = int(param["M"])
    grad_dtype = param["grad_dtype"]
    idx_dtype = param["idx_dtype"]

    # grad_x: (B, C, N)
    grad_x = torch.randn(B, C, N, dtype=getattr(torch, grad_dtype), device=device)

    # idx: (B, N, 3) - indices in [0, M)
    idx = torch.randint(0, M, (B, N, 3), dtype=getattr(torch, idx_dtype), device=device)

    # weight: (B, N, 3) - random weights
    weight = torch.randn(B, N, 3, dtype=getattr(torch, grad_dtype), device=device)

    # m as int64
    m = M

    return (grad_x, idx, weight, m)


def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    grad_dtype = param["grad_dtype"]

    if "float16" == grad_dtype:
        RTOL = 1.0 / 512
    elif "float32" == grad_dtype:
        RTOL = 1.0 / 2048 + 1.0 / 16384
    else:
        RTOL = 1.0 / 2048 + 1.0 / 16384

    abs_diffs = []
    rel_diffs = []
    is_pass = True

    for out, out_new in zip(outputs, outputs_new):
        if out.numel() == 0:
            continue
        out_f = out.float()
        out_new_f = out_new.float()
        abs_diff = (out_f - out_new_f).abs()
        abs_max = out_f.abs().max()
        abs_diffs.append(abs_diff.flatten())
        if abs_max > 0:
            rel_diff = abs_diff / abs_max
            rel_diffs.append(rel_diff.flatten())
        else:
            rel_diffs.append(torch.zeros_like(abs_diff).flatten())

        max_abs = abs_diff.max().item()
        max_rel = rel_diffs[-1].max().item() if abs_max > 0 else 0.0
        if max_abs > RTOL * abs_max.item() and max_abs > 1e-6:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])

    return is_pass, all_abs_diff, all_rel_diff
