import torch
import numpy as np
from typing import List, Optional


def get_inputs(param, device=None):
    """根据 test_cases.csv 行生成输入张量。返回与 Model.forward() 签名匹配的元组。"""
    if device is None:
        device = torch.device("npu:0")

    tokens_num = int(param["tokens_num"])
    hidden_size = int(param["hidden_size"])
    topk_num = int(param["topk_num"])
    dtype_str = str(param["dtype"])
    has_probs = int(param["has_probs"])
    range_start = int(param["range_start"])
    range_end = int(param["range_end"])

    if dtype_str == "float32" or dtype_str == "torch.float32":
        dtype = torch.float32
    elif dtype_str == "float16" or dtype_str == "torch.float16":
        dtype = torch.float16
    elif dtype_str == "bfloat16" or dtype_str == "torch.bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    total_indices = tokens_num * topk_num

    # Generate sorted_indices: valid indices in [0, total_indices)
    sorted_indices = torch.randperm(total_indices)[:total_indices].int().to(device)

    # unpermuted_tokens_grad: (tokens_num, hidden_size)
    unpermuted_tokens_grad = torch.randn(tokens_num, hidden_size, dtype=dtype, device=device)

    # permuted_tokens: (total_indices, hidden_size)
    permuted_tokens = torch.randn(total_indices, hidden_size, dtype=dtype, device=device)

    # probs
    if has_probs:
        probs = torch.randn(tokens_num, topk_num, dtype=dtype, device=device)
    else:
        probs = None

    padded_mode = False
    restore_shape = [1, 1]
    if range_start < 0 or range_end < 0:
        range_vals = [0, total_indices]
    else:
        range_vals = [range_start, range_end]

    return (
        unpermuted_tokens_grad,
        sorted_indices,
        permuted_tokens,
        probs,
        padded_mode,
        restore_shape,
        range_vals,
        topk_num,
    )


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    rtol = 0.0
    dtype_str = str(param["dtype"])
    if dtype_str == "float16" or dtype_str == "torch.float16":
        rtol = 1.0 / 512
    elif dtype_str == "bfloat16" or dtype_str == "torch.bfloat16":
        rtol = 1.0 / 256 + 1.0 / 16384
    elif dtype_str == "float32" or dtype_str == "torch.float32":
        rtol = 1.0 / 2048 + 1.0 / 16384

    atol = rtol

    abs_diffs = []
    rel_diffs = []

    for out, out_new in zip(outputs, outputs_new):
        if out.numel() == 0 or out_new.numel() == 0:
            continue
        # Convert to float32 for comparison
        out_f = out.float()
        out_new_f = out_new.float()
        diff = torch.abs(out_f - out_new_f)
        abs_diffs.append(diff.flatten())

        max_val = torch.max(torch.abs(out_f), torch.abs(out_new_f))
        max_val = torch.clamp(max_val, min=1e-8)
        rel_diff = diff / max_val
        rel_diffs.append(rel_diff.flatten())

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])

    max_abs = torch.max(all_abs_diff).item()
    max_rel = torch.max(all_rel_diff).item()
    is_pass = max_abs <= atol and max_rel <= rtol

    return is_pass, all_abs_diff, all_rel_diff
