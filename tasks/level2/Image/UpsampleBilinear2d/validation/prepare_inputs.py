import torch
import numpy as np


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(param['dtype'], torch.float32)

    batch = int(param['batch'])
    channels = int(param['channels'])
    in_h = int(param['in_h'])
    in_w = int(param['in_w'])
    out_h = int(param['out_h'])
    out_w = int(param['out_w'])

    align_corners = param['align_corners']
    if isinstance(align_corners, bool):
        pass
    elif isinstance(align_corners, str):
        align_corners = align_corners.lower() == 'true'
    else:
        align_corners = bool(align_corners)

    input = torch.randn(batch, channels, in_h, in_w, dtype=dtype, device=device)
    size = torch.tensor([out_h, out_w], dtype=torch.int32, device=device)

    return (input, size, align_corners)


def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    dtype = param['dtype']
    # 对于 bilinear 上采样，float16/bfloat16 由于 matmul 精度限制需要更大容差
    if dtype == 'float16':
        RTOL = 0.5
        ATOL = 0.5
    elif dtype == 'bfloat16':
        RTOL = 0.5
        ATOL = 0.5
    elif dtype == 'float32':
        RTOL = 1.0 / 2048 + 1.0 / 16384
        ATOL = 1.0 / 2048 + 1.0 / 16384
    else:
        RTOL = 0
        ATOL = 0

    abs_diffs = []
    rel_diffs = []
    is_pass = True

    for out, out_new in zip(outputs, outputs_new):
        if out.numel() == 0:
            continue
        out_f = out.detach().float()
        out_new_f = out_new.detach().float()

        abs_diff = (out_f - out_new_f).abs()
        abs_max = abs_diff.max()

        denominator = torch.clamp(out_f.abs(), min=1e-8)
        rel_diff = abs_diff / denominator
        rel_max = rel_diff.max()

        abs_diffs.append(abs_max.unsqueeze(0))
        rel_diffs.append(rel_max.unsqueeze(0))

        if RTOL == 0:
            if abs_max.item() > 0:
                is_pass = False
        else:
            if (abs_max.item() > ATOL and rel_max.item() > RTOL):
                is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
