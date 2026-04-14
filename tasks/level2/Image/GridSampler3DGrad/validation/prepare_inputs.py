import torch
import numpy as np


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(param['dtype'], torch.float32)

    N = int(param['N'])
    C = int(param['C'])
    D_in = int(param['D_in'])
    H_in = int(param['H_in'])
    W_in = int(param['W_in'])
    D_out = int(param['D_out'])
    H_out = int(param['H_out'])
    W_out = int(param['W_out'])

    # x: (N, C, D_in, H_in, W_in)
    x = torch.randn(N, C, D_in, H_in, W_in, dtype=dtype, device=device)

    # grid: (N, D_out, H_out, W_out, 3) - values in [-1, 1]
    grid = torch.rand(N, D_out, H_out, W_out, 3, dtype=dtype, device=device) * 2 - 1

    # grad: (N, C, D_out, H_out, W_out)
    grad = torch.randn(N, C, D_out, H_out, W_out, dtype=dtype, device=device)

    interpolation_mode = param.get('interpolation_mode', 'bilinear')
    padding_mode = param.get('padding_mode', 'zeros')
    align_corners = param.get('align_corners', 'False') == 'True'

    return [grad, x, grid, interpolation_mode, padding_mode, align_corners]


def get_init_inputs(param, device=None):
    """返回初始化参数。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    dtype = param.get('dtype', 'float32')

    if dtype == 'float32':
        RTOL = 1 / 2048 + 1 / 16384
    elif dtype == 'float16':
        RTOL = 1 / 512
    elif dtype == 'bfloat16':
        RTOL = 1 / 256 + 1 / 16384
    else:
        RTOL = 1 / 2048 + 1 / 16384

    is_pass = True
    all_abs_diff = torch.tensor(0.0)
    all_rel_diff = torch.tensor(0.0)

    for output, output_new in zip(outputs, outputs_new):
        if output is None or output_new is None:
            continue
        # Cast to float32 for comparison
        o = output.detach().float()
        o_new = output_new.detach().float()

        abs_diff = torch.abs(o - o_new).max()
        denom = torch.max(torch.abs(o), torch.abs(o_new)).clamp(min=1e-6)
        rel_diff = (torch.abs(o - o_new) / denom).max()

        all_abs_diff = torch.max(all_abs_diff, abs_diff)
        all_rel_diff = torch.max(all_rel_diff, rel_diff)

        if rel_diff.item() > RTOL:
            is_pass = False

    return is_pass, all_abs_diff.reshape(1), all_rel_diff.reshape(1)
