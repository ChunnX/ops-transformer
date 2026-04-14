import torch
from typing import List


def get_inputs(param, device=None):
    if device is None:
        device = torch.device("npu")

    dtype = param['dtype']
    N = param['N']
    C = param['C']
    D = param['D']
    H = param['H']
    W = param['W']
    output_size = [param['output_d'], param['output_h'], param['output_w']]
    scale_d = float(param.get('scale_d', 0.0))
    scale_h = float(param.get('scale_h', 0.0))
    scale_w = float(param.get('scale_w', 0.0))

    torch_dtype = getattr(torch, dtype)
    x = torch.randn((N, C, D, H, W), dtype=torch_dtype, device=device)

    return (x, output_size, scale_d, scale_h, scale_w)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype = param['dtype']
    if dtype == 'float16':
        rtol = 1.0 / 512
    elif dtype == 'bfloat16':
        rtol = 1.0 / 256 + 1.0 / 16384
    else:
        rtol = 1.0 / 2048 + 1.0 / 16384

    abs_diffs = []
    rel_diffs = []
    is_pass = True

    for out, out_new in zip(outputs, outputs_new):
        if out.numel() == 0:
            continue
        abs_diff = (out - out_new).abs()
        abs_diffs.append(abs_diff.flatten())
        rel_diff = abs_diff / (out.abs().clamp(min=1e-8) + 1e-8)
        rel_diffs.append(rel_diff.flatten())
        max_abs = abs_diff.max().item()
        max_rel = rel_diff.max().item()
        if max_abs > rtol * 10 and max_rel > rtol:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
