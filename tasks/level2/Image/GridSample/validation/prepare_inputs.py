import torch
import math


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    input_shape = eval(param.get('input_shape', '[1, 3, 4, 4]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    mode = param.get('mode', 'bilinear')
    padding_mode = param.get('padding_mode', 'zeros')
    align_corners = param.get('align_corners', 'False')
    align_corners = align_corners == 'True'

    # Generate input tensor
    x = torch.randn(input_shape, device=device, dtype=dtype)

    # Determine grid shape based on input dimensionality
    dim = len(input_shape)
    if dim == 4:
        # 2D: grid shape is (N, outH, outW, 2)
        N = input_shape[0]
        outH = max(input_shape[2] // 2, 2)
        outW = max(input_shape[3] // 2, 2)
        grid_shape = (N, outH, outW, 2)
    elif dim == 5:
        # 3D: grid shape is (N, outD, outH, outW, 3)
        N = input_shape[0]
        outD = max(input_shape[2] // 2, 2)
        outH = max(input_shape[3] // 2, 2)
        outW = max(input_shape[4] // 2, 2)
        grid_shape = (N, outD, outH, outW, 3)
    else:
        grid_shape = (input_shape[0], 2, 2, 2)

    # Generate grid in range [-1, 1] which is the standard range for grid_sample
    # Use uniform distribution to avoid degenerate cases
    grid = torch.rand(grid_shape, device=device, dtype=dtype) * 2 - 1

    return [x, grid, mode, padding_mode, align_corners]


def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    dtype_str = param.get('dtype', 'float16')

    if 'float16' in dtype_str:
        RTOL = 1.0 / 512.0
        ATOL = 0.01
    elif 'bfloat16' in dtype_str:
        RTOL = 1.0 / 256.0 + 1.0 / 16384.0
        ATOL = 0.02
    elif 'float32' in dtype_str:
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0
        ATOL = 0.001
    else:
        RTOL = 0.0
        ATOL = 0.0

    if not isinstance(outputs, torch.Tensor):
        outputs = torch.tensor(outputs)
    if not isinstance(outputs_new, torch.Tensor):
        outputs_new = torch.tensor(outputs_new)

    outputs_f = outputs.float()
    outputs_new_f = outputs_new.float()

    abs_diff = (outputs_f - outputs_new_f).abs()

    # Use combined tolerance (allclose-style): abs_diff <= ATOL + RTOL * |output|
    # This handles near-zero outputs properly (grid_sample with zeros padding)
    combined_tol = ATOL + RTOL * outputs_f.abs()
    is_pass = bool((abs_diff <= combined_tol).all().item())

    # For reporting, also compute relative error
    abs_outputs = outputs_f.abs().clamp(min=1e-8)
    rel_diff = abs_diff / abs_outputs

    return is_pass, abs_diff, rel_diff
