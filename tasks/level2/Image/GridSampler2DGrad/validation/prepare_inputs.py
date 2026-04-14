import torch


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    x_shape = eval(param.get('x_shape', '[1, 3, 4, 4]'))
    grid_shape = eval(param.get('grid_shape', '[1, 2, 2, 2]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)
    interpolation_mode = param.get('interpolation_mode', 'bilinear')
    padding_mode = param.get('padding_mode', 'zeros')
    align_corners = param.get('align_corners', 'False')
    align_corners = align_corners == 'True'

    # Generate input tensors
    x = torch.randn(x_shape, device=device, dtype=dtype)

    # Generate grid in range [-1, 1]
    grid = torch.rand(grid_shape, device=device, dtype=dtype) * 2 - 1

    # Generate grad tensor: shape is (N, C, outH, outW) where outH=outW from grid shape
    N = grid_shape[0]
    C = x_shape[1]
    outH = grid_shape[1]
    outW = grid_shape[2]
    grad_shape = [N, C, outH, outW]
    grad = torch.randn(grad_shape, device=device, dtype=dtype)

    return [grad, x, grid, interpolation_mode, padding_mode, align_corners]


def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, abs_diff_tensor, rel_diff_tensor)。"""
    dtype_str = param.get('dtype', 'float16')

    if 'bfloat16' in dtype_str:
        RTOL = 1.0 / 64.0
    elif 'float16' in dtype_str:
        RTOL = 1.0 / 512.0
    elif 'float32' in dtype_str:
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0
    else:
        RTOL = 0.0

    is_pass = True
    abs_diffs = []
    rel_diffs = []

    # outputs and outputs_new are tuples of (dx, dgrid)
    for out_ref, out_new in zip(outputs, outputs_new):
        if not isinstance(out_ref, torch.Tensor):
            out_ref = torch.tensor(out_ref)
        if not isinstance(out_new, torch.Tensor):
            out_new = torch.tensor(out_new)

        out_ref_f = out_ref.float().flatten()
        out_new_f = out_new.float().flatten()

        abs_diff = (out_ref_f - out_new_f).abs()
        abs_outputs = out_ref_f.abs().clamp(min=1e-8)
        rel_diff = abs_diff / abs_outputs

        abs_diffs.append(abs_diff)
        rel_diffs.append(rel_diff)

        cur_pass = bool(abs_diff.max().item() < 0.1 and rel_diff.max().item() < RTOL)
        is_pass = is_pass and cur_pass

    # Concatenate into single tensors for framework compatibility
    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])

    return is_pass, all_abs_diff, all_rel_diff
