import torch

def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    dtype_map = {
        'float32': torch.float32,
        'float16': torch.float16,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map[param['dtype']]
    shape = eval(param['input_shape'])
    x = torch.randn(shape, dtype=dtype, device=device)
    output_size = eval(param['output_size'])
    align_corners = param['align_corners'] == 'True'
    return (x, output_size, align_corners)

def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []

def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    dtype = param['dtype']
    if dtype == 'float32':
        RTOL = 1.0 / 2048 + 1.0 / 16384
    elif dtype == 'float16':
        RTOL = 1.0 / 512
    elif dtype == 'bfloat16':
        RTOL = 1.0 / 64
    else:
        RTOL = 1.0 / 2048 + 1.0 / 16384

    is_pass = True
    abs_diffs = []
    rel_diffs = []

    for out, out_new in zip(outputs, outputs_new):
        out_f = out.float()
        out_new_f = out_new.float()
        abs_diff = (out_f - out_new_f).abs()
        abs_diffs.append(abs_diff.flatten())

        # Compute relative difference where golden != 0
        nonzero_mask = out_f.abs() > 1e-6
        if nonzero_mask.any():
            rel_diff = abs_diff[nonzero_mask] / out_f[nonzero_mask].abs()
            rel_diffs.append(rel_diff.flatten())
        else:
            rel_diffs.append(torch.tensor([0.0]))

        max_abs = abs_diff.max().item()
        max_rel = rel_diffs[-1].max().item() if rel_diffs[-1].numel() > 0 else 0.0

        if max_abs > RTOL and max_rel > RTOL:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
