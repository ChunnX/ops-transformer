import torch


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    dtype_map = {
        'float16': torch.float16,
        'float32': torch.float32,
        'bfloat16': torch.bfloat16,
    }
    dtype = dtype_map.get(param['dtype'], torch.float32)

    n = int(param['n'])
    c = int(param['c'])
    h_in = int(param['h_in'])
    w_in = int(param['w_in'])
    h_out = int(param['h_out'])
    w_out = int(param['w_out'])
    align_corners_val = str(param['align_corners']).strip().lower()
    align_corners = align_corners_val in ('true', '1')

    # grad_output has the spatial size of the forward output (h_out, w_out)
    grad_output = torch.randn(n, c, h_out, w_out, dtype=dtype, device=device)
    # output_size: spatial size of the forward output
    output_size = [h_out, w_out]
    # input_size: full shape of the grad_input (= forward input shape)
    input_size = [n, c, h_in, w_in]

    return (grad_output, output_size, input_size, align_corners)


def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    dtype = str(param['dtype']).strip()
    if dtype == 'float16':
        rtol = 1.0 / 200
    elif dtype == 'bfloat16':
        rtol = 1.0 / 50
    elif dtype == 'float32':
        rtol = 1.0 / 2048 + 1.0 / 16384
    else:
        rtol = 1.0 / 2048 + 1.0 / 16384

    abs_diffs = []
    rel_diffs = []
    is_pass = True

    for out, out_new in zip(outputs, outputs_new):
        if out_new is None:
            is_pass = False
            continue
        out_f = out.float()
        out_new_f = out_new.float()
        abs_diff = (out_f - out_new_f).abs()
        abs_max = abs_diff.max().item()
        denom = out_f.abs().clamp(min=1e-8)
        rel_diff = (abs_diff / denom)
        rel_max = rel_diff.max().item()

        abs_diffs.append(abs_diff.flatten())
        rel_diffs.append(rel_diff.flatten())

        if abs_max > rtol and rel_max > rtol:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
