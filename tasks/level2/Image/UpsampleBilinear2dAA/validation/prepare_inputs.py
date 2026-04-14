import torch


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    if device is None:
        device = "npu:0"

    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(param["dtype"], torch.float32)

    batch = int(param["batch"])
    channels = int(param["channels"])
    in_h = int(param["in_h"])
    in_w = int(param["in_w"])
    out_h = int(param["out_h"])
    out_w = int(param["out_w"])
    align_corners = param["align_corners"] == "True"

    input_tensor = torch.randn(batch, channels, in_h, in_w, dtype=dtype, device=device)
    output_size = [out_h, out_w]

    return (input_tensor, output_size, align_corners)


def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。"""
    dtype_map = {
        "float16": torch.float16,
        "float32": torch.float32,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(param["dtype"], torch.float32)

    if dtype == torch.float16:
        RTOL = 1.0 / 512.0
    elif dtype == torch.bfloat16:
        RTOL = 1.0 / 256.0 + 1.0 / 16384.0
    elif dtype == torch.float32:
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0
    else:
        RTOL = 0.0

    is_pass = True
    abs_diffs = []
    rel_diffs = []

    for out, out_new in zip(outputs, outputs_new):
        if dtype == torch.bfloat16 or dtype == torch.float16:
            out_f = out.float()
            out_new_f = out_new.float()
        else:
            out_f = out
            out_new_f = out_new

        abs_diff = (out_f - out_new_f).abs()
        abs_diffs.append(abs_diff.flatten())

        rel_diff = abs_diff / (out_f.abs() + 1e-8)
        rel_diffs.append(rel_diff.flatten())

        max_abs = abs_diff.max().item()
        max_rel = rel_diff.max().item()

        if max_abs > RTOL and max_rel > RTOL:
            is_pass = False

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])
    return is_pass, all_abs_diff, all_rel_diff
