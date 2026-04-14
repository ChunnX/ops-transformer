import torch


def get_inputs(param, device=None):
    """根据 test_cases.csv 的行生成输入张量。"""
    if device is None:
        device = torch.device("npu:0")

    # 读取参数
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    dtype = dtype_map.get(param["dtype"], torch.float32)

    batch = int(param["batch"])
    channels = int(param["channels"])
    input_w = int(param["input_w"])
    output_w = int(param["output_w"])

    ac = param["align_corners"]
    align_corners = bool(ac) if not isinstance(ac, str) else (ac.lower() == "true")

    x = torch.randn(batch, channels, input_w, dtype=dtype, device=device)
    return (x, output_w, align_corners)


def get_init_inputs(param, device=None):
    """返回空列表。"""
    return []


def custom_check_precision(param, outputs, outputs_new):
    """自定义精度检查。返回 (is_pass, all_abs_diff, all_rel_diff)。
    使用 atol + rtol * |ref| 组合判据，避免近零值导致相对误差爆炸。
    """
    dtype = param.get("dtype", "float32")

    if dtype == "float32":
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0
        ATOL = 1e-5
    elif dtype == "float16":
        RTOL = 1.0 / 512.0
        ATOL = 1e-3
    elif dtype == "bfloat16":
        RTOL = 1.0 / 256.0 + 1.0 / 16384.0
        ATOL = 8e-3
    else:
        RTOL = 1.0 / 2048.0 + 1.0 / 16384.0
        ATOL = 1e-5

    abs_diffs = []
    rel_diffs = []
    ref_abs_list = []

    for out, out_new in zip(outputs, outputs_new):
        out_f = out.float()
        out_new_f = out_new.float()
        abs_diff = (out_f - out_new_f).abs()
        abs_diffs.append(abs_diff.flatten())

        out_abs = out_f.abs()
        rel_diff = abs_diff / (out_abs + 1e-8)
        rel_diffs.append(rel_diff.flatten())

        ref_abs_list.append(out_abs.flatten())

    all_abs_diff = torch.cat(abs_diffs) if abs_diffs else torch.tensor([0.0])
    all_rel_diff = torch.cat(rel_diffs) if rel_diffs else torch.tensor([0.0])

    # 组合判据：|diff| <= ATOL + RTOL * |ref|，逐元素检查
    ref_abs = torch.cat(ref_abs_list) if ref_abs_list else torch.tensor([0.0])
    element_pass = all_abs_diff <= ATOL + RTOL * ref_abs
    is_pass = element_pass.all().item()

    return is_pass, all_abs_diff, all_rel_diff
