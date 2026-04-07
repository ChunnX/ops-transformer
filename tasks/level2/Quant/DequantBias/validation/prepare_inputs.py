import torch

GE_DT_FP16 = 1
GE_DT_BF16 = 27


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _bias_from_dtype(dtype_str, shape, device):
    if dtype_str in ("none", "", "null"):
        return None
    if dtype_str == "int32":
        return torch.randint(-32, 32, shape, device=device, dtype=torch.int32)
    if dtype_str == "float32":
        return (torch.rand(shape, device="cpu", dtype=torch.float32) * 0.2 - 0.1).to(device)
    if dtype_str == "float16":
        return (torch.rand(shape, device="cpu", dtype=torch.float32) * 0.2 - 0.1).to(device=device, dtype=torch.float16)
    if dtype_str == "bfloat16":
        return (torch.rand(shape, device="cpu", dtype=torch.float32) * 0.2 - 0.1).to(device=device, dtype=torch.bfloat16)
    raise ValueError(f"unknown bias_dtype: {dtype_str}")


def get_inputs(param, device=None):
    shape = eval(param.get("input_shape", "[4, 16]"), {"__builtins__": {}})
    m, n = int(shape[0]), int(shape[1])
    has_activate = _parse_bool(param.get("has_activate", True))
    has_bias = _parse_bool(param.get("has_bias", False))
    bias_dtype = str(param.get("bias_dtype", "none")).strip().lower()
    output_dtype = int(param.get("output_dtype", GE_DT_FP16))

    x = torch.randint(-64, 64, (m, n), device=device, dtype=torch.int32)
    # EZ1001 白名单：BF16 输出 ↔ weightScale BF16；FP16 输出 ↔ weightScale FLOAT（见精度日志 SupportInfo）
    w_cpu = torch.rand(n, dtype=torch.float32, device="cpu") * 0.05 + 0.01
    if output_dtype == GE_DT_BF16:
        weight_scale = w_cpu.to(device=device, dtype=torch.bfloat16)
    else:
        weight_scale = w_cpu.to(device=device, dtype=torch.float32)

    activate_scale = None
    if has_activate:
        activate_scale = (torch.rand(m, device="cpu", dtype=torch.float32) * 0.1 + 0.95).to(device=device)

    bias = None
    if has_bias:
        bias = _bias_from_dtype(bias_dtype, (n,), device)

    return (x, weight_scale, activate_scale, bias)


def get_init_inputs(param, device=None):
    output_dtype = int(param.get("output_dtype", GE_DT_FP16))
    return [output_dtype]


def custom_check_precision(param, ref_output, test_output):
    """半精度比较：在 float32 上比绝对误差，避免 reshape 广播问题。"""
    outputs = ref_output if isinstance(ref_output, (list, tuple)) else [ref_output]
    outputs_new = test_output if isinstance(test_output, (list, tuple)) else [test_output]
    max_abs = 0.05 if int(param.get("output_dtype", GE_DT_FP16)) == GE_DT_BF16 else 0.02
    is_pass = 1
    all_abs_diff, all_rel_diff = [], []
    for o, n in zip(outputs, outputs_new):
        o32 = o.detach().float().cpu().reshape(-1)
        n32 = n.detach().float().cpu().reshape(-1)
        ad = torch.abs(o32 - n32)
        rd = ad / (torch.abs(o32) + 1e-5)
        if (ad > max_abs).any():
            is_pass = 0
        all_abs_diff.append(ad)
        all_rel_diff.append(rd)
    return is_pass, torch.cat(all_abs_diff), torch.cat(all_rel_diff)
