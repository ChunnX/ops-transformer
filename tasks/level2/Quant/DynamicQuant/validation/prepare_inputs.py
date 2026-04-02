import torch


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def get_inputs(param, device=None):
    shape = eval(param.get("input_shape", "[8, 32]"), {"__builtins__": {}})
    dtype_str = param.get("x_dtype", "float16")
    x_dtype = getattr(torch, dtype_str)

    x = (torch.rand(shape, device="cpu", dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=x_dtype)
    if _parse_bool(param.get("has_smooth", False)):
        last = shape[-1]
        smooth = (torch.rand((last,), device="cpu", dtype=torch.float32) * 0.6 + 0.2).to(
            device=device, dtype=x_dtype
        )
    else:
        smooth = None
    return (x, smooth)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    """
    y：int8；大 shape 分块路径下偶发 ±1 舍入差，允许逐元素 |Δ|≤1。
    scale：float，与 framework.check_precision 一致（abs 与 rel 不同时超阈则通过）。
    """
    from framework.utils import check_precision

    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outputs_new = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    ry, rs = outputs[0].cpu(), outputs[1].cpu()
    ny, ns = outputs_new[0].cpu(), outputs_new[1].cpu()
    if ny.dtype != ry.dtype:
        ny = ny.to(ry.dtype)
    y_ad = (ry.to(torch.int16) - ny.to(torch.int16)).abs().to(torch.float32).reshape(-1)
    y_ok = bool((y_ad <= 1).all().item())
    y_rd = y_ad / 128.0

    acc_s, ad_s, rd_s = check_precision([rs], [ns], 0.001, 0.001)

    is_pass = 1 if (y_ok and acc_s == 1) else 0
    all_ad = torch.cat([y_ad, ad_s])
    all_rd = torch.cat([y_rd, rd_s])
    return is_pass, all_ad, all_rd
