import torch


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def get_init_inputs(param, device=None):
    dst_type = int(param.get("dst_type", 2))
    return (dst_type,)


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

    if _parse_bool(param.get("has_group_index", False)):
        # MoE 路径需与 tiling 一致；默认关闭，避免未覆盖的 tiling key
        group_index = torch.zeros(shape[0], device=device, dtype=torch.int32)
    else:
        group_index = None

    return (x, smooth, group_index)


def custom_check_precision(param, outputs, outputs_new):
    """y：int8 允许 |Δ|≤1；scale/offset：与 check_precision 一致。"""
    from framework.utils import check_precision as cp

    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outputs_new = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    ry, rs, ro = outputs[0].cpu(), outputs[1].cpu(), outputs[2].cpu()
    ny, ns, no = outputs_new[0].cpu(), outputs_new[1].cpu(), outputs_new[2].cpu()
    if ny.dtype != ry.dtype:
        ny = ny.to(ry.dtype)
    y_ad = (ry.to(torch.int16) - ny.to(torch.int16)).abs().to(torch.float32).reshape(-1)
    y_ok = bool((y_ad <= 1).all().item())
    y_rd = y_ad / 128.0

    acc_s, ad_s, rd_s = cp([rs], [ns], 0.001, 0.001)
    acc_o, ad_o, rd_o = cp([ro], [no], 0.001, 0.001)

    is_pass = 1 if (y_ok and acc_s == 1 and acc_o == 1) else 0
    all_ad = torch.cat([y_ad, ad_s, ad_o])
    all_rd = torch.cat([y_rd, rd_s, rd_o])
    return is_pass, all_ad, all_rd
