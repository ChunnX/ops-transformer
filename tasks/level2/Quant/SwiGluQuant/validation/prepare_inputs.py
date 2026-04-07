import torch


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def get_init_inputs(param, device=None):
    static_mode = str(param.get("static_mode", "none"))
    if str(param.get("quant_mode", "dynamic")) == "dynamic":
        static_mode = "none"
    return (
        _parse_bool(param.get("activate_left", False)),
        str(param.get("quant_mode", "dynamic")),
        int(param.get("group_list_type", 0)),
        int(param.get("dst_type", 2)),
        static_mode,
    )


def _half_cols(param, shape):
    last = int(shape[-1])
    if last % 2 != 0:
        raise ValueError("x last dim must be even")
    h = last // 2
    if int(param.get("dst_type", 2)) == 29 and last % 4 != 0:
        raise ValueError("int4 requires x last dim divisible by 4")
    return h


def get_inputs(param, device=None):
    shape = eval(param.get("input_shape", "[4, 32]"), {"__builtins__": {}})
    if len(shape) < 2:
        raise ValueError("x must be at least 2D")
    x_dtype = getattr(torch, str(param.get("x_dtype", "float16")))
    half = _half_cols(param, shape)
    qm = str(param.get("quant_mode", "dynamic"))
    static_mode = str(param.get("static_mode", "per_channel"))
    has_group = _parse_bool(param.get("has_group", False))
    torch.manual_seed(int(param.get("case_id", 0)))

    x = (torch.rand(shape, device="cpu", dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=x_dtype)

    if has_group:
        g = int(param.get("num_groups", 2))
        rn = 1
        for d in shape[:-1]:
            rn *= int(d)
        # cumsum 边界，末元为 rn
        part = max(1, rn // g)
        bounds = []
        c = 0
        for i in range(g - 1):
            c = min(c + part, rn - 1)
            bounds.append(c)
        bounds.append(rn)
        group_index = torch.tensor(bounds, dtype=torch.int32, device=device)
        smooth = (torch.rand((g, half), device="cpu", dtype=torch.float32) * 0.4 + 0.3).to(device=device)
    else:
        group_index = None
        if qm == "dynamic" or static_mode == "per_channel":
            smooth = (torch.rand((1, half), device="cpu", dtype=torch.float32) * 0.4 + 0.3).to(device=device)
        else:
            # per_tensor: 1 维 [1]
            smooth = (torch.rand(1, device="cpu", dtype=torch.float32) * 0.4 + 0.3).to(device=device)

    if qm == "static":
        if has_group:
            g = int(param.get("num_groups", 2))
            if static_mode == "per_tensor":
                off = (torch.rand(g, device="cpu", dtype=torch.float32) * 0.02).to(device=device)
            else:
                off = (torch.rand((g, half), device="cpu", dtype=torch.float32) * 0.02).to(device=device)
        else:
            if static_mode == "per_tensor":
                off = (torch.rand(1, device="cpu", dtype=torch.float32) * 0.02).to(device=device)
            else:
                off = (torch.rand((1, half), device="cpu", dtype=torch.float32) * 0.02).to(device=device)
    else:
        off = None

    return (x, smooth, off, group_index)


def custom_check_precision(param, outputs, outputs_new):
    from framework.utils import check_precision as cp

    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outputs_new = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    ry, rs = outputs[0].cpu(), outputs[1].cpu()
    ny, ns = outputs_new[0].cpu(), outputs_new[1].cpu()
    if ny.dtype != ry.dtype:
        ny = ny.to(ry.dtype)
    y_ad = (ry.to(torch.int16) - ny.to(torch.int16)).abs().to(torch.float32).reshape(-1)
    y_ok = bool((y_ad <= 1).all().item())
    y_rd = y_ad / 128.0

    if str(param.get("quant_mode", "dynamic")) == "static":
        acc_s = 1 if (rs.abs() < 1e-5).all() and (ns.abs() < 1e-5).all() else 0
        ad_s = (rs - ns).abs().reshape(-1).float()
        rd_s = ad_s
    else:
        acc_s, ad_s, rd_s = cp([rs], [ns], 0.08, 0.08)

    is_pass = 1 if (y_ok and acc_s == 1) else 0
    all_ad = torch.cat([y_ad, ad_s])
    all_rd = torch.cat([y_rd, rd_s])
    return is_pass, all_ad, all_rd
