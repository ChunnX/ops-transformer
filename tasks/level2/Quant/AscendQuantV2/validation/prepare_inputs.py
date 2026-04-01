import torch


GE_DT_INT8 = 2


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _make_scale_offset(shape, scale_mode, x_dtype, device, has_offset):
    nd = len(shape)
    if scale_mode == "per_tensor":
        scale_shape = (1,) * nd
        scale_v = torch.tensor(0.11, device=device, dtype=x_dtype).reshape(scale_shape)
    elif scale_mode == "per_channel":
        c = shape[-1]
        scale_shape = (1,) * (nd - 1) + (c,)
        base = torch.rand(c, device="cpu", dtype=torch.float32) * 0.14 + 0.06
        scale_v = base.to(device=device, dtype=x_dtype).reshape(scale_shape)
    else:
        raise ValueError(f"unknown scale_mode: {scale_mode}")

    if has_offset:
        offset_v = (torch.rand(scale_shape, device="cpu", dtype=torch.float32) * 0.5 - 0.25).to(
            device=device, dtype=x_dtype
        )
    else:
        offset_v = None
    return scale_v, offset_v


def get_inputs(param, device=None):
    shape = eval(param.get("input_shape", "[8, 32]"), {"__builtins__": {}})
    dtype_str = param.get("x_dtype", "float16")
    x_dtype = getattr(torch, dtype_str)
    scale_mode = str(param.get("scale_mode", "per_channel")).strip()
    has_offset = _parse_bool(param.get("has_offset", False))

    x = (torch.rand(shape, device="cpu", dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=x_dtype)
    scale, offset = _make_scale_offset(shape, scale_mode, x_dtype, device, has_offset)
    return (x, scale, offset)


def get_init_inputs(param, device=None):
    sqrt_mode = _parse_bool(param.get("sqrt_mode", False))
    round_mode = str(param.get("round_mode", "round")).strip()
    return [sqrt_mode, round_mode, GE_DT_INT8]


def custom_check_precision(param, outputs, outputs_new):
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outputs_new = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]

    is_pass = 1
    all_abs_diff, all_rel_diff = [], []
    for out, out_new in zip(outputs, outputs_new):
        o = out.cpu()
        n = out_new.cpu()
        if o.dtype != n.dtype:
            n = n.to(o.dtype)
        if not torch.equal(o, n):
            is_pass = 0
        o32 = o.to(torch.float32).reshape(-1)
        n32 = n.to(torch.float32).reshape(-1)
        ad = torch.abs(o32 - n32)
        rd = ad / (torch.abs(o32) + 1e-7)
        all_abs_diff.append(ad)
        all_rel_diff.append(rd)

    return is_pass, torch.cat(all_abs_diff), torch.cat(all_rel_diff)
