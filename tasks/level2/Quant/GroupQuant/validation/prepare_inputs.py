import torch


def _parse_bool(v):
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def _make_group_index(s: int, e: int, device, dtype: torch.dtype):
    """单调非减边界：group_index[i] 为第 i 个 expert 负责行的右端点（开区间），末尾等于 S。"""
    if e < 1:
        raise ValueError("num_experts must be >= 1")
    base = s // e
    rem = s % e
    g = []
    cur = 0
    for i in range(e):
        cur += base + (1 if i < rem else 0)
        g.append(cur)
    assert g[-1] == s
    return torch.tensor(g, device=device, dtype=dtype)


def get_init_inputs(param, device=None):
    dst_type = int(param.get("dst_type", 2))
    return (dst_type,)


def get_inputs(param, device=None):
    shape = eval(param.get("input_shape", "[16, 32]"), {"__builtins__": {}})
    if len(shape) != 2:
        raise ValueError("GroupQuant x must be 2D [S, H]")
    s, h = int(shape[0]), int(shape[1])
    e = int(param.get("num_experts", 2))
    dst_type = int(param.get("dst_type", 2))
    if dst_type == 29 and h % 2 != 0:
        raise ValueError("int4 output requires even H")

    x_dtype_str = param.get("x_dtype", "float16")
    scale_dtype_str = param.get("scale_dtype", x_dtype_str)
    gi_dtype_str = param.get("group_index_dtype", "int32")
    x_dtype = getattr(torch, x_dtype_str)
    scale_dtype = getattr(torch, scale_dtype_str)
    gi_dtype = getattr(torch, gi_dtype_str)

    torch.manual_seed(int(param.get("case_id", 0)))
    x = (torch.rand((s, h), device="cpu", dtype=torch.float32) * 2.0 - 1.0).to(device=device, dtype=x_dtype)
    scale = (torch.rand((e, h), device="cpu", dtype=torch.float32) * 0.5 + 0.1).to(device=device, dtype=scale_dtype)
    group_index = _make_group_index(s, e, device, gi_dtype)

    if _parse_bool(param.get("has_offset", False)):
        off = torch.tensor([0.03], device=device, dtype=scale_dtype)
    else:
        off = None

    return (x, scale, group_index, off)


def custom_check_precision(param, outputs, outputs_new):
    """整型输出：与 AscendC 两次 CAST_RINT 路径一致时仍可能差 1，int8/int4 均允许 |Δ|≤1。"""
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outputs_new = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    ry = outputs[0].cpu()
    ny = outputs_new[0].cpu()
    if ny.dtype != ry.dtype:
        ny = ny.to(ry.dtype)
    diff = (ry.to(torch.int16) - ny.to(torch.int16)).abs().to(torch.float32).reshape(-1)
    y_ok = bool((diff <= 1).all().item())
    rd = diff / 128.0
    is_pass = 1 if y_ok else 0
    return is_pass, diff, rd
