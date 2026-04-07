import torch


def _parse_optional_shape(v):
    if v is None:
        return None
    try:
        import pandas as pd

        if pd.isna(v):
            return None
    except Exception:
        pass
    if isinstance(v, float) and v != v:
        return None
    s = str(v).strip()
    if not s or s.lower() == "nan":
        return None
    return eval(s, {"__builtins__": {}})


def get_init_inputs(param, device=None):
    round_mode = int(param.get("round_mode", 0))
    return (round_mode,)


def get_inputs(param, device=None):
    scale_shape = eval(param.get("scale_shape", "[8]"), {"__builtins__": {}})
    scale = (torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.5 + 0.01).contiguous()

    off_shape = _parse_optional_shape(param.get("offset_shape", None))
    if off_shape is None:
        offset = None
    else:
        offset = (torch.rand(off_shape, device=device, dtype=torch.float32) * 0.2 - 0.1).contiguous()

    return (scale, offset)


def custom_check_precision(param, outputs, outputs_new):
    """输出为 uint64 打包值；NPU 侧可能为 int64，按位与 golden 对齐（precision-debug §1）。"""
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outputs_new = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    ro = outputs[0].cpu()
    no = outputs_new[0].cpu()
    ru = ro.view(torch.uint64) if ro.dtype == torch.int64 else ro
    nu = no.view(torch.uint64) if no.dtype == torch.int64 else no
    if nu.dtype != ru.dtype:
        nu = nu.to(dtype=ru.dtype)
    exact = bool(torch.equal(ru, nu))
    is_pass = 1 if exact else 0
    ad = (ru.to(torch.int64) - nu.to(torch.int64)).abs().float().reshape(-1)
    rd = ad / (ru.to(torch.float64).abs() + 1.0).to(torch.float32).reshape(-1)
    return is_pass, ad, rd
