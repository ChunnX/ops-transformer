import ast

import torch


def _parse_fill_scalar(param, dtype: torch.dtype):
    raw = param.get("fill_value", 0)
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.lower() in ("true", "false"):
            v = raw.lower() == "true"
        else:
            v = ast.literal_eval(raw)
    else:
        v = raw
    if dtype == torch.bool:
        return bool(int(v)) if not isinstance(v, bool) else v
    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        return int(v)
    return float(v)


def _standard_fill_tensor(fill_scalar, dtype: torch.dtype, device):
    """与 x 同 device、dtype；shape [1]、contiguous，与 aclnn 对 fillValue 的 dtype 要求一致。"""
    dev = device if device is not None else torch.device("cpu")
    t = torch.tensor([fill_scalar], dtype=dtype, device=dev)
    return t.reshape(1).contiguous()


def get_inputs(param, device=None):
    shape = ast.literal_eval(param.get("input_shape", "[4, 4]"))
    shape = tuple(int(x) for x in shape)
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)
    # 显式 bool，避免 pandas 读入整型后传入 pybind 时类型含糊
    wrap_raw = param.get("wrap", 0)
    wrap = bool(int(wrap_raw))
    fill_scalar = _parse_fill_scalar(param, dtype)

    if dtype in (torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64):
        low, high = -100, 100
        if dtype == torch.int8:
            low, high = -128, 127
        elif dtype == torch.uint8:
            low, high = 0, 255
        x = torch.randint(low, high, shape, device=device, dtype=dtype)
    elif dtype == torch.bool:
        x = torch.randint(0, 2, shape, device=device, dtype=torch.bool)
    else:
        x = torch.randn(shape, device=device, dtype=dtype)

    fill_tensor = _standard_fill_tensor(fill_scalar, dtype, device)
    return (x, fill_tensor, wrap)


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(row, ref_output, test_output):
    from framework.kernel_gen_config import config
    from framework.utils import check_precision

    ma = config.eval.max_abs_error
    mr = config.eval.max_rel_error
    if row.get("dtype") in ("float16", "bfloat16"):
        ma = max(ma, 0.125)
        mr = max(mr, 0.05)
    return check_precision(ref_output, test_output, ma, mr)
