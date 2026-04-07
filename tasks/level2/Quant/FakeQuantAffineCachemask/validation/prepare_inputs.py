import torch

from framework.utils import check_precision


def get_init_inputs(param, device=None):
    axis = int(param.get("axis", 0))
    qmin = int(param.get("quant_min", 0))
    qmax = int(param.get("quant_max", 255))
    return (axis, qmin, qmax)


def get_inputs(param, device=None):
    x_shape = eval(param.get("x_shape", "[8,16]"), {"__builtins__": {}})
    x_dtype_str = str(param.get("x_dtype", "float16")).lower()
    x_dtype = getattr(torch, x_dtype_str) if hasattr(torch, x_dtype_str) else torch.float16

    scale_shape = eval(param.get("scale_shape", "[8]"), {"__builtins__": {}})
    if len(scale_shape) != 1:
        raise ValueError("scale_shape must be 1D [N] to match tiling headNum on dim0")
    n = scale_shape[0]
    if x_shape[0] != n:
        raise ValueError("x_shape[0] must equal scale_shape[0] (per-channel on dim0)")

    qmin = int(param.get("quant_min", 0))
    qmax = int(param.get("quant_max", 255))

    x = (torch.rand(x_shape, device=device, dtype=torch.float32) * 2.0 - 1.0).to(x_dtype)
    scale = (torch.rand(scale_shape, device=device, dtype=torch.float32) * 0.5 + 0.05).to(x_dtype)
    zero_point = torch.randint(qmin, qmax + 1, scale_shape, device=device, dtype=torch.int32)
    return (x, scale, zero_point)


def custom_check_precision(param, outputs, outputs_new):
    """y：float 半精度链路与 fake_quant 对齐；mask：bool 全量一致。"""
    outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
    outputs_new = outputs_new if isinstance(outputs_new, (list, tuple)) else [outputs_new]
    ry, rm = outputs[0].cpu(), outputs[1].cpu()
    ny, nm = outputs_new[0].cpu(), outputs_new[1].cpu()
    if ny.dtype != ry.dtype:
        ny = ny.to(ry.dtype)
    acc_y, ad_y, rd_y = check_precision([ry], [ny], 0.01, 0.01)
    rm_b = rm.bool() if rm.dtype != torch.bool else rm
    nm_b = nm.bool() if nm.dtype != torch.bool else nm
    mask_ok = bool(torch.equal(rm_b, nm_b))
    is_pass = 1 if (acc_y == 1 and mask_ok) else 0
    return is_pass, ad_y.reshape(-1), rd_y.reshape(-1)
