import random

import numpy as np
import torch

from npukernelbench.stages.precision.comparator import check_precision


def _seed(case_id: int):
    torch.manual_seed(case_id)
    random.seed(case_id)
    np.random.seed(case_id)


def _parse_bool(v, default=False):
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(int(v))
    s = str(v).strip().lower()
    if s in ("true", "1", "yes"):
        return True
    if s in ("false", "0", "no"):
        return False
    return default


def get_inputs(param, device=None):
    case_id = int(param.get("case_id", 0))
    _seed(case_id)

    x_shape = eval(param.get("x_shape", "[1,2048]"))
    expert_idx_shape = eval(param.get("expert_idx_shape", "[1,8]"))
    scale_shape = eval(param.get("scale_shape", "[1]"))
    offset_shape = eval(param.get("offset_shape", "[1]"))
    active_num = int(param.get("active_num", 8))
    expert_capacity = int(param.get("expert_capacity", 1))
    expert_num = int(param.get("expert_num", 256))
    drop_pad_mode = int(param.get("drop_pad_mode", 1))
    expert_tokens_num_type = int(param.get("expert_tokens_num_type", 1))
    expert_tokens_num_flag = _parse_bool(param.get("expert_tokens_num_flag", True), True)
    quant_mode = int(param.get("quant_mode", 0))
    active_expert_range = eval(param.get("active_expert_range", "[0,256]"))
    row_idx_type = int(param.get("row_idx_type", 0))
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    lo, hi = int(active_expert_range[0]), int(active_expert_range[1])
    if hi <= lo:
        raise ValueError("active_expert_range 必须满足 end > start")

    if dtype == torch.int8:
        x = torch.randint(-5, 5, size=x_shape, dtype=dtype, device=device)
    else:
        x = torch.randn(x_shape, dtype=dtype, device=device)

    # 专家下标落在 active 区间内，保证 golden 中 actual_expert_total_num 与算子语义一致
    expert_idx = torch.randint(lo, hi, size=expert_idx_shape, dtype=torch.int32, device=device)

    scale = torch.randn(scale_shape, dtype=torch.float32, device=device)
    offset = torch.randn(offset_shape, dtype=torch.float32, device=device)

    if quant_mode == -1 and drop_pad_mode == 1:
        scale = None
        offset = None

    return (
        x,
        expert_idx,
        scale,
        offset,
        active_num,
        expert_capacity,
        expert_num,
        drop_pad_mode,
        expert_tokens_num_type,
        expert_tokens_num_flag,
        quant_mode,
        active_expert_range,
        row_idx_type,
    )


def get_init_inputs(param, device=None):
    return []


def custom_check_precision(param, outputs, outputs_new):
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    if dtype == torch.float32:
        abs_e, rel_e = 1.0e-4, 1.0e-4
    elif dtype == torch.float16:
        abs_e, rel_e = 5.0e-3, 5.0e-3
    elif dtype == torch.bfloat16:
        abs_e, rel_e = 1.0e-2, 1.0e-2
    elif dtype == torch.int8:
        abs_e, rel_e = 1.0, 0.5
    else:
        abs_e, rel_e = 1.0e-4, 1.0e-4

    outs, outs_new = [], []
    for a, b in zip(outputs, outputs_new):
        if a is None or b is None:
            continue
        if a.numel() == 0 and b.numel() == 0:
            continue
        if a.dtype != b.dtype:
            b = b.to(a.dtype)
        outs.append(a)
        outs_new.append(b)

    if not outs:
        return 1, torch.tensor([0.0]), torch.tensor([0.0])

    return check_precision(outs, outs_new, max_abs_error=abs_e, max_rel_error=rel_e)
