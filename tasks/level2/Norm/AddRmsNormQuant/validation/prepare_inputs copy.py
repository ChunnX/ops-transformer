import os
import random

import torch
from framework.utils import check_precision

# 与 DynamicQuantUpdateScatter 等任务一致：按 case_id 固定 RNG，避免多次跑批时同一 case 随机输入不同导致精度时好时坏。
_DEFAULT_PREPARE_RNG_BASE = 34


def _case_id_as_int(param):
    cid = param.get("case_id", 0)
    try:
        import pandas as pd

        if pd.isna(cid):
            return 0
    except Exception:
        pass
    return int(cid)


def _seed_prepare_rng(param, stream_id: int):
    base = int(os.environ.get("NKB_PREPARE_INPUTS_SEED", str(_DEFAULT_PREPARE_RNG_BASE)))
    cid = _case_id_as_int(param)
    seed = (base + cid * 7919 + stream_id * 9973) % (2**31)
    torch.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np

        np.random.seed(seed)
    except Exception:
        pass
    try:
        torch.npu.manual_seed(seed)
    except Exception:
        pass


def _csv_bool(param, key, default=True):
    v = param.get(key, default)
    if isinstance(v, bool):
        return v
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ("false", "0", "no"):
        return False
    return True


def get_inputs(param, device=None):
    """
    根据 DataFrame 行中的参数生成模型的输入张量列表和标量。
    """
    _seed_prepare_rng(param, stream_id=1)
    shape = eval(param.get('input_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    # x1 = torch.rand(shape, device=device, dtype=dtype)
    # x2 = torch.rand(shape, device=device, dtype=dtype)
    
    x1 = (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)  # 区间 [-1, 1)
    x2 = (torch.rand(shape, device=device, dtype=dtype) * 2 - 1)

    return (x1, x2)


def get_init_inputs(param, device=None):
    """
    Extract initialization parameters for the model from DataFrame row.
    No special initialization needed for foreach_mul_scalar.

    Args:
        param (dict): Parameters from a pandas DataFrame row

    Returns:
        list: Empty list as no special initialization inputs needed
    """
    _seed_prepare_rng(param, stream_id=0)
    shape = eval(param.get('normalized_shape', '[1]'))
    dtype_str = param.get('dtype', 'float16')
    dtype = getattr(torch, dtype_str)

    scales_dtype_str = param.get('scales_dtype', 'float32')
    scales_dtype = getattr(torch, scales_dtype_str)

    zeros_dtype_str = param.get('zeros_dtype', 'int32')
    zeros_dtype = getattr(torch, zeros_dtype_str)

    gamma = torch.rand(shape, device=device, dtype=dtype)

    scales1 = torch.rand(shape, device=device, dtype=scales_dtype)
    scales2 = None # c++底层实现默认为空

    has_zp1 = _csv_bool(param, "has_zero_points1", True)
    zero_points1 = (
        torch.rand(shape, device=device, dtype=zeros_dtype) if has_zp1 else None
    )
    zero_points2 = None # c++底层实现默认为空

    axis = -1

    epsilon = param.get('epsilon', 1e-5)
    div_mode = _csv_bool(param, "div_mode", True)

    return [gamma, scales1, scales2, zero_points1, zero_points2, axis, epsilon, div_mode]

def custom_check_precision(param, outputs, outputs_new):
    is_pass = 1
    LOSS = 3e-3
    all_abs_diff, all_rel_diff = [], []
    for out, out_new in zip(outputs, outputs_new):
        result = torch.abs(out - out_new)

        # Get maximum of absolute values for relative error denominator
        deno = torch.maximum(torch.abs(out), torch.abs(out_new))

        # Calculate absolute error check
        result_atol = result

        # Calculate relative error check
        result_rtol = result / (deno + 1e-7)
        print(torch.sum(result_rtol > LOSS))
        print(torch.sum(result_atol > LOSS))

        # Count failures where error exceeds tolerance
        if torch.sum(result_rtol > LOSS) > out.numel() * LOSS and \
        torch.sum(result_atol > LOSS) > out.numel() * LOSS:
            is_pass = 0
        all_abs_diff.append(result_atol)
        all_rel_diff.append(result_rtol)
    all_abs_diff = torch.cat(all_abs_diff)
    all_rel_diff = torch.cat(all_rel_diff)
    return is_pass, all_abs_diff, all_rel_diff    