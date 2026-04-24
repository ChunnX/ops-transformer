import random
from typing import Optional, Tuple

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


def _expert_counts_sum_to_a(
    A: int, N: int, E: int, device: Optional[torch.device], case_id: int
) -> torch.Tensor:
    """生成 (N,E) 非负整型，元素和为 A（与 token 条数一致）。"""
    g = torch.Generator(device="cpu")
    g.manual_seed(case_id)
    total = N * E
    if total <= 0:
        raise ValueError("N*E 必须为正")
    w = torch.rand(total, generator=g)
    draws = torch.multinomial(w, A, replacement=True)
    flat = torch.bincount(draws, minlength=total).to(torch.int32)
    out = flat.view(N, E)
    if device is not None:
        out = out.to(device)
    return out


def get_inputs(param, device=None) -> Tuple:
    case_id = int(param.get("case_id", 0))
    _seed(case_id)

    A = int(param.get("A", 4))
    H = int(param.get("H", 2))
    N = int(param.get("N", 16))
    E = int(param.get("E", 16))
    idx_type = int(param.get("idx_type", 0))
    expert_token_num_type = int(param.get("expert_token_num_type", 1))
    has_scales = _parse_bool(param.get("has_scales", False), False)
    dtype_str = param.get("dtype", "float16")
    dtype = getattr(torch, dtype_str)

    if dtype == torch.int8:
        tokens = torch.randint(-5, 5, size=(A, H), dtype=dtype, device=device)
    else:
        tokens = torch.randn(A, H, dtype=dtype, device=device)

    expert_token_num_per_rank = _expert_counts_sum_to_a(A, N, E, device, case_id)

    per_token_scales = None
    if has_scales:
        g = torch.Generator(device="cpu")
        g.manual_seed(case_id + 17)
        per_token_scales = torch.randn(A, generator=g, dtype=torch.float32, device="cpu")
        if device is not None:
            per_token_scales = per_token_scales.to(device)

    return (tokens, expert_token_num_per_rank, per_token_scales, expert_token_num_type, idx_type)


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

    float_ref, float_new = [], []
    for a, b in zip(outputs, outputs_new):
        if b.dtype != a.dtype:
            b = b.to(a.dtype)
        if a.dtype in (torch.int32, torch.int64):
            if not torch.equal(a.cpu(), b.cpu()):
                z = torch.tensor([1.0])
                return 0, z, z
            continue
        float_ref.append(a.float() if a.dtype == torch.int8 else a)
        float_new.append(b.float() if b.dtype == torch.int8 else b)

    if not float_ref:
        return 1, torch.tensor([0.0]), torch.tensor([0.0])

    return check_precision(float_ref, float_new, max_abs_error=abs_e, max_rel_error=rel_e)
