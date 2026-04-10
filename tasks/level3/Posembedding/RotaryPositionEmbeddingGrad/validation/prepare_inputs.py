import os
import random
from typing import Any, List, Optional, Tuple

import torch

from framework.utils import check_precision

_DEFAULT_PREPARE_RNG_BASE = 41


def _case_id_as_int(param: Any) -> int:
    cid = param.get("case_id", 0)
    try:
        import pandas as pd

        if pd.isna(cid):
            return 0
    except Exception:
        pass
    return int(cid)


def _seed_prepare_rng(param: Any) -> None:
    base = int(os.environ.get("NKB_PREPARE_INPUTS_SEED", str(_DEFAULT_PREPARE_RNG_BASE)))
    seed = (base + _case_id_as_int(param) * 7919) % (2**31)
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


def _dtype_from_str(name: str) -> torch.dtype:
    name = str(name).strip().lower()
    if name in ("float16", "fp16"):
        return torch.float16
    if name in ("bfloat16", "bf16"):
        return torch.bfloat16
    if name in ("float32", "fp32"):
        return torch.float32
    return torch.float16


def get_init_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[int]:
    mode = int(param.get("mode", 0))
    return (mode,)


def get_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[torch.Tensor, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    b = int(param["B"])
    n = int(param["N"])
    s = int(param["S"])
    d = int(param["D"])
    dtype = _dtype_from_str(param.get("dtype", "float16"))
    mode = int(param.get("mode", 0))

    if mode in (0, 1) and d % 2 != 0:
        raise ValueError("mode 0/1 require even D")

    shape = (b, n, s, d)
    dy = (torch.rand(shape, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    x = (torch.rand(shape, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    cos_t = (torch.rand(1, 1, s, d, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    sin_t = (torch.rand(1, 1, s, d, device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    dx = torch.empty_like(dy)
    dcos = torch.empty_like(cos_t)
    dsin = torch.empty_like(sin_t)
    return (dy, cos_t, sin_t, x, dx, dcos, dsin)


def custom_check_precision(
    param: Any,
    ref_output: List[torch.Tensor],
    test_output: List[torch.Tensor],
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    dtype_str = str(param.get("dtype", "float16"))
    if dtype_str == "bfloat16":
        atol, rtol = 2e-2, 2e-2
    elif dtype_str == "float32":
        atol, rtol = 1e-4, 1e-4
    else:
        atol, rtol = 5e-3, 5e-3

    outs = ref_output if isinstance(ref_output, (list, tuple)) else [ref_output]
    outs_n = test_output if isinstance(test_output, (list, tuple)) else [test_output]
    acc = 1
    all_abs: List[torch.Tensor] = []
    all_rel: List[torch.Tensor] = []
    for o, n in zip(outs, outs_n):
        a, ad, rd = check_precision([o], [n], atol, rtol)
        acc = 1 if (acc == 1 and a == 1) else 0
        all_abs.append(ad)
        all_rel.append(rd)
    return acc, torch.cat(all_abs), torch.cat(all_rel)
