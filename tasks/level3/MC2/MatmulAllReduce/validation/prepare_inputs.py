import os
import random
from typing import Any, Optional, Tuple

import torch

from npukernelbench.stages.precision.comparator import check_precision

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
    return torch.float16


def _parse_bool(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    return str(v).strip().lower() in ("true", "1", "yes")


def get_init_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[str, str, int]:
    group = str(param.get("group", "hccl_world_group"))
    reduce_op = str(param.get("reduce_op", "sum"))
    comm_turn = int(param.get("comm_turn", 0))
    return (group, reduce_op, comm_turn)


def get_inputs(
    param: Any, device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], str, str, int]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    m = int(param.get("M", 1024))
    k = int(param.get("K", 1024))
    n = int(param.get("N", 1024))
    dtype = _dtype_from_str(param.get("dtype", "float16"))
    has_bias = _parse_bool(param.get("has_bias", False))

    group = str(param.get("group", "hccl_world_group"))
    reduce_op = str(param.get("reduce_op", "sum"))
    comm_turn = int(param.get("comm_turn", 0))

    x1 = (torch.rand((m, k), device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)
    x2 = (torch.rand((k, n), device=dev, dtype=torch.float32) * 2.0 - 1.0).to(dtype=dtype)

    bias = None
    if has_bias:
        bias = (torch.rand((n,), device=dev, dtype=torch.float32) * 0.2 - 0.1).to(dtype=dtype)

    return (x1, x2, bias, group, reduce_op, comm_turn)


def custom_check_precision(param: Any, outputs, outputs_new):
    return check_precision(outputs, outputs_new, max_abs_error=0.02, max_rel_error=0.02)
