import os
import random
from typing import Any, List, Optional, Tuple

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
    seed = (base + _case_id_as_int(param)) % (2**31)
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


def get_init_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[Any, ...]:
    row = param
    return (
        float(row["scale_value"]),
        int(row["head_num"]),
        float(row["keep_prob"]),
        int(row["pre_tockens"]),
        int(row["next_tockens"]),
        int(row["inner_precise"]),
        int(row["sparse_mode"]),
    )


def get_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[Any, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    b = int(param["B"])
    n = int(param["N"])
    s1 = int(param["S"])
    s2 = int(param["S"])
    d = int(param["D"])
    head_num = int(param["head_num"])
    if head_num != n:
        raise ValueError(f"head_num({head_num}) 须等于 N({n}) for BNSD 用例")

    g_rng = torch.Generator(device=dev)
    g_rng.manual_seed((_case_id_as_int(param) * 10007 + 13) % (2**31))

    query = torch.randn(b, n, s1, d, dtype=torch.float16, device=dev, generator=g_rng)
    key = torch.randn(b, n, s2, d, dtype=torch.float16, device=dev, generator=g_rng)
    value = torch.randn(b, n, s2, d, dtype=torch.float16, device=dev, generator=g_rng)

    softmax_max = torch.zeros(b, n, s1, 8, dtype=torch.float32, device=dev)
    softmax_sum = torch.zeros(b, n, s1, 8, dtype=torch.float32, device=dev)
    softmax_out = torch.zeros(b, n, s1, s2, dtype=torch.float16, device=dev)
    attention_out = torch.zeros(b, n, s1, d, dtype=torch.float16, device=dev)

    return (query, key, value, softmax_max, softmax_sum, softmax_out, attention_out)


def custom_check_precision(
    param: Any,
    ref_output: List[torch.Tensor],
    test_output: List[torch.Tensor],
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """FlashAttention 数值与朴素实现存在算法差异，主要校验 attention_out。"""
    del param
    outs = ref_output if isinstance(ref_output, (list, tuple)) else [ref_output]
    outs_n = test_output if isinstance(test_output, (list, tuple)) else [test_output]
    return check_precision(outs[3], outs_n[3], max_abs_error=0.08, max_rel_error=0.05)
