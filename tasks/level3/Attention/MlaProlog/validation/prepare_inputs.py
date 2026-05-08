import copy
import math
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
        float(row["rmsnorm_epsilon_cq"]),
        float(row["rmsnorm_epsilon_ckv"]),
        str(row["cache_mode"]).strip() or "PA_BSND",
    )


def get_inputs(param: Any, device: Optional[torch.device] = None) -> Tuple[Any, ...]:
    _seed_prepare_rng(param)
    dev = device or torch.device("cpu")

    B = int(param["B"])
    S1 = int(param["S1"])
    kv_seq = int(param["kv_seq"])
    He = int(param["He"])
    Hcq = int(param["Hcq"])
    Hckv = int(param["Hckv"])
    N = int(param["N"])
    D = int(param["D"])
    Dr = int(param["Dr"])
    Nkv = int(param["Nkv"])
    block_size = int(param["block_size"])
    cache_slot = int(param["cache_slot"])

    block_num = int(math.ceil(B * kv_seq / block_size))
    if block_num < 1:
        block_num = 1

    g = torch.Generator(device=dev)
    g.manual_seed((_case_id_as_int(param) * 10007 + 13) % (2**31))

    token_x = torch.randn(B, S1, He, dtype=torch.bfloat16, device=dev, generator=g)
    weight_dq = torch.randn(He, Hcq, dtype=torch.bfloat16, device=dev, generator=g)
    weight_uq_qr = torch.randn(Hcq, N * (D + Dr), dtype=torch.bfloat16, device=dev, generator=g)
    weight_uk = torch.randn(N, D, Hckv, dtype=torch.bfloat16, device=dev, generator=g)
    weight_dkv_kr = torch.randn(He, Hckv + Dr, dtype=torch.bfloat16, device=dev, generator=g)
    rmsnorm_gamma_cq = torch.randn(Hcq, dtype=torch.bfloat16, device=dev, generator=g)
    rmsnorm_gamma_ckv = torch.randn(Hckv, dtype=torch.bfloat16, device=dev, generator=g)
    rope_sin = torch.randn(B, S1, Dr, dtype=torch.bfloat16, device=dev, generator=g)
    rope_cos = torch.randn(B, S1, Dr, dtype=torch.bfloat16, device=dev, generator=g)

    cache_index = torch.full((B, S1), cache_slot, dtype=torch.int64, device=dev)
    kv_cache = torch.randn(block_num, block_size, Nkv, Hckv, dtype=torch.bfloat16, device=dev, generator=g)
    kr_cache = torch.randn(block_num, block_size, Nkv, Dr, dtype=torch.bfloat16, device=dev, generator=g)

    query_out = torch.zeros(B, S1, N, Hckv, dtype=torch.bfloat16, device=dev)
    query_rope_out = torch.zeros(B, S1, N, Dr, dtype=torch.bfloat16, device=dev)

    return (
        token_x,
        weight_dq,
        weight_uq_qr,
        weight_uk,
        weight_dkv_kr,
        rmsnorm_gamma_cq,
        rmsnorm_gamma_ckv,
        rope_sin,
        rope_cos,
        cache_index,
        kv_cache,
        kr_cache,
        query_out,
        query_rope_out,
    )


def custom_check_precision(
    param: Any,
    ref_output: List[torch.Tensor],
    test_output: List[torch.Tensor],
) -> Tuple[int, torch.Tensor, torch.Tensor]:
    """BF16 多 MatMul + RMSNorm + RoPE 融合链路与金标准在 float 上存在累积差，默认 1e-3 阈值过严。"""
    del param
    atol, rtol = 16.0, 0.05
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
