# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0.
# See LICENSE in the root of the software repository.
"""PR 12177 shared STC definition; standard library only, safe on a CPU host."""
from copy import deepcopy
from pathlib import Path
import runpy

TESTS = Path(__file__).resolve().parents[1]
GOLDEN_DIR = TESTS / "pytest" / "bsa_fullquant_mxfp8_test"
PR_HEAD = "af49ad74f2f72043f837c14203bcf5f2b247c2b3"
LAYOUTS = ("TND", "BSND", "BNSD")


def _cases():
    existing = runpy.run_path(str(GOLDEN_DIR / "qbsa_mxfp8_layout_cases.py"))["TestCases"]
    cases = deepcopy(existing)
    for case in cases:
        case["name"] = case["name"].replace("LAYOUT_", "STC12177_", 1)
        case["design_source"] = "compact_layout_boundary_and_relationship"
    return cases


TestCases = _cases()


def select_cases(tier="all", layout="all"):
    return [deepcopy(c) for c in TestCases
            if (tier == "all" or c["test_tier"] == tier)
            and (layout == "all" or c["layout_q"] == layout)]


def ceil_div(n, d):
    return (n + d - 1) // d


def tensor_shapes(case):
    """TTK's 16 positional inputs, in bsa_ttk_ops signature order."""
    b, s, n, nk, d = (case[k] for k in ("B", "S1_max", "N1", "N2", "D"))
    pa, sb = case["block_size"], case["sparse_kv_block_size"]
    pages = case["blocknum"] or sum(ceil_div(k, pa) for k in case["seqused_kv"])
    width = max(case["max_block_per_batch"], max(ceil_div(k, pa) for k in case["seqused_kv"]))
    qb, kb = ceil_div(s, case["sparse_q_block_size"]), max(ceil_div(k, sb) for k in case["seqused_kv"])
    q = {"TND": (b * s, n, d), "BSND": (b, s, n, d), "BNSD": (b, n, s, d)}[case["layout_q"]]
    return (q, (pages, nk, pa, d), (pages, nk, pa, d), (*q[:-1], 2, 2),
            (pages, nk, pa, 2, 2), (pages, nk, pa // 64, d, 2),
            (b, n, qb, kb), (b, n, qb), (0,) if case["p_scale_value"] is None else (1,),
            (b + 1,) if case["layout_q"] == "TND" else (0,), (0,), (0,), (b,),
            (b, width), (2048, 2048) if case["mask_mode"] == 3 else (0,), (0,))


def ttk_attributes(case):
    return {
        "p_scale_value": case["p_scale_value"],
        **{k + "_value": case[k] for k in ("cu_seqlens_q", "cu_seqlens_kv", "seqused_q", "seqused_kv")},
        "quant_mode": 2, "softmax_scale": case["softmax_scale"], "mask_mode": case["mask_mode"],
        "blocksize": case["block_size"], "sparse_block_size_q": case["sparse_q_block_size"],
        "sparse_block_size_kv": case["sparse_kv_block_size"], "layout_q": case["layout_q"],
        "layout_kv": "PA_BNBD", "layout_out": case["layout_out"],
        "layout_sparse_indices": "B_N_Qb_Kb", "return_softmax_lse": case["return_softmax_lse"],
        "batch_size": case["B"], "num_heads_q": case["N1"], "num_heads_kv": case["N2"],
        "head_dim": 128, "sparse_mode": case["sparse_mode"], "seed": case["seed"],
    }
