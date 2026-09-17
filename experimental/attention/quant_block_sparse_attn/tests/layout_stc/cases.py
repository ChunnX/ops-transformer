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
        case["design_source"] = "layout_boundary"
    base = deepcopy(next(c for c in cases if c["layout_scenario"] == "Q129_SB128"))
    # Reference: note/stc设计/qbsa_mxfp8_stc.csv, 128-Q / 512-KV alignment.
    # EQ-UA cannot exist: a multiple of 512 is necessarily a multiple of 128.
    relations = (
        ("EQ_AA", 512, 512), ("EQ_AU", 128, 128), ("EQ_UU", 129, 129),
        ("LT_AA", 128, 512), ("LT_AU", 128, 513), ("LT_UA", 129, 512),
        ("LT_UU", 129, 513), ("GT_AA", 1024, 512), ("GT_AU", 1024, 513),
        ("GT_UA", 641, 512), ("GT_UU", 641, 257),
    )
    ranges = ([-1, 1], [-0.01, 1], [-0.01, 0.1], [0, 0.001],
              [0.001, 0.01], [1, 2], [-0.001, 0], [-1, -0.01],
              [-3, 3], [-5, 5], [0, 0])
    for i, (relation, sq, kv) in enumerate(relations):
        batch, nq, nkv = ((1, 3, 3), (2, 4, 2), (3, 5, 1))[i % 3]
        for layout in LAYOUTS:
            case = deepcopy(base)
            case.update(
                name=f"STC12177_REF_{relation}_{layout}",
                layout_scenario=f"REF_{relation}", test_tier="P1",
                design_source="reference_stc_relationship_and_range", B=batch,
                N1=nq, N2=nkv, S1_max=sq, seqused_kv=[kv] * batch,
                layout_q=layout, layout_out=layout, mask_mode=3 if i % 2 else 0,
                cu_seqlens_q=[b * sq for b in range(batch + 1)] if layout == "TND" else [],
                sparse_mode="random" if i % 3 == 0 else "dense", seed=1217700 + i,
                data_range_q=ranges[i], data_range_k=ranges[i], data_range_v=ranges[i],
                case_group="layout_stc",
            )
            cases.append(case)
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
