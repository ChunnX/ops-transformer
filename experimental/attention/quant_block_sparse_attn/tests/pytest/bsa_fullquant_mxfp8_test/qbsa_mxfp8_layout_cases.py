# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# See LICENSE in the root of the software repository for the full text of the License.
"""Matched TND/BSND/BNSD cases for the MXFP8 layout feature.

Use with qbsa_mxfp8_golden.py --case_files qbsa_mxfp8_layout_cases.py.
Executing this file prints comma-separated case names, optionally selected by tier
and layout, for the golden runner's --case_name argument. No NPU imports needed.
"""
import argparse
from copy import deepcopy


_BASE = {
    "enable": True,
    "B": 2, "N1": 4, "N2": 2, "D": 128,
    "cu_seqlens_q": [], "cu_seqlens_kv": [], "seqused_q": [],
    "seqused_kv": [257, 193], "S1_max": 129,
    "s2_base_size": 512, "block_size": 128, "max_block_per_batch": 0,
    "mask_mode": 0, "fp8_dtype": "float8_e4m3fn", "scale_dtype": "float8_e8m0fnu",
    "quant_group_size": 32, "layout_kv": "PA_BNBD",
    "layout_sparse_indices": "B_N_Qb_Kb", "kv_cache_layout": "BnNBsD",
    "p_scale_value": None, "p_scale_type": "float8_e8m0fnu",
    "softmax_scale": 128 ** -0.5, "return_softmax_lse": True,
    "seed": 11704, "data_range_q": [-1.0, 1.0],
    "data_range_k": [-1.0, 1.0], "data_range_v": [-1.0, 1.0],
    "device_id": 0, "sparse_q_block_size": 128, "sparse_kv_block_size": 128,
    "sparse_mode": "dense", "quant_mode": 2, "blocknum": 0,
}

TestCases = []


def _add_triplet(scenario, tier="P1", **overrides):
    logical = deepcopy(_BASE)
    logical.update(overrides)
    for layout in ("TND", "BSND", "BNSD"):
        case = deepcopy(logical)
        case.update(
            name=f"LAYOUT_{scenario}_{layout}", layout_q=layout, layout_out=layout,
            case_group="mxfp8_layout", layout_scenario=scenario, test_tier=tier,
        )
        if layout == "TND":
            case["cu_seqlens_q"] = [b * case["S1_max"] for b in range(case["B"] + 1)]
        TestCases.append(case)


# Around vector/tile boundaries, keep B>1 and N1>1 so swapped axes are observable.
for _block in (64, 128):
    for _sq in (1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 128, 129):
        _add_triplet(
            f"Q{_sq}_SB{_block}", tier="P0" if _sq == 129 else "P1",
            S1_max=_sq, block_size=_block,
            sparse_q_block_size=_block, sparse_kv_block_size=_block,
        )

# Dense, unmasked rows force coverage across 256/512 KV boundaries and >1 task.
for _kv in (255, 256, 257, 511, 512, 513, 1025):
    _add_triplet(
        f"KV{_kv}", tier="P0" if _kv == 513 else "P1",
        S1_max=17, seqused_kv=[_kv, _kv],
    )

_add_triplet("MHA_B3", B=3, N1=3, N2=3, S1_max=65, seqused_kv=[257, 193, 129])
_add_triplet("MQA_G5", N1=5, N2=1)
_add_triplet("GQA_G5_B3", tier="P0", B=3, N1=15, N2=3, S1_max=257,
             seqused_kv=[513, 491, 257])
_add_triplet("CAUSAL_LT", mask_mode=3, seqused_kv=[513, 193])
_add_triplet("CAUSAL_EQ", mask_mode=3, seqused_kv=[129, 129])
_add_triplet("CAUSAL_GT", tier="P0", S1_max=257, mask_mode=3, seqused_kv=[129, 193])
_add_triplet("KV_ZERO_BATCH", seqused_kv=[0, 257])
_add_triplet("PA512_SHARED", block_size=512, sparse_q_block_size=64,
             sparse_kv_block_size=64, seqused_kv=[1025, 769], blocknum=2)
_add_triplet("SPARSE_RANDOM", S1_max=257, seqused_kv=[1025, 513], sparse_mode="random")
_add_triplet("NO_LSE", tier="P0", block_size=64, sparse_q_block_size=64,
             sparse_kv_block_size=64, mask_mode=3, return_softmax_lse=False)
_add_triplet("P_E8M0", p_scale_value=0.03125)
_add_triplet("P_FP32", p_scale_value=0.75, p_scale_type="float32")
_add_triplet("SOFTMAX_SCALE", softmax_scale=0.125)
_add_triplet("B1", B=1, seqused_kv=[257])
_add_triplet("N1", N1=1, N2=1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("P0", "P1", "all"), default="all")
    parser.add_argument("--layout", choices=("TND", "BSND", "BNSD", "all"), default="all")
    args = parser.parse_args()
    print(",".join(case["name"] for case in TestCases
                   if (args.tier == "all" or case["test_tier"] == args.tier)
                   and (args.layout == "all" or case["layout_q"] == args.layout)))
