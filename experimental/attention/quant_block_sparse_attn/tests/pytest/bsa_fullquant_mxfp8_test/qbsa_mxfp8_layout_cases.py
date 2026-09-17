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


# Twelve small scenarios, each expanded into three layouts: 36 cases total.
# Most cases keep B>1, N1>1 and S!=N1 so an axis swap is observable.
_add_triplet("Q1_N1", B=1, N1=1, N2=1, S1_max=1, seqused_kv=[1],
             data_range_q=[0, 0], data_range_k=[0, 0], data_range_v=[0, 0])
_add_triplet("Q63_SB64", S1_max=63, seqused_kv=[127, 65], block_size=64,
             sparse_q_block_size=64, sparse_kv_block_size=64)
_add_triplet("Q64_SB64_MHA", N1=2, N2=2, S1_max=64, seqused_kv=[128, 64],
             block_size=64, sparse_q_block_size=64, sparse_kv_block_size=64)
_add_triplet("Q65_SB64_MQA", tier="P0", N2=1, S1_max=65, seqused_kv=[129, 65],
             block_size=64, sparse_q_block_size=64, sparse_kv_block_size=64,
             p_scale_value=0.03125)
_add_triplet("Q127_SB128", S1_max=127, seqused_kv=[255, 129])
_add_triplet("Q128_SB128", S1_max=128, seqused_kv=[256, 128])
_add_triplet("Q129_SB128", tier="P0")
_add_triplet("CAUSAL_LT", S1_max=17, mask_mode=3, seqused_kv=[129, 65])
_add_triplet("CAUSAL_EQ", N1=2, N2=2, S1_max=65, mask_mode=3, seqused_kv=[65, 65],
             data_range_q=[-0.01, 0.1], data_range_k=[-0.01, 0.1], data_range_v=[-0.01, 0.1])
_add_triplet("CAUSAL_GT", tier="P0", mask_mode=3, seqused_kv=[65, 97])
_add_triplet("KV_ZERO_NO_LSE", tier="P0", S1_max=17, seqused_kv=[0, 129],
             return_softmax_lse=False)
_add_triplet("PA256_RANDOM_P_FP32", S1_max=65, block_size=256, sparse_q_block_size=64,
             sparse_kv_block_size=64, blocknum=2, sparse_mode="random",
             p_scale_value=0.75, p_scale_type="float32", softmax_scale=0.125)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", choices=("P0", "P1", "all"), default="all")
    parser.add_argument("--layout", choices=("TND", "BSND", "BNSD", "all"), default="all")
    args = parser.parse_args()
    print(",".join(case["name"] for case in TestCases
                   if (args.tier == "all" or case["test_tier"] == args.tier)
                   and (args.layout == "all" or case["layout_q"] == args.layout)))
