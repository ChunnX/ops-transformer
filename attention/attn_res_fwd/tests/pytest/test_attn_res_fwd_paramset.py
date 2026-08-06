# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------
"""AttnResFwd NPU 精度用例表（H=7168 为主）。"""

from __future__ import annotations

from typing import Any, Dict, List

Case = Dict[str, Any]

# 生产
PRODUCTION_CASE: Case = {
    "id": "prod",
    "T": 544,
    "N": 12,
    "H": 7168,
    "seed": 0,
    "expect_branch": "RELOAD/40010",
}

# 历史失败 / 边界（回归）
PREVIOUS_FAIL_CASES: List[Case] = [
    {"id": "fail_inf", "T": 1024, "N": 1, "H": 7168, "seed": 0, "expect_branch": "RELOAD/40010"},
    {"id": "fail_huge_4096_4", "T": 4096, "N": 4, "H": 7168, "seed": 0, "expect_branch": "RELOAD/40010"},
    {"id": "fail_huge_6144_1", "T": 6144, "N": 1, "H": 7168, "seed": 0, "expect_branch": "RELOAD/40010"},
]

# pick10（T×N 采样）
PICK10_CASES: List[Case] = [
    {"id": "pick_1_1", "T": 1, "N": 1, "H": 7168, "seed": 0},
    {"id": "pick_1_8", "T": 1, "N": 8, "H": 7168, "seed": 0},
    {"id": "pick_16_4", "T": 16, "N": 4, "H": 7168, "seed": 0},
    {"id": "pick_128_2", "T": 128, "N": 2, "H": 7168, "seed": 0},
    {"id": "pick_512_6", "T": 512, "N": 6, "H": 7168, "seed": 0},
    {"id": "pick_1024_1", "T": 1024, "N": 1, "H": 7168, "seed": 0},
    {"id": "pick_2048_8", "T": 2048, "N": 8, "H": 7168, "seed": 0},
    {"id": "pick_4096_4", "T": 4096, "N": 4, "H": 7168, "seed": 0},
    {"id": "pick_8192_2", "T": 8192, "N": 2, "H": 7168, "seed": 0},
    {"id": "pick_16384_8", "T": 16384, "N": 8, "H": 7168, "seed": 0},
]

# 驻留分支（小 H）
RESIDENT_CASES: List[Case] = [
    {"id": "resident_512", "T": 544, "N": 12, "H": 512, "seed": 0, "expect_branch": "RESIDENT/40020"},
    {"id": "resident_256", "T": 1, "N": 12, "H": 256, "seed": 0, "expect_branch": "RESIDENT/40020"},
]

# 正向 H 扫描（T=1, N=12, seed=0）
FORWARD_H_SWEEP_CASES: List[Case] = [
    {"id": f"h_sweep_{h}", "T": 1, "N": 12, "H": h, "seed": 0}
    for h in (64, 128, 256, 512, 1024, 2048, 4096, 7168)
]

# 泛化精度：H=7168，N=1..12，T 对数采样覆盖 [1, 16384]
# 全整数枚举约 19 万 case，不可行；采样含边界 / 生产 / 历史失败 T
GRID_T_VALUES = (
    1,
    2,
    4,
    8,
    16,
    32,
    64,
    128,
    256,
    512,
    544,
    1024,
    2048,
    4096,
    6144,
    8192,
    12288,
    16384,
)
GRID_N_VALUES = tuple(range(1, 13))


def build_grid_tn_7168_cases() -> List[Case]:
    cases: List[Case] = []
    for t in GRID_T_VALUES:
        for n in GRID_N_VALUES:
            cases.append(
                {
                    "id": f"grid_t{t}_n{n}",
                    "T": t,
                    "N": n,
                    "H": 7168,
                    "seed": 0,
                    "expect_branch": "RELOAD/40010",
                }
            )
    return cases


GRID_TN_7168_CASES: List[Case] = build_grid_tn_7168_cases()

# CI 仅跑生产；扩展集 = pick10 + 历史失败
CI_NPU_CASES: List[Case] = [PRODUCTION_CASE]
EXTENDED_NPU_CASES: List[Case] = PICK10_CASES + PREVIOUS_FAIL_CASES
