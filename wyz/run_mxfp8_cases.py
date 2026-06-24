#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Run the four FLOAT8_E4M3FN per-block GQA cases from the local testcase table.
"""

from mxfp8_golden import run_mxfp8_case


CASES = [
    # {
    #     "name": "FP16_FLOAT8_E4M3FN_FP16_1_1_1_64_2048_55_BNSD_GQA_preto1024_nxtto1024_pre0_000001",
    #     "B": 1,
    #     "Nq": 10,
    #     "Nkv": 1,
    #     "Sq": 64,
    #     "Skv": 192,
    #     "D": 128,
    #     "D_rope":64,
    #     "scale": 0.08838834764831843,
    #     "layout": "TND",
    # },
    {
        "name": "b_1_s1_1_s2_256_PA_sparse3_n1_1_n2_1_rope",
        "B": 1,
        "Nq": 1,
        "Nkv": 1,
        "Sq": 1,
        "Skv": 256,
        "D": 128,
        "enable_rope": False,
        "D_rope":64,
        "scale": 0.08838834764831843,
        "layout": "TND",
        "enable_pa": True,
        "kv_cache_layout": "BnNBsD",
        "block_size": 512,
        "sparse_mode": 3,
    },
    # {
    #     "name": "FP16_FLOAT8_E4M3FN_FP16_1_14_1_64_2048_64_BNSD_GQA_preto1024_nxtto1024_pre0_000002",
    #     "B": 1,
    #     "Nq": 14,
    #     "Nkv": 1,
    #     "Sq": 64,
    #     "Skv": 2048,
    #     "D": 64,
    #     "scale": 0.125,
    #     "layout": "BNSD",
    # },
    # {
    #     "name": "FP16_FLOAT8_E4M3FN_FP16_1_1_1_64_512_128_BNSD_GQA_preto1024_nxtto1024_pre0_000003",
    #     "B": 1,
    #     "Nq": 1,
    #     "Nkv": 1,
    #     "Sq": 64,
    #     "Skv": 2048,
    #     "D": 128,
    #     "scale": 0.08838834764831843,
    #     "layout": "BNSD",
    # },
    # {
    #     "name": "FP16_FLOAT8_E4M3FN_FP16_2_7_1_128_2048_64_BNSD_GQA_preto1024_nxtto1024_pre0_000004",
    #     "B": 2,
    #     "Nq": 7,
    #     "Nkv": 1,
    #     "Sq": 128,
    #     "Skv": 2048,
    #     "D": 64,
    #     "scale": 0.125,
    #     "layout": "BNSD",
    # },
]


def main():
    passed_count = 0
    total_count = len(CASES)
    for case in CASES:
        result = run_mxfp8_case(case=case, run_npu=True, catch_npu_error=True)
    #     status = "PASSED" if result["passed"] else "FAILED"
    #     if result["npu_out"] is None:
    #         status = "NPU_SKIPPED"
    #     if result["passed"]:
    #         passed_count += 1
    #     print(f"[SUMMARY] {case['name']} {status}")
    # print(f"[SUMMARY] TOTAL {passed_count}/{total_count} PASSED")


if __name__ == "__main__":
    main()
