#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Run MXFP8 golden cases and print a final pass summary."""

import argparse
import importlib
import subprocess
import sys


CASES = [
    {
        "name": "PA_NZ_S31_Q4_Nq1_Nkv1_D128",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "actual_seq_q": [4],
        "actual_seq_kv": [31],
        "enable_pa": True,
        "kv_cache_layout": "PA_NZ",
        "block_size": 512,
        "sparse_mode": 3,
        "p_scale": 1.0,
    },
    {
        "name": "PA_NZ_S48_Q48_Nq1_Nkv1_D128",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "actual_seq_q": [48],
        "actual_seq_kv": [48],
        "enable_pa": True,
        "kv_cache_layout": "PA_NZ",
        "block_size": 512,
        "sparse_mode": 3,
        "p_scale": 1.0,
    },
    {
        "name": "PA_NZ_S64_Q64_Nq1_Nkv1_D128",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "actual_seq_q": [64],
        "actual_seq_kv": [64],
        "enable_pa": True,
        "kv_cache_layout": "PA_NZ",
        "block_size": 512,
        "sparse_mode": 3,
        "p_scale": 1.0,
    },
    {
        "name": "PA_NZ_S66_Q66_Nq1_Nkv1_D128",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "actual_seq_q": [66],
        "actual_seq_kv": [66],
        "enable_pa": True,
        "kv_cache_layout": "PA_NZ",
        "block_size": 512,
        "sparse_mode": 3,
        "p_scale": 1.0,
    },
    {
        "name": "PA_NZ_S86_Q86_Nq1_Nkv1_D128",
        "B": 1,
        "N_q": 1,
        "N_kv": 1,
        "D": 128,
        "actual_seq_q": [86],
        "actual_seq_kv": [86],
        "enable_pa": True,
        "kv_cache_layout": "PA_NZ",
        "block_size": 512,
        "sparse_mode": 3,
        "p_scale": 1.0,
    },
    {
        "name": "TND_S48_Q48_Nq8_Nkv1_D128_AUTO_TND",
        "B": 1,
        "N_q": 8,
        "N_kv": 1,
        "D": 128,
        "actual_seq_q": [48],
        "actual_seq_kv": [64],
        "enable_pa": False,
        "kv_cache_layout": "BnNBsD",
        "block_size": 512,
        "sparse_mode": 3,
        "p_scale": 1.0,
    },
    {
        "name": "TND_S66_Q66_Nq8_Nkv1_D128_AUTO_TND",
        "B": 1,
        "N_q": 8,
        "N_kv": 1,
        "D": 128,
        "actual_seq_q": [66],
        "actual_seq_kv": [86],
        "enable_pa": False,
        "kv_cache_layout": "BnNBsD",
        "block_size": 512,
        "sparse_mode": 3,
        "p_scale": 1.0,
    },
    {
        "name": "TND_S48_Q48_Nq8_Nkv1_D128_SPARSE4_AUTO_TND",
        "B": 1,
        "N_q": 8,
        "N_kv": 1,
        "D": 128,
        "actual_seq_q": [48],
        "actual_seq_kv": [64],
        "enable_pa": False,
        "kv_cache_layout": "BnNBsD",
        "q_scale_layout": "TND",
        "block_size": 512,
        "sparse_mode": 4,
        "pre_tokens": 32,
        "next_tokens": 8,
        "p_scale": 1.0,
    },
    {
        "name": "PA_NZ_S1023_Q1023_Nq80_Nkv8_D128_AUTO_TND",
        "B": 1,
        "N_q": 80,
        "N_kv": 8,
        "D": 128,
        "actual_seq_q": [1023],
        "actual_seq_kv": [1024],
        "enable_pa": True,
        "kv_cache_layout": "PA_NZ",
        "block_size": 512,
        "sparse_mode": 3,
        "p_scale": 1.0,
    },
]


def _set_case(module, case):
    module.B = int(case["B"])
    module.N_q = int(case["N_q"])
    module.N_kv = int(case["N_kv"])
    module.D = int(case["D"])
    module.ACTUAL_SEQ_Q = [int(item) for item in case["actual_seq_q"]]
    module.ACTUAL_SEQ_KV = [int(item) for item in case["actual_seq_kv"]]
    module.ENABLE_PA = bool(case.get("enable_pa", module.ENABLE_PA))
    module.BLOCK_SIZE = int(case.get("block_size", module.BLOCK_SIZE))
    module.SPARSE_MODE = int(case.get("sparse_mode", module.SPARSE_MODE))
    module.PRE_TOKENS = int(case.get("pre_tokens", 2147483647))
    module.NEXT_TOKENS = int(case.get("next_tokens", 0))
    module.INPUT_LAYOUT = case.get("input_layout", "TND")
    module.Q_SCALE_LAYOUT = case.get("q_scale_layout", "AUTO")
    module.KV_CACHE_LAYOUT = case.get("kv_cache_layout", "PA_NZ")
    module.P_SCALE = float(case["p_scale"])


def _format_compare_error(compare_result):
    if isinstance(compare_result, (list, tuple)):
        return "compare_result=" + ", ".join(str(item) for item in compare_result)
    return f"compare_result={compare_result}"


def run_case(module, case, run_npu=False):
    _set_case(module, case)
    try:
        data = module.generate_data()
        q_fp8, k_fp8, v_fp8, deq_q, deq_k, deq_v, p_scale, qr_bf16, kr_bf16, block_table_torch = data
        cpu_out = module.cpu_mxfp8_golden(
            q_fp8,
            k_fp8,
            v_fp8,
            deq_q,
            deq_k,
            deq_v,
            p_scale,
            module.ACTUAL_SEQ_Q,
            module.ACTUAL_SEQ_KV,
            qr_bf16,
            kr_bf16,
        )
        if run_npu:
            atten_out = module.npu_mxfp8_fa(
                q_fp8,
                k_fp8,
                v_fp8,
                deq_q,
                deq_k,
                deq_v,
                p_scale,
                module.ACTUAL_SEQ_Q,
                module.ACTUAL_SEQ_KV,
                block_table_torch,
                qr_bf16,
                kr_bf16,
            )
            module.torch_npu.npu.synchronize()
            cpu_tnd = module.convert_q_bnsd_to_layout(cpu_out, module.ACTUAL_SEQ_Q, "TND")
            compare_result = module.result_compare_method.check_result(cpu_tnd, atten_out)
            compare_status = compare_result[0] if isinstance(compare_result, (list, tuple)) else compare_result
            if compare_status != "Pass":
                return {
                    "name": case["name"],
                    "status": "FAILED",
                    "error_message": _format_compare_error(compare_result),
                }
        return {"name": case["name"], "status": "PASSED", "error_message": ""}
    except Exception as exc:
        return {"name": case["name"], "status": "FAILED", "error_message": f"{type(exc).__name__}: {exc}"}


def run_case_in_subprocess(case, args):
    cmd = [sys.executable, __file__, "--npu", "--no-isolate", "--case-name", case["name"]]
    if args.graph_path is not None:
        cmd.extend(["--graph-path", str(args.graph_path)])
    if args.device_id is not None:
        cmd.extend(["--device-id", str(args.device_id)])

    print(f"\n[SUBPROCESS] {case['name']}")
    ret = subprocess.run(cmd, check=False)
    if ret.returncode == 0:
        return {"name": case["name"], "status": "PASSED", "error_message": ""}
    return {
        "name": case["name"],
        "status": "FAILED",
        "error_message": f"subprocess exit code {ret.returncode}",
    }


def select_cases(args):
    selected = CASES
    if args.case_name:
        selected = [case for case in selected if case["name"] in args.case_name]
    if args.case:
        selected = [case for case in selected if any(item in case["name"] for item in args.case)]
    return selected


def main():
    parser = argparse.ArgumentParser(description="Run MXFP8 batch cases")
    parser.add_argument("--case", action="append", default=None, help="Run only matching case name substring")
    parser.add_argument("--case-name", action="append", default=None, help="Run exact case name")
    parser.add_argument("--npu", "--with-npu", action="store_true", help="Run NPU path and compare with CPU golden")
    parser.add_argument("--no-isolate", action="store_true", help="Run NPU batch in the current process")
    parser.add_argument("--graph-path", type=int, default=None, choices=[0, 3, 5, 6, 7], help="Override GRAPH_PATH")
    parser.add_argument("--device-id", type=int, default=None, help="Override DEVICE_ID for NPU execution")
    args = parser.parse_args()

    selected = select_cases(args)
    if not selected:
        raise ValueError("No cases selected")

    results = []
    if args.npu and not args.no_isolate:
        for case in selected:
            results.append(run_case_in_subprocess(case, args))
    else:
        module = importlib.import_module("fia_fullquant_mxfp8_golden")
        if args.graph_path is not None:
            module.GRAPH_PATH = args.graph_path
        if args.device_id is not None:
            module.DEVICE_ID = args.device_id

        for case in selected:
            mode = "CPU+NPU" if args.npu else "CPU"
            print(f"\n[CASE] {case['name']} [{mode}]")
            results.append(run_case(module, case, run_npu=args.npu))

    passed = sum(item["status"] == "PASSED" for item in results)
    print("\n[SUMMARY]")
    print(f"{'case':<48} {'status':<8} error_message")
    for item in results:
        print(f"{item['name']:<48} {item['status']:<8} {item['error_message']}")
    print(f"[SUMMARY] TOTAL {passed}/{len(results)} PASSED")
    if passed != len(results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
