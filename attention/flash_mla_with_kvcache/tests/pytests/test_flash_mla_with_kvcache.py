#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

"""flash_mla_with_kvcache 功能矩阵入口（MLA 版 harness，布局 TND/NZ/NTD per R1）。

运行： cd attention/flash_mla_with_kvcache/tests/pytests && python test_flash_mla_with_kvcache.py [--case_id ...] [--static_graph]
矩阵覆盖：prefill(FD=false)/decode(FD)、PA(block_table)、LSE 开关、静态图 vs 动态图等价(M9)、
>1-section、mask_mode {0,3}、fp16/bf16。
"""

import sys
import os
import gc
import argparse
import torch
from pathlib import Path

from core.data import flash_mla_with_kvcache_inputs
from core.case_loader import load_case_modules, normalize_params, resolve_case_ids
from core.orchestrator import run_case
from core.reporter import Reporter

_DEFAULT_CASE_FILES = [os.path.splitext(f)[0] for f in os.listdir(
    os.path.join(os.path.dirname(__file__), "test_cases"))
    if f.endswith(".py") and not f.startswith("_")]


def _build_primary(args):
    from core.backends.npu import NPUBackend
    primary = NPUBackend(args.device_id, meta_only=args.meta_only)
    if not primary.is_available():
        return None, ("NPU 后端不可用：torch_npu/cann_ops_transformer 缺失或 "
                      "torch.ops.cann_ops_transformer.flash_mla_with_kvcache 未注册"
                      "（需先构建并安装 flash_mla_with_kvcache 算子包）")
    return primary, None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FlashMlaWithKvcache 功能矩阵精度测试（TND/NZ/NTD）")
    parser.add_argument("--case_files", type=str, default=",".join(_DEFAULT_CASE_FILES))
    parser.add_argument("--case_id", type=str, default="all")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--verbose_diff", action="store_true")
    parser.add_argument("--meta_only", action="store_true",
                        help="只调用 flash_mla_metadata 并打印分核结果，跳过 kernel")
    parser.add_argument("--static_graph", action="store_true",
                        help="对每个 case 追加：确定性（同输入两次）+ 静态图(torch.export) vs 动态图等价（M9）")
    parser.add_argument("--result_csv", type=str,
                        default="attention/flash_mla_with_kvcache/tests/pytests/result.csv")
    parser.add_argument("--report_interval", type=int, default=20)
    args = parser.parse_args()

    module_names = ["test_cases." + x.strip() for x in args.case_files.split(",")]
    all_cases = load_case_modules(module_names)
    run_cases = resolve_case_ids(args.case_id, all_cases)
    if not run_cases:
        print("[ERROR] 没有可运行的 case。")
        sys.exit(1)

    primary, err = _build_primary(args)
    from core.backends.cpu import CPUBackend
    golden = CPUBackend()

    reporter = Reporter()
    if args.result_csv:
        csv_path = Path(args.result_csv)
        csv_path.parent.mkdir(parents=True, exist_ok=True)
        reporter.set_csv(csv_path)

    primary_unavailable = primary is None
    if primary_unavailable:
        print(f"[WARN] {err}\n       矩阵行将在 CPU-golden 侧验证管道，NPU 行门控到 F6。")

    for i, name in enumerate(run_cases):
        params = normalize_params(all_cases[name])
        dtype = params.get("Dtype", "fp16")
        b = params.get("B", 1)
        n1 = params["N1"]
        s1 = params.get("S1", 1)
        s2 = params.get("S2", 1)
        print(f"\n{'='*80}")
        print(f"  Case: {name}  B={b} N1={n1} S1={s1} S2={s2} "
              f"dtype={dtype} mask={params.get('mask_mode', 0)} "
              f"lse={params.get('return_softmax_lse', False)}")
        print(f"{'='*80}")

        if primary_unavailable:
            # 无 NPU：只做 pipeline 冒烟（数据生成 + golden 自身可跑通），
            # 设备侧结果标记 gated，不伪造通过。
            try:
                cpu_inputs = flash_mla_with_kvcache_inputs.generate(params, device=torch.device("cpu"),
                                                       layout=params.get("input_layout", "TND"))
                cpu_inputs["block_table"] = params["block_table"]
                golden_out = golden.compute(cpu_inputs, params)
                print(f"  [golden] CPU 参考计算 OK  out={tuple(golden_out['out'].shape)} "
                      f"lse={tuple(golden_out['lse'].shape)}")
                del cpu_inputs, golden_out
                gc.collect()
            except Exception as e:
                import traceback
                err_txt = f"{e}\n{traceback.format_exc()}"
                print(f"[ERROR] {name} CPU-golden 冒烟异常: {e}")
            result = {"attn": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                               "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')},
                      "lse": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                              "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}}
            reporter.record(name, result,
                            error="gated: NPU backend unavailable (F6)",
                            dtype=dtype)
            continue

        try:
            result = run_case(params=params, primary=primary, golden=golden,
                              spec=flash_mla_with_kvcache_inputs, case_name=name,
                              verbose_diff=args.verbose_diff,
                              static_graph=args.static_graph)
            err = None
        except Exception as e:
            import traceback
            err = f"{e}\n{traceback.format_exc()}"
            print(f"[ERROR] {name} 运行异常: {e}")
            traceback.print_exc()
            result = {"attn": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                               "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')},
                      "lse": {"passed": False, "max_abs": float('nan'), "mean_abs": float('nan'),
                              "fail_cnt": -1, "total": -1, "fail_ratio": float('nan')}}

        reporter.record(name, result, error=err, dtype=dtype)
        del result, params
        gc.collect()
        primary.clear_cache()

        if args.report_interval > 0 and (i + 1) % args.report_interval == 0 \
                and (i + 1) < len(run_cases):
            reporter.print_interim(len(run_cases))

    fail_cnt = reporter.print_final()
    if args.result_csv:
        print(f"[result] {args.result_csv}")
    sys.exit(0 if fail_cnt == 0 else 1)