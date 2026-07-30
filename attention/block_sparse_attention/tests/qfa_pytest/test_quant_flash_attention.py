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
# 调用方式:
#   cd ops-transformer/attention/flash_attn/tests/pytests2
#   python test_flash_attn.py
#   python test_flash_attn.py --case_files base
#   python test_flash_attn.py --case_id BASE_04
#   python test_flash_attn.py --device_id 1 --case_files base --case_id BASE_04
# ======================================================================================================================

import os
import gc
import argparse
import importlib
import itertools
import inspect
import re
import torch
import torch_npu

from quant_flash_attn_golden import run_fia_eager

_DEFAULT_CASE_FILES = [os.path.splitext(f)[0] for f in os.listdir(
    os.path.join(os.path.dirname(__file__), "test_cases"))
    if f.endswith(".py") and not f.startswith("_")]

class Color:
    RED = '\033[91m'
    GREEN = '\033[92m'
    BOLD = '\033[1m'
    END = '\033[0m'

def visible_length(s):
    return len(re.sub(r'\033\[\d+(;\d+)*m', '', s))

def center_visible(s, width):
    visible = re.sub(r'\033\[\d+(;\d+)*m', '', s)
    padding = width - len(visible)
    left_pad = padding // 2
    right_pad = padding - left_pad
    return ' ' * left_pad + s + ' ' * right_pad

def load_case_modules(module_names):
    all_cases = {}
    for mod_name in module_names:
        mod = importlib.import_module(mod_name)
        for case_name, case_dict in mod.TestCases.items():
            keys = list(case_dict.keys())
            values = [case_dict[k] for k in keys]
            combos = list(itertools.product(*values))
            if len(combos) == 1:
                all_cases[f"{mod_name}/{case_name}"] = dict(zip(keys, combos[0]))
            else:
                for i, combo in enumerate(combos):
                    all_cases[f"{mod_name}/{case_name}_{i}"] = dict(zip(keys, combo))
    return all_cases


def resolve_case_ids(case_id_arg, all_cases):
    if case_id_arg == "all":
        def extract_number(case_name):
            """从 case 名称中提取最后一个数字（参数值）"""
            parts = case_name.rsplit('_', 1)  # 按最后一个下划线分割
            if len(parts) == 2:
                try:
                    return int(parts[1])
                except ValueError:
                    # 如果最后一部分不是纯数字，尝试转为 float
                    try:
                        return float(parts[1])
                    except ValueError:
                        pass
            return 0  # 无法提取时返回 0
        
        return sorted(all_cases.keys(), key=extract_number)
    
    ids = [x.strip() for x in case_id_arg.split(",")]
    result = []
    missing = []
    for cid in ids:
        full_names = [k for k in all_cases if k.endswith(f"/{cid}") or k == cid or k.rsplit("/", 1)[-1].startswith(f"{cid}_")]
        if len(full_names) >= 1:
            result.extend(full_names)
        else:
            missing.append(cid)
    if missing:
        print(f"[WARN] 以下 case 不存在: {missing}")
    return result

# ─────────────────── main ───────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="QuantFlashAttn 精度测试")
    parser.add_argument("--case_files", type=str, default=",".join(_DEFAULT_CASE_FILES),
                        help=f"逗号分隔的 test case 模块名（不含 .py）")
    parser.add_argument("--case_id",   type=str, default="all", help="case名称，多个用逗号分隔")
    parser.add_argument("--device_id", type=int, default=0,    help="NPU device id")
    args = parser.parse_args()

    module_names = ["test_cases." + x.strip() for x in args.case_files.split(",")]
    all_cases = load_case_modules(module_names)
    run_cases = resolve_case_ids(args.case_id, all_cases)
    if not run_cases:
        print("[ERROR] 没有可运行的 case。")
        exit(1)

    results_table = []

    for name in run_cases:
        gc.collect()
        torch.npu.synchronize()
        torch.npu.empty_cache()
        config = all_cases[name]
        print(f"\n{'='*66}")
        print(f"  Case: {name}  "
            f"B={config.get('b')} N1={config.get('n1')} N2={config.get('n2')} "
            f"S1={config.get('s1')} S2={config.get('s2')} D={config.get('qk_d')} "
            f"layout={config.get('input_layout')} q_dtype={config.get('q_dtype')}")

        sig = inspect.signature(run_fia_eager)
        kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name in config:
                kwargs[param_name] = config[param_name]
            elif param.default is not inspect.Parameter.empty:
                kwargs[param_name] = param.default
            else:
                raise TypeError(
                    f"{run_fia_eager.__name__}() missing required argument: '{param_name}'"
                )
        
        # 直接获取返回值
        fulfill_percent, result = run_fia_eager(**kwargs)
        results_table.append([name, fulfill_percent, result])
        
        print(f"{'='*66}")

    # 循环结束后打印表格
    col1_width = 40   # Case Name 列宽
    col2_width = 18   # Fulfill Percent 列宽
    col3_width = 15   # Result 列宽
    total_width = col1_width + col2_width + col3_width

    print("\n" + "=" * total_width)
    print("SUMMARY TABLE".center(total_width))
    print("=" * total_width)

    # 表头
    header1 = "Case Name".center(col1_width)
    header2 = "Fulfill Percent".center(col2_width)
    header3 = "Result".center(col3_width)
    print(f"{header1} {header2} {header3}")

    print("-" * total_width)

    for row in results_table:
        # 去掉 test_cases. 前缀
        case_name = row[0].replace("test_cases.", "", 1) if row[0].startswith("test_cases.") else row[0]
        
        percent_str = f"{row[1]:.4f}" if isinstance(row[1], float) else str(row[1])
        
        # 处理 Result 列的显示
        result = row[2]
        if result.upper() == "PASS":
            result_display = f"{Color.GREEN}✓ PASS{Color.END}"
        elif result.upper() == "FAILED":
            result_display = f"{Color.RED}{Color.BOLD}✗ FAILED{Color.END}"
        else:
            result_display = result
        
        # 居中对齐
        c1 = case_name.center(col1_width)
        c2 = percent_str.center(col2_width)
        c3 = center_visible(result_display, col3_width)
        
        print(f"{c1} {c2} {c3}")

    print("=" * total_width)