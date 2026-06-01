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
import argparse
import importlib
import itertools

_DEFAULT_CASE_FILES = [os.path.splitext(f)[0] for f in os.listdir(
    os.path.join(os.path.dirname(__file__), "test_cases"))
    if f.endswith(".py") and not f.startswith("_")]


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
        return sorted(all_cases.keys())
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
    parser = argparse.ArgumentParser(description="FlashAttn 精度测试")
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

    # ============================================================
    # TODO: 在此处添加你的测试执行逻辑
    #   - all_cases[name] 是当前用例的参数字典
    #   - 示例:
    #       result = your_test_func(name, **all_cases[name])
    # ============================================================
    for name in run_cases:
        config = all_cases[name]
        print(f"[{name}] config: {config}")
