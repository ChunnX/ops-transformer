#!/usr/bin/python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
QBSA MXFP8 Tiling 拦截校验测试脚本

只调 NPU 算子接口，不跑 CPU golden。期望每条用例被 tiling check 拦截（抛异常）。
用法与 qbsa_mxfp8_golden.py 一致:
  python qbsa_mxfp8_tiling_check.py --case_files qbsa_mxfp8_tiling_check_cases
  python qbsa_mxfp8_tiling_check.py --case_files qbsa_mxfp8_tiling_check_cases --case_name TC_EXIST_01_mask_mode_0_with_mask
  python qbsa_mxfp8_tiling_check.py --case_files qbsa_mxfp8_tiling_check_cases --list_cases
"""

import argparse
import importlib.util
import logging
import os
import sys
import time

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTEST_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PYTEST_DIR not in sys.path:
    sys.path.insert(0, PYTEST_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import qbsa_mxfp8_golden as golden


def _load_case_file(case_file):
    if not case_file.endswith(".py"):
        case_file = case_file + ".py"
    case_path = os.path.join(CURRENT_DIR, case_file) if not os.path.isabs(case_file) else case_file
    if not os.path.exists(case_path):
        raise FileNotFoundError(f"case file not found: {case_path}")
    module_name = f"_tiling_check_cases_{abs(hash(case_path))}"
    spec = importlib.util.spec_from_file_location(module_name, case_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if not hasattr(module, "TestCases"):
        raise AttributeError(f"{case_path} must define TestCases")
    return module.TestCases


def _iter_cases(test_cases):
    if isinstance(test_cases, dict):
        for name, case in test_cases.items():
            item = dict(case)
            item.setdefault("name", name)
            yield item
    else:
        for case in test_cases:
            yield dict(case)


def run_one_tiling_check(case_id, case, case_idx, total):
    start = time.time()
    name = case.get("name", case_id)
    logger.info("")
    logger.info("=" * 80)
    logger.info("[TILING CHECK %d/%d] %s", case_idx, total, name)
    logger.info("=" * 80)

    try:
        golden.set_active_case(case)
        data = golden._generate_reference_style_mxfp8_inputs()
        atten_out, lse_out = golden.npu_mxfp8_fa(
            data["query"],
            data["key"],
            data["value"],
            data["q_descale"],
            data["k_descale"],
            data["v_descale"],
            data["p_scale"],
            data["q_lengths"],
            data["kv_lengths"],
            data["cu_seqlens_q"],
            data["cu_seqlens_kv"],
            block_table_torch=data["block_table"],
            sparse_indices=data["sparse_indices"],
            sparse_seq_len=data["sparse_seq_len"],
        )
        elapsed = time.time() - start
        logger.info("[RESULT] %s -> UNEXPECTED PASS (算子未拦截，耗时 %.2fs)", name, elapsed)
        return {"case": name, "status": "UnexpectedPass", "time": elapsed, "error": ""}
    except Exception as err:
        elapsed = time.time() - start
        err_msg = str(err).strip().replace("\n", " ")[:200]
        logger.info("[RESULT] %s -> BLOCKED (%.2fs) %s", name, elapsed, err_msg)
        return {"case": name, "status": "Blocked", "time": elapsed, "error": err_msg}


def _print_summary(results):
    total = len(results)
    blocked = sum(1 for r in results if r["status"] == "Blocked")
    unexpected = sum(1 for r in results if r["status"] == "UnexpectedPass")
    errors = sum(1 for r in results if r["status"] == "Error")

    logger.info("")
    logger.info("=" * 80)
    logger.info("Tiling Check Summary")
    logger.info("=" * 80)
    logger.info("  Total          : %d", total)
    logger.info("  Blocked (PASS)  : %d", blocked)
    logger.info("  UnexpectedPass  : %d", unexpected)
    logger.info("  Error           : %d", errors)
    logger.info("  PassRate        : %.2f%%  (%d/%d)",
                (blocked / total * 100) if total else 0, blocked, total)
    logger.info("-" * 80)
    for r in results:
        tag = "OK" if r["status"] == "Blocked" else "FAIL"
        logger.info("  [%s] %-50s %.2fs  %s", tag, r["case"], r["time"], r["error"][:80])
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="QBSA MXFP8 Tiling 拦截校验测试")
    parser.add_argument(
        "--case_files",
        default="qbsa_mxfp8_tiling_check_cases",
        help="case 文件路径（不含 .py 后缀也可）；默认 qbsa_mxfp8_tiling_check_cases",
    )
    parser.add_argument(
        "--case_name",
        "--case-name",
        dest="case_name",
        default=None,
        help="只跑指定 case（支持逗号分隔多个）；不传时跑所有 enable=True 的 case",
    )
    parser.add_argument(
        "--list_cases",
        action="store_true",
        help="只列出 case，不执行",
    )
    args = parser.parse_args()

    test_cases = _load_case_file(args.case_files)
    all_cases = {}
    for case in _iter_cases(test_cases):
        cid = case.get("name", str(len(all_cases)))
        all_cases[cid] = case

    if args.list_cases:
        for cid, case in all_cases.items():
            print(f"{cid}  enable={case.get('enable', True)}")
        sys.exit(0)

    if args.case_name:
        names = [n.strip() for n in args.case_name.split(",") if n.strip()]
        run_ids = [n for n in names if n in all_cases]
        missing = [n for n in names if n not in all_cases]
        if missing:
            logger.info("[WARN] case not found: %s", ", ".join(missing))
    else:
        run_ids = [cid for cid, case in all_cases.items() if case.get("enable", True)]

    if not run_ids:
        parser.error("No runnable case selected.")

    total = len(run_ids)
    results = []
    for idx, cid in enumerate(run_ids, 1):
        results.append(run_one_tiling_check(cid, all_cases[cid], idx, total))

    _print_summary(results)

    unexpected = sum(1 for r in results if r["status"] != "Blocked")
    if unexpected:
        sys.exit(1)


if __name__ == "__main__":
    main()
