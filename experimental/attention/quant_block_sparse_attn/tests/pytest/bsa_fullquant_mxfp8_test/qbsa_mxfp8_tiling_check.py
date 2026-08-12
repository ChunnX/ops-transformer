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

用合法 base case 生成数据，然后在调 NPU 前篡改张量/参数，触发 tiling check 拦截。
不跑 CPU golden，只调 NPU 接口。

用法:
  python qbsa_mxfp8_tiling_check.py
  python qbsa_mxfp8_tiling_check.py --case_name TC_EXIST_01
  python qbsa_mxfp8_tiling_check.py --list_cases
"""

import argparse
import logging
import os
import sys
import time

import torch

if not hasattr(torch, "float8_e4m3fn"):
    torch.float8_e4m3fn = torch.float8_e4m3

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTEST_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PYTEST_DIR not in sys.path:
    sys.path.insert(0, PYTEST_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import qbsa_mxfp8_golden as golden


_BASE_CASE = {
    "name": "TILING_CHECK_BASE",
    "enable": True,
    "B": 1,
    "N1": 8,
    "N2": 8,
    "D": 128,
    "actual_seq_q": [128],
    "actual_seq_kv": [256],
    "S1": 128,
    "S2": 256,
    "s2_base_size": 512,
    "block_size": 128,
    "mask_mode": 3,
    "fp8_dtype": "float8_e4m3fn",
    "scale_dtype": "float8_e8m0fnu",
    "quant_group_size": 32,
    "layout_q": "TND",
    "layout_kv": "PA_BNSD",
    "layout_sparse_indices": "B_N_Qb_Kb",
    "layout_out": "TND",
    "kv_cache_layout": "BnNBsD",
    "is_contiguous": True,
    "p_scale_value": 1.0,
    "softmax_scale": 0.125,
    "return_softmax_lse": True,
    "empty_actual_seq": False,
    "seed": 42,
    "data_range_q": 1.0,
    "data_range_k": 1.0,
    "data_range_v": 1.0,
    "device_id": 0,
    "sparse_q_block_size": 128,
    "sparse_kv_block_size": 128,
    "sparse_count": 2,
    "sparse_pattern": "sequential",
    "block_table_pattern": "sequential",
    "quant_mode": 2,
}


def _gen_base_data():
    golden.set_active_case(golden._normalize_case(dict(_BASE_CASE)))
    return golden._generate_reference_style_mxfp8_inputs()


def _call_npu(data, case_override=None):
    if case_override is None:
        case_override = dict(_BASE_CASE)
    golden.set_active_case(golden._normalize_case(case_override))
    return golden.npu_mxfp8_fa(
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


# ============================================================================
# 拦截用例定义：每条 = (name, mutate_fn, case_override_fn)
#   mutate_fn(data): 篡改生成好的 data dict 中的张量
#   case_override_fn(): 返回覆盖后的 case dict（改 attr 值，不改影响数据生成的字段）
# ============================================================================

def _noop(data):
    pass

def _base_case():
    return dict(_BASE_CASE)


TILING_CHECK_CASES = [
    # --- 一、存在性 ---
    ("TC_EXIST_01_block_table_none",
     lambda d: d.__setitem__("block_table", None), _base_case),

    ("TC_EXIST_02_cu_seqlens_q_none",
     lambda d: d.__setitem__("cu_seqlens_q", None), _base_case),

    ("TC_EXIST_03_p_scale_none",
     lambda d: d.__setitem__("p_scale", None), _base_case),

    ("TC_EXIST_04_cu_seqlens_kv_not_null",
     lambda d: d.__setitem__("cu_seqlens_kv", torch.tensor([0, 128], dtype=torch.int32)), _base_case),

    # --- 二、Shape 维度 ---
    ("TC_SHAPE_01_query_T_plus1",
     lambda d: d.__setitem__("query", torch.cat([d["query"], d["query"][:1]], dim=0)), _base_case),

    ("TC_SHAPE_02_query_N_minus1",
     lambda d: d.__setitem__("query", d["query"][:, :7]), _base_case),

    ("TC_SHAPE_03_query_D_plus1",
     lambda d: d.__setitem__("query", torch.cat([d["query"], torch.zeros(d["query"].shape[0], d["query"].shape[1], 1, dtype=d["query"].dtype)], dim=2)), _base_case),

    ("TC_SHAPE_04_key_blocknum_minus1",
     lambda d: d.__setitem__("key", d["key"][:-1]), _base_case),

    ("TC_SHAPE_05_key_N_plus1",
     lambda d: d.__setitem__("key", torch.cat([d["key"], torch.zeros(d["key"].shape[0], 1, d["key"].shape[2], d["key"].shape[3], dtype=d["key"].dtype)], dim=1)), _base_case),

    ("TC_SHAPE_06_key_blocksize_minus1",
     lambda d: d.__setitem__("key", d["key"][:, :, :-1]), _base_case),

    ("TC_SHAPE_07_key_D_plus1",
     lambda d: d.__setitem__("key", torch.cat([d["key"], torch.zeros(d["key"].shape[0], d["key"].shape[1], d["key"].shape[2], 1, dtype=d["key"].dtype)], dim=3)), _base_case),

    ("TC_SHAPE_08_n2_exceed_limit",
     _noop, lambda: {**_BASE_CASE, "N1": 80, "N2": 10}),

    ("TC_SHAPE_09_sparse_indices_b_mismatch",
     lambda d: d.__setitem__("sparse_indices", d["sparse_indices"][:0]), _base_case),

    ("TC_SHAPE_10_sparse_indices_n_mismatch",
     lambda d: d.__setitem__("sparse_indices", d["sparse_indices"][:, :4]), _base_case),

    ("TC_SHAPE_11_sparse_indices_qb_mismatch",
     lambda d: d.__setitem__("sparse_indices", d["sparse_indices"][:, :, :0]), _base_case),

    ("TC_SHAPE_12_sparse_seq_len_b_mismatch",
     lambda d: d.__setitem__("sparse_seq_len", d["sparse_seq_len"][:0]), _base_case),

    ("TC_SHAPE_13_block_table_b_plus1",
     lambda d: d.__setitem__("block_table", torch.cat([d["block_table"], d["block_table"][:1]], dim=0)), _base_case),

    ("TC_SHAPE_14_p_scale_shape_2",
     lambda d: d.__setitem__("p_scale", torch.tensor([1.0, 1.0], dtype=torch.float32)), _base_case),

    ("TC_SHAPE_15_p_scale_shape_1_1",
     lambda d: d.__setitem__("p_scale", torch.tensor([[1.0]], dtype=torch.float32)), _base_case),

    ("TC_SHAPE_16_batch_exceed_limit",
     _noop, lambda: {**_BASE_CASE, "B": 65537, "actual_seq_q": [128] * 65537, "actual_seq_kv": [256] * 65537, "S1": 128, "S2": 256}),

    # --- 三、dim 数量 ---
    ("TC_DIM_01_query_2d",
     lambda d: d.__setitem__("query", d["query"].reshape(d["query"].shape[0], -1)), _base_case),

    ("TC_DIM_02_query_4d",
     lambda d: d.__setitem__("query", d["query"].unsqueeze(0)), _base_case),

    ("TC_DIM_03_key_3d",
     lambda d: d.__setitem__("key", d["key"].reshape(d["key"].shape[0], d["key"].shape[1], -1)), _base_case),

    ("TC_DIM_04_key_5d",
     lambda d: d.__setitem__("key", d["key"].unsqueeze(0)), _base_case),

    ("TC_DIM_05_q_descale_3d",
     lambda d: d.__setitem__("q_descale", d["q_descale"].reshape(d["q_descale"].shape[0], d["q_descale"].shape[1], -1)), _base_case),

    ("TC_DIM_06_k_descale_4d",
     lambda d: d.__setitem__("k_descale", d["k_descale"].reshape(d["k_descale"].shape[0], d["k_descale"].shape[1], d["k_descale"].shape[2], -1)) if d["k_descale"].dim() == 5 else None, _base_case),

    ("TC_DIM_07_sparse_indices_3d",
     lambda d: d.__setitem__("sparse_indices", d["sparse_indices"].reshape(d["sparse_indices"].shape[0], d["sparse_indices"].shape[1], -1)), _base_case),

    ("TC_DIM_08_sparse_indices_5d",
     lambda d: d.__setitem__("sparse_indices", d["sparse_indices"].unsqueeze(0)), _base_case),

    ("TC_DIM_09_sparse_seq_len_2d",
     lambda d: d.__setitem__("sparse_seq_len", d["sparse_seq_len"].reshape(d["sparse_seq_len"].shape[0], -1)), _base_case),

    ("TC_DIM_10_block_table_1d",
     lambda d: d.__setitem__("block_table", d["block_table"].reshape(-1)), _base_case),

    ("TC_DIM_11_block_table_3d",
     lambda d: d.__setitem__("block_table", d["block_table"].unsqueeze(0)), _base_case),

    # --- 四、取值: dtype ---
    ("TC_VAL_01_query_dtype_fp16",
     lambda d: d.__setitem__("query", d["query"].to(torch.float16)), _base_case),

    ("TC_VAL_02_value_dtype_e5m2",
     lambda d: d.__setitem__("value", d["value"].to(torch.float8_e5m2)), _base_case),

    ("TC_VAL_03_q_descale_dtype_fp32",
     lambda d: d.__setitem__("q_descale", d["q_descale"].to(torch.float32)), _base_case),

    ("TC_VAL_04_block_table_dtype_int64",
     lambda d: d.__setitem__("block_table", d["block_table"].to(torch.int64)), _base_case),

    ("TC_VAL_05_sparse_indices_dtype_int64",
     lambda d: d.__setitem__("sparse_indices", d["sparse_indices"].to(torch.int64)), _base_case),

    # --- 四、取值: attr 属性 ---
    ("TC_VAL_06_mask_mode_4",
     _noop, lambda: {**_BASE_CASE, "mask_mode": 4}),

    ("TC_VAL_07_mask_mode_2",
     _noop, lambda: {**_BASE_CASE, "mask_mode": 2}),

    ("TC_VAL_08_quant_mode_3",
     _noop, lambda: {**_BASE_CASE, "quant_mode": 3}),

    ("TC_VAL_09_layout_q_BSND",
     _noop, lambda: {**_BASE_CASE, "layout_q": "BSND"}),

    ("TC_VAL_10_layout_kv_BNSD",
     _noop, lambda: {**_BASE_CASE, "layout_kv": "BNSD"}),

    ("TC_VAL_11_layout_out_BSND",
     _noop, lambda: {**_BASE_CASE, "layout_out": "BSND"}),

    ("TC_VAL_12_layout_sparse_B_N_Qb",
     _noop, lambda: {**_BASE_CASE, "layout_sparse_indices": "B_N_Qb"}),

    # --- 四、取值: 数值范围 ---
    ("TC_VAL_13_head_dim_127",
     _noop, lambda: {**_BASE_CASE, "D": 127}),

    ("TC_VAL_14_n1_not_div_n2",
     _noop, lambda: {**_BASE_CASE, "N1": 7, "N2": 3}),

    ("TC_VAL_15_sparse_block_size_72",
     _noop, lambda: {**_BASE_CASE, "sparse_q_block_size": 72, "sparse_kv_block_size": 72}),

    ("TC_VAL_16_q_kv_block_size_inconsistent",
     _noop, lambda: {**_BASE_CASE, "sparse_q_block_size": 128, "sparse_kv_block_size": 64}),

    ("TC_VAL_17_pa_block_size_not_multiple",
     _noop, lambda: {**_BASE_CASE, "block_size": 100}),

    ("TC_VAL_18_pa_block_size_exceed_limit",
     _noop, lambda: {**_BASE_CASE, "block_size": 2048}),

    ("TC_VAL_19_softmax_scale_2",
     _noop, lambda: {**_BASE_CASE, "softmax_scale": 2.0}),

    ("TC_VAL_20_softmax_scale_0",
     _noop, lambda: {**_BASE_CASE, "softmax_scale": 0.0}),
]


def _log_constructed_case(name, data, case_override):
    logger.info("[CASE] %s", name)
    for key in ["query", "key", "value", "q_descale", "k_descale", "v_descale",
                "p_scale", "sparse_indices", "sparse_seq_len", "block_table",
                "cu_seqlens_q", "cu_seqlens_kv"]:
        val = data.get(key)
        if val is None:
            logger.info("  %-18s = None", key)
        elif hasattr(val, "shape"):
            logger.info("  %-18s = shape=%s dtype=%s", key, tuple(val.shape), val.dtype)
        else:
            logger.info("  %-18s = %s", key, val)
    diff = {k: v for k, v in case_override.items() if _BASE_CASE.get(k) != v}
    if diff:
        logger.info("  [attr overrides]")
        for k, v in diff.items():
            logger.info("    %-18s = %s", k, v)


def run_one_tiling_check(name, mutate_fn, case_fn, idx, total):
    start = time.time()
    logger.info("")
    logger.info("=" * 80)
    logger.info("[TILING CHECK %d/%d] %s", idx, total, name)
    logger.info("=" * 80)

    try:
        data = _gen_base_data()
        mutate_fn(data)
        case_override = case_fn()
        _log_constructed_case(name, data, case_override)
        _call_npu(data, case_override)
        elapsed = time.time() - start
        logger.info("[RESULT] %s -> UNEXPECTED PASS (算子未拦截)", name)
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

    logger.info("")
    logger.info("=" * 80)
    logger.info("Tiling Check Summary")
    logger.info("=" * 80)
    logger.info("  Total           : %d", total)
    logger.info("  Blocked (PASS)  : %d", blocked)
    logger.info("  UnexpectedPass  : %d", unexpected)
    logger.info("  PassRate        : %.2f%%  (%d/%d)",
                (blocked / total * 100) if total else 0, blocked, total)
    logger.info("-" * 80)
    for r in results:
        tag = "OK" if r["status"] == "Blocked" else "FAIL"
        logger.info("  [%s] %-50s %.2fs  %s", tag, r["case"], r["time"], r["error"][:80])
    logger.info("=" * 80)


def main():
    parser = argparse.ArgumentParser(description="QBSA MXFP8 Tiling 拦截校验测试")
    parser.add_argument("--case_name", default=None,
                        help="只跑指定 case（支持逗号分隔多个）")
    parser.add_argument("--list_cases", action="store_true",
                        help="只列出 case，不执行")
    args = parser.parse_args()

    if args.list_cases:
        for name, _, _ in TILING_CHECK_CASES:
            print(name)
        sys.exit(0)

    if args.case_name:
        names = [n.strip() for n in args.case_name.split(",") if n.strip()]
        cases = [(n, m, c) for n, m, c in TILING_CHECK_CASES if n in names]
        missing = [n for n in names if n not in {tc[0] for tc in TILING_CHECK_CASES}]
        if missing:
            logger.info("[WARN] case not found: %s", ", ".join(missing))
    else:
        cases = list(TILING_CHECK_CASES)

    if not cases:
        parser.error("No runnable case selected.")

    total = len(cases)
    results = []
    for idx, (name, mutate_fn, case_fn) in enumerate(cases, 1):
        results.append(run_one_tiling_check(name, mutate_fn, case_fn, idx, total))

    _print_summary(results)

    if any(r["status"] != "Blocked" for r in results):
        sys.exit(1)


if __name__ == "__main__":
    main()
