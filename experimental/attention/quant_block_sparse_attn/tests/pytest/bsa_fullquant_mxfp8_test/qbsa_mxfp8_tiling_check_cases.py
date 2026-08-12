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
QBSA MXFP8 Tiling 拦截校验异常用例

对齐 QuantBlockSparseAttention_MXFP8_拦截校验项.md，
覆盖存在性、Shape维度、dim数量、取值四类拦截项。
运行方式:
  python qbsa_mxfp8_golden.py --case_files qbsa_mxfp8_tiling_check_cases --mode npu
"""

_BASE = {
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
    "case_group": "tiling_check",
    "s1_base_size": 128,
    "n1_n2_ratio": 1,
    "s1_s2_relation": "lt",
    "alignment": "s1_aligned_s2_aligned",
}


def _case(name, **overrides):
    c = dict(_BASE)
    c["name"] = name
    c["enable"] = True
    c.update(overrides)
    return c


TestCases = [
    # ====================================================================
    # 一、存在性校验
    # ====================================================================
    _case("TC_EXIST_01_mask_mode_0_with_mask", mask_mode=0),
    _case("TC_EXIST_02_mask_mode_3_without_mask", mask_mode=3, S1=2048, S2=2048,
          actual_seq_q=[2048], actual_seq_kv=[2048]),
    _case("TC_EXIST_03_metadata_missing", ),
    _case("TC_EXIST_04_p_scale_none", p_scale_value=None),

    # ====================================================================
    # 二、Shape 维度校验
    # ====================================================================
    # 2.1 query shape 异常 (通过 actual_seq_q / N1 / D 触发)
    _case("TC_SHAPE_01_query_T_axis_anomaly", actual_seq_q=[129], S1=129),
    _case("TC_SHAPE_02_query_N_axis_anomaly", N1=7, N2=7),
    _case("TC_SHAPE_03_query_D_axis_anomaly", D=127),

    # 2.2 key/value shape 异常 (通过 block_size / N2 / D 触发)
    _case("TC_SHAPE_04_key_blocknum_anomaly", actual_seq_kv=[128], S2=128),
    _case("TC_SHAPE_05_key_N_axis_anomaly", N1=8, N2=9),
    _case("TC_SHAPE_06_key_blocksize_anomaly", block_size=127,
          sparse_q_block_size=127, sparse_kv_block_size=127),
    _case("TC_SHAPE_07_key_D_axis_anomaly", D=129),
    _case("TC_SHAPE_08_n2_exceed_limit", N1=80, N2=10),

    # 2.3 q_descale shape 异常 (通过 quant_group_size 触发 D/64 不匹配)
    _case("TC_SHAPE_09_q_descale_dgroup_anomaly", quant_group_size=64),

    # 2.4 sparse_indices shape 异常 (通过 sparse_count 触发 max_Kb 不匹配)
    _case("TC_SHAPE_10_sparse_indices_kb_anomaly", sparse_count=3),

    # 2.5 sparse_seq_len / sparse_indices qbMax 不一致
    _case("TC_SHAPE_11_sparse_seq_len_qb_mismatch", sparse_count=1),

    # 2.6 block_table shape 异常 (通过 B 触发)
    _case("TC_SHAPE_12_block_table_b_anomaly", B=2, actual_seq_q=[128, 64], S1=128,
          actual_seq_kv=[256, 128], S2=256),

    # 2.7 p_scale shape 异常
    _case("TC_SHAPE_13_p_scale_shape_2", p_scale_value=2.0),

    # 2.8 batch_size 超限
    _case("TC_SHAPE_14_batch_exceed_limit", B=65537,
          actual_seq_q=[128] * 65537, actual_seq_kv=[256] * 65537,
          S1=128, S2=256),

    # ====================================================================
    # 三、dim 数量校验 (通过 layout 触发 dim 期望不匹配)
    # ====================================================================
    _case("TC_DIM_01_query_layout_NTD", layout_q="NTD"),

    # ====================================================================
    # 四、取值校验
    # ====================================================================
    # 4.1 dtype 异常 (通过 fp8_dtype 触发)
    _case("TC_VAL_01_qkv_dtype_wrong", fp8_dtype="float8_e5m2"),

    # 4.2 scale dtype 异常 (通过 scale_dtype 触发)
    _case("TC_VAL_02_scale_dtype_wrong", scale_dtype="float32"),

    # 4.3 attr 属性值异常
    _case("TC_VAL_03_mask_mode_invalid_4", mask_mode=4),
    _case("TC_VAL_04_mask_mode_invalid_2", mask_mode=2),
    _case("TC_VAL_05_quant_mode_invalid", quant_mode=3),
    _case("TC_VAL_06_layout_q_invalid", layout_q="BSND"),
    _case("TC_VAL_07_layout_kv_invalid", layout_kv="BNSD"),
    _case("TC_VAL_08_layout_out_invalid", layout_out="BSND"),
    _case("TC_VAL_09_layout_sparse_indices_invalid", layout_sparse_indices="B_N_Qb"),

    # 4.4 数值范围异常
    _case("TC_VAL_10_head_dim_not_128", D=127),
    _case("TC_VAL_11_n1_not_divisible_n2", N1=7, N2=3),
    _case("TC_VAL_12_sparse_block_size_invalid", sparse_q_block_size=72, sparse_kv_block_size=72),
    _case("TC_VAL_13_q_kv_block_size_inconsistent", sparse_q_block_size=128, sparse_kv_block_size=64),
    _case("TC_VAL_14_pa_block_size_not_multiple", block_size=100),
    _case("TC_VAL_15_pa_block_size_exceed_limit", block_size=2048),
    _case("TC_VAL_16_softmax_scale_out_of_range", softmax_scale=2.0),
    _case("TC_VAL_17_softmax_scale_zero", softmax_scale=0.0),
]
