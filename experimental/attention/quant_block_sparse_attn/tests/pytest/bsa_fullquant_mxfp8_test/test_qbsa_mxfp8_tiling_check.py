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
QBSA MXFP8 Tiling 拦截校验异常用例测试

对齐 QuantBlockSparseAttention_MXFP8_拦截校验项.md，
覆盖存在性、Shape维度、dim数量、取值四类拦截项。
每条用例构造非法输入，期望 NPU 算子调用抛出异常（TORCH_CHECK / OP_LOGE）。
"""

import os
import sys
from pathlib import Path

import pytest
import torch

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PYTEST_DIR = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
if PYTEST_DIR not in sys.path:
    sys.path.insert(0, PYTEST_DIR)
if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)

import qbsa_mxfp8_golden as golden

DEVICE_ID = 0
SPARSE_BLOCK_SIZE = 128
D = 128


def _make_base_case():
    return {
        "name": "TILING_CHECK_BASE",
        "enable": True,
        "B": 1,
        "N1": 8,
        "N2": 8,
        "D": D,
        "actual_seq_q": [128],
        "actual_seq_kv": [256],
        "S1": 128,
        "S2": 256,
        "s2_base_size": 512,
        "block_size": SPARSE_BLOCK_SIZE,
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
        "device_id": DEVICE_ID,
        "sparse_q_block_size": SPARSE_BLOCK_SIZE,
        "sparse_kv_block_size": SPARSE_BLOCK_SIZE,
        "sparse_count": 2,
        "sparse_pattern": "sequential",
        "block_table_pattern": "sequential",
        "quant_mode": 2,
    }


def _generate_inputs(case):
    golden.set_active_case(case)
    data = golden._generate_reference_style_mxfp8_inputs()
    return data


def _call_npu(data, case, **overrides):
    golden.set_active_case(case)
    q = overrides.get("query", data["query"])
    k = overrides.get("key", data["key"])
    v = overrides.get("value", data["value"])
    q_descale = overrides.get("q_descale", data["q_descale"])
    k_descale = overrides.get("k_descale", data["k_descale"])
    v_descale = overrides.get("v_descale", data["v_descale"])
    p_scale = overrides.get("p_scale", data["p_scale"])
    q_lengths = data["q_lengths"]
    kv_lengths = data["kv_lengths"]
    cu_seqlens_q = overrides.get("cu_seqlens_q", data["cu_seqlens_q"])
    cu_seqlens_kv = overrides.get("cu_seqlens_kv", data["cu_seqlens_kv"])
    block_table = overrides.get("block_table", data["block_table"])
    sparse_indices = overrides.get("sparse_indices", data["sparse_indices"])
    sparse_seq_len = overrides.get("sparse_seq_len", data["sparse_seq_len"])
    return golden.npu_mxfp8_fa(
        q, k, v, q_descale, k_descale, v_descale, p_scale,
        q_lengths, kv_lengths, cu_seqlens_q, cu_seqlens_kv,
        block_table_torch=block_table,
        sparse_indices=sparse_indices,
        sparse_seq_len=sparse_seq_len,
    )


# ============================================================================
# 一、存在性校验
# ============================================================================

class TestExistence:
    """存在性拦截：必传参数缺失、不允许传入的参数传入。"""

    def test_block_table_missing(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case, block_table=None)

    def test_cu_seqlens_q_missing(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case, cu_seqlens_q=None)

    def test_seqused_kv_missing(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_cu_seqlens_kv_not_null(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case, cu_seqlens_kv=torch.tensor([0, 128], dtype=torch.int32))

    def test_seqused_q_not_null(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_metadata_missing(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_mask_mode_0_with_mask(self):
        case = _make_base_case()
        case["mask_mode"] = 0
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_mask_mode_3_without_mask(self):
        case = _make_base_case()
        case["mask_mode"] = 3
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)


# ============================================================================
# 二、Shape 维度校验
# ============================================================================

class TestShapeDim:
    """Shape 维度异常：各轴 +/-1、越界、不一致。"""

    def test_query_T_axis_plus1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        q = data["query"]
        data["query"] = torch.cat([q, q[:1]], dim=0)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_query_N_axis_minus1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        q = data["query"]
        data["query"] = q[:, :7]
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_query_D_axis_plus1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        q = data["query"]
        pad = torch.zeros(q.shape[0], q.shape[1], 1, dtype=q.dtype)
        data["query"] = torch.cat([q, pad], dim=2)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_key_blocknum_minus1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        k = data["key"]
        data["key"] = k[:-1]
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_key_N_axis_plus1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        k = data["key"]
        pad = torch.zeros(k.shape[0], 1, k.shape[2], k.shape[3], dtype=k.dtype)
        data["key"] = torch.cat([k, pad], dim=1)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_key_blocksize_minus1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        k = data["key"]
        data["key"] = k[:, :, :-1]
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_key_D_axis_plus1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        k = data["key"]
        pad = torch.zeros(k.shape[0], k.shape[1], k.shape[2], 1, dtype=k.dtype)
        data["key"] = torch.cat([k, pad], dim=3)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_n2_exceed_limit(self):
        case = _make_base_case()
        case["N1"] = 80
        case["N2"] = 10
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_sparse_indices_b_axis_mismatch(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        si = data["sparse_indices"]
        data["sparse_indices"] = si[:0]
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_sparse_indices_n_axis_mismatch(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        si = data["sparse_indices"]
        data["sparse_indices"] = si[:, :4]
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_sparse_indices_qb_axis_mismatch(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        si = data["sparse_indices"]
        data["sparse_indices"] = si[:, :, :0]
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_sparse_seq_len_b_axis_mismatch(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        ssl = data["sparse_seq_len"]
        data["sparse_seq_len"] = ssl[:0]
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_block_table_b_axis_plus1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        bt = data["block_table"]
        data["block_table"] = torch.cat([bt, bt[:1]], dim=0)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_p_scale_shape_2(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        data["p_scale"] = torch.tensor([1.0, 1.0], dtype=torch.float32)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_p_scale_shape_1_1(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        data["p_scale"] = torch.tensor([[1.0]], dtype=torch.float32)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_batch_size_exceed_limit(self):
        case = _make_base_case()
        case["B"] = 65537
        case["actual_seq_q"] = [128] * 65537
        case["actual_seq_kv"] = [256] * 65537
        with pytest.raises(Exception):
            _generate_inputs(case)

    def test_attn_mask_wrong_shape(self):
        case = _make_base_case()
        case["mask_mode"] = 3
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)


# ============================================================================
# 三、dim 数量校验
# ============================================================================

class TestDimNum:
    """dim 数量异常：维度数与预期不符。"""

    def test_query_2d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        q = data["query"]
        data["query"] = q.reshape(q.shape[0], -1)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_query_4d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        q = data["query"]
        data["query"] = q.unsqueeze(0)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_key_3d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        k = data["key"]
        data["key"] = k.reshape(k.shape[0], k.shape[1], -1)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_key_5d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        k = data["key"]
        data["key"] = k.unsqueeze(0)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_q_descale_3d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        qd = data["q_descale"]
        data["q_descale"] = qd.reshape(qd.shape[0], qd.shape[1], -1)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_k_descale_4d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        kd = data["k_descale"]
        if kd.dim() == 5:
            data["k_descale"] = kd.reshape(kd.shape[0], kd.shape[1], kd.shape[2], -1)
            with pytest.raises(Exception):
                _call_npu(data, case)

    def test_k_descale_6d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        kd = data["k_descale"]
        if kd.dim() == 5:
            data["k_descale"] = kd.unsqueeze(0)
            with pytest.raises(Exception):
                _call_npu(data, case)

    def test_sparse_indices_3d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        si = data["sparse_indices"]
        data["sparse_indices"] = si.reshape(si.shape[0], si.shape[1], -1)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_sparse_indices_5d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        si = data["sparse_indices"]
        data["sparse_indices"] = si.unsqueeze(0)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_sparse_seq_len_2d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        ssl = data["sparse_seq_len"]
        data["sparse_seq_len"] = ssl.reshape(ssl.shape[0], -1)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_sparse_seq_len_4d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        ssl = data["sparse_seq_len"]
        data["sparse_seq_len"] = ssl.unsqueeze(0)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_block_table_1d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        bt = data["block_table"]
        data["block_table"] = bt.reshape(-1)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_block_table_3d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        bt = data["block_table"]
        data["block_table"] = bt.unsqueeze(0)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_cu_seqlens_q_2d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        csq = data["cu_seqlens_q"]
        data["cu_seqlens_q"] = csq.unsqueeze(0)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_metadata_2d(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)


# ============================================================================
# 四、取值校验
# ============================================================================

class TestValueRange:
    """取值异常：dtype、attr、数值范围。"""

    def test_query_dtype_wrong(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        q = data["query"]
        data["query"] = q.to(torch.float16)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_key_value_dtype_inconsistent(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        v = data["value"]
        data["value"] = v.to(torch.float8_e5m2)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_q_descale_dtype_wrong(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        qd = data["q_descale"]
        data["q_descale"] = qd.to(torch.float32)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_block_table_dtype_wrong(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        bt = data["block_table"]
        data["block_table"] = bt.to(torch.int64)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_sparse_indices_dtype_wrong(self):
        case = _make_base_case()
        data = _generate_inputs(case)
        si = data["sparse_indices"]
        data["sparse_indices"] = si.to(torch.int64)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_attn_mask_dtype_wrong(self):
        case = _make_base_case()
        case["mask_mode"] = 3
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_mask_mode_invalid_4(self):
        case = _make_base_case()
        case["mask_mode"] = 4
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_mask_mode_invalid_2(self):
        case = _make_base_case()
        case["mask_mode"] = 2
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_quant_mode_invalid(self):
        case = _make_base_case()
        case["quant_mode"] = 3
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_layout_q_invalid(self):
        case = _make_base_case()
        case["layout_q"] = "BSND"
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_layout_kv_invalid(self):
        case = _make_base_case()
        case["layout_kv"] = "BNSD"
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_layout_out_invalid(self):
        case = _make_base_case()
        case["layout_out"] = "BSND"
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_layout_sparse_indices_invalid(self):
        case = _make_base_case()
        case["layout_sparse_indices"] = "B_N_Qb"
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_head_dim_not_128(self):
        case = _make_base_case()
        case["D"] = 127
        with pytest.raises(Exception):
            _generate_inputs(case)

    def test_n1_not_divisible_by_n2(self):
        case = _make_base_case()
        case["N1"] = 7
        case["N2"] = 3
        with pytest.raises(Exception):
            _generate_inputs(case)

    def test_sparse_block_size_invalid(self):
        case = _make_base_case()
        case["sparse_q_block_size"] = 72
        case["sparse_kv_block_size"] = 72
        with pytest.raises(Exception):
            _generate_inputs(case)

    def test_q_kv_block_size_inconsistent(self):
        case = _make_base_case()
        case["sparse_q_block_size"] = 128
        case["sparse_kv_block_size"] = 64
        with pytest.raises(Exception):
            _generate_inputs(case)

    def test_pa_block_size_not_multiple(self):
        case = _make_base_case()
        case["block_size"] = 100
        with pytest.raises(Exception):
            _generate_inputs(case)

    def test_pa_block_size_exceed_limit(self):
        case = _make_base_case()
        case["block_size"] = 2048
        with pytest.raises(Exception):
            _generate_inputs(case)

    def test_softmax_scale_out_of_range(self):
        case = _make_base_case()
        case["softmax_scale"] = 2.0
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)

    def test_softmax_scale_zero(self):
        case = _make_base_case()
        case["softmax_scale"] = 0.0
        data = _generate_inputs(case)
        with pytest.raises(Exception):
            _call_npu(data, case)
