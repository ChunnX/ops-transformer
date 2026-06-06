#!/usr/bin/python
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

import math

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import CaseConfig

# 下标与 aclnn_sparse_attn_sharedkv.yaml inputs 顺序一致（对齐 sparse_attn_sharedkv_def.cpp）
Q_INDEX = 0
ORI_KV_INDEX = 1
CMP_KV_INDEX = 2
ORI_SPARSE_INDICES_INDEX = 3
CMP_SPARSE_INDICES_INDEX = 4
ORI_BLOCK_TABLE_INDEX = 5
CMP_BLOCK_TABLE_INDEX = 6
CU_SEQLENS_Q_INDEX = 7
CU_SEQLENS_ORI_KV_INDEX = 8
CU_SEQLENS_CMP_KV_INDEX = 9
SEQUSED_Q_INDEX = 10
SEQUSED_KV_INDEX = 11
SINKS_INDEX = 12
METADATA_INDEX = 13
CMP_RATIO_INDEX = 15
LAYOUT_Q_INDEX = 22
LAYOUT_KV_INDEX = 23

N1 = 64
N2 = 1
D = 512
BLOCK_SIZE = 128
ORI_BLOCK_NUM = 65
CMP_BLOCK_NUM = 17
SCFA_TOPK = 512


def _template_mode_from_cmp_ratio(cmp_ratio):
    if cmp_ratio == 1:
        return "SWA"
    if cmp_ratio == 128:
        return "CFA"
    if cmp_ratio == 4:
        return "SCFA"
    return "SWA"


def _cmp_ratio_from_case(case_config: CaseConfig) -> int:
    cmp_ratio = case_config.inputs[CMP_RATIO_INDEX].range_values
    if isinstance(cmp_ratio, list):
        cmp_ratio = cmp_ratio[0] if len(cmp_ratio) == 1 else cmp_ratio
    return int(cmp_ratio)


def _range_values_as_int_list(val, default):
    if val is None:
        return default
    if isinstance(val, list):
        if len(val) == 0:
            return default
        if isinstance(val[0], (list, tuple)):
            return [int(x) for x in val[0]]
        return [int(x) for x in val]
    return [int(val)]


def _lengths_from_case_config(case_config: CaseConfig):
    """序列参数沿用 case 配置（yaml/JSON），不按 decode/prefill 改写。"""
    t1 = int(case_config.inputs[Q_INDEX].shape[0])
    seqused_kv = _range_values_as_int_list(
        case_config.inputs[SEQUSED_KV_INDEX].range_values, [t1])
    cu_seqlens = _range_values_as_int_list(
        case_config.inputs[CU_SEQLENS_Q_INDEX].range_values, [0, t1])
    return t1, t1, cu_seqlens, seqused_kv


@GENERATOR_REGISTRY.register("ascend_generate_sparse_attn_sharedkv")
class SparseAttnSharedkvGenerator(CaseGenerator):
    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        cmp_ratio = _cmp_ratio_from_case(case_config)
        template_mode = _template_mode_from_cmp_ratio(cmp_ratio)
        t1, s1, cu_seqlens, seqused_kv = _lengths_from_case_config(case_config)

        b = 1
        ori_max_s2 = max(seqused_kv)
        ori_block_num_per_batch = math.ceil(ori_max_s2 / BLOCK_SIZE)

        q_dtype = case_config.inputs[Q_INDEX].dtype
        case_config.inputs[ORI_KV_INDEX].dtype = q_dtype
        case_config.inputs[CMP_KV_INDEX].dtype = q_dtype

        case_config.inputs[Q_INDEX].shape = [t1, N1, D]
        case_config.inputs[ORI_KV_INDEX].shape = [ORI_BLOCK_NUM, BLOCK_SIZE, N2, D]
        case_config.inputs[ORI_BLOCK_TABLE_INDEX].shape = [b, ori_block_num_per_batch]
        case_config.inputs[CU_SEQLENS_Q_INDEX].shape = [b + 1]
        case_config.inputs[SEQUSED_KV_INDEX].shape = [b]
        case_config.inputs[SINKS_INDEX].shape = [N1]
        case_config.inputs[METADATA_INDEX].shape = [1024]

        # def.cpp: ori_sparse_indices 当前不支持，保持空 tensor
        case_config.inputs[ORI_SPARSE_INDICES_INDEX].shape = [0]
        case_config.inputs[CU_SEQLENS_ORI_KV_INDEX].shape = [0]
        case_config.inputs[CU_SEQLENS_CMP_KV_INDEX].shape = [0]
        case_config.inputs[SEQUSED_Q_INDEX].shape = [0]

        if template_mode == "SCFA":
            cmp_max_s2 = math.floor(ori_max_s2 / cmp_ratio)
            cmp_block_num_per_batch = math.ceil(cmp_max_s2 / BLOCK_SIZE)
            case_config.inputs[CMP_KV_INDEX].shape = [CMP_BLOCK_NUM, BLOCK_SIZE, N2, D]
            case_config.inputs[CMP_BLOCK_TABLE_INDEX].shape = [b, cmp_block_num_per_batch]
            case_config.inputs[CMP_SPARSE_INDICES_INDEX].shape = [t1, N2, SCFA_TOPK]
            case_config.inputs[CMP_SPARSE_INDICES_INDEX].dtype = "int32"
        elif template_mode == "CFA":
            cmp_max_s2 = math.floor(ori_max_s2 / cmp_ratio)
            cmp_block_num_per_batch = math.ceil(cmp_max_s2 / BLOCK_SIZE)
            case_config.inputs[CMP_KV_INDEX].shape = [CMP_BLOCK_NUM, BLOCK_SIZE, N2, D]
            case_config.inputs[CMP_BLOCK_TABLE_INDEX].shape = [b, cmp_block_num_per_batch]
            case_config.inputs[CMP_SPARSE_INDICES_INDEX].shape = [0]
        else:
            case_config.inputs[CMP_KV_INDEX].shape = [0]
            case_config.inputs[CMP_BLOCK_TABLE_INDEX].shape = [0]
            case_config.inputs[CMP_SPARSE_INDICES_INDEX].shape = [0]

        case_config.inputs[CU_SEQLENS_Q_INDEX].range_values = cu_seqlens
        case_config.inputs[SEQUSED_KV_INDEX].range_values = seqused_kv
        return case_config
