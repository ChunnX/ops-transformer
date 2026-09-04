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

import torch


def build_mtp_short_query_case(
    query_seq_len, sparse_count, qk_dtype=torch.float8_e4m3fn
):
    batch_size = 2
    key_seq_len = 4096
    cmp_ratio = 4
    return {
        "batch_size": [batch_size],
        "q_seq": [query_seq_len],
        "k_seq": [key_seq_len],
        "q_t_size": [batch_size * query_seq_len],
        "k_t_size": [batch_size * key_seq_len],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128],
        "block_num": [16],
        "qk_dtype": [qk_dtype],
        "dequant_dtype": [torch.float32],
        "actual_seq_dtype": [torch.int32],
        "act_seq_q": [[query_seq_len * (b_idx + 1) for b_idx in range(batch_size)]],
        "act_seq_k": [[key_seq_len] * batch_size],
        "query_quant_mode": [0],
        "key_quant_mode": [0],
        "layout_query": ["TND"],
        "layout_key": ["PA_BSND"],
        "sparse_count": [sparse_count],
        "sparse_mode": [3],
        "query_datarange": [[-448, 448]],
        "key_datarange": [[-20, 20]],
        "weights_datarange": [[-123, 123]],
        "q_scale_datarange": [[0, 255]],
        "k_scale_datarange": [[0, 65504]],
        "cmp_ratio": [cmp_ratio],
    }


# 定义测试参数组合
TEST_PARAMS = {
    # 基础场景
    "quant_li_default_a5": {
        "batch_size": [8],
        "q_seq": [15],
        "k_seq": [111],
        "q_t_size": [8],
        "k_t_size": [15],  # 压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [512],  # 取16的整数倍，最多支持到1024
        "block_num": [8],
        "qk_dtype": [torch.float8_e4m3fn],
        "dequant_dtype": [torch.float32],
        "actual_seq_dtype": [torch.int32],
        "act_seq_q": [[3, 3, 3, 3, 3, 3, 3, 3]],
        "act_seq_k": [
            [28, 24, 80, 96, 47, 76, 0, 111]
        ],  # PA场景非前缀和，表示每个batch_size的实际token数
        "query_quant_mode": [0],
        "key_quant_mode": [0],
        "layout_query": ["BSND"],
        "layout_key": ["PA_BSND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange": [[-448, 448]],
        "key_datarange": [[-20, 20]],
        "weights_datarange": [[-123, 123]],
        "q_scale_datarange": [[0, 255]],
        "k_scale_datarange": [[0, 65504]],
        "cmp_ratio": [1],  # 1/2/4/8/16/32/64/128
    },
    "quant_li_default_hifp8_a5": {
        "batch_size": [8],
        "q_seq": [15],
        "k_seq": [111],
        "q_t_size": [8],
        "k_t_size": [15],  # 压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [512],  # 取16的整数倍，最多支持到1024
        "block_num": [8],
        "qk_dtype": [torch.uint8],
        "dequant_dtype": [torch.float32],
        "actual_seq_dtype": [torch.int32],
        "act_seq_q": [[3, 3, 3, 3, 3, 3, 3, 3]],
        "act_seq_k": [
            [28, 24, 80, 96, 47, 76, 0, 111]
        ],  # PA场景非前缀和，表示每个batch_size的实际token数
        "query_quant_mode": [0],
        "key_quant_mode": [0],
        "layout_query": ["BSND"],
        "layout_key": ["PA_BSND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange": [[-448, 448]],
        "key_datarange": [[-20, 20]],
        "weights_datarange": [[-123, 123]],
        "q_scale_datarange": [[0, 255]],
        "k_scale_datarange": [[0, 65504]],
        "cmp_ratio": [1],  # 1/2/4/8/16/32/64/128
    },
    # DeepSeek-V4 decode: MTP1/2/3 map to per-request query lengths 2/3/4.
    "quant_li_mtp0_tnd_fallback_a5": build_mtp_short_query_case(1, 512),
    "quant_li_mtp1_tnd_top512_a5": build_mtp_short_query_case(2, 512),
    "quant_li_mtp2_tnd_top512_a5": build_mtp_short_query_case(3, 512),
    "quant_li_mtp3_tnd_top1024_a5": build_mtp_short_query_case(4, 1024),
    "quant_li_mtp1_tnd_hifp8_top512_a5": build_mtp_short_query_case(
        2, 512, torch.uint8
    ),
    "quant_li_query_len5_tnd_fallback_a5": build_mtp_short_query_case(5, 512),
    "quant_li_default_a3": {
        "batch_size": [1],
        "q_seq": [1],
        "k_seq": [8192],
        "q_t_size": [1],
        "k_t_size": [8192],  # 压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128],  # 取16的整数倍，最多支持到1024
        "block_num": [17],
        "qk_dtype": [torch.int8],
        "dequant_dtype": [torch.float16],
        "actual_seq_dtype": [torch.int32],
        "act_seq_q": [[1]],
        "act_seq_k": [[8196]],  # PA场景非前缀和，表示每个batch_size的实际token数
        "query_quant_mode": [0],
        "key_quant_mode": [0],
        "layout_query": ["TND"],
        "layout_key": ["PA_BSND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange": [[-100, 100]],
        "key_datarange": [[-100, 100]],
        "weights_datarange": [[-25, 25]],
        "q_scale_datarange": [[0, 255]],
        "k_scale_datarange": [[0, 65504]],
        "cmp_ratio": [4],  # 1/2/4/8/16/32/64/128
    },
}

# 按需选择要启用的测试参数（例如默认启用所有）
properties = torch.npu.get_device_properties()
if "Ascend910_93" in properties.name:
    ENABLED_PARAMS = [TEST_PARAMS["quant_li_default_a3"]]
elif "Ascend950" in properties.name:
    ENABLED_PARAMS = [
        TEST_PARAMS["quant_li_default_a5"],
        TEST_PARAMS["quant_li_mtp0_tnd_fallback_a5"],
        TEST_PARAMS["quant_li_mtp1_tnd_top512_a5"],
        TEST_PARAMS["quant_li_mtp2_tnd_top512_a5"],
        TEST_PARAMS["quant_li_mtp3_tnd_top1024_a5"],
        TEST_PARAMS["quant_li_mtp1_tnd_hifp8_top512_a5"],
        TEST_PARAMS["quant_li_query_len5_tnd_fallback_a5"],
    ]
