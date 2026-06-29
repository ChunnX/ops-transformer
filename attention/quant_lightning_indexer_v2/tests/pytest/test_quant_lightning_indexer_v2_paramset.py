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

# 定义测试参数组合
# 参数规则：
#   cu_seqlens_q/k: TND 时必传 [B+1] 前缀和（首元素=0），非 TND 时为 None
#   seqused_q/k:    每个 batch 的实际有效元素数 [B]
#                   TND 时可选（golden 可从 cu_seqlens 推导）
#                   BSND 时可选（None 则用 q_seq/k_seq 填满）
#                   PA_BBND 时 seqused_k 必传
TEST_PARAMS = {
    # Ascend950 基础场景: BSND query + PA_BBND key
    "quant_li_default_a5":{
        "batch_size": [8],
        "q_seq": [15],
        "k_seq": [111],
        "q_t_size":[8],
        "k_t_size":[15],#压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [512], # 取16的整数倍，最多支持到1024
        "block_num":[8],
        "qk_dtype": [torch.float8_e4m3fn],
        "dequant_dtype": [torch.float32],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[3,3,3,3,3,3,3,3]],
        "seqused_k": [[28,24,80,96,47,76,0,111]], # PA场景每个batch的实际token数
        "cmp_residual_k": [None],
        "max_seqlen_q": [-1],
        "quant_mode": [1],
        "layout_query": ["BSND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange":[[-448,448]],
        "key_datarange":[[-20,20]],
        "weights_datarange":[[-123,123]],
        "q_scale_datarange":[[0,255]],
        "k_scale_datarange":[[0,65504]],
        "cmp_ratio":[1], #1/2/4/8/16/32/64/128
        "return_value":[0]
    },

    # Ascend950 hifp8 场景: BSND query + PA_BBND key
    "quant_li_default_hifp8_a5":{
        "batch_size": [3],
        "q_seq": [13],
        "k_seq": [111],
        "q_t_size":[8],
        "k_t_size":[15],#压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [1024], # 取16的整数倍，最多支持到1024
        "block_num":[100],
        "qk_dtype": [torch.uint8],
        "dequant_dtype": [torch.float32],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[2,5,13]],
        "seqused_k": [[2080,2114,1180]],
        "max_seqlen_q": [-1],
        "cmp_residual_k":[[3,1,3]],
        "quant_mode": [4],
        "layout_query": ["BSND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange":[[-448,448]],
        "key_datarange":[[-20,20]],
        "weights_datarange":[[-123,123]],
        "q_scale_datarange":[[0,255]],
        "k_scale_datarange":[[0,65504]],
        "cmp_ratio":[4], #1/2/4/8/16/32/64/128
        "return_value":[0]
    },

    # Ascend910_93 场景: TND query + PA_BBND key
    "quant_li_default_a3":{
        "batch_size": [1],
        "q_seq": [1],
        "k_seq": [8192],
        "q_t_size":[1],
        "k_t_size":[8192],#压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [1024], # 取16的整数倍，最多支持到1024
        "block_num":[17],
        "qk_dtype": [torch.int8],
        "dequant_dtype": [torch.float16],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [[0, 1]],   # TND: cu_seqlens_q 必传 [B+1]
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [None],        # TND: seqused_q 可选，None 时从 cu_seqlens 推导
        "seqused_k": [[8196]],      # PA_BBND: seqused_k 必传
        "cmp_residual_k": [[1]],    # cmp_ratio=4 时需要
        "quant_mode": [2],          # 910_93 tiling 要求 quant_mode=2
        "layout_query": ["TND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange":[[-100,100]],
        "key_datarange":[[-100,100]],
        "weights_datarange":[[-25,25]],
        "q_scale_datarange":[[0,255]],
        "k_scale_datarange":[[0,65504]],
        "cmp_ratio":[4], #1/2/4/8/16/32/64/128
        "max_seqlen_q": [-1],
        "return_value": [0]
    }

}

# 按需选择要启用的测试参数（例如默认启用所有）
properties = torch.npu.get_device_properties()
if "Ascend910_93" in properties.name:
    ENABLED_PARAMS = [TEST_PARAMS["quant_li_default_a3"]]
elif "Ascend950" in properties.name:
    ENABLED_PARAMS = [TEST_PARAMS["quant_li_default_a5"]]