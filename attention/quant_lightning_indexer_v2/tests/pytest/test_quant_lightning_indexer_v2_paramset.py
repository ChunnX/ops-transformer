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
    "mxfp8_1":{
        "batch_size": [1],
        "q_seq": [4],
        "k_seq": [512],
        "q_t_size":[1],
        "k_t_size":[1],#压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[4],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[4]],
        "seqused_k": [[512]], # PA场景每个batch的实际token数
        "cmp_residual_k": [None],
        "max_seqlen_q": [-1],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["BSND"],
        "sparse_count": [128],
        "sparse_mode": [0],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[1], #1/2/4/8/16/32/64/128
        "return_value":[0],
        "output_idx_offset": [None]
    },
    "mxfp8_2":{
        "batch_size": [2],
        "q_seq": [8],
        "k_seq": [2048],
        "q_t_size":[1],
        "k_t_size":[1],#压缩后的值
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[4],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[8, 8]],
        "seqused_k": [[1024, 2048]], # PA场景每个batch的实际token数
        "cmp_residual_k": [None],
        "max_seqlen_q": [-1],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["BSND"],
        "sparse_count": [128],
        "sparse_mode": [0],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[1], #1/2/4/8/16/32/64/128
        "return_value":[0],
        "output_idx_offset": [None]
    },
}

# 按需选择要启用的测试参数（例如默认启用所有）
properties = torch.npu.get_device_properties()
ENABLED_PARAMS = [
    TEST_PARAMS["mxfp8_1"],
    TEST_PARAMS["mxfp8_2"],
    ]