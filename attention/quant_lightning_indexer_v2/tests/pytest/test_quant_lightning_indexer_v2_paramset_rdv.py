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
    # BSND+PA+mask0+s2整除基本块: b=1,s1=1,s2=2048,g=64,blockSize=128,sparse=2048,mask=0,ratio=1
    "rdv_1":{
        "batch_size": [1],
        "q_seq": [1],
        "k_seq": [2048],
        "q_t_size":[1],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[16],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[1]],
        "seqused_k": [[2048]], # PA场景每个batch的实际token数
        "cmp_residual_k": [[0]],
        "max_seqlen_q": [1],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [1024],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[0]
    },
    # BSND+PA+mask0+s2小于一个基本块: b=1,s1=1,s2=120,g=64,blockSize=128,sparse=100,mask=0,ratio=1
    "rdv_2":{
        "batch_size": [1],
        "q_seq": [1],
        "k_seq": [120],
        "q_t_size":[1],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[1],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[1]],
        "seqused_k": [[120]], # PA场景每个batch的实际token数
        "cmp_residual_k": [[0]],
        "max_seqlen_q": [1],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[0]
    },
    # BSND+PA+mask3+s2不整除基本块: b=1,s1=2,s2=3000,g=32,blockSize=256,sparse=2047,mask=3,ratio=2
    "rdv_3":{
        "batch_size": [1],
        "q_seq": [2],
        "k_seq": [3000],
        "q_t_size":[1],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[22],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[1]],
        "seqused_k": [[2800]], # PA场景每个batch的实际token数
        "cmp_residual_k": [[1]],
        "max_seqlen_q": [1],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [1024],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[0]
    },
    # BSND+PA+mask3+刚好满核: b=64,s1=2,s2=3000,g=32,blockSize=256,sparse=2047,mask=3,ratio=2
    "rdv_4":{
        "batch_size": [64],
        "q_seq": [2],
        "k_seq": [3000],
        "q_t_size":[1],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[1408],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2]],
        "seqused_k": [[2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800]], # PA场景每个batch的实际token数
        "cmp_residual_k": [[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]],
        "max_seqlen_q": [2],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [1024],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[0]
    },
    # BSND+PA+mask3+满核还多: b=65,s1=2,s2=3000,g=32,blockSize=256,sparse=2047,mask=3,ratio=2
    "rdv_5":{
        "batch_size": [65],
        "q_seq": [2],
        "k_seq": [3000],
        "q_t_size":[1],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[1560],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1,2,1]],
        "seqused_k": [[2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800,2800]], # PA场景每个batch的实际token数
        "cmp_residual_k": [[1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1]],
        "max_seqlen_q": [2],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [1024],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[0]
    },
    # TND+PA+mask0+s2切成大于基本块临界值: b=1,s1=3,s2=513,g=4,blockSize=896,sparse=512,mask=0,ratio=64
    "rdv_6":{
        "batch_size": [1],
        "q_seq": [3],
        "k_seq": [513],
        "q_t_size":[3],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[5],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [[0,3]],    # TND: 前缀和 [B+1]
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[3]],
        "seqused_k": [[513]], # PA场景每个batch的实际token数
        "cmp_residual_k": [[1]],
        "max_seqlen_q": [3],
        "quant_mode": [3],
        "layout_query": ["TND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[1]
    },
    # BSND+mask0+不传seqused+s2切成小于基本块临界值: b=2,s1=16,s2=3967,g=16,blockSize=384,sparse=2046,mask=0,ratio=4
    "rdv_7":{
        "batch_size": [2],
        "q_seq": [16],
        "k_seq": [3967],
        "q_t_size":[1],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[32],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # BSND: cu_seqlens_k 不传
        "seqused_q": [None],
        "seqused_k": [None],
        "cmp_residual_k": [[0,0]],
        "max_seqlen_q": [16],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["BSND"],
        "sparse_count": [1024],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[0]
    },
    # TND+mask3+s2为一半基本块: b=4,s1=64,s2=512,g=64,blockSize=768,sparse=511,mask=3,ratio=32
    "rdv_8":{
        "batch_size": [4],
        "q_seq": [64],
        "k_seq": [512],
        "q_t_size":[64],
        "k_t_size":[512],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[8],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [[0,16,32,48,64]],    # TND: 前缀和 [B+1]
        "cu_seqlens_k": [[0,64,128,192,512]], # TND: 前缀和 [B+1]
        "seqused_q": [[16,16,16,2]],
        "seqused_k": [[64,64,64,64]],
        "cmp_residual_k": [[1,2,3,0]],
        "max_seqlen_q": [16],
        "quant_mode": [3],
        "layout_query": ["TND"],
        "layout_key":["TND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[1]
    },
    # TND+PA+mask3+大batch+s2切成大于基本块临界值: b=48,s1=48,s2=2049,g=5,blockSize=1024,sparse=513,mask=3,ratio=128
    "rdv_9":{
        "batch_size": [48],
        "q_seq": [48],
        "k_seq": [2049],
        "q_t_size":[48],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[1440],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [[0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48]], # TND: 前缀和 [B+1]
        "cu_seqlens_k": [None],     # PA_BBND: cu_seqlens_k 不传
        "seqused_q": [[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]],
        "seqused_k": [[2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049,2048,2049]], # PA场景每个batch的实际token数
        "cmp_residual_k": [[2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1,2,3,0,1]],
        "max_seqlen_q": [1],
        "quant_mode": [3],
        "layout_query": ["TND"],
        "layout_key":["PA_BBND"],
        "sparse_count": [512],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[1]
    },
    # BSND+mask0+s2不整除基本块但是整除16: b=2,s1=15,s2=4000,g=16,blockSize=384,sparse=2046,mask=0,ratio=4
    "rdv_10":{
        "batch_size": [2],
        "q_seq": [15],
        "k_seq": [4000],
        "q_t_size":[1],
        "k_t_size":[1],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[40],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [None],     # BSND: cu_seqlens_q 不传
        "cu_seqlens_k": [None],     # BSND: cu_seqlens_k 不传
        "seqused_q": [[10,12]],
        "seqused_k": [[3999,350]],
        "cmp_residual_k": [[1,2]],
        "max_seqlen_q": [12],
        "quant_mode": [3],
        "layout_query": ["BSND"],
        "layout_key":["BSND"],
        "sparse_count": [1024],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[0]
    },
    # TND+mask0+s2仅最后一个batch整除基本块: b=4,s1=49,s2=2051,g=64,blockSize=640,sparse=2048,mask=0,ratio=16
    "rdv_11":{
        "batch_size": [4],
        "q_seq": [49],
        "k_seq": [2051],
        "q_t_size":[49],
        "k_t_size":[2051],
        "q_head_num": [64],
        "k_head_num": [1],
        "head_dim": [128],
        "block_size": [128], # 取16的整数倍，最多支持到1024
        "block_num":[40],
        "qk_dtype": [torch.float8_e4m3fn],
        "weight_dtype": [torch.float],
        "dequant_dtype": [torch.float8_e8m0fnu],
        "actual_seq_dtype": [torch.int32],
        "cu_seqlens_q": [[0,3,9,20,49]],      # TND: 前缀和 [B+1]
        "cu_seqlens_k": [[0,1,2,3,2051]],     # TND: 前缀和 [B+1]
        "seqused_q": [[1,1,11,28]],
        "seqused_k": [[1,1,1,2048]],
        "cmp_residual_k": [[1,2,3,0]],
        "max_seqlen_q": [29],
        "quant_mode": [3],
        "layout_query": ["TND"],
        "layout_key":["TND"],
        "sparse_count": [1024],
        "sparse_mode": [3],
        "query_datarange":[[-1,1]],
        "key_datarange":[[-1,1]],
        "weights_datarange":[[-130,130]],
        "q_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "k_scale_datarange":[[-3,4]],   # 2的指数范围，各次方等概率
        "cmp_ratio":[4],
        "return_value":[0]
    },
}

# 按需选择要启用的测试参数（例如默认启用所有）
properties = torch.npu.get_device_properties()
ENABLED_PARAMS = [
    TEST_PARAMS["rdv_1"],
    TEST_PARAMS["rdv_2"],
    TEST_PARAMS["rdv_3"],
    TEST_PARAMS["rdv_4"],
    TEST_PARAMS["rdv_5"],
    TEST_PARAMS["rdv_6"],
    TEST_PARAMS["rdv_7"],
    TEST_PARAMS["rdv_8"],
    TEST_PARAMS["rdv_9"],
    TEST_PARAMS["rdv_10"],
    TEST_PARAMS["rdv_11"],
    ]
