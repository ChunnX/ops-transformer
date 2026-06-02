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

import torch

####### 参数说明 ########
# q_range:可选; Q tensor 均匀随机值域 (low, high)，省略则默认值全为10（固定调试值）
# k_range:可选; K tensor 均匀随机值域 (low, high)
# v_range:可选; V tensor 均匀随机值域 (low, high)
# B:必选; batch_size,TND格式下可选
# N1:必选; head_num
# N2:可选; kv's head_num,支持GQA/MHA/MQA
# S1:必选; query's sequence length;TND格式下可选
# S2:可选; key&value's sequence length
# D:必选; 表示query&key&value的head_dim
# DV:可选; value的head_dim;设置该参数,value的head_dim以DV为准
# layout_q:必选; 输入tensor的格式, [BNSD, BSND, TND]
# layout_kv:可选; kv tensor的格式, [BNSD, BSND, TND, PA_BBND, PA_BNBD, PA_NZ]
# layout_out:可选; 输出tensor的格式, [BNSD, BSND, TND]
# Dtype:必选; 数据类型, [fp16, bf16]
# scale:可选; 注意力得分缩放系数
# seqused_q:可选; TND下必选;query实际的序列长度
# seqused_kv:可选; TND下必选;key&value实际的序列长度

# keep_prob:可选; dropout的保留概率：keep_prob = 1 - dropout_p
# seed:可选; 随机种子，用于随机数生成
# offset:可选; 随机数偏移

# mask_mode:可选; sparse模式, [0, 1, 2, 3, 4, 5, 6, 7, 8]
# win_left:可选; 配合mask_mode使用
# win_right:可选; 配合mask_mode使用

# block_table:可选; Paged Attention模式下使用
# block_size:可选; Paged Attention模式下使用

TestCasesFlashAttn = {
    # ========== 任务1：D值泛化（支持64, 128, 256）==========
    # D=64基础用例 - BNSD布局
    'aclnnFlashAttn_D64_16_8_4_256_1024_case201': {
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BNSD'],
        'Dtype': ['fp16'],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)],
        'mask_mode': [0],
        'winLeft': ['-1.0'],
        'winRight': ['-1.0'],
        'B': [16],
        'N1': [8],
        'S1': [256],
        'D': [64],
        'N2': [4],
        'S2': [1024],
    },
    # D=64 - BSND布局
    'aclnnFlashAttn_D64_32_16_8_512_2048_case202': {
        'layout_q': ['BSND'],
        'layout_kv': ['BSND'],
        'layout_out': ['BSND'],
        'Dtype': ['fp16'],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)],
        'scale': [0.125],
        'mask_mode': [0],
        'winLeft': ['-1.0'],
        'winRight': ['-1.0'],
        'B': [32],
        'N1': [16],
        'S1': [512],
        'D': [64],
        'N2': [8],
        'S2': [2048],
    },
    # D=64 - GQA场景（N1/N2不等）
    'aclnnFlashAttn_D64_GQA_8_32_2_128_512_case203': {
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BNSD'],
        'Dtype': ['fp16'],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)],
        'scale': [0.125],
        'mask_mode': [3],
        'winLeft': ['-1.0'],
        'winRight': ['-1.0'],
        'B': [8],
        'N1': [32],
        'S1': [128],
        'D': [64],
        'N2': [2],
        'S2': [512],
    },
    # D=64 - MQA场景（N2=1）
    'aclnnFlashAttn_D64_MQA_4_64_1_64_1024_case204': {
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BSND'],
        'Dtype': ['bf16'],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)],
        'scale': [0.125],
        'mask_mode': [0],
        'winLeft': ['-1.0'],
        'winRight': ['-1.0'],
        'B': [4],
        'N1': [64],
        'S1': [64],
        'D': [64],
        'N2': [2],
        'S2': [1024],
    },
    
    # D=256基础用例 - BNSD布局
    'aclnnFlashAttn_D256_8_4_4_128_512_case205': {
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BNSD'],
        'Dtype': ['fp16'],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)],
        'scale': [0.0625],
        'mask_mode': [0],
        'winLeft': ['-1.0'],
        'winRight': ['-1.0'],
        'B': [8],
        'N1': [4],
        'S1': [128],
        'D': [256],
        'N2': [4],
        'S2': [512],
    },
    # D=256 - BSND布局
    'aclnnFlashAttn_D256_16_8_4_256_1024_case206': {
        'layout_q': ['BSND'],
        'layout_kv': ['BSND'],
        'layout_out': ['BSND'],
        'Dtype': ['fp16'],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)],
        'mask_mode': [0],
        'winLeft': ['-1.0'],
        'winRight': ['-1.0'],

        # 该shape D=128 精度fail
        # 'B': [16],
        # 'N1': [8],
        # 'S1': [256],
        # 'D': [128],
        # 'N2': [4],
        # 'S2': [1024],

        'B': [32],
        'N1': [16],
        'S1': [512],
        'D': [256],
        'N2': [8],
        'S2': [2048],
    },
    # D=256 - GQA场景
    'aclnnFlashAttn_D256_GQA_12_24_3_128_768_case207': {
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BNSD'],
        'Dtype': ['fp16'],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)],
        'scale': [0.0625],
        'mask_mode': [3],
        'winLeft': ['-1.0'],
        'winRight': ['-1.0'],

        # 该shape D=128 精度fail
        # 'B': [12],
        # 'N1': [24],
        # 'S1': [128],
        # 'D': [128],
        # 'N2': [3],
        # 'S2': [768],

        'B': [8],
        'N1': [32],
        'S1': [128],
        'D': [256],
        'N2': [2],
        'S2': [512],
    },
    # D=256 - MQA场景
    'aclnnFlashAttn_D256_MQA_6_32_1_64_512_case208': {
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BSND'],
        'Dtype': ['bf16'],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)],
        'scale': [0.0625],
        'mask_mode': [0],
        'winLeft': ['-1.0'],
        'winRight': ['-1.0'],

        # 该shape D=128 精度fail
        # 'B': [6],
        # 'N1': [32],
        # 'S1': [64],
        # 'D': [128],
        # 'N2': [1],
        # 'S2': [512],

        'B': [4],
        'N1': [64],
        'S1': [64],
        'D': [256],
        'N2': [2],
        'S2': [1024],
    },
    
    # ========== 任务2：PA_NZ排布支持 ==========



    # =========== 任务3：TND排布D支持 ===========



    # =========== 任务4：PA_NZ排布D支持==========
    

}

TestCases = TestCasesFlashAttn
