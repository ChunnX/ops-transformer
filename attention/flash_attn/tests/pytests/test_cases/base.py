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

# mask_mode:可选; sparse模式, [0, 3, 4]
# win_left:可选; 配合mask_mode使用
# win_right:可选; 配合mask_mode使用

# block_table:可选; Paged Attention模式下使用
# block_size:可选; Paged Attention模式下使用

TestCases = {
    'TND_MIXED_SEQUSED_01': {
        'B': [4],
        'N1': [1],
        'N2': [1],
        'S1': [128],
        'S2': [128],
        'cu_seqlens_q': [[0, 128, 256, 384, 512]],
        'cu_seqlens_kv': [[0, 128, 256, 384, 512]],
        'seqused_q': [[64, 128, 128, 128]],
        'seqused_kv': [[128, 100, 128, 128]],
        'D': [128],
        'layout_q': ['TND'],
        'layout_kv': ['TND'],
        'layout_out': ['TND'],
        'Dtype': ['fp16'],
        'mask_mode': [0],
        'q_range': [(-10.0, 10.0)],
        'k_range': [(-10.0, 10.0)],
        'v_range': [(-10.0, 10.0)],
    },
    'BASE_04': {
        'B': [1],
        'N1': [19],
        'N2': [19],
        'S1': [1024],
        'S2': [1024],
        'D': [128],
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BNSD'],
        'Dtype': ['fp16'],
        'mask_mode': [0],
        'seqused_q': [[640]],
        'seqused_kv': [[1024]],
        'q_range': [(-10, 10)],
        'k_range': [(-10, 10)],
        'v_range': [(-10, 10)],
    },
    'SECTION_01': {
        'B': [20],
        'N1': [19],
        'N2': [19],
        'S1': [640],
        'S2': [1024],
        'D': [128],
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BNSD'],
        'Dtype': ['fp16'],
        'mask_mode': [0],
        'q_range': [(-10, 10)],
        'k_range': [(-10, 10)],
        'v_range': [(-10, 10)],
    },
    'BSND_01': {
        'B': [1],
        'N1': [1],
        'N2': [1],
        'S1': [64],
        'S2': [256],
        'D': [128],
        'layout_q': ['BSND'],
        'layout_kv': ['BSND'],
        'layout_out': ['BSND'],
        'Dtype': ['fp16'],
        'mask_mode': [0],
        'q_range': [(-10, 10.0)],
        'k_range': [(-10, 10.0)],
        'v_range': [(-10, 10.0)],
    },
    'Transpose_01': {
        'B': [1],
        'N1': [1],
        'N2': [1],
        'S1': [64],
        'S2': [256],
        'D': [128],
        'layout_q': ['BNSD'],
        'layout_kv': ['BNSD'],
        'layout_out': ['BSND'],
        'Dtype': ['fp16'],
        'mask_mode': [0],
        'q_range': [(-10.0, 10.0)],
        'k_range': [(-10.0, 10.0)],
        'v_range': [(-10.0, 10.0)],
    },
    'TND_MIXED_01': {
        'B': [4],
        'N1': [1],
        'N2': [1],
        'S1': [128],
        'S2': [128],
        'cu_seqlens_q': [[0, 128, 256, 384, 512]],
        'cu_seqlens_kv': [[0, 128, 256, 384, 512]],
        'seqused_q': [[128, 128, 128, 128]],
        'seqused_kv': [[128, 128, 128, 128]],
        'D': [128],
        'layout_q': ['TND'],
        'layout_kv': ['TND'],
        'layout_out': ['TND'],
        'Dtype': ['fp16'],
        'mask_mode': [3],
        'q_range': [(-10.0, 10.0)],
        'k_range': [(-10.0, 10.0)],
        'v_range': [(-10.0, 10.0)],
        'return_softmax_lse' : [1],
    },
    'TND_MIXED_02': {
        'B': [8],
        'N1': [4],
        'N2': [4],
        'cu_seqlens_q': [[0, 118, 236, 354, 472, 590, 708, 826, 944]],
        'cu_seqlens_kv': [[0, 118, 236, 354, 472, 590, 708, 826, 944]],
        'seqused_q': [[118, 118, 118, 118, 118, 118, 118, 118]],
        'seqused_kv': [[118, 118, 118, 118, 118, 118, 118, 118]],
        'D': [128],
        'layout_q': ['TND'],
        'layout_kv': ['TND'],
        'layout_out': ['TND'],
        'Dtype': ['fp16'],
        'mask_mode': [3],
        'q_range': [(-10.0, 10.0)],
        'k_range': [(-10.0, 10.0)],
        'v_range': [(-10.0, 10.0)],
    },
}
