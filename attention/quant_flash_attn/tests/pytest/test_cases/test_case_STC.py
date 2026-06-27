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

TestCasesQFA = {
    'BNSD_S1_128_S2_512': {
        'b': [1],
        'n1': [1],
        'n2': [1],
        'g': [1],
        's1': [128],
        's2': [512],
        'qk_d': [128],
        'v_d': [128],
        'input_layout': ['BNSD'],
        'kv_storage_mode': ['continue'],
        'q_dtype': ['fp4_e2m1'],
        'kv_dtype': ['fp4_e2m1'],
        'out_dtype': ['bfloat16'],
        'q_range': [(-1.0, 1.0)],
        'k_range': [(-1.0, 1.0)],
        'v_range': [(-1.0, 1.0)],
        'q_quant_mode': [3],
        'k_quant_mode': [3],
        'v_quant_mode': [3],
        'act_seq_lens_q': [[128]],
        "act_seq_lens_kv": [[512]],
    },
    'BNSD_S1_128_S2_1024': {
        'b': [1],
        'n1': [1],
        'n2': [1],
        'g': [1],
        's1': [128],
        's2': [1024],
        'qk_d': [128],
        'v_d': [128],
        'input_layout': ['BNSD'],
        'kv_storage_mode': ['continue'],
        'q_dtype': ['fp4_e2m1'],
        'kv_dtype': ['fp4_e2m1'],
        'out_dtype': ['bfloat16'],
        'q_range': [(-1.0, 1.0)],
        'k_range': [(-1.0, 1.0)],
        'v_range': [(-1.0, 1.0)],
        'q_quant_mode': [3],
        'k_quant_mode': [3],
        'v_quant_mode': [3],
        'act_seq_lens_q': [[128]],
        "act_seq_lens_kv": [[1024]],
    },
    'BNSD_S1_128_S2_4096': {
        'b': [1],
        'n1': [1],
        'n2': [1],
        'g': [1],
        's1': [128],
        's2': [4096],
        'qk_d': [128],
        'v_d': [128],
        'input_layout': ['BNSD'],
        'kv_storage_mode': ['continue'],
        'q_dtype': ['fp4_e2m1'],
        'kv_dtype': ['fp4_e2m1'],
        'out_dtype': ['bfloat16'],
        'q_range': [(-1.0, 1.0)],
        'k_range': [(-1.0, 1.0)],
        'v_range': [(-1.0, 1.0)],
        'q_quant_mode': [3],
        'k_quant_mode': [3],
        'v_quant_mode': [3],
        'act_seq_lens_q': [[128]],
        "act_seq_lens_kv": [[4096]],
    },
    'BNSD_S1_128_S2_8192': {
        'b': [1],
        'n1': [1],
        'n2': [1],
        'g': [1],
        's1': [128],
        's2': [8192],
        'qk_d': [128],
        'v_d': [128],
        'input_layout': ['BNSD'],
        'kv_storage_mode': ['continue'],
        'q_dtype': ['fp4_e2m1'],
        'kv_dtype': ['fp4_e2m1'],
        'out_dtype': ['bfloat16'],
        'q_range': [(-1.0, 1.0)],
        'k_range': [(-1.0, 1.0)],
        'v_range': [(-1.0, 1.0)],
        'q_quant_mode': [3],
        'k_quant_mode': [3],
        'v_quant_mode': [3],
        'act_seq_lens_q': [[128]],
        "act_seq_lens_kv": [[8192]],
    },
}

TestCases = TestCasesQFA
