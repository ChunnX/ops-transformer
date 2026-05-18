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

TestCasesFIA = {
    'aclnnFusedInferAttentionScoreV5_FIA_15_36_4_630_3072_case3': {
        'layout_q': 'BNSD',
        'layout_kv': 'BNSD',
        'layout_out': 'BNSD',
        'Dtype': 'fp16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 0,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 15,
        'N1': 36,
        'S1': 630,
        'D': 128,
        'N2': 4,
        'S2': 3072,
    },
    'aclnnFusedInferAttentionScoreV5_FIA_16_8_8_4096_4096_case41': {
        'layout_q': 'BNSD',
        'layout_kv': 'BNSD',
        'layout_out': 'BNSD',
        'Dtype': 'bf16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 0,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 16,
        'N1': 4096,
        'S1': 8,
        'D': 128,
        'N2': 4096,
        'S2': 8,
    },
    # 'aclnnFusedInferAttentionScoreV5_FIA_24_16_1_2206_2048_case49': {
    #     'layout_q': 'TND',
    #     'layout_kv': 'TND',
    #     'layout_out': 'TND',
    #     'Dtype': 'bf16',
    #     'q_range': (-5.0, 5.0),
    #     'k_range': (-5.0, 5.0),
    #     'v_range': (-5.0, 5.0),
    #     'mask_mode': 0,
    #     'D': 128,
    #     'N1': 1,
    #     'N2': 1,
    #     'cu_seqlens_q': [0, 586, 1180, 3149, 3186, 5216, 6778, 8031, 10003, 11078, 13108, 14950, 16701, 18718, 19934, 21136, 22634, 23795, 24212, 25687, 25719, 27795, 28192, 29494, 29640],
    #     'cu_seqlens_kv': [0, 1919, 3023, 3774, 4937, 5164, 5576, 7117, 8066, 9923, 11309, 12360, 14040, 14239, 15184, 15339, 16086, 17372, 17372, 19066, 19785, 20802, 21123, 22587, 24241],
    # },
    "aclnnFusedInferAttentionScoreV5_FIA_24_16_1_2206_2048_case49": {
        "N1": 1,
        "N2": 1,
        # cu_seqlens: (B+1,) 累积偏移 [0, s1, s1+s2, ...]，此处与 seqused 精确对应（无 padding）
        # 真实 padding 场景中 cu_seqlens 对应分配量、seqused 对应实际量，两者不同；
        # 此处为保证 CPU/NPU 张量尺寸一致取无 padding 形式，四个参数仍独立提供。
        "cu_seqlens_q": [0, 586, 1180, 3149, 3186, 5216, 6778, 8031, 10003, 11078, 13108, 14950, 16701, 18718, 19934, 21136, 22634, 23795, 24212, 25687, 25719, 27795, 28192, 29494, 29640],
        "cu_seqlens_kv": [0, 1919, 3023, 3774, 4937, 5164, 5576, 7117, 8066, 9923, 11309, 12360, 14040, 14239, 15184, 15339, 16086, 17372, 17372, 19066, 19785, 20802, 21123, 22587, 24241],
        # # seqused: (B,) 各请求实际使用长度
        # "seqused_q":  [118, 118, 118, 118, 118, 118, 118, 118],       # sequsedQOptional
        # "seqused_kv": [118, 118, 118, 118, 118, 118, 118, 118],    # sequsedKvOptional
        "D": 128,
        "layout_q":   "TND",
        "layout_kv":  "TND",
        "layout_out": "TND",
        "Dtype": "bf16",
        "mask_mode": 0,
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    'aclnnFusedInferAttentionScoreV5_FIA_48_2_2_256_1600_case52': {
        'layout_q': 'BSND',
        'layout_kv': 'BSND',
        'layout_out': 'BSND',
        'Dtype': 'fp16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 0,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 48,
        'S1': 256,
        'S2': 256,
        'N1': 2,
        'N2': 2,
        'D': 128,
    },
    'aclnnFusedInferAttentionScoreV5_FIA_16_104_13_224_256_case53': {
        'layout_q': 'BSND',
        'layout_kv': 'BSND',
        'layout_out': 'BSND',
        'Dtype': 'bf16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 0,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 16,
        'N1': 104,
        'S1': 224,
        'D': 128,
        'N2': 13,
        'S2': 256,
    },
    'aclnnFusedInferAttentionScoreV5_FIA_51_160_20_59_256_case56': {
        'layout_q': 'BSND',
        'layout_kv': 'BSND',
        'layout_out': 'BSND',
        'Dtype': 'bf16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 0,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 51,
        'N1': 160,
        'S1': 59,
        'D': 128,
        'N2': 20,
        'S2': 256,
    },
    'aclnnFusedInferAttentionScoreV5_FIA_16_180_12_253_1600_case69': {
        'layout_q': 'BNSD',
        'layout_kv': 'BNSD',
        'layout_out': 'BNSD',
        'Dtype': 'bf16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 0,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 16,
        'N1': 180,
        'S1': 253,
        'D': 128,
        'N2': 12,
        'S2': 1600,
    },
    'aclnnFusedInferAttentionScoreV5_FIA_8_64_4_512_512_case87': {
        'layout_q': 'BNSD',
        'layout_kv': 'BNSD',
        'layout_out': 'BNSD',
        'Dtype': 'fp16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 3,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 8,
        'N1': 64,
        'S1': 512,
        'D': 128,
        'N2': 4,
        'S2': 512,
    },
    'aclnnFusedInferAttentionScoreV5_FIA_32_32_4_1_1600_case110': {
        'layout_q': 'BNSD',
        'layout_kv': 'BSND',
        'layout_out': 'BNSD',
        'Dtype': 'fp16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 0,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 32,
        'N1': 32,
        'S1': 1,
        'D': 128,
        'N2': 1600,
        'S2': 4,
    },
    'aclnnFusedInferAttentionScoreV5_FIA_30_160_5_4_3583_case119': {
        'layout_q': 'BSND',
        'layout_kv': 'BSND',
        'layout_out': 'BSND',
        'Dtype': 'bf16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 3,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 30,
        'N1': 160,
        'S1': 4,
        'D': 128,
        'N2': 5,
        'S2': 3583,
    },
    'aclnnFusedInferAttentionScoreV5_FIA_30_64_2_3089_4096_case123': {
        'layout_q': 'BNSD',
        'layout_kv': 'BNSD',
        'layout_out': 'BNSD',
        'Dtype': 'fp16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 3,
        'winLeft': '-1.0',
        'winRight': '-1.0',
        'B': 30,
        'N1': 64,
        'S1': 3089,
        'D': 128,
        'N2': 2,
        'S2': 4096,
    },
}