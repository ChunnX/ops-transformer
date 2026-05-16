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

TestCases = {
    "BASE_01": {
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 64,
        "S2": 256,
        "D": 128,
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "Dtype": "fp16",
        "mask_mode": 1,
        "q_range": (-10, 10.0),
        "k_range": (-10, 10.0),
        "v_range": (-10, 10.0),
    },
    "BASE_02": {
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 1024,
        "S2": 128,
        "D": 128,
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "Dtype": "fp16",
        "mask_mode": 1,
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    "BASE_03": {
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 512,
        "S2": 1024,
        "D": 128,
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "Dtype": "fp16",
        "mask_mode": 1,
        "q_range": (1, 1.0),
        "k_range": (1, 1.0),
        "v_range": (1, 1.0),
    },
    "BASE_04": {
        "B": 1,
        "N1": 19,
        "N2": 19,
        "S1": 640,
        "S2": 1024,
        "D": 128,
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "Dtype": "fp16",
        "mask_mode": 0,
        "q_range": (-10, 10),
        "k_range": (-10, 10),
        "v_range": (-10, 10),
    },
    "SECTION_01": {
        "B": 20,
        "N1": 19,
        "N2": 19,
        "S1": 640,
        "S2": 1024,
        "D": 128,
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "Dtype": "fp16",
        "mask_mode": 1,
        "q_range": (-10, 10),
        "k_range": (-10, 10),
        "v_range": (-10, 10),
    },
    "BSND_01": {
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 64,
        "S2": 256,
        "D": 128,
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "layout_out": "BSND",
        "Dtype": "fp16",
        "mask_mode": 0,
        "q_range": (-10, 10.0),
        "k_range": (-10, 10.0),
        "v_range": (-10, 10.0),
    },
    "Transpose_01": {
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 64,
        "S2": 256,
        "D": 128,
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "layout_out": "BNSD",
        "Dtype": "fp16",
        "mask_mode": 0,
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    "Transpose_02": {
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 64,
        "S2": 256,
        "D": 128,
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BSND",
        "Dtype": "fp16",
        "mask_mode": 1,
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    "BNSD_01": {
        "B": 1,
        "N1": 1,
        "N2": 1,
        "S1": 128,
        "S2": 128,
        "D": 64,
        "layout_q": "BNSD",
        "layout_kv": "BNSD",
        "layout_out": "BNSD",
        "Dtype": "fp16",
        "mask_mode": 1,
    },
    "BSND_02": {
        "B": 4,
        "N1": 16,
        "N2": 8,
        "S1": 256,
        "S2": 256,
        "D": 128,
        "layout_q": "BSND",
        "layout_kv": "BSND",
        "layout_out": "BSND",
        "Dtype": "fp16",
        "mask_mode": 1,
    },
    "TND_MIXED_01": {
        "B":  4,
        "N1": 1,
        "N2": 1,
        "S1": 128,
        "S2": 128,
        # cu_seqlens: (B+1,) 累积偏移 [0, s1, s1+s2, ...]，此处与 seqused 精确对应（无 padding）
        # 真实 padding 场景中 cu_seqlens 对应分配量、seqused 对应实际量，两者不同；
        # 此处为保证 CPU/NPU 张量尺寸一致取无 padding 形式，四个参数仍独立提供。
        "cu_seqlens_q":  [0, 128, 256, 384, 512],  # (B+1,) cumsum([200,1,128,1])
        "cu_seqlens_kv": [0, 128, 256, 384, 512],   # (B+1,) cumsum([400,128,300,64])   fia:128, 256, 384, 512  1、tiling：tilingdata/tilingkey  TND    2、处理逻辑、
        # # seqused: (B,) 各请求实际使用长度
        "seqused_q":  [128, 128, 128, 128],       # sequsedQOptional
        "seqused_kv": [128, 128, 128, 128],    # sequsedKvOptional
        "D": 128,
        "layout_q":   "TND",
        "layout_kv":  "TND",
        "layout_out": "TND",
        "Dtype": "fp16",
        "mask_mode": 3,
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    "TND_MIXED_02": {
        "B": 8,
        "N1": 4,
        "N2": 4,
        # cu_seqlens: (B+1,) 累积偏移 [0, s1, s1+s2, ...]，此处与 seqused 精确对应（无 padding）
        # 真实 padding 场景中 cu_seqlens 对应分配量、seqused 对应实际量，两者不同；
        # 此处为保证 CPU/NPU 张量尺寸一致取无 padding 形式，四个参数仍独立提供。
        "cu_seqlens_q":  [0, 118, 236, 354, 472, 590, 708, 826, 944],  # (B+1,) cumsum([200,1,128,1])
        "cu_seqlens_kv": [0, 118, 236, 354, 472, 590, 708, 826, 944],   # (B+1,) cumsum([400,128,300,64])   fia:128, 256, 384, 512  1、tiling：tilingdata/tilingkey  TND    2、处理逻辑、
        # # seqused: (B,) 各请求实际使用长度
        "seqused_q":  [118, 118, 118, 118, 118, 118, 118, 118],       # sequsedQOptional
        "seqused_kv": [118, 118, 118, 118, 118, 118, 118, 118],    # sequsedKvOptional
        "D": 128,
        "layout_q":   "TND",
        "layout_kv":  "TND",
        "layout_out": "TND",
        "Dtype": "fp16",
        "mask_mode": 0,
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    "TND_MIXED_03": {
        "N1": 16,
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
    "ACTUAL_BNSD_01": {
        "layout_q": 'BNSD',
        "layout_kv": 'BNSD',
        "layout_out": 'BNSD',
        "Dtype": 'fp16',
        "q_range": (-5.0, 5.0),
        "k_range": (-5.0, 5.0),
        "v_range": (-5.0, 5.0),
        "mask_mode": 3,
        "cu_seqlens_q": [0, 174,1866,2088,2639,2914,734,525,2282,1945,2514,911,1645,2291,2739,2913,342,1613,114,2470,1653,3074,2182,2285,1509,1135,291,1818,2511,2508,747],
        "cu_seqlens_kv": [0, 1725,2862,4079,2690,646,0,270,3561,2155,121,185,2129,3018,2007,1898,2142,3211,2158,3388,2646,2047,566,275,4030,995,2893,3107,3097,3607,214],
        "B": 30,
        "N1": 64,
        "S1": 3089,
        "D": 128,
        "N2": 2,
        "S2": 4096,
    },
    "triton_fia_case_1": {
        "B": 8,
        "N1": 16,
        "N2": 16,
        "cu_seqused_q": [0,64,128,192,256,320,384,448,512],
        "cu_seqused_kv": [0,64,128,192,256,320,384,448,512],
        "seqused_q":  [64, 64, 64, 64, 64, 64, 64, 64],
        "seqused_kv":  [64, 64, 64, 64, 64, 64, 64, 64],
        "D": 32,
        "layout_q": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
        "Dtype": "fp16",
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    'aclnnFusedInferAttentionScoreV5_FIA_24_16_1_2206_2048_case49': {
        'layout_q': 'TND',
        'layout_kv': 'TND',
        'layout_out': 'TND',
        'Dtype': 'fp16',
        'q_range': (-5.0, 5.0),
        'k_range': (-5.0, 5.0),
        'v_range': (-5.0, 5.0),
        'mask_mode': 0,
        'D': 128,
        'N1': 24,
        'N2': 1,
        'cu_seqlens_q': [0, 586, 1180, 3149, 3186, 5216, 6778, 8031, 10003, 11078, 13108, 14950, 16701, 18718, 19934, 21136, 22634, 23795, 24212, 25687, 25719, 27795, 28192, 29494, 29640],
        'cu_seqlens_kv': [0, 1919, 1104, 751, 1163, 227, 412, 1541, 949, 1857, 1386, 1051, 1680, 199, 945, 155, 747, 1286, 0, 1694, 719, 1017, 321, 1464, 1654],
    },
    'FIA_8_64_4_512_512_case87': {
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
    "TND_01": {
        "N1": 8,
        "N2": 4,
        "cu_seqused_q": [0, 128, 256, 512],
        "D": 128,
        "layout_q": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
        "Dtype": "bf16",
        "mask_mode": 1,
    },
    "TND_02": {
        "N1": 8,
        "cu_seqused_q": [0, 2048],
        "D": 192,
        "layout_q": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
        "Dtype": "bf16",
        "mask_mode": 1,
    },
    "TND_03": {
        "N1": 8,
        "N2": 4,
        "cu_seqused_q": [0, 256, 512, 768],
        "cu_seqused_kv": [0, 256, 512, 768],
        "D": 128,
        "layout_q": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
        "Dtype": "bf16",
        "mask_mode": 1,
    },
    "TND_04": {
        "N1": 8,
        "N2": 4,
        "cu_seqused_q": [128, 256, 512],
        "cu_seqused_kv": [256, 512, 768],
        "D": 128,
        "layout_q": "TND",
        "layout_kv": "TND",
        "layout_out": "TND",
        "Dtype": "bf16",
        "mask_mode": 1,
    },
    # ── TND Decode：模拟推理解码阶段 ─────────────────────────────────────────
    # 8 个独立请求，每个请求 Q=1 token（逐 token 解码），KV 为已有 cache，长度各不相同。
    # GQA: N1=32, N2=8 (4:1)；无因果 mask（decode 对全 KV 做 attention）。
    # cu_seqused_q  = [0,1,2,...,8]  → per-batch Q seqlen 全为 1，total_q = 8
    # cu_seqused_kv = [0,64,128,256,384,640,896,1408,2432]
    #   → KV seqlens: [64, 64, 128, 128, 256, 256, 512, 1024], total_kv = 2432
    "TND_DECODE_05": {
        "N1": 32,
        "N2": 8,
        "cu_seqused_q":  [0, 1, 2, 3, 4, 5, 6, 7, 8],
        "cu_seqused_kv": [0, 64, 128, 256, 384, 640, 896, 1408, 2432],
        "D": 128,
        "layout_q":  "TND",
        "layout_kv": "TND",
        "layout_out":"TND",
        "Dtype": "bf16",
        "mask_mode": 1,
        "q_range": (-1.0, 1.0),
        "k_range": (-1.0, 1.0),
        "v_range": (-1.0, 1.0),
    },
    # ── TND Prefill + 因果掩码：模拟变长 batch 的首次前向 ───────────────────
    # 4 个请求，seqlen 分别为 128/256/384/512（Q=KV，自注意力）。
    # GQA: N1=16, N2=4 (4:1)；mask_mode=2 = 上三角因果 mask。
    # cu_seqused_q = cu_seqused_kv = [0,128,384,768,1280]
    #   → per-batch seqlens: [128, 256, 384, 512], max=512, total=1280
    "TND_PREFILL_06": {
        "N1": 16,
        "N2": 4,
        "cu_seqused_q":  [0, 128, 384, 768, 1280],
        "cu_seqused_kv": [0, 128, 384, 768, 1280],
        "D": 128,
        "layout_q":  "TND",
        "layout_kv": "TND",
        "layout_out":"TND",
        "Dtype": "fp16",
        "mask_mode": 2,
        "q_range": (-0.5, 0.5),
        "k_range": (-0.5, 0.5),
        "v_range": (-0.5, 0.5),
    },
    # ── TND Mixed：Prefill + Decode 混合批次（四个 optional tensor 正确语义）───────────────
    # 模拟生产推理：同一批次 prefill + decode，使用 padding 内存模式。
    # 四个参数语义：
    #   cu_seqlens_q  → cuSeqlensQOptional  : (B+1,) 累积偏移 [0,s1,s1+s2,...]，定义 buffer 布局
    #   cu_seqlens_kv → cuSeqlensKvOptional : (B+1,) 累积偏移
    #   seqused_q     → sequsedQOptional    : (B,) 各请求实际使用 Q 长度（≤ 分配）
    #   seqused_kv    → sequsedKvOptional   : (B,) 各请求实际使用 KV 长度（≤ 分配）
    # 请求明细（4 个请求，GQA N1=16 N2=8）：
    #   req0: 分配 Q=256 KV=512，实际使用 Q=200 KV=400（prefill）
    #   req1: 分配 Q=256 KV=512，实际使用 Q=  1 KV=128（decode）
    #   req2: 分配 Q=256 KV=512，实际使用 Q=128 KV=300（prefill）
    #   req3: 分配 Q=256 KV=512，实际使用 Q=  1 KV= 64（decode）
    "TND_MIXED_07": {
        "B": 4,
        "N1": 1,
        "N2": 1,
        # "S1": 128,
        # "S2": 128,
        # cu_seqlens: (B+1,) 累积偏移 [0, s1, s1+s2, ...]，此处与 seqused 精确对应（无 padding）
        # 真实 padding 场景中 cu_seqlens 对应分配量、seqused 对应实际量，两者不同；
        # 此处为保证 CPU/NPU 张量尺寸一致取无 padding 形式，四个参数仍独立提供。
        "cu_seqlens_q":  [0, 128, 256, 384, 1024],  # (B+1,) cumsum([200,1,128,1])
        "cu_seqlens_kv": [0, 128, 256, 384, 1024],   # (B+1,) cumsum([400,128,300,64])   fia:128, 256, 384, 512  1、tiling：tilingdata/tilingkey  TND    2、处理逻辑、
        # # seqused: (B,) 各请求实际使用长度
        # "seqused_q":  [0,128, 256, 384, 512],       # sequsedQOptional
        # "seqused_kv": [0, 128, 256, 384, 512],    # sequsedKvOptional
        "D": 128,
        "layout_q":   "TND",
        "layout_kv":  "TND",
        "layout_out": "TND",
        "Dtype": "fp16",
        "mask_mode": 0,
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    "TND_MIXED_08": {
        # "B": 8,
        "N1": 1,
        "N2": 1,
        # cu_seqlens: (B+1,) 累积偏移 [0, s1, s1+s2, ...]，此处与 seqused 精确对应（无 padding）
        # 真实 padding 场景中 cu_seqlens 对应分配量、seqused 对应实际量，两者不同；
        # 此处为保证 CPU/NPU 张量尺寸一致取无 padding 形式，四个参数仍独立提供。
        # "cu_seqlens_q": [0, 586, 1180, 3149, 3186, 5216, 6778, 8031, 10003, 11078, 13108, 14950, 16701, 18718, 19934, 21136, 22634, 23795, 24212, 25687, 25719, 27795, 28192, 29494, 29640],
        # "cu_seqlens_kv": [0, 1919, 3023, 3774, 4937, 5164, 5576, 7117, 8066, 9923, 11309, 12360, 14040, 14239, 15184, 15339, 16086, 17372, 17372, 19066, 19785, 20802, 21123, 22587, 24241],
        "cu_seqlens_q": [0, 586, 1180, 3149, 3186, 5216],
        "cu_seqlens_kv": [0, 1919, 3023, 3774, 4937, 5164],

        # # seqused: (B,) 各请求实际使用长度
        # "seqused_q":  [128, 256, 384, 512, 640, 768, 896, 944],       # sequsedQOptional
        # "seqused_kv": [128, 256, 384, 512, 640, 768, 896, 944],    # sequsedKvOptional
        "D": 128,
        "layout_q":   "TND",
        "layout_kv":  "TND",
        "layout_out": "TND",
        "Dtype": "fp16",
        "mask_mode": 0,
        "q_range": (-10.0, 10.0),
        "k_range": (-10.0, 10.0),
        "v_range": (-10.0, 10.0),
    },
    "PA_BBND_05": {
        "B": 4,
        "N1": 8,
        "N2": 4,
        "S1": 256,
        "S2": 512,
        "D": 128,
        "layout_q": "BNSD",
        "layout_kv": "PA_BBND",
        "layout_out": "BNSD",
        "Dtype": "bf16",
        "mask_mode": 1,
        "seqused_kv": [256, 256, 512, 512],
        "block_size": 64,
        "block_table": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    },
    "PA_BNBD_06": {
        "B": 4,
        "N1": 8,
        "N2": 4,
        "S1": 256,
        "S2": 512,
        "D": 128,
        "layout_q": "BNSD",
        "layout_kv": "PA_BNBD",
        "layout_out": "BNSD",
        "Dtype": "fp16",
        "mask_mode": 1,
        "seqused_kv": [256, 256, 512, 512],
        "block_size": 64,
        "block_table": [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]],
    },
}
