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
# B:必选; batch_size,TND格式下可选
# N1:必选; head_num
# N2:可选; kv's head_num,支持GQA/MHA/MQA
# S1:必选; query's sequence length;TND格式下可选
# S2:可选; key&value's sequence length
# D:必选; 表示query&key&value的head_dim
# DV:可选; value's head_dim;设置该参数,value的head_dim以DV为准
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

BASE = {"B": [1],
        "N1": [1],
        "N2": [1],
        "S1": [128],
        "S2": [256],
        "D": [64],
        "DV": [64],
        "Dtype": ["fp16"],
        "layout_q": ["BNSD"],
        "layout_kv": ["BNSD"],
        "layout_out": ["BNSD"],
        "block_size": [-1],
        "return_softmax_lse": [False],
        "enable_learnable_sink": [False],
        "cu_seqlens_q": [[None]],
        "cu_seqlens_kv": [[None]],
        "seqused_q": [[None]],
        "seqused_kv": [[None]],
        "mask_mode": [0],
        "win_left": [-1],
        "win_right": [-1],
        'q_range': [(-5.0, 5.0)],
        'k_range': [(-5.0, 5.0)],
        'v_range': [(-5.0, 5.0)]}

layout = "BSND"
TestCases = {
    # ------type
    "BF16_01": {**BASE, "Dtype": ["bf16"]},
    "FP16_02": {**BASE, "Dtype": ["fp16"]},
    # ------layout
    "BSND_03": {
        **BASE,
        "layout_q": ["BSND"],
        "layout_kv": ["BSND"],
        "layout_out": ["BSND"],
        "H": [16],
        "W": [8],
    },
    "BNSD": {
        **BASE,
        "B": [20],
        "S1": [1111],
        "S2": [4096],
        "N1": [9],
        "layout_q": [layout],
        "layout_kv": [layout],
        "layout_out": [layout],
        "H": [64],
        "W": [64],
    },
    "TND_05": {
        **BASE,
        "layout_q": ["TND"],
        "layout_kv": ["TND"],
        "layout_out": ["TND"],
        "cu_seqlens_q": [[0, 128, 256]],
        "cu_seqlens_kv": [[0, 1024, 2048]],
        "H": [32],
        "W": [32],
    },
    "TND_06": {
        **BASE,
        "layout_q": ["TND"],
        "layout_kv": ["TND"],
        "layout_out": ["TND"],
        "cu_seqlens_q": [[0, 64, 128, 192, 256, 320, 384, 448, 512]],
        "cu_seqlens_kv": [[0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]],
        "seqused_q": [[60] * 8],
        "seqused_kv": [[1000] * 8],
    },
    "PA_07": {
        **BASE,
        "layout_kv": ["PA_BBND", "PA_BNBD"],  # 2条
        # "layout_kv": ["PA_BBND", "PA_BNBD", "PA_NZ"], # 3条
        "block_size": [128],
    },
    "REL_S2_640_H20W32": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [640], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [20], "W": [32],
    },
    "REL_S2_640_H10W64": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [640], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [10], "W": [64],
    },
    "REL_S2_640_H32W20": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [640], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [32], "W": [20],
    },
    "REL_S2_640_H16W40": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [640], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [16], "W": [40],
    },
    "REL_S2_512_H32W16": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [512], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [32], "W": [16],
    },
    "REL_S2_640": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [640], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [40], "W": [16],
    },
    "REL_S2_768": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [768], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [48], "W": [16],
    },
    "REL_S2_1024": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [1024], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [64], "W": [16],
    },
    "REL_S2_2048": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [2048], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [128], "W": [16],
    },
    "REL_S2_4096_H64W64": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [4096], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [64], "W": [64],
    },
    "REL_S2_4096_H256W16": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [4096], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [256], "W": [16],
    },
    "REL_S2_4096_H128W32": {
        **BASE,
        "B": [1], "S1": [4096], "S2": [4096], "N1": [1], "N2": [1], "D": [64],
        "layout_q": ["BNSD"], "layout_kv": ["BNSD"], "layout_out": ["BNSD"],
        "H": [128], "W": [32],
    },
}
