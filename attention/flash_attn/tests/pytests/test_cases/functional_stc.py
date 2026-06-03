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

BASE = {"B": [2],
        "N1": [8],
        "N2": [1],
        "S1": [64],
        "S2": [1024],
        "D": [128],
        "DV": [128],
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

# 共95条
TestCases = {
    # ------type
    "BF16_01"  : {**BASE,
                  "Dtype": ["bf16"]},
    "FP16_02"  : {**BASE,
                  "Dtype": ["fp16"]},
    # ------layout
    "BSND_03"  : {**BASE,
                  "layout_q": ["BSND"],
                  "layout_kv": ["BSND"],
                  "layout_out": ["BSND"]},
    "BSND_04"  : {**BASE,
                  "layout_out": ["BSND"]},
    "TND_05"   : {**BASE,
                  "layout_q": ["TND"],
                  "layout_kv": ["TND"],
                  "layout_out": ["TND"],
                  "cu_seqlens_q": [[0, 64, 128, 192, 256, 320, 384, 448, 512]],
                  "cu_seqlens_kv": [[0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]]},
    "TND_06"   : {**BASE,
                  "layout_q": ["TND"],
                  "layout_kv": ["TND"],
                  "layout_out": ["TND"],
                  "cu_seqlens_q": [[0, 64, 128, 192, 256, 320, 384, 448, 512]],
                  "cu_seqlens_kv": [[0, 1024, 2048, 3072, 4096, 5120, 6144, 7168, 8192]],
                  "seqused_q": [[60] * 8],
                  "seqused_kv": [[1000] * 8]},
    "PA_07"    : {**BASE,
                  "layout_kv": ["PA_BBND", "PA_BNBD"], # 2条
                  # "layout_kv": ["PA_BBND", "PA_BNBD", "PA_NZ"], # 3条
                  "block_size": [128]},
    # PA时，block_size<s2Base，s2Base不整除block_size
    # PA时，block_size>s2Base，block_size不整除s2Base
    # 边界值 典型值
    # "PA_08"    : {**BASE,
    #               "layout_kv": ["PA_BBND"],
    #               "block_size": [96, 192, 16, 1024, 512]}, # 5条
    # ------lse
    "LSE_09"   : {**BASE,
                  "return_softmax_lse": [True]},
    # ------sink
    # "SINK_10"  : {**BASE,
    #               "enable_learnable_sink": [True]},

    # ------D
    # 边界值1
    # 边界值512
    # 特殊值D%32=7、8、15、16、31、0
    # D等长，典型值64、128、256
    # D不等长，典型值qkD=192，vD=128
    # D不等长，典型值qkD=64，vD=128
    # "D_11"     : {**BASE,
    #               "D": [1, 7, 8, 15, 16 ,31 ,40, 64, 96, 128, 256, 512]}, # 12条
    # "D_12"     : {**BASE,
    #               "D": [192],
    #               "DV": [128]},
    # "D_13"     : {**BASE,
    #               "D": [64],
    #               "DV": [128]},
    # ------GS1
    # GS1类型，S1<mBase，mBase%S1==0，GS1<mBase
    # GS1类型，S1<mBase，mBase%S1==0，GS1=mBase
    # GS1类型，S1<mBase，mBase%S1==0，GS1>mBase，GS1%mBase!=0
    # GS1类型，S1<mBase，mBase%S1==0，GS1>mBase，GS1%mBase==0
    "GS1_14"   : {**BASE,
                  "N1": [1, 2, 3, 4], # 4条
                  "N2": [1],
                  "S1": [64]},
    # GS1类型，S1<mBase，mBase%S1!=0，GS1<mBase
    # GS1类型，S1<mBase，mBase%S1!=0，GS1>mBase
    "GS1_15"   : {**BASE,
                  "N1": [1, 3], # 2条
                  "N2": [1],
                  "S1": [65]},
    # GS1类型，S1=mBase，GS1>=mBase
    # GS1类型，S1>mBase，S1%mBase==0
    "GS1_16"   : {**BASE,
                  "S1": [128, 256]}, # 2条
    # GS1类型，S1>mBase，S1%mBase!=0，G=1
    "GS1_17"   : {**BASE,
                  "N1": [1],
                  "N2": [1],
                  "S1": [257]},
    # GS1类型，S1>mBase，S1%mBase!=0，G>=2
    "GS1_18"   : {**BASE,
                  "N1": [2],
                  "N2": [1],
                  "S1": [129]},
    # S1G类型，G<mBase，mBase%G==0，GS1<mBase
    # S1G类型，G<mBase，mBase%G==0，GS1=mBase
    # S1G类型，G<mBase，mBase%G==0，GS1>mBase，GS1%mBase!=0
    # S1G类型，G<mBase，mBase%G==0，GS1>mBase，GS1%mBase==0
    "GS1_19"   : {**BASE,
                  "N1": [64],
                  "N2": [1],
                  "S1": [1, 2, 3, 4], # 4条
                  "layout_q": ["BSND"],
                  "layout_kv": ["BSND"],
                  "layout_out": ["BSND"]},
    # S1G类型，G<mBase，mBase%G!=0，GS1<mBase
    # S1G类型，G<mBase，mBase%G!=0，GS1>mBase
    "GS1_20"   : {**BASE,
                  "N1": [65],
                  "N2": [1],
                  "S1": [1, 3], # 2条
                  "layout_q": ["BSND"],
                  "layout_kv": ["BSND"],
                  "layout_out": ["BSND"]},
    # S1G类型，G=mBase，GS1>=mBase
    # S1G类型，G>mBase，G%mBase==0
    "GS1_21"   : {**BASE,
                  "N1": [128],
                  "N2": [256],
                  "layout_q": ["BSND"],
                  "layout_kv": ["BSND"],
                  "layout_out": ["BSND"]},
    # S1G类型，G>mBase，G%mBase!=0，S1=1
    "GS1_22"   : {**BASE,
                  "N1": [257],
                  "N2": [1],
                  "S1": [1],
                  "layout_q": ["BSND"],
                  "layout_kv": ["BSND"],
                  "layout_out": ["BSND"]},
    # S1G类型，G>mBase，G%mBase!=0，S1>=2
    "GS1_23"   : {**BASE,
                  "N1": [129],
                  "N2": [1],
                  "S1": [2],
                  "layout_q": ["BSND"],
                  "layout_kv": ["BSND"],
                  "layout_out": ["BSND"]},
    # Mmad在M=1时可能会当做GEMV处理，对左矩阵的L0A格式要求不是NZ，需要特殊测试。
    "GS1_24"   : {**BASE,
                  "N1": [1],
                  "N2": [1],
                  "S1": [1]},
    # ------ s2
    # 前面部分batch为0
    "S2_25"    : {**BASE,
                  "B": [5],
                  "seqused_kv": [[0, 0, 1024, 1024, 1024]]},
    # 尾部部分batch为0
    "S2_26"    : {**BASE,
                  "B": [5],
                  "seqused_kv": [[1024, 1024, 1024, 0, 0]]},
    # 所有batch为0
    "S2_27"    : {**BASE,
                  "B": [5],
                  "seqused_kv": [[0, 0, 0, 0, 0]]},
    # 中间部分batch为0
    "S2_28"    : {**BASE,
                  "B": [5],
                  "seqused_kv": [[1024, 0, 0, 0, 1024]]},
    # S2<mBase，S2仅一轮循环
    # S2=mBase，S2仅一轮循环
    "S2_29"    : {**BASE,
                  "B": [8],
                  "seqused_kv": [[1, 2, 3, 5, 31, 32, 64, 128, 256]]},
    # S2>mBase，S2%mBase==0，S2多轮循环且不存在尾块
    # S2>mBase，0<S2%mBase<8，S2多轮循环且存在尾块
    # S2>mBase，S2%mBase=8，S2多轮循环且存在尾块
    # S2>mBase，8<S2%mBase<16，S2多轮循环且存在尾块
    # S2>mBase，S2%mBase=16，S2多轮循环且存在尾块
    # S2>mBase，16<S2%mBase<32，S2多轮循环且存在尾块
    # S2>mBase，S2%mBase=32，S2多轮循环且存在尾块
    "S2_30"    : {**BASE,
                  "B": [8],
                  "seqused_kv": [[512, 513, 520, 524, 528, 531, 544, 550, 511]]},
    "S2_31"    : {**BASE,
                  "B": [8],
                  "seqused_kv": [[7, 513, 9, 524, 19, 531, 35, 550, 259]]},
    # ------mask
    # mask_mode=3, 覆盖：S1<=S2和S1>S2（存在行无效）
    "MASK_32"  : {**BASE,
                  "S1": [64],
                  "S2": [1024],
                  "mask_mode": [3]},
    "MASK_33"  : {**BASE,
                  "S1": [1024],
                  "S2": [64],
                  "mask_mode": [3]},
    # mask_mode=4，win_left/win_right生效，s1Base=Ceil(mBase，g)
    # 场景1:s1Base<s2Base
    # win_left : 1、0; 2、0~s1Base; 3、>=s1Base;
    # win_right : 1、0; 2、0~s2Base-s1Base; 3、s2Base-s1Base; 4、s2Base-s1Base~s2Base; 5、s2Base
    # "MASK_34"  : {**BASE,
    #               "N1": [32], # mBase = 128, G = 32, s1Base = 4
    #               "N2": [1],
    #               "mask_mode": [4],
    #               "win_left": [0, 2, 4], # 14条，其中0 0为无效用例
    #               "win_right": [0, 29, 124, 126, 128]},
    # 场景2:s1Base==s2Base
    # win_left : 1、0; 2、0~s1Base; 3、>=s1Base;
    # win_right : 1、0; 2、0~s2Base; 3、>=s2Base
    #      "MASK_35"  : {**BASE,
    #                    "N1": [1], # mBase = 128, G = 1, s1Base = 128
    #                    "N2": [1],
    #                    "mask_mode": [4],
    #                    "win_left": [0, 63, 128],
    #                    "win_right": [0, 92, 128]}, # 8条，其中0 0为无效用例
    # 场景3:s1Base>s2Base
    # win_left : 1、0; 2、0~s1Base-s2Base; 3、s1Base-s2Base; 4、s1Base-s2Base~s1Base; 5、s1Base
    # win_right : 1、0; 2、0~s2Base; 3、>=s2Base;
    # "MASK_36"  : {**BASE,
    #               "N1": [1], # mBase = 128, G = 1, s1Base = 128
    #               "N2": [1],
    #               "S2": [64],
    #               "mask_mode": [4],
    #               "win_left": [0, 63, 64, 90, 128],
    #               "win_right": [0, 15, 64]}, # 14条，其中0 0为无效用例
}
