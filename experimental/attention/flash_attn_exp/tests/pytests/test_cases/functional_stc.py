# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

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

# H:可选; rel_h末维大小(图像高方向网格数), 与W同时传入, 需满足S2=H*W且0<H<=64
# W:可选; rel_w末维大小(图像宽方向网格数), 与H同时传入, 需满足S2=H*W且0<W<=64
# rel_mode:可选; "zero"时rel_h/rel_w生成全0张量(零偏置等价性用例)
# rel_range:可选; (lo, hi)元组, 覆盖rel_h/rel_w默认[-1,1]幅值(数值稳定性用例)

####### 用例分组(场景说明详见 docs/rel_h_rel_w_design.md §8.2, 仅含rel用例) ########
# 组A: rel基础功能(BNSD正向下钻)             —— REL_BASE_*
# 组B: H/W分解方式(j//W与j%W映射正确性)      —— REL_HW_*
# 组C: S2块边界与规模扫描(sInner=128整/尾块) —— REL_S2_*
# 组D: S1(M轴)边界                          —— REL_S1_* / BNSD
# 组E: seqused_q截断(BNSD跨head分段拷贝)     —— REL_SEQ_*
# 组F: 数值稳定性与零偏置                    —— REL_ZERO_01 / REL_RANGE_02
# 组G: 全特性交叉                           —— REL_CROSS_01
# 组H: 其他layout扩展(含rel,非BNSD主线)      —— BSND_03 / TND_05

BASE = {
    "B": [1],
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
    "q_range": [(-5.0, 5.0)],
    "k_range": [(-5.0, 5.0)],
    "v_range": [(-5.0, 5.0)],
}

TestCases = {
    # ==================== 组A: rel基础功能(BNSD正向下钻) ====================
    # 最小闭环: MHA, S2=64=8x8, rel基础叠加精度
    "REL_BASE_01": {
        **BASE,
        "N1": [2],
        "N2": [2],
        "S1": [128],
        "S2": [64],
        "H": [8],
        "W": [8],
    },
    # dtype覆盖: bf16下bias叠加精度
    "REL_BASE_02": {
        **BASE,
        "Dtype": ["bf16"],
        "N1": [2],
        "N2": [2],
        "S1": [128],
        "S2": [64],
        "H": [8],
        "W": [8],
    },
    # GQA(g=4): rel按n1=head索引正确性, 多batch隔离
    "REL_BASE_03": {
        **BASE,
        "B": [2],
        "N1": [8],
        "N2": [2],
        "S1": [256],
        "S2": [576],
        "H": [24],
        "W": [24],
    },
    # MQA(N2=1, g=8): GQA极端
    "REL_BASE_04": {
        **BASE,
        "Dtype": ["bf16"],
        "N1": [8],
        "N2": [1],
        "S1": [512],
        "S2": [1024],
        "H": [32],
        "W": [32],
    },
    # LSE输出: return_softmax_lse=True, 含偏置的log-sum-exp精度
    "REL_BASE_05": {
        **BASE,
        "N1": [2],
        "N2": [2],
        "S1": [128],
        "S2": [256],
        "H": [16],
        "W": [16],
        "return_softmax_lse": [True],
    },
    # BNSD输入→BSND输出: InOutLayoutType=BNSD_BSND分支叠加rel
    "REL_BASE_06": {
        **BASE,
        "N1": [4],
        "N2": [2],
        "S1": [256],
        "S2": [512],
        "H": [16],
        "W": [32],
        "layout_out": ["BSND"],
    },
    # 自定义scale=0.03: 验证先scale后加bias的顺序
    # (scale小时softmax偏平坦, P矩阵fp16量化的绝对误差~1.3e-4会超过小golden值的rtol阈值,
    #  放大q/k量程使QK*scale恢复默认case的分布, 保持顺序校验意图的同时保证fp16精度裕量)
    "REL_BASE_07": {
        **BASE,
        "N1": [2],
        "N2": [2],
        "S1": [128],
        "S2": [256],
        "H": [16],
        "W": [16],
        "scale": [0.03],
        "q_range": [(-12, 12)],
        "k_range": [(-12, 12)],
    },
    # ==================== 组B: H/W分解方式(j//W与j%W映射正确性) ====================
    # 方形网格H=W=16基准
    "REL_HW_SQUARE_01": {**BASE, "S1": [128], "S2": [256], "H": [16], "W": [16]},
    # 高瘦网格H=64上界: h行索引跨64行
    "REL_HW_TALL_02": {**BASE, "S1": [128], "S2": [256], "H": [64], "W": [4]},
    # 矮胖网格W=64上界: w列索引跨64列
    "REL_HW_WIDE_03": {**BASE, "S1": [128], "S2": [256], "H": [4], "W": [64]},
    # 最小网格1x1: S2=1单KV token边界
    "REL_HW_MIN_04": {**BASE, "S1": [16], "S2": [1], "H": [1], "W": [1]},
    # 质数分解13x17: H/W均非2的幂, 整除/取余正确性
    "REL_HW_PRIME_05": {**BASE, "S1": [128], "S2": [221], "H": [13], "W": [17]},
    # 同S2=640异分解a(20x32): 固定S2改变分解
    "REL_HW_S640_A": {**BASE, "S1": [256], "S2": [640], "H": [20], "W": [32]},
    # 同S2=640异分解b(32x20): 行/列分解互换, 验证不串轴
    "REL_HW_S640_B": {**BASE, "S1": [256], "S2": [640], "H": [32], "W": [20]},
    # 同S2=640异分解c(10x64): W上界组合
    "REL_HW_S640_C": {**BASE, "S1": [256], "S2": [640], "H": [10], "W": [64]},
    # 同S2=640异分解d(64x10): H上界组合
    "REL_HW_S640_D": {**BASE, "S1": [256], "S2": [640], "H": [64], "W": [10]},
    # ==================== 组C: S2块边界与规模扫描(sInner=128整块/尾块/多块) ====================
    # S2=128恰好1个sInner块, 无尾块
    "REL_S2_EXACT_01": {**BASE, "S1": [128], "S2": [128], "H": [8], "W": [16]},
    # S2=200尾块(200%128=72)+奇数W=25非对齐
    "REL_S2_TAIL_02": {**BASE, "S1": [128], "S2": [200], "H": [8], "W": [25]},
    # 规模扫描: 大S1=4096, S2=640=40x16非整块
    "REL_S2_SWEEP_640": {**BASE, "S1": [4096], "S2": [640], "H": [40], "W": [16]},
    # 规模扫描: S2=768=6x128整块
    "REL_S2_SWEEP_768": {**BASE, "S1": [4096], "S2": [768], "H": [48], "W": [16]},
    # 规模扫描: S2=1024=8x128整块+H上界
    "REL_S2_SWEEP_1024": {**BASE, "S1": [4096], "S2": [1024], "H": [64], "W": [16]},
    # 规模扫描: S2=2048多块大循环
    "REL_S2_SWEEP_2048": {**BASE, "S1": [4096], "S2": [2048], "H": [64], "W": [32]},
    # 规模扫描: S2=4096=64x64, H/W双上界+32块大循环(S2切分/FD路径)
    "REL_S2_SWEEP_4096": {**BASE, "S1": [4096], "S2": [4096], "H": [64], "W": [64]},
    # ==================== 组D: S1(M轴)边界 ====================
    # 单query行: S1=1时M轴最小块
    "REL_S1_ONE_01": {**BASE, "S1": [1], "S2": [64], "H": [8], "W": [8]},
    # M轴非对齐: S1=33<基本块且非32对齐, 转置mStride尾部
    "REL_S1_UNALIGN_02": {**BASE, "S1": [33], "S2": [256], "H": [16], "W": [16]},
    # M尾块(1111%128=87)+GQA(g=9)+seqused截断+H/W双上界+S2=4096组合(真BNSD)
    "BNSD": {
        **BASE,
        "B": [4],
        "S1": [1111],
        "S2": [4096],
        "N1": [9],
        "layout_q": ["BNSD"],
        "layout_kv": ["BNSD"],
        "layout_out": ["BNSD"],
        "seqused_q": [[512] * 4],
        "H": [64],
        "W": [64],
    },
    # 大S1=4096: M轴32块大循环
    "REL_S1_LARGE_04": {**BASE, "S1": [4096], "S2": [1024], "H": [64], "W": [16]},
    # ==================== 组E: seqused_q截断(BNSD跨head分段拷贝路径) ====================
    # 各batch混合截断[512,1024,300,1]+GQA(g=4): M子块跨head边界分段拷贝(b04b0d23b修复路径), b3截断至1行
    "REL_SEQ_MIX_01": {
        **BASE,
        "B": [4],
        "N1": [4],
        "N2": [1],
        "S1": [1024],
        "S2": [1024],
        "H": [32],
        "W": [32],
        "seqused_q": [[512, 1024, 300, 1]],
    },
    # 截断至单行: actS1Size=1的分段拷贝边界
    "REL_SEQ_ONE_02": {
        **BASE,
        "N1": [2],
        "N2": [2],
        "S1": [128],
        "S2": [256],
        "H": [16],
        "W": [16],
        "seqused_q": [[1]],
    },
    # 传seqused但等于全长: 截断边界=S1
    "REL_SEQ_FULL_03": {
        **BASE,
        "N1": [2],
        "N2": [2],
        "S1": [256],
        "S2": [256],
        "H": [16],
        "W": [16],
        "seqused_q": [[256]],
    },
    # ==================== 组F: 数值稳定性与零偏置 ====================
    # 零偏置: rel全0时NPU==CPU(golden加0), 验证rel路径对零偏置的退化正确性
    "REL_ZERO_01": {
        **BASE,
        "N1": [2],
        "N2": [2],
        "S1": [128],
        "S2": [64],
        "H": [8],
        "W": [8],
        "rel_mode": ["zero"],
    },
    # 大幅偏置[-10,10]: softmax减max溢出保护, FP16量化误差不放大
    "REL_RANGE_02": {
        **BASE,
        "N1": [2],
        "N2": [2],
        "S1": [128],
        "S2": [1024],
        "H": [32],
        "W": [32],
        "rel_range": [(-10.0, 10.0)],
    },
    # ==================== 组G: 全特性交叉 ====================
    # rel x GQA(g=4) x seqused截断 x LSE x BNSD→BSND输出 x bf16综合
    "REL_CROSS_01": {
        **BASE,
        "B": [2],
        "N1": [16],
        "N2": [4],
        "S1": [2048],
        "S2": [1024],
        "H": [32],
        "W": [32],
        "Dtype": ["bf16"],
        "return_softmax_lse": [True],
        "layout_out": ["BSND"],
        "seqused_q": [[1024, 2048]],
    },
    # ==================== 组H: 其他layout扩展(含rel,非BNSD主线) ====================
    # BSND+rel(H16x8): BSND的rel连续拷贝路径(修正原用例S2与H*W不一致问题)
    "BSND_03": {
        **BASE,
        "S2": [128],
        "layout_q": ["BSND"],
        "layout_kv": ["BSND"],
        "layout_out": ["BSND"],
        "H": [16],
        "W": [8],
    },
    # TND+rel变长(H32x32): TND的rel前缀偏移拷贝路径
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
}
