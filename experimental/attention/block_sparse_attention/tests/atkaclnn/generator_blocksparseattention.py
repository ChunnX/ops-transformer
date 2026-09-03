# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import random
import copy
import math
from typing import Union, List
import numpy as np
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig


@GENERATOR_REGISTRY.register("generate_block_sparse_attention_mxfp4")
class BlockSparseAttentionMxfp4Generator(CaseGenerator):
    # pad_kind -> (Q 侧取长方式, KV 侧取长方式)
    PAD_KIND_MAP = {
        "both": ("rand", "rand"),
        "kv_only": ("full", "rand"),
        "q_only": ("rand", "full"),
        "decode": ("decode", "rand"),
        "block_aligned": ("block_aligned", "block_aligned"),
        # tight: 张量 S 维紧贴最长 batch(S_pad == max(actual)), 不额外留 padding
        "tight": ("wide", "wide"),
        # spread: per-batch 实际长度数量级差异(压 tiling 负载不均衡)
        "spread": ("spread", "spread"),
    }
    # 张量 S 维取 max(actual) 而非 S_ceil 的形态
    TIGHT_PAD_KINDS = ("tight",)

    # CPU golden 规模预算, 量纲为 B*N1*S1*S2*D。golden 是 CPU 上的分块 flash + MXFP4 qdq,
    # 且每个用例要跑两遍(benchmark + 真值), 超预算的组合会让单用例耗时到分钟级以上。
    # 超出时按 batch -> S2 -> S1 的顺序往更小的取值退, 保证大序列只与小 batch/小头数组合出现。
    # COST_BUDGET = 1_500_000_000  # 已禁用(不再做规模退避)

    def __init__(self, config):
        super().__init__(config)
        self.input_layout_range = ["BNSD", "BSND", "TND"]
        self.headDim_range = [64, 128]
        # blockShapeX/Y 各自在 {64, 128, 256} 内随机, 组合全覆盖(64/128/256 x 64/128/256)。
        self.block_shape_range = [64, 128, 256]
        self.attention_out_dtype_range = ["bf16", "fp16"]
        self.quant_mode_range = [2, 3]
        self.E8M0_BIAS = 127

        # 序列长度/规模: 连续范围随机采样(不写死离散列表)。
        # s1(qs) / s2(kvs) 各自在 [1, s1_hi] / [1, s2_hi] 内随机;
        # batch/g/n2 在各自范围随机, 超预算由 COST_BUDGET 退避。
        self.s1_hi = 4096  # 查询侧 seq len 上限 (s1 = qs)
        self.s2_hi = 25000  # KV 侧 seq len 上限 (s2 = kvs)
        self.batch_hi = 8
        self.g_hi = 4
        self.n2_hi = 8

    @staticmethod
    def gen_actual_len(s_padded, block, kind):
        """padded 场景下单个 batch 的 actualSeqLength(<= 张量 S 维)。

        除 decode 外实际长度不低于 padded 的一半: padding 区算子不写、比对时两侧都置 0,
        占比过大会稀释精度指标。decode 极端形态(act=1)另由 pad_kind 把 padded S 限制到单块内。
        """
        if kind == "full":
            return s_padded
        if kind == "decode":
            return 1
        if kind == "block_aligned":
            block_num = s_padded // block
            if block_num >= 1:
                return random.randint(max(1, block_num // 2), block_num) * block
            return s_padded
        if kind == "wide":
            return random.randint(1, s_padded)
        if kind == "spread":
            return random.choice(
                [
                    1,
                    max(1, s_padded // 8),
                    max(1, s_padded // 4),
                    max(1, s_padded // 2),
                    s_padded,
                ]
            )
        return random.randint(max(1, s_padded // 2), s_padded)

    # def fit_cost_budget(self):
    #     """把规模压回 CPU golden 可承受范围: 依次把 batch / S2 / S1 减半退回。"""
    #     while self.B * self.N1 * self.S1_ceil * self.S2_ceil * self.D > self.COST_BUDGET:
    #         if self.B > 1:
    #             self.B = max(1, self.B // 2)
    #         elif self.S2_ceil > 1:
    #             self.S2_ceil = max(1, self.S2_ceil // 2)
    #         elif self.S1_ceil > 1:
    #             self.S1_ceil = max(1, self.S1_ceil // 2)
    #         else:
    #             break

    def param_generator(self):
        self.B = 0
        self.N1 = 0
        self.N2 = 0
        self.D = 0
        self.S1_ceil = 0
        self.S2_ceil = 0
        self.S1_aligned = 0
        self.S2_aligned = 0
        self.is_varied_len = 0
        self.q_seq_list = []
        self.kv_seq_list = []
        self.q_input_layout = "BNSD"
        self.kv_input_layout = "BNSD"
        self.softmax_lse_flag = 0
        self.T1 = 0
        self.T2 = 0
        self.inner_prec = 4
        self.block_shape_x = 128
        self.block_shape_y = 128

        self.block_shape_x = random.choice(self.block_shape_range)
        self.block_shape_y = random.choice(self.block_shape_range)
        self.block_shape_list = [self.block_shape_x, self.block_shape_y]

        self.D = random.choice(self.headDim_range)

        # 序列长度: s1 = qs(查询侧 seq len), s2 = kvs(KV 侧 seq len)
        self.B = random.randint(1, self.batch_hi)
        self.S1_ceil = random.randint(1, self.s1_hi)
        self.S2_ceil = random.randint(1, self.s2_hi)

        # GQA 头数: n2 = kvn(KV 头数), n1 = qn(查询头数)。
        # n1 = n2 * g, 保证 n1 是 n2 的整数倍(n1 % n2 == 0 整除约束)。
        self.N2 = random.randint(1, self.n2_hi)
        self.G = random.randint(1, self.g_hi)
        self.N1 = self.N2 * self.G
        assert self.N1 % self.N2 == 0

        # self.fit_cost_budget()  # 已禁用(不做规模退避)

        self.q_input_layout = random.choice(self.input_layout_range)
        self.kv_input_layout = self.q_input_layout

        # 变长场景:
        # TND: is_varied_len=1 时每个 batch 的 seq 长度不同(T = sum(actualSeqLengths))
        # BNSD/BSND: is_padded=1 时张量 S 维为 padded 大小, actualSeqLengths 指示更小的实际有效长度
        self.pad_kind = "both"
        if self.q_input_layout == "TND":
            self.is_varied_len = random.choice([0, 1])
            self.is_padded = 0
        else:
            self.is_varied_len = 0
            self.is_padded = random.choice([0, 1])
            self.pad_kind = random.choice(list(self.PAD_KIND_MAP))
            if self.is_padded and self.pad_kind == "decode":
                # decode: act_q=1, Q 张量 padded 到单块内(对齐参考脚本 act_q=1 / s1=128)
                self.S1_ceil = min(self.S1_ceil, self.block_shape_x)
        # 注: BNSD/BSND 下算子允许 actualSeqLengths 与 Kv 同时不配置(按张量 S 维全长计算), 但
        # tensor 输入那套 range_values=["None"] 的写法对 attr_tuple 的 int 元素不成立 —— 不会
        # 报错, 而是被物化成负数哨兵传下来, 当作真实长度用会造出负维度。这条覆盖暂不支持。

        self.quant_mode = random.choice(self.quant_mode_range)
        if self.quant_mode == 3:
            # quant_mode=3 时 dstTypeMax 可取 0.0 或 [6.0, 12.0] 内的随机值
            self.dst_type_max = (
                0.0 if random.random() < 0.5 else round(random.uniform(6.0, 12.0), 2)
            )
        else:
            self.dst_type_max = 0.0

        self.attention_out_dtype = random.choice(self.attention_out_dtype_range)

        for b in range(self.B):
            S1_cur = self.S1_ceil
            S2_cur = self.S2_ceil
            if self.is_varied_len:
                S1_cur = random.randint(1, self.S1_ceil)
                S2_cur = random.randint(1, self.S2_ceil)
            elif self.is_padded:
                # BNSD/BSND padded: actualSeq <= 张量S维(S_ceil)，每 batch 独立取值
                q_kind, kv_kind = self.PAD_KIND_MAP[self.pad_kind]
                S1_cur = self.gen_actual_len(self.S1_ceil, self.block_shape_x, q_kind)
                S2_cur = self.gen_actual_len(self.S2_ceil, self.block_shape_y, kv_kind)
            self.T1 += S1_cur
            self.T2 += S2_cur
            self.q_seq_list.append(S1_cur)
            self.kv_seq_list.append(S2_cur)

        self.S1_aligned = max(self.q_seq_list)
        self.S2_aligned = max(self.kv_seq_list)
        # BNSD/BSND padded: 张量S维用 S_ceil(padded大小)，actualSeq 用实际值。
        # tight 形态例外: 张量 S 维就取 max(actual), 最长的 batch 一行 padding 都没有。
        if self.is_padded and self.pad_kind not in self.TIGHT_PAD_KINDS:
            self.S1_aligned = self.S1_ceil
            self.S2_aligned = self.S2_ceil
        self.x_block_num_aligned = (
            self.S1_aligned + self.block_shape_x - 1
        ) // self.block_shape_x
        self.y_block_num_aligned = (
            self.S2_aligned + self.block_shape_y - 1
        ) // self.block_shape_y

        self.q_scale_blocks = (self.D + 31) // 32
        self.kv_scale_blocks_s = (self.S2_aligned + 31) // 32

    def expand_list_config(self, config_list, value_list):
        if not config_list:
            return
        template = config_list[0]
        config_list.clear()
        elem_num = len(value_list)
        for i in range(elem_num):
            new_item = copy.deepcopy(template)
            new_item.range_values = [value_list[i]]
            config_list.append(new_item)

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:
        self.param_generator()
        inputs_map = {
            item[0].name if isinstance(item, list) and item else item.name: item
            for item in case_config.inputs
        }

        query = inputs_map.get("query")
        key = inputs_map.get("key")
        value = inputs_map.get("value")
        attentionOut = inputs_map.get("attentionOut")
        blockSparseMask = inputs_map.get("blockSparseMask")
        attenMask = inputs_map.get("attenMask")
        blockShape = inputs_map.get("blockShape")
        actualSeqLengths = inputs_map.get("actualSeqLengths")
        actualSeqLengthsKv = inputs_map.get("actualSeqLengthsKv")
        blockTable = inputs_map.get("blockTable")
        softmaxLse = inputs_map.get("softmaxLse")

        qInputLayout = inputs_map.get("qInputLayout")
        kvInputLayout = inputs_map.get("kvInputLayout")
        numKeyValueHeads = inputs_map.get("numKeyValueHeads")
        maskType = inputs_map.get("maskType")
        scaleValue = inputs_map.get("scaleValue")
        innerPrecise = inputs_map.get("innerPrecise")
        blockSize = inputs_map.get("blockSize")
        preTokens = inputs_map.get("preTokens")
        nextTokens = inputs_map.get("nextTokens")
        softmaxLseFlag = inputs_map.get("softmaxLseFlag")
        quantMode = inputs_map.get("quantMode")
        dstTypeMax = inputs_map.get("dstTypeMax")

        query.dtype = "uint8"
        key.dtype = "uint8"
        value.dtype = "uint8"
        attentionOut.dtype = self.attention_out_dtype

        if self.q_input_layout == "TND":
            qInputLayout.range_values = ["TND"]
            query.shape = [self.T1, self.N1, self.D // 2]
            attentionOut.shape = [self.T1, self.N1, self.D]
        elif self.q_input_layout == "BNSD":
            qInputLayout.range_values = ["BNSD"]
            query.shape = [self.B, self.N1, self.S1_aligned, self.D // 2]
            attentionOut.shape = [self.B, self.N1, self.S1_aligned, self.D]
        elif self.q_input_layout == "BSND":
            qInputLayout.range_values = ["BSND"]
            query.shape = [self.B, self.S1_aligned, self.N1, self.D // 2]
            attentionOut.shape = [self.B, self.S1_aligned, self.N1, self.D]

        if self.kv_input_layout == "TND":
            kvInputLayout.range_values = ["TND"]
            key.shape = [self.T2, self.N2, self.D // 2]
            value.shape = [self.T2, self.N2, self.D // 2]
        elif self.kv_input_layout == "BNSD":
            kvInputLayout.range_values = ["BNSD"]
            key.shape = [self.B, self.N2, self.S2_aligned, self.D // 2]
            value.shape = [self.B, self.N2, self.S2_aligned, self.D // 2]
        elif self.kv_input_layout == "BSND":
            kvInputLayout.range_values = ["BSND"]
            key.shape = [self.B, self.S2_aligned, self.N2, self.D // 2]
            value.shape = [self.B, self.S2_aligned, self.N2, self.D // 2]

        blockSparseMask.dtype = "int8"
        blockSparseMask.shape = [
            self.B,
            self.N1,
            self.x_block_num_aligned,
            self.y_block_num_aligned,
        ]
        blockSparseMask.range_values = [0, 1]

        self.expand_list_config(blockShape, self.block_shape_list)
        self.expand_list_config(actualSeqLengths, self.q_seq_list)
        self.expand_list_config(actualSeqLengthsKv, self.kv_seq_list)

        # attenMask (blockEffRows): int32 [B, N1, maxBlockNum, 2], 末维 [effRowsQ, effRowsKV]。
        # kernel 只读 KV 槽(index 1); 内容由 function 按 case_seed 重建(且仅在 blockShapeY==128
        # 时开启), 这里只负责分配形状 —— dim2 须 >= max(xBlockNumAligned, yBlockNumAligned),
        # 对齐 tiling ParseAttenMask 的 dim2 >= max(maxQBlockNum_, maxKvBlockNum_) 校验。
        max_block_num = max(self.x_block_num_aligned, self.y_block_num_aligned)
        attenMask.dtype = "int32"
        attenMask.shape = [self.B, self.N1, max_block_num, 2]
        attenMask.range_values = [0, self.block_shape_y]
        blockTable.range_values = ["None"]

        numKeyValueHeads.range_values = [self.N2]
        maskType.range_values = [0]
        scaleValue.range_values = [1 / (self.D**0.5)]
        innerPrecise.range_values = [self.inner_prec]
        blockSize.range_values = [0]
        preTokens.range_values = [2147483647]
        nextTokens.range_values = [2147483647]
        softmaxLseFlag.range_values = [self.softmax_lse_flag]
        quantMode.range_values = [self.quant_mode]
        dstTypeMax.range_values = [self.dst_type_max]

        softmaxLse.range_values = ["None"]

        # MXFP4 E8M0 dequant scales: float8_e8m0fnu 存储为 uint8
        # Q/K scale: [B, N, S, D//64, 2] (BNSD) — 5D
        # V scale: [B, N2, ceil(S2/64), D, 2] (BNSD) — 5D
        qk_scale_d_dim = self.D // 64
        v_scale_s_dim = (self.S2_aligned + 63) // 64

        qDequantScale = inputs_map.get("qDequantScale")
        kDequantScale = inputs_map.get("kDequantScale")
        vDequantScale = inputs_map.get("vDequantScale")

        qDequantScale.dtype = "uint8"
        kDequantScale.dtype = "uint8"
        vDequantScale.dtype = "uint8"

        if self.q_input_layout == "TND":
            qDequantScale.shape = [self.T1, self.N1, qk_scale_d_dim, 2]
            kDequantScale.shape = [self.T2, self.N2, qk_scale_d_dim, 2]
            vDequantScale.shape = [v_scale_s_dim * self.B, self.N2, self.D, 2]
        elif self.q_input_layout == "BNSD":
            qDequantScale.shape = [self.B, self.N1, self.S1_aligned, qk_scale_d_dim, 2]
            kDequantScale.shape = [self.B, self.N2, self.S2_aligned, qk_scale_d_dim, 2]
            vDequantScale.shape = [self.B, self.N2, v_scale_s_dim, self.D, 2]
        elif self.q_input_layout == "BSND":
            qDequantScale.shape = [self.B, self.S1_aligned, self.N1, qk_scale_d_dim, 2]
            kDequantScale.shape = [self.B, self.S2_aligned, self.N2, qk_scale_d_dim, 2]
            vDequantScale.shape = [self.B, v_scale_s_dim, self.N2, self.D, 2]

        qDequantScale.range_values = [0, 254]
        kDequantScale.range_values = [0, 254]
        vDequantScale.range_values = [0, 254]

        return case_config
