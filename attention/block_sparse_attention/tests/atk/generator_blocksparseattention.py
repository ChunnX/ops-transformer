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
    def __init__(self, config):
        super().__init__(config)
        self.input_layout_range = ["BNSD", "BSND", "TND"]
        self.headDim_range = [64, 128]
        self.block_shape_x_range = [64, 128]
        self.block_shape_y_range = [128, 256]
        self.attention_out_dtype_range = ["bf16", "fp16"]
        self.quant_mode_range = [2, 3]
        self.E8M0_BIAS = 127

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

        self.block_shape_x = random.choice(self.block_shape_x_range)
        self.block_shape_y = random.choice(self.block_shape_y_range)
        self.block_shape_list = [self.block_shape_x, self.block_shape_y]

        self.D = random.choice(self.headDim_range)

        self.B = random.randint(1, 8)
        self.S1_ceil = random.choice([128, 256, 512, 1024])
        self.S2_ceil = random.choice([128, 256, 512, 1024, 4096])
        self.N2 = random.randint(1, 4)
        self.G = random.randint(1, 4)
        self.N1 = self.N2 * self.G
        self.q_input_layout = random.choice(self.input_layout_range)
        self.kv_input_layout = self.q_input_layout
        self.is_varied_len = random.choice([0, 1]) if self.q_input_layout == "TND" else 0

        self.quant_mode = random.choice(self.quant_mode_range)
        if self.quant_mode == 3:
            self.dst_type_max = random.choice([0.0, 6.0, 8.0, 10.0])
        else:
            self.dst_type_max = 0.0

        self.attention_out_dtype = random.choice(self.attention_out_dtype_range)

        for b in range(self.B):
            S1_cur = self.S1_ceil
            S2_cur = self.S2_ceil
            if self.is_varied_len:
                S1_cur = random.randint(1, self.S1_ceil)
                S2_cur = random.randint(1, self.S2_ceil)
            self.T1 += S1_cur
            self.T2 += S2_cur
            self.q_seq_list.append(S1_cur)
            self.kv_seq_list.append(S2_cur)

        self.S1_aligned = max(self.q_seq_list)
        self.S2_aligned = max(self.kv_seq_list)
        self.x_block_num_aligned = (self.S1_aligned + self.block_shape_x - 1) // self.block_shape_x
        self.y_block_num_aligned = (self.S2_aligned + self.block_shape_y - 1) // self.block_shape_y

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

        query = inputs_map.get('query')
        key = inputs_map.get('key')
        value = inputs_map.get('value')
        attentionOut = inputs_map.get('attentionOut')
        blockSparseMask = inputs_map.get('blockSparseMask')
        attenMask = inputs_map.get('attenMask')
        blockShape = inputs_map.get('blockShape')
        actualSeqLengths = inputs_map.get('actualSeqLengths')
        actualSeqLengthsKv = inputs_map.get('actualSeqLengthsKv')
        blockTable = inputs_map.get('blockTable')
        softmaxLse = inputs_map.get('softmaxLse')

        qInputLayout = inputs_map.get('qInputLayout')
        kvInputLayout = inputs_map.get('kvInputLayout')
        numKeyValueHeads = inputs_map.get('numKeyValueHeads')
        maskType = inputs_map.get('maskType')
        scaleValue = inputs_map.get('scaleValue')
        innerPrecise = inputs_map.get('innerPrecise')
        blockSize = inputs_map.get('blockSize')
        preTokens = inputs_map.get('preTokens')
        nextTokens = inputs_map.get('nextTokens')
        softmaxLseFlag = inputs_map.get('softmaxLseFlag')
        quantMode = inputs_map.get('quantMode')
        dstTypeMax = inputs_map.get('dstTypeMax')

        # MXFP4: QKV packed as uint8 (FP4 nibble-packed, last_dim = D//2)
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
        blockSparseMask.shape = [self.B, self.N1, self.x_block_num_aligned, self.y_block_num_aligned]
        blockSparseMask.range_values = [0, 1]

        self.expand_list_config(blockShape, self.block_shape_list)
        self.expand_list_config(actualSeqLengths, self.q_seq_list)
        self.expand_list_config(actualSeqLengthsKv, self.kv_seq_list)

        attenMask.range_values = ["None"]
        blockTable.range_values = ["None"]

        numKeyValueHeads.range_values = [self.N2]
        maskType.range_values = [0]
        scaleValue.range_values = [1 / (self.D ** 0.5)]
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

        qDequantScale = inputs_map.get('qDequantScale')
        kDequantScale = inputs_map.get('kDequantScale')
        vDequantScale = inputs_map.get('vDequantScale')

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
