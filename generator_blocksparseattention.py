import random
import copy
import math
from typing import Union, List
import numpy as np
from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig

@GENERATOR_REGISTRY.register("generate_block_sparse_attention")
class BlockSparseAttentionGenerator(CaseGenerator):
    def __init__(self, config):
        super().__init__(config)
        self.input_layout_range = ["TND", "BNSD", "BSND"]
        self.dtype_range = ["fp8e4m3"]
        self.attention_out_dtyp_range = ["fp16", "bf16"]
        self.headDim_range = [64, 128]

    def param_generator(self):
        self.B = 0
        self.N1 = 0
        self.N2 = 0
        self.D = 0
        self.S1_ceil = 0
        self.S2_ceil = 0
        self.S1_aligned = 0
        self.S2_aligned = 0
        self.actseq_aval = 0
        self.is_varied_len = 0
        self.q_seq_list = []
        self.kv_seq_list = []
        self.q_input_layout = "TND"
        self.kv_input_layout = "TND"
        self.softmax_lse_flag = 0
        self.T1 = 0
        self.T2 = 0
        self.inner_prec = 4
        self.block_shape_x = 128
        self.block_shape_y = 128
        self.block_shape_aval = 1

        self.block_shape_x = random.choice([128])
        self.block_shape_y = random.choice([256, 512])
        self.block_shape_list = [self.block_shape_x, self.block_shape_y]

        self.D = random.choice(self.headDim_range)
        self.dtype_qkv = random.choice(self.dtype_range)

        self.B = random.randint(1, 3)
        self.S1_ceil = random.randint(1024, 4096)
        self.S2_ceil = random.randint(1024, 4096)
        self.N2 = random.randint(1, 3)
        self.G = random.randint(1, 3)
        self.N1 = self.N2 * self.G
        self.N2 = self.N1
        self.q_input_layout = random.choice(self.input_layout_range)
        self.kv_input_layout = self.q_input_layout
        self.is_varied_len = random.choice([0, 1])

        # 生成seqlen相关参数
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
    
    def expand_list_config(self, config_list, value_list):
        if not config_list: return
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
        # qkvo + blockSparseMask
        query.dtype = self.dtype_qkv
        key.dtype = self.dtype_qkv
        value.dtype = self.dtype_qkv
        attentionOut.dtype = random.choice(self.attention_out_dtyp_range)
        if self.q_input_layout == "TND":
            qInputLayout.range_values = ["TND"]
            query.shape = [self.T1, self.N1, self.D]
            attentionOut.shape = [self.T1, self.N1, self.D]
        elif self.q_input_layout == "BNSD":
            qInputLayout.range_values = ["BNSD"]
            query.shape = [self.B, self.N1, self.S1_aligned, self.D]
            attentionOut.shape = [self.B, self.N1, self.S1_aligned, self.D]
        elif self.q_input_layout == "BSND":
            qInputLayout.range_values = ["BSND"]
            query.shape = [self.B, self.S1_aligned, self.N1, self.D]
            attentionOut.shape = [self.B, self.S1_aligned, self.N1, self.D]

        if self.kv_input_layout == "TND":
            kvInputLayout.range_values = ["TND"]
            key.shape = [self.T2, self.N2, self.D]
            value.shape = [self.T2, self.N2, self.D]
        elif self.kv_input_layout == "BNSD":
            kvInputLayout.range_values = ["BNSD"]
            key.shape = [self.B, self.N2, self.S2_aligned, self.D]
            value.shape = [self.B, self.N2, self.S2_aligned, self.D]
        elif self.kv_input_layout == "BSND":
            kvInputLayout.range_values = ["BSND"]
            key.shape = [self.B, self.S2_aligned, self.N2, self.D]
            value.shape = [self.B, self.S2_aligned, self.N2, self.D]
        blockSparseMask.dtype = "int8"
        blockSparseMask.shape = [self.B, self.N1, self.x_block_num_aligned, self.y_block_num_aligned]
        blockSparseMask.range_values = [0, 1]
        # blockShape
        if self.block_shape_aval:
            self.expand_list_config(blockShape, self.block_shape_list)
        else:
            blockShape.range_values = ["None"]
        # actseq
        self.expand_list_config(actualSeqLengths, self.q_seq_list)
        self.expand_list_config(actualSeqLengthsKv, self.kv_seq_list)
        # attenMask & blockTable & lse
        # attenMask.range_values = ["None"]
        # blockEffRows shape: [batch, numHeads, maxBlockNum, 2], only when numHeads == kvHeads
        if self.N1 == self.N2:
            max_block_num = max(self.x_block_num_aligned, self.y_block_num_aligned)
            attenMask.shape = [self.B, self.N1, max_block_num, 2]
            attenMask.dtype = 'int32'
        blockTable.range_values = ["None"]
        # attrs
        numKeyValueHeads.range_values = [self.N2]
        maskType.range_values = [0]
        scaleValue.range_values = [1 / (self.D ** 0.5)]
        innerPrecise.range_values = [self.inner_prec]
        blockSize.range_values = [0]
        preTokens.range_values = [2147483647]
        nextTokens.range_values = [2147483647]
        softmaxLseFlag.range_values = [self.softmax_lse_flag]

        if self.softmax_lse_flag == 1:
            if self.q_input_layout == "TND":
                softmaxLse.shape = [self.T1, self.N1, 1]
                softmaxLse.range_values = [-5 ,5]
            elif self.q_input_layout == "BNSD":
                softmaxLse.shape = [self.B, self.N1, self.S1_aligned, 1]
                softmaxLse.range_values = [-5 ,5]
            elif self.q_input_layout == "BSND":
                softmaxLse.shape = [self.B, self.S1_aligned, self.N1, 1]
                softmaxLse.range_values = [-5 ,5]
        else:
            softmaxLse.range_values = ["None"]

        qDequantScale = inputs_map.get('qDequantScale')
        kDequantScale = inputs_map.get('kDequantScale')
        vDequantScale = inputs_map.get('vDequantScale')

        qDequantScale.dtype = "fp32"
        qDequantScale.shape = [self.B, self.N1, self.x_block_num_aligned, 1]
        qDequantScale.range_values = [0, 1]

        kDequantScale.dtype = "fp32"
        kDequantScale.shape = [self.B, self.N2, self.y_block_num_aligned, 1]
        kDequantScale.range_values = [0, 1]

        vDequantScale.dtype = "fp32"
        vDequantScale.shape = [self.B, self.N2, self.y_block_num_aligned, 1]
        vDequantScale.range_values = [0, 1]

        return case_config
