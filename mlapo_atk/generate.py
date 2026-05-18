# Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
import random
from typing import Union, List

from atk.case_generator.generator.generate_types import GENERATOR_REGISTRY
from atk.case_generator.generator.base_generator import CaseGenerator
from atk.configs.case_config import InputCaseConfig, CaseConfig


@GENERATOR_REGISTRY.register("ascend_generate_aclnn_mla_preprocess")
class MlaPreprocessGenerator(CaseGenerator):

    def __init__(self, config):
        super().__init__(config)
        self.tensor_dim = 0
        self.range_is_null = False

    def after_case_config(self, case_config: CaseConfig) -> CaseConfig:

        input_token_nums = [16, 32, 48, 64, 128, 256]
        # input_token_nums = [32]

        input_token_num = random.choice(input_token_nums)
        # hiddenStrate = 256 * rand_hidden_base  hideenSize 2K-8K 256对齐，超过会出现越界错误；
        # hiddenSizes = [2048, 2304, 2560, 2816, 3072, 3328, 3584, 3840, 4096, 4352, 4608, 4864, 5120, 5376, 5632, 5888, 6144, 6400, 6656, 6912, 7168]
        hiddenSizes = [7424, 7680, 7936, 8192, 8448, 8704, 8960, 9216, 9472, 9728, 9984, 10240]
        # hiddenSizes = [7168]
        hiddenStrate = random.choice(hiddenSizes)
        # 已有128
        # headNums = [16,32,64,128]
        # headNums = [11]
        # headNums = [128]

        headNum = random.randint(1,128) #headnum 128泛化
        blockNum = 192
        blockSize = 128

        qlora = random.randint(1, 4096 // 32) * 32
        nope = random.randint(1, 256 // 16) * 16

        # qlora = 0
        # nope = 0

        mm1_out_shape = qlora + 576
        mm2_out_shape = nope + 64

        cache_mode = case_config.inputs[33].range_values
        # input
        case_config.inputs[0].shape = [input_token_num, hiddenStrate]
        in_dtype = case_config.inputs[0].dtype
        # gamma0
        case_config.inputs[1].shape = [hiddenStrate]
        case_config.inputs[1].dtype = in_dtype
        # beta0
        case_config.inputs[2].shape = [hiddenStrate]
        case_config.inputs[2].dtype = in_dtype
        # quantScale0
        case_config.inputs[3].shape = [1]
        case_config.inputs[3].dtype = in_dtype
        # quantOffset0
        case_config.inputs[4].shape = [1]
        # wdqkv
        # case_config.inputs[5].shape = [2112, hiddenStrate]
        case_config.inputs[5].shape = [mm1_out_shape, hiddenStrate]
        # deScale0
        # case_config.inputs[6].shape = [2112]
        case_config.inputs[6].shape = [mm1_out_shape]
        # if in_dtype == 'fp16':
        #     case_config.inputs[6].dtype = 'int64'
        # elif in_dtype == 'bf16':
        #     case_config.inputs[6].dtype = 'fp32'
        # bias0
        # case_config.inputs[7].shape = [2112]
        case_config.inputs[7].shape = [mm1_out_shape]
        # gamma1
        # case_config.inputs[8].shape = [1536]
        case_config.inputs[8].shape = [qlora]
        case_config.inputs[8].dtype = in_dtype
        # beta1
        # case_config.inputs[9].shape = [1536]
        case_config.inputs[9].shape = [qlora]
        case_config.inputs[9].dtype = in_dtype
        # quantScale1
        case_config.inputs[10].shape = [1]
        case_config.inputs[10].dtype = in_dtype
        # quantOffset1
        case_config.inputs[11].shape = [1]
        # wuq
        # case_config.inputs[12].shape = [headNum * 192, 1536]
        case_config.inputs[12].shape = [headNum * mm2_out_shape, qlora]
        #todo
        # deScale1
        # case_config.inputs[13].shape = [headNum * 192]
        case_config.inputs[13].shape = [headNum * mm2_out_shape]
        # bias1
        # case_config.inputs[14].shape = [headNum * 192]
        case_config.inputs[14].shape = [headNum * mm2_out_shape]
        # gamma2
        case_config.inputs[15].shape = [512]
        case_config.inputs[15].dtype = in_dtype
        # cos
        case_config.inputs[16].shape = [input_token_num, 64]
        case_config.inputs[16].dtype = in_dtype
        # sin
        case_config.inputs[17].shape = [input_token_num, 64]
        case_config.inputs[17].dtype = in_dtype
        # wuk   
        # case_config.inputs[18].shape = [headNum, 128, 512] # ND
        case_config.inputs[18].shape = [headNum, nope, 512]
        # nz格式
        # case_config.inputs[18].shape = [headNum, 32, 128, 16] #NZ
        case_config.inputs[18].dtype = in_dtype
        
        # 19:kvCache , 20:kvCacheRope
        if cache_mode == 0:
            case_config.inputs[19].shape = [blockNum, blockSize, 1, 576]
            case_config.inputs[20].shape = [blockNum, blockSize, 1, 576]
        elif cache_mode == 1:
            #标杆需要尾轴是576，aclnn在这个模式是512
            case_config.inputs[19].shape = [blockNum, blockSize, 1, 512]
            case_config.inputs[20].shape = [blockNum, blockSize, 1, 64]
        elif cache_mode == 2:# ND 表示，需转NZ
            # case_config.inputs[19].shape = [blockNum, 512//32,blockSize, 32]
            # case_config.inputs[19].dtype = 'int8'
            # case_config.inputs[20].shape = [blockNum, 64 // 16 ,blockSize, 16]
            case_config.inputs[19].shape = [blockNum, blockSize, 1, 512]
            case_config.inputs[20].shape = [blockNum, blockSize, 1, 64]
        elif cache_mode == 3:
            # case_config.inputs[19].shape = [blockNum, 512//16,blockSize, 16]
            # case_config.inputs[20].shape = [blockNum, 64 // 16 ,blockSize, 16]
            case_config.inputs[19].shape = [blockNum, blockSize, 1, 512]
            case_config.inputs[20].shape = [blockNum, blockSize, 1, 64]
        case_config.inputs[19].dtype = in_dtype
        if cache_mode == 2:
            case_config.inputs[19].dtype = 'int8'
        case_config.inputs[20].dtype = in_dtype
        # slotmapping
        case_config.inputs[21].shape = [input_token_num]
        case_config.inputs[21].range_values = random.randint(0, 192 * 128 - 1)
        # ctkvScale
        case_config.inputs[22].shape = [1]
        case_config.inputs[22].dtype = in_dtype
        # qNopeScale
        case_config.inputs[23].shape = [headNum]
        case_config.inputs[23].dtype = in_dtype

        return case_config