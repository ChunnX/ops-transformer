 #!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

import sys
import torch
import numpy as np
import torch_npu
import random
import os
import logging

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult

from atk.case_generator.generator.data_types import DATATYPE_REGISTRY
from atk.case_generator.generator.data_types.data_base import BaseDataType
from atk.case_generator.utils.enums import TorchDtype
from atk.configs.case_config import InputCaseConfig

@DATATYPE_REGISTRY.register("torch_npu_npu_nsa_compress_infer_slot_mapping_int32")
class NpuNsaCompressInfer(BaseDataType):
    def __init__(self, input_config: InputCaseConfig):
        super(NpuNsaCompressInfer, self).__init__(input_config)
        random.seed(2023)
        np.random.seed(2023)
        torch.manual_seed(2023)
        os.environ["PYTHONHASHSEED"] = str(2023)

    def gen_data(self):
        dtype = self.config.dtype.split("_")[-1]
        torch_dtype = TorchDtype[dtype.upper()].value
        slot_mapping_tensor = self.generate_unique_random_slot_mapping(self.config.shape[0],self.config.range_values[1])
        return slot_mapping_tensor

    def generate_unique_random_slot_mapping(self, slot_mapping_count, result_len):
        numbers = np.arange(result_len)
        if len(numbers) < slot_mapping_count:
            raise ValueError(
                "Requested number of unique values is larger than possible unique values at this precision.")
        slot_mapping_np = np.random.choice(numbers, size=slot_mapping_count, replace=False).astype(np.int32)
        slot_mapping = torch.from_numpy(slot_mapping_np)

        return slot_mapping
@register("execute_aclnnNsaCompressWithCache")
class MethodTorchNsaCompressWithCacheApi(BaseApi):
    def __init__(self, task_result: TaskResult): # 固定写法
        super(MethodTorchNsaCompressWithCacheApi, self).__init__(task_result)
    def get_cpp_func_signature_type(self): # 获取aclnn第一段接口签名
       return "aclnnStatus aclnnNsaCompressWithCacheGetWorkspaceSize(const aclTensor *input, const aclTensor *weight, const aclTensor *slotMapping, const aclIntArray *actSeqLenOptional,const aclTensor *blockTableOptional, char *layoutOptional, int64_t compressBlockSize, int64_t compressStride,int64_t actSeqLenType, int64_t pageBlockSize, aclTensor *outputCache, uint64_t *workspaceSize,aclOpExecutor **executor);"
    def __call__(self, input_data: InputDataset, with_output: bool = False): # torch标杆实现
        ipt = input_data.kwargs["input"]
        weight = input_data.kwargs["weight"]
        slot_mapping = input_data.kwargs["slot_mapping"]
        output_cache = input_data.kwargs["output_cache"]
        act_seq_lens = input_data.kwargs["act_seq_len"]
        block_table = input_data.kwargs["block_table"]
        ipt_dtype = ipt.dtype
        batch_size = slot_mapping.shape[0]
        compress_block_size = weight.shape[0]
        compress_stride=input_data.kwargs["compress_stride"]
        page_block_size = ipt.shape[1]
        ipt = ipt.to(torch.float)
        weight = weight.to(torch.float)
        output_cache = output_cache.to(torch.float)
        for batch_idx in range(batch_size):
            if(act_seq_lens[batch_idx] >= compress_block_size and (act_seq_lens[batch_idx] - compress_block_size) % compress_stride == 0):
                last_kv = self.get_last_kv(ipt, act_seq_lens, block_table, batch_idx, compress_block_size, page_block_size)
                weight_last_kv = last_kv * weight.unsqueeze(2)
                compress_last_kv = weight_last_kv.sum(axis=0)
                output_cache[slot_mapping[batch_idx]] = compress_last_kv
        return output_cache.to(ipt_dtype)

    def get_last_kv(self, input_kv, act_seq_lens, block_table, batch_idx, compress_block_size, page_block_size):
        act_seq_len = act_seq_lens[batch_idx]
        block_num = (act_seq_len - 1) // page_block_size
        tile_act_seq_len = act_seq_len - block_num * page_block_size
        if tile_act_seq_len < compress_block_size:
            cur_block_idx = block_table[batch_idx, block_num]
            pre_block_idx = block_table[batch_idx, block_num - 1]
            input_kv_cur = input_kv[cur_block_idx, 0:tile_act_seq_len]
            input_kv_pre = input_kv[pre_block_idx, page_block_size - compress_block_size + tile_act_seq_len:page_block_size]
            return torch.concatenate((input_kv_pre, input_kv_cur), axis=0)

        block_idx = block_table[batch_idx, block_num]
        return input_kv[block_idx, tile_act_seq_len - compress_block_size:tile_act_seq_len]


 # 下面用于处理输入即输出的函数，一般情况下不用实现
@register("aclnn_execute_aclnnNsaCompressWithCache")
class AclnnFunctionApi(AclnnBaseApi):
    def __call__(self):
        super().__call__()

    def init_by_input_data(self, input_data): # 处理输入即输出的情况
        input_args, output_packages = super().init_by_input_data(input_data)
        if "is_nd" in input_data.kwargs:
            print(len(input_args))
            is_nd = input_data.kwargs["is_nd"]
            if is_nd is False:
                input_args.pop(-2)  # 去除入参列表中的最后一个值
        # 把最后一个入参去除（默认处理会把标杆的输出拼到入参后，但实际上算子并无此参数）
        for i in range(len(output_packages)):
            input_args.pop()
        # 将前1个输入参数标记为输出
        output_packages[:] = [input_args[10]]

        return input_args, output_packages

    def get_format(self, input_data: InputDataset, index=None, name=None):
        """
        :param input_data: 参数列表
        :param index: 参数位置
        :param name: 参数名字
        :return:
        format at this index or name
        """
        from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat
        if "is_nd" in input_data.kwargs:
            is_nd = input_data.kwargs["is_nd"]
            if is_nd is False and name == "input":
                return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND

    def after_call(self, output_packages):
        output = super().after_call(output_packages)
        return output