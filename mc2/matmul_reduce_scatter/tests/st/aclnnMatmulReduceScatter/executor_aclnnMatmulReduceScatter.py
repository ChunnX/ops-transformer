#!/usr/bin/env python3
# -*- coding: utf-8 -*-
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
import torch.distributed as dist
from torch.distributed import ReduceOp
try:
   import torch_npu
except ImportError:
   pass
import ctypes

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.configs.results_config import TaskResult
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

@register("execute_aclnnMatmulReduceScatter")
class DistFunctionApi(BaseApi):
    """Matmul + ReduceScatter 真值/标杆/融合算子实现"""

    def __init__(self, task_result):
        super(DistFunctionApi, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = int(self.dist_task_info.rank)
        world_size = self.dist_task_info.world_size
        input_chunk = input_data.kwargs['x1']
        weight_chunk = input_data.kwargs['x2']

        # 权重以转置形式提供时，先转回 (K, N)
        if weight_chunk.shape[0]!=input_chunk.shape[1] and weight_chunk.shape[1] == input_chunk.shape[1]:
            weight_chunk = weight_chunk.transpose(0, 1)

        if self.name == "cpu":
            # CPU 真值
            input_chunk = input_chunk.cpu().to(torch.float32)
            weight_chunk = weight_chunk.cpu().to(torch.float32).contiguous()
            output = torch.matmul(input_chunk, weight_chunk)
            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            scatter_shape_m = input_chunk.shape[0] // world_size
            scatter_output = output.narrow(0, rank_id * scatter_shape_m, scatter_shape_m)
            return scatter_output

        if self.dist_task_info.is_bm:
            # NPU 标杆：用基础小算子级联实现 matmul + reduce-scatter
            if input_chunk.shape == []:
                return torch.tensor([])
            tensor_scatter_shape = [input_chunk.shape[0] // world_size, weight_chunk.shape[1]]
            tensor_scatter = torch.zeros(tensor_scatter_shape, dtype=input_chunk.dtype).npu()
            output = torch.matmul(input_chunk, weight_chunk)
            dist._reduce_scatter_base(tensor_scatter, output, op=ReduceOp.SUM)
            scatter_output = tensor_scatter
            return scatter_output
        else:
            # NPU 融合算子：直接调用 mc2 融合接口 npu_mm_reduce_scatter_base
            if dist.is_available():
                from torch.distributed.distributed_c10d import _get_default_group
                default_pg = _get_default_group()
                if torch.__version__ > '2.0.1':
                    hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
                else:
                    hcomm_info = default_pg.get_hccl_comm_name(rank_id)
            else:
                bias = None
            output_npu = torch_npu.npu_mm_reduce_scatter_base(input_chunk, weight_chunk,
                                                      hcomm_info, world_size, reduce_op="sum", bias=bias)
            return output_npu

    def init_by_input_data(self, input_data: InputDataset):
        rank_id = self.dist_task_info.rank
        if self.device == 'pyaclnn' and dist.is_available():
            from torch.distributed.distributed_c10d import _get_default_group
            default_pg = _get_default_group()
            if torch.__version__ > '2.0.1':
                hcomm_info = default_pg._get_backend(torch.device("npu")).get_hccl_comm_name(rank_id)
            else:
                hcomm_info = default_pg.get_hccl_comm_name(rank_id)
            input_data.kwargs['group'] = hcomm_info
        OpsDataset.seed_everything()
        x1 = input_data.kwargs['x1']
        x2 = input_data.kwargs["x2"]
        if x2.shape[0] != x1.shape[1] and x2.shape[1] == x1.shape[1]:
            input_data.kwargs["x2"] = input_data.kwargs["x2"].transpose(0, 1)


@register("aclnn_matmul_reduce_scatter")
class AclnnMatmulReduceScatter(AclnnBaseApi):
    """aclnn 版 MatmulReduceScatter"""

    def __init__(self, task_result: TaskResult, backend):
        super(AclnnMatmulReduceScatter, self).__init__(task_result, backend)
        self.dist_task_info = task_result.dist_task_info

    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["reduceOp"] = "sum"
        # 转化为 ascendc 入参
        input_args, output_packages = super().init_by_input_data(input_data)
        torch.npu.synchronize()
        return input_args, output_packages

    def get_null_tensor_pte(self):
        # 构造一个 AclTensor 的空指针，用于占位可选输入
        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, acl_tensor_ptr)
        return null_tensor_ptr