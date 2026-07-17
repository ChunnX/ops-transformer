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
import torch_npu

import ctypes
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor
from atk.tasks.backends.lib_interface.acl_wrapper import AclFormat

torch.use_deterministic_algorithms(True)
@register("function_matmul_reduce_satter_v2")
class MatmulReduceScatterV2(BaseApi):
    def __init__(self, task_result: TaskResult):
        super(MatmulReduceScatterV2, self).__init__(task_result)
        self.dist_task_info = task_result.dist_task_info

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        rank_id = int(self.dist_task_info.rank)
        world_size = self.dist_task_info.world_size
        x1 = input_data.kwargs["x1"]
        x2 = input_data.kwargs["x2"]
        bias = input_data.kwargs["bias"]
        has_bias = input_data.kwargs["hasBias"]
        x1_scale = input_data.kwargs["x1_scale"]
        x2_scale = input_data.kwargs["x2_scale"]
        output_dtype = input_data.kwargs['output_dtype']
        is_format_nz = input_data.kwargs["isFormatNz"]
        use_int64 = input_data.kwargs['use_int64']
        comm_mode = input_data.kwargs['commMode']

        # 反量化类型: 0-非量化, 1-per-channel, 2-per-token + per-channel
        if x2_scale is None:
            dequantType = 0
        elif x1_scale is not None:
            dequantType = 2
        else:
            dequantType = 1
        # 权重以转置形式提供时，先转回 (K, N)
        if x2.shape[0]!=x1.shape[1] and x2.shape[1] == x1.shape[1]:
            x2 = x2.transpose(0, 1)

        if self.name == "cpu":
            # CPU 真值：不支持量化的算子路径在 CPU 上用普通 matmul 模拟
            if dequantType == 0:
                x1 = x1.cpu().to(torch.float32)
                x2 = x2.cpu().to(torch.float32).contiguous()
                output = torch.matmul(x1, x2)
            else:
                # 量化场景：int8 输入用 int32 累加后再乘 scale 反量化
                x1 = x1.cpu().to(torch.int32)
                x2 = x2.cpu().to(torch.int32)
                output = torch.matmul(x1, x2).to(torch.int32)
                x2_scale = x2_scale.cpu()
                if dequantType == 1:
                    output = (output * x2_scale).to(torch.float32)
                elif dequantType==2:
                    x1_scale = x1_scale.cpu()
                    output = (output * x2_scale * x1_scale).to(torch.float32)
            if has_bias and bias is not None:
                output = output + bias.cpu().to(output.dtype)

            dist.all_reduce(output, op=dist.ReduceOp.SUM)
            scatter_shape_m = x1.shape[0] // world_size
            scatter_output = output.narrow(0, rank_id * scatter_shape_m, scatter_shape_m)

            return scatter_output

        if self.dist_task_info.is_bm:
            # NPU 标杆：用基础小算子级联实现 matmul + reduce-scatter
            if x1.shape == []:
                return torch.tensor([])

            tensor_scatter_shape = [x1.shape[0] // world_size, x2.shape[1]]
            if dequantType:
                if use_int64:
                    output_dtype = torch.float16
                if output_dtype is None:
                    output_dtype = torch.bfloat16
            if dequantType==0:
                output = torch.matmul(x1, x2)
            elif dequantType==1:
                output = torch_npu.npu_quant_matmul(x1=x1, x2=x2, scale=x2_scale.squeeze(0), output_dtype=output_dtype)
            else:
                output = torch_npu.npu_quant_matmul(x1=x1, x2=x2, scale=x2_scale.squeeze(0), pertoken_scale=x1_scale.squeeze(-1), output_dtype=output_dtype)
            if has_bias and bias is not None:
                output = output + bias.to(output.dtype)


            if output_dtype is None:
                output_dtype = output.dtype
            
            tensor_scatter = torch.zeros(tensor_scatter_shape, dtype=output.dtype).npu()
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
            if is_format_nz:
                x2 = torch_npu.npu_format_cast(x2, 29)
            if use_int64:
                x2_scale = torch_npu.npu_trans_quant_param(x2_scale)
            bias_arg = bias if has_bias else None
            output_npu = torch_npu.npu_mm_reduce_scatter_base(x1, x2, 
                                                                hcomm_info, world_size, reduce_op="sum", bias=bias_arg, comm_mode=comm_mode,
                                                                x1_scale=x1_scale, x2_scale=x2_scale, output_dtype=output_dtype)
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
        has_bias = input_data.kwargs.get("hasBias", False)
        if x2.shape[0] != x1.shape[1] and x2.shape[1] == x1.shape[1]:
            input_data.kwargs["x2"] = input_data.kwargs["x2"].transpose(0, 1)
        if not has_bias:
            input_data.kwargs['bias'] = None
        if x1.dtype != torch.int8:
            input_data.kwargs['x1_scale'] = None 
            input_data.kwargs['x2_scale'] = None 
            input_data.kwargs['output_dtype'] = None
            input_data.kwargs['use_int64'] = None
            return
        if not input_data.kwargs['use_x1_scale']:
            input_data.kwargs['x1_scale'] = None
        if input_data.kwargs['x2_scale'].dtype == torch.int64:
            input_data.kwargs['output_dtype'] = None


@register("aclnn_matmul_reduce_scatter_v2")
class AclnnMatmulReduceScatterV2(AclnnBaseApi):
    """aclnn 版 MatmulReduceScatter"""

    def __init__(self, task_result: TaskResult, backend):
        super(AclnnMatmulReduceScatterV2, self).__init__(task_result, backend)
        self.dist_task_info = task_result.dist_task_info
        self.is_format_nz = False

    def init_by_input_data(self, input_data: InputDataset):
        input_data.kwargs["reduceOp"] = "sum"
        input_data.kwargs["x1"] = input_data.kwargs["x1"]
        input_data.kwargs["x2"] = input_data.kwargs["x2"]
        if input_data.kwargs["bias"] is not None:
            input_data.kwargs["bias"] = input_data.kwargs["bias"]
        if input_data.kwargs["isFormatNz"]:
            input_data.kwargs['x2'] = torch_npu.npu_format_cast(input_data.kwargs['x2'], 29)
            self.is_format_nz = True

        use_int64 = input_data.kwargs["use_int64"]
        use_x1_scale = input_data.kwargs["use_x1_scale"]
        has_bias = input_data.kwargs["hasBias"]
        if use_int64:
            input_data.kwargs["x2_scale"] = torch_npu.npu_trans_quant_param(input_data.kwargs["x2_scale"])
        input_data.kwargs.pop("output_dtype")
        input_data.kwargs.pop("isFormatNz")
        input_data.kwargs.pop("use_x1_scale")
        input_data.kwargs.pop("use_int64")
        input_data.kwargs.pop("hasBias")
        input_args, output_packages = super().init_by_input_data(input_data)

        if not has_bias:
            input_args[2] = self.get_null_tensor_pte()
        input_args[5] = self.get_null_tensor_pte()
        if input_data.kwargs["x1"].dtype != torch.int8:
            input_args[3] = self.get_null_tensor_pte()
            input_args[4] = self.get_null_tensor_pte()
        if not use_x1_scale:
            input_args[3] = self.get_null_tensor_pte()

        input_args.insert(len(input_args), self.get_null_tensor_pte())
        torch.npu.synchronize()
        return input_args, output_packages

    def get_format(self, input_data: InputDataset, index=None, name=None):
        if name == "x2" and self.is_format_nz:
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND

    def get_null_tensor_pte(self):
        # 构造一个 AclTensor 的空指针，用于占位可选输入
        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_void_ptr = ctypes.c_void_p(None)
        null_tensor_ptr = ctypes.cast(null_void_ptr, acl_tensor_ptr)
        return null_tensor_ptr