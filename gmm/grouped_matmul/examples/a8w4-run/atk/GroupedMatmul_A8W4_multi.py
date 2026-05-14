#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 1.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
import torch
import torch_npu
import ctypes
import logging
import numpy as np
import random
from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.dataset.base_dataset import OpsDataset
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import aclnn, ascendcl, pointer, AclIntArray, AclDataType, \
AclnnStatus, AclFormat, AclrtMemMallocPolicy, nnopbase, Int64, Uint64, TensorPtr, VoidPtr, OpExecutor, AclTensorStruct, AclrtMemcpyKind, AclTensorList
#import tensorflow as tf
#bf16 = tf.bfloat16.as_numpy_dtype
from ml_dtypes import bfloat16
def non_quant_golden_fakequant(x, weight, scale, perTokenScale, groupList, bias, isBF16, isNew = False):
    groupNum, k, n = weight.shape
    quantGroupNum = scale.shape[1]
    index = np.cumsum(groupList)
    xSplit = np.split(x, index, axis=0)
    perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
    weightGroup = weight.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
    mmOuts = []
    atomic = np.float16
    if isNew:
        atomic = np.float32  # using A8W8
    
    for i in range(groupNum):
        xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
        mmi = np.zeros([xi.shape[0], n], dtype=atomic)
        for j in range(quantGroupNum):
            mm = np.matmul(xi[:, j, :].astype(np.float64), weightGroup[i, j, ...].astype(np.float64)).astype(np.float32)
            mm = mm.astype(np.float32) * scale[i, j].reshape(1, -1)
            mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)

        mmi = mmi.astype(np.float32)
        mmi = mmi * perTokenScaleSplit[i]
        mmOuts.append(mmi)
    golden = np.concatenate(mmOuts, axis=0)
    golden_tensor = torch.from_numpy(golden)
    # out_dtype = random.choice([torch.float16, torch.bfloat16])
    out_dtype = torch.float16
    return golden_tensor.to(out_dtype)

# pylint:disable = huawei-too-many-arguments
def gmm_a8w4_golden_fakequant(x_in, weight_in, bias_in, scale_in, groupList_in, perTokenScale_in, isBF16):
    weightNz = weight_in.astype(np.int8)
    groupNum = groupList_in.shape[0]
    m = x_in.shape[0]
    k = x_in.shape[1]
    n = scale_in.shape[2]
    isNew = scale_in.shape[1] == 1
    weight = weightNz.reshape(groupNum, k, n)
    scaleUint32 = scale_in.astype(np.uint32)
    scaleUint32.dtype = np.float32
    out = non_quant_golden_fakequant(x_in, weight, scaleUint32, perTokenScale_in, groupList_in, bias_in, isBF16, isNew)
    return out

def gmm_a8w4_golden(x_in, weight_in, bias_in, scale_in, groupList_in, perTokenScale_in, tuningConfig, offset=None):
    weightNz = weight_in.astype(np.int8)
    groupNum = groupList_in.shape[0]
    m = x_in.shape[0]
    k = x_in.shape[1]
    n = scale_in.shape[2]
    isNew = ((tuningConfig !=0 and tuningConfig <= 64) or (groupNum == 256 and m == 1024 and k == 512 and n == 32768)) and k // scale_in.shape[1] == 256 and k % scale_in.shape[1] == 0 and k%64==0 and n%64==0
    if offset is not None:
        isNew = False

    weight = weightNz.reshape(groupNum, k, n)
    xC12 = np.concatenate([x_in.reshape(m, 1, k) // 16, (x_in.reshape(m, 1, k) & 0x0F) - 8], axis=1).reshape(m * 2, k)
    scaleUint32 = scale_in.astype(np.uint32)
    scaleUint32.dtype = np.float32
    out = non_quant_golden(xC12, weight, scaleUint32, perTokenScale_in, groupList_in, bias_in, isNew, offset)
    return out

def non_quant_golden(x, weight, scale, perTokenScale, groupList, bias, isNew, offset=None):
    groupNum, k, n = weight.shape
    quantGroupNum = scale.shape[1]
    index = np.cumsum(groupList)
    xSplit = np.split(x, index * 2, axis=0)
    perTokenScaleSplit = np.split(perTokenScale, index, axis=0)
    weightGroup = weight.reshape(groupNum, quantGroupNum, k // quantGroupNum, n).astype(np.int32)
    mmOuts = []
    atomic = np.float16
    if isNew:
        atomic = np.float32
    print("[gmm_a8w4_golden] isNew:", isNew, groupNum, k, n)

    for i in range(groupNum):
        xi = xSplit[i].reshape(-1, quantGroupNum, k // quantGroupNum).astype(np.int32)
        mmi = np.zeros([xi.shape[0], n], dtype=atomic)
        for j in range(quantGroupNum):
            mm = np.matmul(xi[:, j, :], weightGroup[i, j, ...])
            if offset is None:
                mm = mm.astype(np.float32) * scale[i, j].reshape(1, -1)
            else:
                mm = mm.astype(np.float32) * scale[i].reshape(1, -1)  # 从scale[i, j]改为scale[i]
            mmi = (mmi.astype(atomic) + mm.astype(atomic)).astype(atomic)
        mmi = mmi.reshape(-1, 2, n).astype(np.float32)
        mmi = mmi[:, 0, :] * 16 + mmi[:, 1, :] + bias[i].reshape(1, n)
        mmi_xo = np.zeros(mmi.shape, dtype=np.float32)
        if offset is not None:
            xi_o = xSplit[i].astype(np.int32).reshape(-1, 2, k)
            xi_o = xi_o[:, 0, :] * 16 + xi_o[:, 1, :] + 8
            xi_o = xi_o.astype(np.float16).reshape(-1, quantGroupNum, k // quantGroupNum)
            for j in range(quantGroupNum):
                mm = xi_o[:, j, :].sum(axis=1, keepdims=True).astype(np.float32)
                mm = np.matmul(mm, offset[i].reshape(1, -1))
                mmi_xo += mm.astype(np.float32)
        mmi += mmi_xo
        xxx = mmi.astype(np.float32) + mmi_xo.astype(np.float32)
        mmi = mmi * perTokenScaleSplit[i]
        mmOuts.append(mmi)
    golden = np.concatenate(mmOuts, axis=0)
    golden_tensor = torch.from_numpy(golden)
    # out_dtype = random.choice([torch.float16, torch.bfloat16])
    out_dtype = torch.float16
    return golden_tensor.to(out_dtype)

def pack_4bit_to_bytes(data):
    data = data.astype(np.int8)
    data[data < 0] += 16
    packed = (data[..., 1::2] << 4) | (data[..., ::2] & 0x0F)
    return packed

@register("function_aclnn_grouped_matmul_V5_a8w4")
class AclnnGroupedMatmulV4A8W4(BaseApi):
    def init_by_input_data(self, input_data: InputDataset):
        """
        该接口可实现部门场景下api的初始化需要依赖于当前的输入数据，且不希望计入耗时，
        可以在此接口实现
        :param input_data:
        :return:
        """
        weight = input_data.kwargs['weight'][0]
        bias = input_data.kwargs['biasOptional'][0]
        scaleOptional = input_data.kwargs['scaleOptional'][0]
        E,quantGroupNum,N = scaleOptional.shape[0],scaleOptional.shape[1],scaleOptional.shape[2]
        M = input_data.kwargs['x'][0].shape[0]
        input_data.kwargs['tuningConfigOptional'] = [M // E]
        k = weight.shape[1]

        self.weight_single = weight.clone()
        self.bias_single = bias.clone()
        self.scale_single = scaleOptional.clone()
        self.multi_weight = (E < 128 and E > 1) and random.choice([True, False])

        self.withOffset = quantGroupNum == 1
        if self.withOffset:
            offset = torch.randint(-5, 5, scaleOptional.shape, dtype=torch.float32, device="cpu")
            self.offset_single = offset.clone()
        else:
            offset = None
            self.offset_single = None

        if self.device == "pyaclnn":

            
            scale_np = scaleOptional.cpu().numpy()
            scaleUint32 = scale_np.astype(bfloat16).astype(np.float32)
            scaleUint32.dtype = np.uint32
            scaleUint64 = np.zeros((E, quantGroupNum, N*2), dtype=np.uint32)
            scaleUint64[...,::2] = scaleUint32
            scaleUint64.dtype = np.int64
            scale = torch.from_numpy(scaleUint64).npu()
            #K = weight.shape[1]
            #weight_in = weight.cpu().numpy()
            if input_data.kwargs.get('format') == 'NZ':
                print("!! transpose to NZ")
                weight = torch.from_numpy(weight.cpu().numpy().reshape([E, k // 16, 16, N // 64, 64]).transpose([0, 3, 1, 2, 4])).to(torch.float32).reshape(E,k,N).npu()
            # weight = torch.from_numpy(weight_in).npu()
            weight_quant = torch_npu.npu_quantize(weight.to(torch.float32), torch.tensor([1.], device='npu'), None,
                                                  torch.quint4x2, -1, False)
            # print("weight_quant.shape",weight_quant.shape)
            bias_npu = bias.to('npu')
            if self.multi_weight:
                input_data.kwargs['weight'] = [weight_quant[i] for i in range(E)]
                input_data.kwargs['scaleOptional'] = [scale[i] for i in range(E)]
                input_data.kwargs['biasOptional'] = [bias_npu[i] for i in range(E)]
            else:
                input_data.kwargs['weight'][0] = weight_quant
                input_data.kwargs['scaleOptional'][0] = scale
                input_data.kwargs['biasOptional'][0] = bias_npu
            if self.withOffset:
                offset_npu = offset.to('npu')
                if self.multi_weight:
                    input_data.kwargs['offsetOptional'] = [offset_npu[i] for i in range(E)]
                else:
                    input_data.kwargs['offsetOptional'] = [offset_npu]
            else:
                input_data.kwargs['offsetOptional'] = None
            input_data.kwargs['groupListOptional'] = input_data.kwargs['groupListOptional'].to('npu')
            input_data.kwargs['perTokenScaleOptional'][0] = input_data.kwargs['perTokenScaleOptional'][0].to('npu')
        else:
            if self.multi_weight:
                input_data.kwargs['weight'] = [weight[i].clone() for i in range(E)]
                input_data.kwargs['scaleOptional'] = [scaleOptional[i].clone() for i in range(E)]
                input_data.kwargs['biasOptional'] = [bias[i].clone() for i in range(E)]
            if self.withOffset:
                if self.multi_weight:
                    input_data.kwargs['offsetOptional'] = [offset[i].clone() for i in range(E)]
                else:
                    input_data.kwargs['offsetOptional'] = [offset.clone()]
            else:
                input_data.kwargs['offsetOptional'] = None
            

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self.device == "gpu":
            device = f"cuda:{self.device_id}"
        elif self.device == "npu":
            device = f"{self.device}:{self.device_id}"
        else:
            device = "cpu"

        x = input_data.kwargs['x'][0]
        M = x.shape[0]
        weight = self.weight_single
        biasOptional = self.bias_single
        scaleOptional = self.scale_single
        E,k,N = scaleOptional.shape[0],scaleOptional.shape[1],scaleOptional.shape[2]
        scale_np = scaleOptional.cpu().numpy()
        scaleUint32 = scale_np.astype(bfloat16).astype(np.float32)
        scaleUint32.dtype = np.uint32
        scaleUint64 = np.zeros((E, k, N*2), dtype=np.uint32)
        scaleUint64[...,::2] = scaleUint32
        scaleUint64.dtype = np.int64

        perTokenScaleOptional = input_data.kwargs['perTokenScaleOptional'][0]
        groupListOptional = input_data.kwargs['groupListOptional']
        tuningConfig = input_data.kwargs['tuningConfigOptional'][0]

        x_in = x.cpu().numpy()
        weight_in = weight.cpu().numpy()
        bias_in = biasOptional.cpu().numpy()
        scale_in = scaleUint64
        groupList_in = groupListOptional.cpu().numpy()
        perTokenScale_in = perTokenScaleOptional.cpu().numpy()

        if self.withOffset:
            offset_in = self.offset_single.cpu().numpy()
        else:
            offset_in = None

        # const bool isMSD = (tuningConfig == 0L || avg_m == 0U || n / avg_m > 4U || withOffset == true) &&
        #             !(biasPtr == nullptr || biasPtr->GetStorageShape().GetShapeSize() == 0);
        if tuningConfig == 0:
            avg_m = M // E
        else:
            avg_m = tuningConfig
            
        isMSD = tuningConfig == 0 or avg_m == 0 or N // avg_m > 4 or self.withOffset
        # print("isMSD", isMSD, "tuningConfig", tuningConfig, "N // avg_m (>4)", N // avg_m)
        # print("x_in", x_in.flatten()[:128])
        # print("weight_in", weight_in.flatten()[:128])
        # print("bias_in", bias_in.flatten()[:128])
        # print("scale_in", scale_in.flatten()[:128])
        # print("groupList_in", groupList_in.flatten()[:2])
        # print("perTokenScale_in", perTokenScale_in.flatten()[:16])
        if isMSD:
            t1 = gmm_a8w4_golden(x_in=x_in, weight_in=weight_in, bias_in=bias_in,
                                    scale_in=scale_in, groupList_in=groupList_in, perTokenScale_in=perTokenScale_in, tuningConfig=tuningConfig, offset=offset_in)
        else:
            t1 = gmm_a8w4_golden_fakequant(x_in=x_in, weight_in=weight_in, bias_in=bias_in,
                                    scale_in=scale_in, groupList_in=groupList_in, perTokenScale_in=perTokenScale_in, isBF16 = False)

        t2 = torch.zeros(M, N, dtype=t1.dtype, device=t1.device)
        t2[:t1.shape[0]] = t1

        return t2

    def get_cpp_func_signature_type(self):
        return "aclnnStatus aclnnGroupedMatmulV5GetWorkspaceSize(const aclTensorList *x, const aclTensorList *weight, \
        const aclTensorList *biasOptional, const aclTensorList *scaleOptional, const aclTensorList *offsetOptional, \
        const aclTensorList *antiquantScaleOptional, const aclTensorList *antiquantOffsetOptional, const aclTensorList *perTokenScaleOptional, \
        const aclTensor *groupListOptional, const aclTensorList *activationInputOptional, const aclTensorList *activationQuantScaleOptional,  \
        const aclTensorList *activationQuantOffsetOptional, int64_t splitItem, int64_t groupType, int64_t groupListType, int64_t actType, aclIntArray *tuningConfigOptional, \
        aclTensorList *out, aclTensorList *activationFeatureOutOptional, aclTensorList *dynQuantScaleOutOptional, uint64_t *workspaceSize, \
        aclOpExecutor **executor)"

@register("function_pyaclnn_grouped_matmul_V5_a8w4")
class PyaclnnGroupedMatmulV4A8W4(AclnnBaseApi):
    def __init__(self,task_result:TaskResult,backend):
        super().__init__(task_result,backend)
        self.input_args = None

    def init_by_input_data(self, input_data: InputDataset):
        from atk.tasks.backends.lib_interface.acl_wrapper import AclIntArray
        self.task_result.output_info_list = [self.task_result.output_info_list]
        input_args, output_packages = super().init_by_input_data(input_data)
        self.input_args, tmp = super().init_by_input_data(input_data)
        format = input_data.kwargs["format"]
        for name, kwarg in input_data.kwargs.items():
            if name == "weight" and format == "NZ":
                print("*** weight NZ!")
                if len(kwarg[0].shape) == 2:
                    isSingleWeight = False
                    for i in range(len(kwarg)):
                        kwarg[i] = self.nd_to_nz(kwarg[i],isSingleWeight)
                else:
                    isSingleWeight = True
                    data = self.nd_to_nz(kwarg[0],isSingleWeight)
                    kwarg[0] = data
        input_args[4] = ctypes.POINTER(AclTensorList)()
        input_args[5] = ctypes.POINTER(AclTensorList)()
        input_args[6] = ctypes.POINTER(AclTensorList)()
        input_args[9] = ctypes.POINTER(AclTensorList)()
        input_args[10] = ctypes.POINTER(AclTensorList)()
        input_args[11] = ctypes.POINTER(AclTensorList)()
        input_args.append(ctypes.POINTER(AclTensorList)())
        input_args.append(ctypes.POINTER(AclTensorList)())
        void_p = ctypes.c_void_p(0)
        AclIntArrayPtr = ctypes.POINTER(AclIntArray)
        # input_args[16] = ctypes.cast(void_p, AclIntArrayPtr)

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

        weight = input_data.kwargs["weight"]
        format = input_data.kwargs.get("format")
        if name == "weight" and format == 'NZ':
            # print('******************************',name)
            return AclFormat.ACL_FORMAT_FRACTAL_NZ
        return AclFormat.ACL_FORMAT_ND
    def nd_to_nz(self, weight, isSingleWeight = True):
        # 1. 调用aclnnCalculateMatmulWeightSizeV2
        print('****************weight.shape',weight.shape)
        if isSingleWeight:
            e, k, n = weight.shape
        else:
            k, n = weight.shape
            e = 1
        # weightShape = nnopbase.create_x_list([Int64(e), Int64(k), Int64(n)])
        size = Uint64(weight.numel())

        # 2. 调用aclCreateTensor
        device_addr = VoidPtr()
        ascendcl.aclrtMalloc(ctypes.byref(device_addr), size, AclrtMemMallocPolicy.ACL_MEM_MALLOC_HUGE_FIRST)
        ascendcl.aclrtMemcpy(device_addr, size.value, weight.data_ptr(), n * k * e,
                             AclrtMemcpyKind.ACL_MEMCPY_HOST_TO_DEVICE)
        # 计算连续tensor的strides
        strides = (Int64 * len(weight.shape))(*[1] * len(weight.shape))
        for i in range(len(weight.shape) - 2, -1, -1):
            strides[i] = weight.shape[i + 1] * strides[i + 1]

        storageShape = [e, n//64 ,k//16,16,64]
        # 调用aclCreateTensor接口创建aclTensor
        aclTensor = nnopbase.aclCreateTensor(
            (Int64 * len(weight.shape))(*weight.shape), len(weight.shape), AclDataType.ACL_INT4, strides, 0,
        AclFormat.ACL_FORMAT_FRACTAL_NZ, (Int64 * len(storageShape))(*storageShape), len(storageShape), device_addr
        )
        # print("type(aclTensor) = ",type(aclTensor))

        return [AclTensorStruct(aclTensor, device_addr, [size.value], AclDataType.ACL_INT4)]
    def after_call(self, output_packages):
        groupListOptional = self.acl_tensor_to_torch(self.input_args[8])
        E = groupListOptional.shape[0]
        output = []
        t1 = self.backend.acl_tensorlist_to_torch(output_packages[0])[0]
        t2 = torch.zeros_like(t1)
        t2[:E,:] = t1[:E,:]
        output.append(t2)
        return output
