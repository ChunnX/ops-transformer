#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software: you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the License).
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
PoolKeyIndexer ATK PyAclnn 适配器 (AclnnBaseApi)。

注册名: "aclnn_pool_key_indexer" (对应 YAML aclnn_api_type)

职责(主要完成 aclnn 接口的测试):
  1. init_by_input_data: 构造 aclnn C 接口入参, 调用 aclnnPoolKeyIndexer;
     - poolTailK / actualSeqQ / actualSeqK 为 ValueDepend 输入,
       传 aclIntArray*(Host 侧 int64 值), 未启用时传空指针;
     - blockTable / qDescale / kDescale 为可选 aclTensor*, 未启用传空指针;
     - 属性顺序与 op_def 声明一致: topk, poolSize, layoutQuery, layoutKey,
       maskMode, quantMode, returnValue, keyStride0, kDescaleStride0;
     - 输出 sparseIndicesOut(INT32) / sparseValuesOut(FLOAT, return_value=False
       时为空 tensor), 规格由标杆节点运行结果(output_info_list)提供。
  2. 算子输出保持原始结果(不做任何归一化/重排后处理), 由精度比较插件
     (pool_key_indexer_compare.py) 按 pytest 口径比对。

aclnn 接口签名(与 op_host/pool_key_indexer_def.cpp 声明顺序一致):
  aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize(
      const aclTensor* query, const aclTensor* poolKey, const aclTensor* weights,
      const aclIntArray* poolTailK,
      const aclIntArray* actualSeqLengthsQueryOptional,
      const aclIntArray* actualSeqLengthsKeyOptional,
      const aclTensor* blockTableOptional,
      const aclTensor* qDescaleOptional, const aclTensor* kDescaleOptional,
      int64_t topk, int64_t poolSize,
      char* layoutQueryOptional, char* layoutKeyOptional,
      int64_t maskMode, int64_t quantMode, bool returnValue,
      int64_t keyStride0, int64_t kDescaleStride0,
      const aclTensor* sparseIndicesOut, const aclTensor* sparseValuesOut,
      uint64_t* workspaceSize, aclOpExecutor** executor)
"""

import ctypes

import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr, AclIntArray

import pool_key_indexer_api  # noqa: F401  注册 BaseApi(标杆/真值执行器)
from pool_key_indexer_api import prepare_pki_inputs, make_pa_key_non_contig

# E8M0 dtype 映射补充: atk acl_wrapper 的 TORCH_TO_ACLTYPE 未收录
# torch.float8_e8m0fnu(仅 e4m3/e5m2), 量化 mxFP8(quant_mode=1) descale 需要
# ACL_FLOAT8_E8M0 —— 增量补充, 不影响既有映射与场景。
# 兼容性守卫: 部分 atk 版本(如 26.4.30)的 AclDataType 未收录 ACL_FLOAT8_E8M0,
# 此时跳过补充(该版本不支持 mxFP8 descale, 非量化场景不受影响),
# 避免模块级 AttributeError 导致整个适配器注册失败。
from atk.tasks.backends.lib_interface import acl_wrapper as _acl_wrapper

_ACL_FLOAT8_E8M0 = getattr(
    getattr(_acl_wrapper, "AclDataType", None), "ACL_FLOAT8_E8M0", None
)
if (
    _ACL_FLOAT8_E8M0 is not None
    and hasattr(_acl_wrapper, "TORCH_TO_ACLTYPE")
    and "torch.float8_e8m0fnu" not in _acl_wrapper.TORCH_TO_ACLTYPE
):
    _acl_wrapper.TORCH_TO_ACLTYPE["torch.float8_e8m0fnu"] = _ACL_FLOAT8_E8M0

# ValueDepend 可选输入的空指针类型(与 aclIntArray* 签名校验匹配)
IntAryPtr = ctypes.POINTER(AclIntArray)

_TORCH_OUTPUT_DTYPES = (
    "torch.int32",
    "torch.float32",
    "int32",
    "fp32",
    "float32",
)


def _parse_torch_dtype(raw, default):
    mapping = {
        "torch.int32": torch.int32,
        "torch.float32": torch.float32,
        "int32": torch.int32,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    return mapping.get(str(raw), default)


def _recover_device(device_id):
    """aicore 异常后复位并重新初始化 NPU 设备(进程内后续用例不受污染)。"""
    import acl
    import os
    import time

    try:
        torch.npu.empty_cache()
    except Exception:
        pass

    try:
        acl.rt.reset_device(device_id)
        time.sleep(1)
        acl.rt.set_device(device_id)
    except Exception:
        pass

    try:
        torch.npu.set_device(device_id)
        torch.npu.empty_cache()
        _ = torch.zeros(1, device=f"npu:{device_id}")
        torch.npu.synchronize()
    except Exception:
        os._exit(1)


@register("aclnn_pool_key_indexer")
class PKIAclnnApi(AclnnBaseApi):
    def __init__(self, task_result, backend):
        super().__init__(task_result, backend)
        self.run_modes = []

    @classmethod
    def get_cpp_func_signature_type(cls):
        return (
            "aclnnStatus aclnnPoolKeyIndexerGetWorkspaceSize("
            "const aclTensor* query, "
            "const aclTensor* poolKey, "
            "const aclTensor* weights, "
            "const aclIntArray* poolTailK, "
            "const aclIntArray* actualSeqLengthsQueryOptional, "
            "const aclIntArray* actualSeqLengthsKeyOptional, "
            "const aclTensor* blockTableOptional, "
            "const aclTensor* qDescaleOptional, "
            "const aclTensor* kDescaleOptional, "
            "int64_t topk, "
            "int64_t poolSize, "
            "char* layoutQueryOptional, "
            "char* layoutKeyOptional, "
            "int64_t maskMode, "
            "int64_t quantMode, "
            "bool returnValue, "
            "int64_t keyStride0, "
            "int64_t kDescaleStride0, "
            "const aclTensor* sparseIndicesOut, "
            "const aclTensor* sparseValuesOut, "
            "uint64_t* workspaceSize, "
            "aclOpExecutor** executor)"
        )

    def _get_device_id(self):
        device_id = 0
        if hasattr(self.backend, "device_id") and self.backend.device_id is not None:
            device_id = self.backend.device_id
        elif getattr(self.task_result, "device_id", None) is not None:
            device_id = self.task_result.device_id
        return int(device_id)

    def init_by_input_data(self, input_data: InputDataset):
        self._saved_device_id = self._get_device_id()
        device_id = self._saved_device_id
        dev = f"npu:{device_id}"

        try:
            torch.npu.set_device(device_id)
            torch.npu.synchronize()
        except RuntimeError:
            _recover_device(device_id)

        # 三节点一致的输入规整(TND 前缀和 / PA block_table / 场景标志);
        # 非连续 key 由本适配器在目标设备构造(输入在 npu:0, 跨设备搬运会
        # 拍平视图, prepare_pki_inputs 传 build_non_contig=False)
        prep = prepare_pki_inputs(input_data, build_non_contig=False)

        def _to_dev(tensor):
            if tensor is None or not hasattr(tensor, "to"):
                return tensor
            if tensor.device.type == "npu" and tensor.device.index == device_id:
                return tensor
            return tensor.to(dev)

        # PA key 0 轴非连续: 搬到目标设备后构造视图, keyStride0 取实际 stride
        key = _to_dev(prep["pool_key"])
        if (
            prep["key_non_contig"]
            and prep["layout_k"] == "PA_BBND"
            and key.is_contiguous()
        ):
            key = make_pa_key_non_contig(key)
        key_stride0 = -1
        if not key.is_contiguous():
            key_stride0 = int(key.stride(0))  # 元素单位, 与 torch_extension csrc 一致

        def _tensor(tensor):
            # aclTensor 输入(已搬到目标设备, 避免 create_acl_tensor 静默跨设备)
            result = self.backend.convert_input_data(_to_dev(tensor), name="tensor")
            if isinstance(result, list):
                result = result[0]
            return result

        def _int_array(tensor):
            # ValueDepend 输入: aclnn 期望 aclIntArray*(Host 侧 int64 值)
            if tensor is None:
                return IntAryPtr()  # 空 aclIntArray 指针(可选输入)
            values = [
                ctypes.c_int64(int(v))
                for v in tensor.detach().cpu().reshape(-1).tolist()
            ]
            return self.backend.convert_input_data(values)[0]

        input_args = []
        # 9 个输入: query/poolKey/weights/blockTable 为 aclTensor,
        # poolTailK/actualSeqQ/actualSeqK 为 aclIntArray(Host 侧取值)
        input_args.append(_tensor(prep["query"]))
        input_args.append(_tensor(key))
        input_args.append(_tensor(prep["weights"]))

        input_args.append(_int_array(prep["pool_tail_k"]))
        input_args.append(_int_array(prep["actual_seq_q"]))
        input_args.append(_int_array(prep["actual_seq_k"]))

        if prep["block_table"] is not None:
            input_args.append(_tensor(prep["block_table"]))
        else:
            input_args.append(TensorPtr())

        # qDescale / kDescale: 量化场景传实际 tensor, 非量化为空指针
        #   mode=0: FLOAT scale 直传;
        #   mode=1: uint8 raw 视图为 float8_e8m0fnu(E8M0, 1B 元素, 视图不改布局)
        if prep["q_descale"] is not None:
            qd = _to_dev(prep["q_descale"])
            if prep["quant_mode"] == 1:
                qd = qd.view(torch.float8_e8m0fnu)
            input_args.append(_tensor(qd))
        else:
            input_args.append(TensorPtr())  # qDescale
        if prep["k_descale"] is not None:
            kd = _to_dev(prep["k_descale"])
            if prep["quant_mode"] == 1:
                kd = kd.view(torch.float8_e8m0fnu)
            input_args.append(_tensor(kd))
        else:
            input_args.append(TensorPtr())  # kDescale

        # 9 个 attr 入参(与 op_def 声明顺序一致):
        # topk, poolSize, layoutQuery, layoutKey, maskMode, quantMode,
        # returnValue, keyStride0, kDescaleStride0
        # keyStride0: 非连续 key 传实际 stride(0)(元素单位); 连续传 -1
        # (tiling 回退 shape 推导); kDescaleStride0 非量化场景恒 -1
        input_args.append(ctypes.c_int64(prep["topk"]))
        input_args.append(ctypes.c_int64(prep["pool_size"]))
        input_args.append(ctypes.c_char_p(prep["layout_q"].encode("utf-8")))
        input_args.append(ctypes.c_char_p(prep["layout_k"].encode("utf-8")))
        input_args.append(ctypes.c_int64(prep["mask_mode"]))
        input_args.append(ctypes.c_int64(prep["quant_mode"]))
        input_args.append(ctypes.c_bool(prep["return_value"]))
        input_args.append(ctypes.c_int64(key_stride0))
        input_args.append(ctypes.c_int64(-1))  # kDescaleStride0

        # 输出 tensor: indices INT32, values FLOAT(规格来自标杆节点 output_info_list)
        self._saved_output_structs = []
        output_packages = []
        for index in range(len(self.task_result.output_info_list)):
            out = self.task_result.output_info_list[index]
            out_shape = (
                list(out.shape)
                if hasattr(out, "shape")
                else list(out.get("shape", [1]))
            )
            if index == 0:
                torch_dtype = torch.int32  # sparseIndicesOut
            elif index == 1:
                torch_dtype = torch.float32  # sparseValuesOut (固定 FLOAT)
            else:
                out_dtype = (
                    out.dtype
                    if hasattr(out, "dtype")
                    else out.get("dtype", "torch.float32")
                )
                torch_dtype = _parse_torch_dtype(out_dtype, torch.float32)

            pytensor = torch.empty(out_shape, dtype=torch_dtype, device=dev)
            acl_struct = self.backend.torch_tensor_to_acl(pytensor)
            self._saved_output_structs.append(acl_struct)
            output_packages.append(acl_struct)

        input_args.extend(output_packages)
        return input_args, output_packages

    def __call__(self):
        device_id = getattr(self, "_saved_device_id", 0)
        try:
            torch.npu.set_device(device_id)
            super().__call__()
            torch.npu.synchronize()
        except RuntimeError as exc:
            msg = str(exc)
            if (
                "507015" in msg
                or "EZ9999" in msg
                or "EE9999" in msg
                or "aicore" in msg.lower()
            ):
                _recover_device(device_id)
            raise

    def after_call(self, output_packages):
        # 直接返回 NPU 侧输出 tensor(算子原始结果, 不做归一化),
        # 后续 save_output_data 阶段统一 .cpu() 落盘
        outputs = []
        for output_pack in output_packages:
            if hasattr(output_pack, "pytensor"):
                outputs.append(output_pack.pytensor)
        return outputs
