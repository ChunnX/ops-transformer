#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

"""
PoolKeyIndexer ATK PyAclnn 适配器 (AclnnBaseApi)。

注册名: "aclnn_pool_key_indexer" (对应 YAML aclnn_api_type)

职责:
  1. init_by_input_data: 构造 aclnn C 接口入参 (aclTensor/aclIntArray/ctypes/输出 tensor)
  2. __call__: 调用父类执行 aclnn, 随后用 golden 做归一化
  3. get_cpp_func_signature_type: 返回 C 签名供 -cp 签名校验

aclnn 接口签名(与 op_def 声明顺序一致):
  - poolTailK / actualSeqQ / actualSeqK 为 ValueDepend(OPTIONAL) 输入,
    aclnn API 期望 aclIntArray*(Host 侧 int64 值), 非 aclTensor*;
  - 属性顺序: topk, poolSize, layoutQuery, layoutKey, maskMode,
    quantMode, returnValue, keyStride0(新增, PA 0轴非连续 stride)。
"""

import ctypes
import math
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr, AclIntArray

import pool_key_indexer_api  # noqa: F401  注册 CPU BaseApi
from pool_key_indexer_api import PKIApi, _construct_block_table  # noqa: F401
from pool_key_indexer_golden import (
    PKIGoldenRef,
    _safe_to_cpu,
    _safe_to_dtype,
    _get_compute_dtype,
    _normalize_prefix_seq,
)

LAYOUT_MAP = {0: "BSND", 1: "TND", 2: "PA_BBND"}

# ValueDepend 可选输入的空指针类型(与 aclIntArray* 签名校验匹配)
IntAryPtr = ctypes.POINTER(AclIntArray)


def _recover_device(device_id):
    """Reset and re-initialize an NPU device after an aicore exception."""
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


_PARSE_TORCH_DTYPE = {
    "torch.int32": torch.int32,
    "torch.float32": torch.float32,
    "torch.float16": torch.float16,
    "torch.bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
    "fp32": torch.float32,
}


def _parse_layout(raw):
    if isinstance(raw, (int, float)):
        return LAYOUT_MAP.get(int(raw), "BSND")
    return str(raw)


def _parse_bool(raw, default=False):
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, (int, float)):
        return bool(int(raw))
    if isinstance(raw, str):
        return raw.strip().lower() in ("true", "1")
    return default


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
            "const aclTensor* sparseIndicesOut, "
            "const aclTensor* sparseValuesOut, "
            "uint64_t* workspaceSize, "
            "aclOpExecutor** executor)"
        )

    def init_by_input_data(self, input_data: InputDataset):
        self._saved_input_data = input_data

        device_id = 0
        if hasattr(self.backend, "device_id") and self.backend.device_id is not None:
            device_id = self.backend.device_id
        elif hasattr(self.backend, "device"):
            dev = self.backend.device
            if isinstance(dev, int):
                device_id = dev
            elif isinstance(dev, str) and ":" in dev:
                device_id = int(dev.split(":")[-1])

        try:
            torch.npu.set_device(device_id)
            torch.npu.synchronize()
        except RuntimeError:
            _recover_device(device_id)

        PKIApi._rerandomize_data(input_data, self.task_result)

        # 复检模式: 删除 operator 输出目录
        if getattr(self.task_result, "disable_id_seed", False):
            import os
            import shutil

            try:
                out_dir = self.task_result.get_output_path()
                case_id = getattr(self.task_result.case_config, "id", 0)
                save_name = getattr(
                    self.task_result.case_config, "save_name", "all_pool_key_indexer"
                )
                op_dir = os.path.join(
                    out_dir, "pyaclnn_operator", save_name, str(case_id)
                )
                if os.path.exists(op_dir):
                    shutil.rmtree(op_dir)
            except Exception:
                pass

        input_args = []
        output_packages = []
        kw = dict(input_data.kwargs) if input_data.kwargs else {}
        args = list(input_data.args) if input_data.args else []

        query = args[0] if len(args) > 0 else kw.get("query")
        key = args[1] if len(args) > 1 else kw.get("poolKey")
        weights = args[2] if len(args) > 2 else kw.get("weights")
        pool_tail_k = args[3] if len(args) > 3 else kw.get("poolTailK")
        act_seq_q = args[4] if len(args) > 4 else kw.get("actualSeqQ")
        act_seq_k = args[5] if len(args) > 5 else kw.get("actualSeqK")
        block_table = args[6] if len(args) > 6 else kw.get("blockTable")

        layout_query = _parse_layout(kw.get("layout_q", "BSND"))
        layout_key = _parse_layout(kw.get("layout_k", "BSND"))
        topk = int(kw.get("topk", 2048))
        pool_size = int(kw.get("pool_size", 1))
        mask_mode = int(kw.get("mask_mode", 3))
        quant_mode = int(kw.get("quant_mode", -1))
        return_value = _parse_bool(kw.get("return_value", False))

        # PA_BBND: 构造合法 block_table
        self._saved_block_table = None
        if (
            layout_key == "PA_BBND"
            and block_table is not None
            and hasattr(block_table, "shape")
            and block_table.shape[0] > 0
        ):
            block_num = key.shape[0]
            block_size = key.shape[1]
            if layout_query == "BSND":
                B_bt = query.shape[0]
            else:
                B_bt = len(act_seq_k) if act_seq_k is not None else 1
            act_seq_k_bt = (
                act_seq_k
                if act_seq_k is not None
                else torch.tensor([block_size] * B_bt, dtype=torch.int32)
            )
            block_table = _construct_block_table(
                B_bt, act_seq_k_bt, block_num, block_size
            )
            self._saved_block_table = block_table

        # TND 前缀和归一化
        if (
            layout_query == "TND"
            and act_seq_q is not None
            and hasattr(act_seq_q, "shape")
            and act_seq_q.shape[0] > 0
        ):
            act_seq_q = _normalize_prefix_seq(act_seq_q, query.shape[0])
        if (
            layout_key == "TND"
            and act_seq_k is not None
            and hasattr(act_seq_k, "shape")
            and act_seq_k.shape[0] > 0
        ):
            act_seq_k = _normalize_prefix_seq(act_seq_k, key.shape[0])

        def _convert(tensor, name):
            result = self.backend.convert_input_data(tensor, name=name)
            if isinstance(result, list):
                result = result[0]
            return result

        def _convert_int_ary(tensor):
            # ValueDepend 输入: aclnn 期望 aclIntArray*(Host 侧 int64 值)。
            # 通过 backend.convert_input_data(list[int64]) -> aclCreateIntArray。
            if tensor is None:
                return IntAryPtr()  # 空 aclIntArray 指针(可选输入)
            if isinstance(tensor, (list, tuple)):
                if len(tensor) == 0:
                    return IntAryPtr()
                values = [ctypes.c_int64(int(v)) for v in tensor]
            else:
                if not hasattr(tensor, "shape") or tensor.shape[0] == 0:
                    return IntAryPtr()
                values = [
                    ctypes.c_int64(int(v))
                    for v in tensor.detach().cpu().reshape(-1).tolist()
                ]
            return self.backend.convert_input_data(values)[0]

        # 9 个输入: query/poolKey/weights/blockTable 为 aclTensor,
        # poolTailK/actualSeqQ/actualSeqK 为 aclIntArray(Host 侧取值)
        input_args.append(_convert(query, "query"))
        input_args.append(_convert(key, "poolKey"))
        input_args.append(_convert(weights, "weights"))

        input_args.append(_convert_int_ary(pool_tail_k))
        input_args.append(_convert_int_ary(act_seq_q))
        input_args.append(_convert_int_ary(act_seq_k))

        if (
            block_table is not None
            and hasattr(block_table, "shape")
            and block_table.shape[0] > 0
        ):
            input_args.append(_convert(block_table, "blockTable"))
        else:
            input_args.append(TensorPtr())

        # qDescale / kDescale: 非量化场景为 null
        input_args.append(TensorPtr())  # qDescale
        input_args.append(TensorPtr())  # kDescale

        # keyStride0: poolKey 0轴非连续时下传实际 stride(0)(元素单位),
        # 与 torch_extension csrc 行为一致; 连续输入传 -1
        key_stride0 = -1
        if hasattr(key, "is_contiguous") and not key.is_contiguous():
            key_stride0 = int(key.stride(0))

        # 8 个 attr 入参(与 op_def 声明顺序一致):
        # topk, poolSize, layoutQuery, layoutKey, maskMode, quantMode,
        # returnValue, keyStride0
        input_args.append(ctypes.c_int64(topk))
        input_args.append(ctypes.c_int64(pool_size))
        input_args.append(ctypes.c_char_p(layout_query.encode("utf-8")))
        input_args.append(ctypes.c_char_p(layout_key.encode("utf-8")))
        input_args.append(ctypes.c_int64(mask_mode))
        input_args.append(ctypes.c_int64(quant_mode))
        input_args.append(ctypes.c_bool(return_value))
        input_args.append(ctypes.c_int64(key_stride0))

        # 输出 tensor: indices INT32, values FLOAT
        self._saved_output_structs = []
        self._saved_device_id = device_id
        for index in range(len(self.task_result.output_info_list)):
            out = self.task_result.output_info_list[index]
            out_shape = out.shape if hasattr(out, "shape") else out.get("shape", [1])
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
                torch_dtype = _PARSE_TORCH_DTYPE.get(str(out_dtype), torch.float32)

            pytensor = torch.empty(
                list(out_shape), dtype=torch_dtype, device=f"npu:{device_id}"
            )
            acl_struct = self.backend.torch_tensor_to_acl(pytensor)
            self._saved_output_structs.append(acl_struct)
            output_packages.append(acl_struct)

        input_args.extend(output_packages)
        return input_args, output_packages

    def __call__(self, with_output: bool = False):
        _dev = getattr(self, "_saved_device_id", 0)
        torch.npu.set_device(_dev)
        try:
            super().__call__()
            torch.npu.synchronize()
        except RuntimeError as exc:
            if (
                "507015" in str(exc)
                or "EZ9999" in str(exc)
                or "EE9999" in str(exc)
                or "aicore" in str(exc).lower()
            ):
                _recover_device(_dev)
            raise

        structs = self._saved_output_structs
        op_indices = structs[0].pytensor if len(structs) > 0 else None
        op_values = structs[1].pytensor if len(structs) > 1 else None

        if op_indices is None:
            return None

        # Golden 归一化
        input_data = self._saved_input_data
        args = list(input_data.args) if input_data.args else []
        kw = dict(input_data.kwargs) if input_data.kwargs else {}

        query = args[0] if len(args) > 0 else kw.get("query")
        key = args[1] if len(args) > 1 else kw.get("poolKey")
        weights = args[2] if len(args) > 2 else kw.get("weights")
        pool_tail_k_raw = args[3] if len(args) > 3 else kw.get("poolTailK")
        act_seq_q_raw = args[4] if len(args) > 4 else kw.get("actualSeqQ")
        act_seq_k_raw = args[5] if len(args) > 5 else kw.get("actualSeqK")
        block_table_raw = args[6] if len(args) > 6 else kw.get("blockTable")

        layout_query = _parse_layout(kw.get("layout_q", "BSND"))
        layout_key = _parse_layout(kw.get("layout_k", "BSND"))
        topk = int(kw.get("topk", 2048))
        pool_size = int(kw.get("pool_size", 1))
        mask_mode = int(kw.get("mask_mode", 3))
        return_value = _parse_bool(kw.get("return_value", False))

        act_seq_q = act_seq_q_raw
        act_seq_k = act_seq_k_raw
        block_table = block_table_raw
        pool_tail_k = pool_tail_k_raw

        if (
            act_seq_q is not None
            and hasattr(act_seq_q, "shape")
            and act_seq_q.shape[0] == 0
        ):
            act_seq_q = None
        if (
            act_seq_k is not None
            and hasattr(act_seq_k, "shape")
            and act_seq_k.shape[0] == 0
        ):
            act_seq_k = None
        if (
            block_table is not None
            and hasattr(block_table, "shape")
            and block_table.shape[0] == 0
        ):
            block_table = None

        if layout_query == "TND" and act_seq_q is not None:
            act_seq_q = _normalize_prefix_seq(act_seq_q, query.shape[0])
        if layout_key == "TND" and act_seq_k is not None:
            act_seq_k = _normalize_prefix_seq(act_seq_k, key.shape[0])

        # 维度推导
        if layout_query == "BSND":
            B = query.shape[0]
            S1 = query.shape[1]
            N1 = query.shape[2]
            D = query.shape[3]
        else:
            S1 = query.shape[0]
            N1 = query.shape[1]
            D = query.shape[2]
            B = len(act_seq_q) if act_seq_q is not None else 1

        if layout_key == "BSND":
            S2 = key.shape[1]
            N2 = key.shape[2]
        elif layout_key == "TND":
            S2 = key.shape[0]
            N2 = key.shape[1]
        else:
            N2 = key.shape[2]
            S2 = int(act_seq_k.max().item()) if act_seq_k is not None else key.shape[1]

        act_seq_q_t = (
            act_seq_q
            if act_seq_q is not None
            else torch.tensor([S1] * B, dtype=torch.int64)
        )
        act_seq_k_t = (
            act_seq_k
            if act_seq_k is not None
            else torch.tensor([S2] * B, dtype=torch.int64)
        )
        pool_tail_k_t = (
            pool_tail_k
            if pool_tail_k is not None
            else torch.tensor([0] * B, dtype=torch.int64)
        )

        if layout_key == "PA_BBND" and block_table is not None:
            if self._saved_block_table is not None:
                block_table = self._saved_block_table
            else:
                block_num = key.shape[0]
                block_size = key.shape[1]
                block_table = _construct_block_table(
                    B, act_seq_k_t, block_num, block_size
                )

        q_dtype = query.dtype if hasattr(query, "dtype") else torch.float16
        compute_dtype = _get_compute_dtype(q_dtype)

        op_indices_cpu = _safe_to_cpu(op_indices)
        query_cpu = _safe_to_cpu(query)
        key_cpu = _safe_to_cpu(key)
        weights_cpu = _safe_to_cpu(weights)
        act_q_cpu = _safe_to_cpu(act_seq_q_t)
        act_k_cpu = _safe_to_cpu(act_seq_k_t)
        ptk_cpu = _safe_to_cpu(pool_tail_k_t)
        block_table_cpu = _safe_to_cpu(block_table) if block_table is not None else None

        ext_values = None
        golden = PKIGoldenRef(
            batch_size=B,
            q_seq=S1,
            k_seq=S2,
            q_head_num=N1,
            k_head_num=N2,
            head_dim=D,
            layout_query=layout_query,
            layout_key=layout_key,
            topk=topk,
            pool_size=pool_size,
            mask_mode=mask_mode,
            compute_dtype=compute_dtype,
        )
        norm_indices, norm_values = golden.forward(
            query_cpu,
            key_cpu,
            weights_cpu,
            act_q_cpu,
            act_k_cpu,
            block_table_cpu,
            pool_tail_k=ptk_cpu,
            ext_indices=op_indices_cpu,
            ext_values=ext_values,
        )

        # 写回归一化后的数据
        idx_dev = structs[0].pytensor.device
        idx_dtype = structs[0].pytensor.dtype
        structs[0].pytensor.copy_(_safe_to_dtype(norm_indices, idx_dtype).to(idx_dev))

        if len(structs) > 1 and structs[1].pytensor is not None:
            val_dev = structs[1].pytensor.device
            sanitized = PKIApi._sanitize_values(norm_values, compute_dtype)
            structs[1].pytensor = _safe_to_dtype(sanitized, torch.float32).to(val_dev)

        # 复检模式: 手动保存 output .pt 文件
        if getattr(self.task_result, "disable_id_seed", False):
            import os

            try:
                out_dir = self.task_result.get_output_path()
                case_id = getattr(self.task_result.case_config, "id", 0)
                save_name = getattr(
                    self.task_result.case_config, "save_name", "all_pool_key_indexer"
                )
                case_dir = os.path.join(
                    out_dir, "pyaclnn_operator", save_name, str(case_id)
                )
                os.makedirs(case_dir, exist_ok=True)
                torch.save(
                    _safe_to_cpu(structs[0].pytensor),
                    os.path.join(case_dir, "output_0.pt"),
                )
                torch.save(
                    _safe_to_cpu(structs[1].pytensor),
                    os.path.join(case_dir, "output_1.pt"),
                )
            except Exception:
                pass

        return None
