#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PoolKeyIndexer ATK PyAclnn 适配器 (AclnnBaseApi)。

注册名: "aclnn_pool_key_indexer" (对应 YAML aclnn_api_type)

职责:
  1. init_by_input_data: 构造 aclnn C 接口入参 (aclTensor/ctypes/输出 tensor)
  2. __call__: 调用父类执行 aclnn, 随后用 golden 做归一化
  3. get_cpp_func_signature_type: 返回 C 签名供 -cp 签名校验
"""

import ctypes
import math
import torch

from atk.configs.dataset_config import InputDataset
from atk.tasks.api_execute import register
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi
from atk.tasks.backends.lib_interface.acl_wrapper import TensorPtr

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
            "const aclTensor* poolTailK, "
            "const aclTensor* actualSeqLengthsQueryOptional, "
            "const aclTensor* actualSeqLengthsKeyOptional, "
            "const aclTensor* blockTableOptional, "
            "const aclTensor* qDescaleOptional, "
            "const aclTensor* kDescaleOptional, "
            "char* layoutQueryOptional, "
            "char* layoutKeyOptional, "
            "int64_t topk, "
            "int64_t poolSize, "
            "int64_t maskMode, "
            "int64_t quantMode, "
            "bool returnValue, "
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
            pass

        PKIApi._rerandomize_data(input_data, self.task_result) if hasattr(PKIApi, '_rerandomize_data') else None

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
        if layout_key == "PA_BBND" and block_table is not None and \
                hasattr(block_table, "shape") and block_table.shape[0] > 0:
            block_num = key.shape[0]
            block_size = key.shape[1]
            if layout_query == "BSND":
                B_bt = query.shape[0]
            else:
                B_bt = len(act_seq_k) if act_seq_k is not None else 1
            act_seq_k_bt = act_seq_k if act_seq_k is not None else \
                torch.tensor([block_size] * B_bt, dtype=torch.int32)
            block_table = _construct_block_table(B_bt, act_seq_k_bt, block_num, block_size)
            self._saved_block_table = block_table

        # TND 前缀和归一化
        if layout_query == "TND" and act_seq_q is not None and \
                hasattr(act_seq_q, "shape") and act_seq_q.shape[0] > 0:
            act_seq_q = _normalize_prefix_seq(act_seq_q, query.shape[0])
        if layout_key == "TND" and act_seq_k is not None and \
                hasattr(act_seq_k, "shape") and act_seq_k.shape[0] > 0:
            act_seq_k = _normalize_prefix_seq(act_seq_k, key.shape[0])

        def _convert(tensor, name):
            result = self.backend.convert_input_data(tensor, name=name)
            if isinstance(result, list):
                result = result[0]
            return result

        # 9 个 tensor 入参
        input_args.append(_convert(query, "query"))
        input_args.append(_convert(key, "poolKey"))
        input_args.append(_convert(weights, "weights"))

        if pool_tail_k is not None and hasattr(pool_tail_k, "shape") and pool_tail_k.shape[0] > 0:
            input_args.append(_convert(pool_tail_k, "poolTailK"))
        else:
            input_args.append(TensorPtr())

        if act_seq_q is not None and hasattr(act_seq_q, "shape") and act_seq_q.shape[0] > 0:
            input_args.append(_convert(act_seq_q, "actualSeqQ"))
        else:
            input_args.append(TensorPtr())

        if act_seq_k is not None and hasattr(act_seq_k, "shape") and act_seq_k.shape[0] > 0:
            input_args.append(_convert(act_seq_k, "actualSeqK"))
        else:
            input_args.append(TensorPtr())

        if block_table is not None and hasattr(block_table, "shape") and block_table.shape[0] > 0:
            input_args.append(_convert(block_table, "blockTable"))
        else:
            input_args.append(TensorPtr())

        # qDescale / kDescale: 非量化场景为 null
        input_args.append(TensorPtr())  # qDescale
        input_args.append(TensorPtr())  # kDescale

        # 7 个 attr 入参
        input_args.append(ctypes.c_char_p(layout_query.encode("utf-8")))
        input_args.append(ctypes.c_char_p(layout_key.encode("utf-8")))
        input_args.append(ctypes.c_int64(topk))
        input_args.append(ctypes.c_int64(pool_size))
        input_args.append(ctypes.c_int64(mask_mode))
        input_args.append(ctypes.c_int64(quant_mode))
        input_args.append(ctypes.c_bool(return_value))

        # 输出 tensor: indices INT32, values FLOAT
        self._saved_output_structs = []
        self._saved_device_id = device_id
        for index in range(len(self.task_result.output_info_list)):
            out = self.task_result.output_info_list[index]
            out_shape = out.shape if hasattr(out, "shape") else out.get("shape", [1])
            if index == 0:
                torch_dtype = torch.int32   # sparseIndicesOut
            elif index == 1:
                torch_dtype = torch.float32  # sparseValuesOut (固定 FLOAT)
            else:
                out_dtype = out.dtype if hasattr(out, "dtype") else out.get("dtype", "torch.float32")
                torch_dtype = _PARSE_TORCH_DTYPE.get(str(out_dtype), torch.float32)

            pytensor = torch.empty(list(out_shape), dtype=torch_dtype, device=f"npu:{device_id}")
            acl_struct = self.backend.torch_tensor_to_acl(pytensor)
            self._saved_output_structs.append(acl_struct)
            output_packages.append(acl_struct)

        input_args.extend(output_packages)
        return input_args, output_packages

    def __call__(self, with_output: bool = False):
        _dev = getattr(self, '_saved_device_id', 0)
        torch.npu.set_device(_dev)
        super().__call__()
        torch.npu.synchronize()

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

        if act_seq_q is not None and hasattr(act_seq_q, "shape") and act_seq_q.shape[0] == 0:
            act_seq_q = None
        if act_seq_k is not None and hasattr(act_seq_k, "shape") and act_seq_k.shape[0] == 0:
            act_seq_k = None
        if block_table is not None and hasattr(block_table, "shape") and block_table.shape[0] == 0:
            block_table = None

        if layout_query == "TND" and act_seq_q is not None:
            act_seq_q = _normalize_prefix_seq(act_seq_q, query.shape[0])
        if layout_key == "TND" and act_seq_k is not None:
            act_seq_k = _normalize_prefix_seq(act_seq_k, key.shape[0])

        # 维度推导
        if layout_query == "BSND":
            B = query.shape[0]; S1 = query.shape[1]; N1 = query.shape[2]; D = query.shape[3]
        else:
            S1 = query.shape[0]; N1 = query.shape[1]; D = query.shape[2]
            B = len(act_seq_q) if act_seq_q is not None else 1

        if layout_key == "BSND":
            S2 = key.shape[1]; N2 = key.shape[2]
        elif layout_key == "TND":
            S2 = key.shape[0]; N2 = key.shape[1]
        else:
            N2 = key.shape[2]
            S2 = int(act_seq_k.max().item()) if act_seq_k is not None else key.shape[1]

        act_seq_q_t = act_seq_q if act_seq_q is not None else torch.tensor([S1] * B, dtype=torch.int32)
        act_seq_k_t = act_seq_k if act_seq_k is not None else torch.tensor([S2] * B, dtype=torch.int32)
        pool_tail_k_t = pool_tail_k if pool_tail_k is not None else torch.tensor([0] * B, dtype=torch.int32)

        if layout_key == "PA_BBND" and block_table is not None:
            if self._saved_block_table is not None:
                block_table = self._saved_block_table
            else:
                block_num = key.shape[0]
                block_size = key.shape[1]
                block_table = _construct_block_table(B, act_seq_k_t, block_num, block_size)

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
            batch_size=B, q_seq=S1, k_seq=S2,
            q_head_num=N1, k_head_num=N2, head_dim=D,
            layout_query=layout_query, layout_key=layout_key,
            topk=topk, pool_size=pool_size,
            mask_mode=mask_mode, compute_dtype=compute_dtype,
        )
        norm_indices, norm_values = golden.forward(
            query_cpu, key_cpu, weights_cpu, act_q_cpu, act_k_cpu,
            block_table_cpu, pool_tail_k=ptk_cpu,
            ext_indices=op_indices_cpu, ext_values=ext_values,
        )

        # 写回归一化后的数据
        idx_dev = structs[0].pytensor.device
        idx_dtype = structs[0].pytensor.dtype
        structs[0].pytensor.copy_(_safe_to_dtype(norm_indices, idx_dtype).to(idx_dev))

        if len(structs) > 1 and structs[1].pytensor is not None:
            val_dev = structs[1].pytensor.device
            sanitized = PKIApi._sanitize_values(norm_values, compute_dtype)
            structs[1].pytensor = _safe_to_dtype(sanitized, torch.float32).to(val_dev)

        return None
