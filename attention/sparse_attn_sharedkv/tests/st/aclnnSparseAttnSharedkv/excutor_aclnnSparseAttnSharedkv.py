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

import ctypes
import os
import sys
import time

import torch
import custom_ops as ops

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import OutputData, TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

_PYTEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pytest"))
if _PYTEST_DIR not in sys.path:
    sys.path.insert(0, _PYTEST_DIR)

import sparse_attn_sharedkv_golden as golden

N1 = 64
N2 = 1
D = 512
BLOCK_SIZE1 = 128
BLOCK_SIZE2 = 128
ORI_BLOCK_NUM = 65
CMP_BLOCK_NUM = 17
SCFA_TOPK = 512
SOFTMAX_SCALE = 0.04419417
ORI_MASK_MODE = 4
CMP_MASK_MODE = 3
ORI_WIN_LEFT = 127
ORI_WIN_RIGHT = 0

# aclnn 入参须为 nullptr：ori_sparse_indices(3)、cu_seqlens_ori/cmp_kv(8,9)、seqused_q(10)
_ACLNN_NULLPTR_INPUT_IDX = (3, 8, 9, 10)

# 与 generate_sparse_attn_sharedkv.py / yaml inputs 下标一致
_Q_INDEX = 0
_CMP_SPARSE_INDICES_INDEX = 4
_CU_SEQLENS_Q_INDEX = 7
_SEQUSED_KV_INDEX = 11
_CMP_RATIO_INDEX = 15

_SHARED_GOLDEN_CASE = {"key": None, "case": None}


def _int_list_from_value(val, default):
    if isinstance(val, torch.Tensor) and val.numel() > 0:
        return [int(x) for x in val.reshape(-1).tolist()]
    if isinstance(val, (list, tuple)) and len(val) > 0:
        flat = []
        for x in val:
            if isinstance(x, (list, tuple)):
                flat.extend(int(v) for v in x)
            else:
                flat.append(int(x))
        return flat
    return default


def _range_values_as_int_list(val, default):
    if val is None:
        return default
    if isinstance(val, list):
        if len(val) == 0:
            return default
        if isinstance(val[0], (list, tuple)):
            return [int(x) for x in val[0]]
        return [int(x) for x in val]
    return [int(val)]


def _input_range_values(task_result, index):
    cc = getattr(task_result, "case_config", None) if task_result else None
    if cc is None or not getattr(cc, "inputs", None):
        return None
    return cc.inputs[index].range_values


def _input_shape(task_result, index):
    cc = getattr(task_result, "case_config", None) if task_result else None
    if cc is None or not getattr(cc, "inputs", None):
        return None
    return cc.inputs[index].shape


def _lengths_from_kwargs(kwargs):
    """ATK 已按 JSON 生成的 tensor 规格。"""
    q = kwargs.get("q")
    t1 = int(q.shape[0]) if isinstance(q, torch.Tensor) and q.dim() >= 1 else 1
    seqused_kv = _int_list_from_value(kwargs.get("seqused_kv"), None)
    cu_seqlens_q = _int_list_from_value(kwargs.get("cu_seqlens_q"), None)
    if seqused_kv is None:
        seqused_kv = [t1]
    if cu_seqlens_q is None:
        cu_seqlens_q = [0, t1]
    return t1, t1, cu_seqlens_q, seqused_kv


def _resolve_lengths(kwargs, task_result=None):
    """序列参数：JSON case_config 优先，缺项用 kwargs 张量补全。"""
    kw_t1, kw_s1, kw_cu, kw_sk = _lengths_from_kwargs(kwargs)
    t1_shape = _input_shape(task_result, _Q_INDEX)
    t1 = int(t1_shape[0]) if t1_shape else kw_t1

    seqused_kv = _range_values_as_int_list(
        _input_range_values(task_result, _SEQUSED_KV_INDEX), None)
    if seqused_kv is None:
        seqused_kv = kw_sk

    cu_seqlens_q = _range_values_as_int_list(
        _input_range_values(task_result, _CU_SEQLENS_Q_INDEX), None)
    if cu_seqlens_q is None:
        cu_seqlens_q = kw_cu

    return t1, t1, cu_seqlens_q, seqused_kv


def _resolve_cmp_ratio(kwargs, task_result=None):
    from_json = _range_values_as_int_list(
        _input_range_values(task_result, _CMP_RATIO_INDEX), None)
    if from_json is not None and len(from_json) > 0:
        return int(from_json[0])
    return int(kwargs.get("cmp_ratio", 1))


def _resolve_scfa_topk(kwargs, task_result=None):
    shape = _input_shape(task_result, _CMP_SPARSE_INDICES_INDEX)
    if shape and len(shape) >= 3 and int(shape[-1]) > 0:
        return int(shape[-1])
    idx = kwargs.get("cmp_sparse_indices")
    if isinstance(idx, torch.Tensor) and idx.dim() >= 3:
        return int(idx.shape[-1])
    return SCFA_TOPK


def _golden_cache_key(kwargs, task_result=None):
    cmp_ratio = _resolve_cmp_ratio(kwargs, task_result)
    t1, _, cu, seqused_kv = _resolve_lengths(kwargs, task_result)
    topk = _resolve_scfa_topk(kwargs, task_result) if cmp_ratio == 4 else 0
    return ("golden", cmp_ratio, t1, tuple(cu), tuple(seqused_kv), topk)


def _get_shared_golden_case(kwargs, task_result=None):
    """benchmark / cpu / npu 共用同一份 gen_data（固定 seed）。"""
    cache_key = _golden_cache_key(kwargs, task_result)
    if _SHARED_GOLDEN_CASE["key"] != cache_key or _SHARED_GOLDEN_CASE["case"] is None:
        import numpy as np

        np.random.seed(42)
        torch.manual_seed(42)
        _SHARED_GOLDEN_CASE["case"] = _build_test_data(
            kwargs, skip_forward=False, task_result=task_result)
        _SHARED_GOLDEN_CASE["key"] = cache_key
    return _SHARED_GOLDEN_CASE["case"]


def _template_mode_from_cmp_ratio(cmp_ratio):
    cmp_ratio = int(cmp_ratio)
    if cmp_ratio == 1:
        return "SWA"
    if cmp_ratio == 128:
        return "CFA"
    if cmp_ratio == 4:
        return "SCFA"
    return "SWA"


def _build_test_data(kwargs, skip_forward=False, task_result=None):
    """按当前执行的 JSON case 生成 golden；SWA/CFA/SCFA 仅由 cmp_ratio 决定模板字段。"""
    cmp_ratio_attr = _resolve_cmp_ratio(kwargs, task_result)
    template_mode = _template_mode_from_cmp_ratio(cmp_ratio_attr)
    layout_q = kwargs.get("layout_q", "TND")
    layout_kv = kwargs.get("layout_kv", "PA_ND")
    q = kwargs.get("q")
    q_type = q.dtype if isinstance(q, torch.Tensor) else torch.bfloat16

    b = 1
    t1, s1, cu_seqlens_q, seqused_kv = _resolve_lengths(kwargs, task_result)
    k = block_num2 = block_size2 = cmp_mask_mode = cmp_ratio = None

    if template_mode == "SCFA":
        k = _resolve_scfa_topk(kwargs, task_result)
        cmp_ratio = cmp_ratio_attr
        block_num2 = CMP_BLOCK_NUM
        block_size2 = BLOCK_SIZE2
        cmp_mask_mode = int(kwargs.get("cmp_mask_mode", CMP_MASK_MODE))
    elif template_mode == "CFA":
        cmp_ratio = cmp_ratio_attr
        block_num2 = CMP_BLOCK_NUM
        block_size2 = BLOCK_SIZE2
        cmp_mask_mode = int(kwargs.get("cmp_mask_mode", CMP_MASK_MODE))

    softmax_scale = float(kwargs.get("softmax_scale", SOFTMAX_SCALE))
    ori_mask_mode = int(kwargs.get("ori_mask_mode", ORI_MASK_MODE))
    ori_win_left = int(kwargs.get("ori_win_left", ORI_WIN_LEFT))
    ori_win_right = int(kwargs.get("ori_win_right", ORI_WIN_RIGHT))

    test_data = (
        layout_q, layout_kv, q_type, q_type, q_type, b, s1, t1, N1, N2, D, k,
        ORI_BLOCK_NUM, block_num2, BLOCK_SIZE1, block_size2, cu_seqlens_q, seqused_kv,
        softmax_scale, cmp_ratio, ori_mask_mode, cmp_mask_mode, ori_win_left, ori_win_right,
        "atk_case_" + str(int(time.time() * 1000000)), max(seqused_kv), [-10, 10], [-10, 10], [-10, 10], None,
    )
    return golden.gen_data(test_data, skip_forward=skip_forward)


def _create_tensor_with_stride_padding(src_tensor, pad_len=64):
    if src_tensor is None or src_tensor.dim() != 4:
        return src_tensor
    device = src_tensor.device
    dtype = src_tensor.dtype
    shape = list(src_tensor.shape)
    row_logical_size = shape[1] * shape[2] * shape[3]
    stride_0 = row_logical_size + pad_len
    new_strides = [stride_0, shape[2] * shape[3], shape[3], 1]
    physical_numel = stride_0 * shape[0]
    storage_tensor = torch.full((physical_numel,), float("nan"), dtype=dtype, device=device)
    raw_storage = storage_tensor.untyped_storage()
    target_tensor = torch.empty((0,), dtype=dtype, device=device)
    target_tensor.set_(raw_storage, 0, shape, new_strides)
    for i in range(shape[0]):
        target_tensor[i].copy_(src_tensor[i])
    return target_tensor


def _to_npu_tensor(t):
    if t is None:
        return torch.tensor([]).npu()
    if isinstance(t, torch.Tensor):
        return t.npu() if not t.is_npu else t
    return torch.tensor(t, dtype=torch.int32).npu()


def _generate_metadata(case, device="npu:0"):
    """与 tests/pytest/batch/sparse_attn_sharedkv_process.py 一致。"""
    metadata_input = case["metadata_input"]
    tensor_input = case["input"]
    k = metadata_input["K"]
    cmp_ratio = metadata_input["cmp_ratio"]

    template_idx = 2 if k is not None else (1 if cmp_ratio is not None else 0)
    metadata_kwargs = dict(
        num_heads_q=metadata_input["N1"],
        num_heads_kv=metadata_input["N2"],
        head_dim=metadata_input["D"],
        cu_seqlens_q=_to_npu_tensor(tensor_input["cu_seqlens_q"]),
        cu_seqlens_ori_kv=torch.tensor([]).npu(),
        cu_seqlens_cmp_kv=torch.tensor([]).npu(),
        seqused_q=torch.tensor([]).npu(),
        seqused_kv=_to_npu_tensor(tensor_input["seqused_kv"]),
        batch_size=metadata_input["B"],
        max_seqlen_q=metadata_input["max_seqlen_q"],
        max_seqlen_kv=metadata_input["max_seqlen_kv"],
        ori_mask_mode=metadata_input["ori_mask_mode"],
        ori_win_left=metadata_input["ori_win_left"],
        ori_win_right=metadata_input["ori_win_right"],
        layout_q=tensor_input["layout_q"],
        layout_kv=tensor_input["layout_kv"],
        has_ori_kv=tensor_input["ori_kv"] is not None,
        has_cmp_kv=tensor_input["cmp_kv"] is not None,
        device=device,
    )
    if template_idx >= 1:
        metadata_kwargs["cmp_ratio"] = cmp_ratio
        metadata_kwargs["cmp_mask_mode"] = metadata_input["cmp_mask_mode"]
    if template_idx == 2:
        metadata_kwargs["cmp_topk"] = k
    return torch.ops.custom.npu_sparse_attn_sharedkv_metadata(**metadata_kwargs)


def _sync_case_to_kwargs(kwargs, case, generate_metadata=False):
    """将 golden 输入写入 ATK kwargs（自定义执行方式：覆盖 dataset 随机值）。"""
    inp = case["input"]
    if inp["layout_kv"] == "PA_ND":
        inp["ori_kv"] = _create_tensor_with_stride_padding(inp["ori_kv"])
        if inp["cmp_kv"] is not None:
            inp["cmp_kv"] = _create_tensor_with_stride_padding(inp["cmp_kv"])

    metadata = _generate_metadata(case) if generate_metadata else kwargs.get("metadata")
    kwargs["q"] = inp["q"]
    kwargs["ori_kv"] = inp["ori_kv"]
    kwargs["cmp_kv"] = inp["cmp_kv"]
    kwargs["ori_sparse_indices"] = None
    kwargs["cmp_sparse_indices"] = inp["cmp_sparse_indices"]
    kwargs["ori_block_table"] = inp["ori_block_table"]
    kwargs["cmp_block_table"] = inp["cmp_block_table"]
    kwargs["cu_seqlens_q"] = inp["cu_seqlens_q"]
    kwargs["cu_seqlens_ori_kv"] = None
    kwargs["cu_seqlens_cmp_kv"] = None
    kwargs["seqused_q"] = None
    kwargs["seqused_kv"] = inp["seqused_kv"]
    kwargs["sinks"] = inp["sinks"]
    kwargs["metadata"] = metadata
    kwargs["softmax_scale"] = inp["softmax_scale"]
    kwargs["ori_mask_mode"] = inp["ori_mask_mode"]
    kwargs["cmp_mask_mode"] = inp["cmp_mask_mode"] if inp["cmp_mask_mode"] is not None else CMP_MASK_MODE
    kwargs["layout_q"] = inp["layout_q"]
    kwargs["layout_kv"] = inp["layout_kv"]
    meta = case.get("metadata_input") or {}
    if meta.get("cmp_ratio") is not None:
        kwargs["cmp_ratio"] = int(meta["cmp_ratio"])
    elif inp.get("cmp_ratio") is not None:
        kwargs["cmp_ratio"] = int(inp["cmp_ratio"])
    else:
        kwargs["cmp_ratio"] = int(kwargs.get("cmp_ratio", 1))
    kwargs["ori_win_left"] = inp["ori_win_left"]
    kwargs["ori_win_right"] = inp["ori_win_right"]
    kwargs["return_softmax_lse"] = False
    kwargs["ori_kv_stride"] = inp["ori_kv"].stride(0) if inp.get("ori_kv") is not None else 0
    kwargs["cmp_kv_stride"] = inp["cmp_kv"].stride(0) if inp.get("cmp_kv") is not None else 0


def _prepare_npu_kwargs(kwargs):
    for key, val in kwargs.items():
        if val is None or not isinstance(val, torch.Tensor):
            continue
        kwargs[key] = val.npu() if not val.is_npu else val


def _layout_to_c_char_p(layout):
    buf = ctypes.create_string_buffer(str(layout).encode("utf-8"))
    return ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))


def _fix_pyaclnn_input_args(input_args, kwargs):
    """pyaclnn：nullptr 入参 + -cp 签名（softmax_scale 须 c_double 等）。"""
    from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

    null_ptr = ctypes.cast(ctypes.c_void_p(None), ctypes.POINTER(AclTensor))
    for idx in _ACLNN_NULLPTR_INPUT_IDX:
        if idx < len(input_args):
            input_args[idx] = null_ptr
    if len(input_args) > 14:
        input_args[14] = ctypes.c_double(float(kwargs["softmax_scale"]))
    if len(input_args) > 15:
        input_args[15] = ctypes.c_int64(int(kwargs["cmp_ratio"]))
    if len(input_args) > 16:
        input_args[16] = ctypes.c_int64(int(kwargs["ori_mask_mode"]))
    if len(input_args) > 17:
        input_args[17] = ctypes.c_int64(int(kwargs["cmp_mask_mode"]))
    if len(input_args) > 18:
        input_args[18] = ctypes.c_int64(int(kwargs.get("ori_kv_stride", 0)))
    if len(input_args) > 19:
        input_args[19] = ctypes.c_int64(int(kwargs.get("cmp_kv_stride", 0)))
    if len(input_args) > 20:
        input_args[20] = ctypes.c_int64(int(kwargs["ori_win_left"]))
    if len(input_args) > 21:
        input_args[21] = ctypes.c_int64(int(kwargs["ori_win_right"]))
    if len(input_args) > 22:
        input_args[22] = _layout_to_c_char_p(kwargs["layout_q"])
    if len(input_args) > 23:
        input_args[23] = _layout_to_c_char_p(kwargs["layout_kv"])
    if len(input_args) > 24:
        input_args[24] = ctypes.c_bool(bool(kwargs.get("return_softmax_lse", False)))


def _setup_output_info_list(task_result, q_tensor):
    q_shape = list(q_tensor.shape)
    out_dtype = str(q_tensor.dtype)
    lse_shape = [q_shape[0], q_shape[1]] if len(q_shape) >= 2 else [1]
    lse_dtype = str(torch.float32)
    if not task_result.output_info_list:
        task_result.output_info_list = [
            OutputData(dtype=out_dtype, shape=q_shape),
            OutputData(dtype=lse_dtype, shape=lse_shape),
        ]
        return
    task_result.output_info_list[0].dtype = out_dtype
    task_result.output_info_list[0].shape = q_shape
    if len(task_result.output_info_list) < 2:
        task_result.output_info_list.append(OutputData(dtype=lse_dtype, shape=lse_shape))
    else:
        task_result.output_info_list[1].dtype = lse_dtype
        task_result.output_info_list[1].shape = lse_shape


def _npu_forward(kwargs):
    """与 sparse_attn_sharedkv_process.py SCFA 主算子调用一致。"""
    out, _lse = torch.ops.custom.npu_sparse_attn_sharedkv(
        kwargs["q"],
        ori_kv=kwargs["ori_kv"],
        cmp_kv=kwargs["cmp_kv"],
        cmp_sparse_indices=kwargs["cmp_sparse_indices"],
        ori_block_table=kwargs["ori_block_table"],
        cmp_block_table=kwargs["cmp_block_table"],
        cu_seqlens_q=kwargs["cu_seqlens_q"],
        seqused_kv=kwargs["seqused_kv"],
        sinks=kwargs["sinks"],
        metadata=kwargs["metadata"],
        softmax_scale=kwargs["softmax_scale"],
        cmp_ratio=int(kwargs["cmp_ratio"]),
        ori_mask_mode=int(kwargs["ori_mask_mode"]),
        cmp_mask_mode=int(kwargs["cmp_mask_mode"]),
        ori_win_left=int(kwargs["ori_win_left"]),
        ori_win_right=int(kwargs["ori_win_right"]),
        layout_q=kwargs["layout_q"],
        layout_kv=kwargs["layout_kv"],
    )
    torch.npu.synchronize()
    return out


@register("ascend_function_sparse_attn_sharedkv")
class SparseAttnSharedkvGoldenApi(BaseApi):
    """CPU 标杆：init 注入 golden 输入，__call__ 返回 cpu_output。"""

    def init_by_input_data(self, input_data: InputDataset):
        self._case = _get_shared_golden_case(input_data.kwargs, self.task_result)
        _sync_case_to_kwargs(input_data.kwargs, self._case, generate_metadata=False)

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        out = self._case["cpu_output"]
        q = self._case["input"]["q"]
        if isinstance(out, torch.Tensor) and isinstance(q, torch.Tensor) and out.dtype != q.dtype:
            out = out.to(q.dtype)
        return out


@register("AclnnBaseApi_aclnn_sparse_attn_sharedkv")
class SparseAttnSharedkvNpuApi(AclnnBaseApi):
    """NPU：init 用 golden 覆盖 kwargs（自定义输入）；执行走 custom 主算子（与 aclnn 等价）。"""

    def init_by_input_data(self, input_data: InputDataset):
        case = _get_shared_golden_case(input_data.kwargs, self.task_result)
        _sync_case_to_kwargs(input_data.kwargs, case, generate_metadata=True)
        self._kwargs = input_data.kwargs

        pyaclnn = self.device == "pyaclnn"
        if pyaclnn:
            # super 之前：自定义 kwargs 上 NPU，并配置输出 dtype/shape
            _prepare_npu_kwargs(input_data.kwargs)
            _setup_output_info_list(self.task_result, input_data.kwargs["q"])

        input_args, output_packages = super().init_by_input_data(input_data)

        if pyaclnn:
            _fix_pyaclnn_input_args(input_args, self._kwargs)
            self.input_args = input_args
        return input_args, output_packages

    def __call__(self):
        if self.device == "pyaclnn" and getattr(self, "input_args", None):
            _fix_pyaclnn_input_args(self.input_args, self._kwargs)
        return super().__call__()

    def after_call(self, output_packages):
        out = _npu_forward(self._kwargs).cpu()
        q = self._kwargs["q"]
        if out.dtype != q.dtype:
            out = out.to(q.dtype)
        return out
