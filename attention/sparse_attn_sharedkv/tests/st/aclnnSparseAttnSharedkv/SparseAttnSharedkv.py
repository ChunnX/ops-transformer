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
import custom_ops as ops  # noqa: F401 加载 custom_transformer 算子库

from atk.configs.dataset_config import InputDataset
from atk.configs.results_config import OutputData, TaskResult
from atk.tasks.api_execute import register
from atk.tasks.api_execute.base_api import BaseApi
from atk.tasks.api_execute.aclnn_base_api import AclnnBaseApi

_PYTEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../pytest"))
if _PYTEST_DIR not in sys.path:
    sys.path.insert(0, _PYTEST_DIR)

import sparse_attn_sharedkv_golden as golden  # noqa: E402

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

_TENSOR_KEYS = (
    "q", "ori_kv", "cmp_kv", "ori_sparse_indices", "cmp_sparse_indices",
    "ori_block_table", "cmp_block_table", "cu_seqlens_q", "cu_seqlens_ori_kv",
    "cu_seqlens_cmp_kv", "seqused_q", "seqused_kv", "sinks", "metadata",
)

# aclnnSparseAttnSharedkvGetWorkspaceSize 入参下标（0-based），须为 nullptr（与 C++ 示例 / process.py 一致）
# 3: ori_sparse_indices（tiling 强制 nullptr）；8/9: cu_seqlens_ori/cmp_kv；10: seqused_q（decode 未开 flag）
_ACLNN_NULLPTR_INPUT_IDX = (3, 8, 9, 10)


def _template_mode_from_cmp_ratio(cmp_ratio):
    cmp_ratio = int(cmp_ratio)
    if cmp_ratio == 1:
        return "SWA"
    if cmp_ratio == 128:
        return "CFA"
    if cmp_ratio == 4:
        return "SCFA"
    return "SWA"


def _infer_phase(kwargs):
    """与 paramset 一致：seqused_kv>=8193 为 decode。"""
    sk = kwargs.get("seqused_kv")
    if isinstance(sk, torch.Tensor) and sk.numel() > 0:
        val = int(sk.reshape(-1)[0].item())
    elif isinstance(sk, (list, tuple)) and len(sk) > 0:
        val = int(sk[0])
    else:
        q = kwargs.get("q")
        if isinstance(q, torch.Tensor):
            return "decode" if q.shape[0] <= 1 else "prefill"
        return "prefill"
    return "decode" if val >= 8193 else "prefill"


def _phase_shapes(phase):
    if phase == "decode":
        return 1, 1, [0, 1], [8193]
    return 8192, 8192, [0, 8192], [8192]


def _paramset_scfa_decode_test_data():
    """sparse_attn_sharedkv_paramset.py 第一个用例 scfa_decode。"""
    q_type = torch.bfloat16
    return (
        "TND", "PA_ND", q_type, q_type, q_type,
        1, 1, 1, N1, N2, D, SCFA_TOPK,
        ORI_BLOCK_NUM, CMP_BLOCK_NUM, BLOCK_SIZE1, BLOCK_SIZE2,
        [0, 1], [8193],
        SOFTMAX_SCALE, 4, ORI_MASK_MODE, CMP_MASK_MODE, ORI_WIN_LEFT, ORI_WIN_RIGHT,
        "scfa_decode_atk", 8193, [-10, 10], [-10, 10], [-10, 10], None,
    )


def _build_test_data(kwargs, skip_forward=False, force_scfa_decode=False):
    if force_scfa_decode or os.getenv("ATK_SAS_CASE", "") == "scfa_decode":
        test_data = _paramset_scfa_decode_test_data()
        return golden.gen_data(test_data, skip_forward=skip_forward)

    cmp_ratio_attr = int(kwargs["cmp_ratio"])
    template_mode = _template_mode_from_cmp_ratio(cmp_ratio_attr)
    phase = _infer_phase(kwargs)
    layout_q = kwargs.get("layout_q", "TND")
    layout_kv = kwargs.get("layout_kv", "PA_ND")
    q = kwargs.get("q")
    q_type = q.dtype if isinstance(q, torch.Tensor) else torch.bfloat16

    b = 1
    t1, s1, cu_seqlens_q, seqused_kv = _phase_shapes(phase)
    k = None
    block_num2 = None
    block_size2 = None
    cmp_mask_mode = None
    cmp_ratio = None

    if template_mode == "SCFA":
        k = SCFA_TOPK
        cmp_ratio = 4
        block_num2 = CMP_BLOCK_NUM
        block_size2 = BLOCK_SIZE2
        cmp_mask_mode = CMP_MASK_MODE
    elif template_mode == "CFA":
        cmp_ratio = 128
        block_num2 = CMP_BLOCK_NUM
        block_size2 = BLOCK_SIZE2
        cmp_mask_mode = CMP_MASK_MODE

    testcase_name = "atk_case_" + str(int(time.time() * 1000000))
    test_data = (
        layout_q, layout_kv, q_type, q_type, q_type, b, s1, t1, N1, N2, D, k,
        ORI_BLOCK_NUM, block_num2, BLOCK_SIZE1, block_size2, cu_seqlens_q, seqused_kv,
        SOFTMAX_SCALE, cmp_ratio, ORI_MASK_MODE, cmp_mask_mode, ORI_WIN_LEFT, ORI_WIN_RIGHT,
        testcase_name, max(seqused_kv), [-10, 10], [-10, 10], [-10, 10], None,
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


def _generate_metadata(input_data, device="npu:0"):
    """与 tests/pytest/batch/sparse_attn_sharedkv_process.py 一致，走 custom 算子库。"""
    metadata_input = input_data["metadata_input"]
    tensor_input = input_data["input"]
    k = metadata_input["K"]
    cmp_ratio = metadata_input["cmp_ratio"]
    layout_q = tensor_input["layout_q"]
    layout_kv = tensor_input["layout_kv"]

    cu_seqlens_q = tensor_input["cu_seqlens_q"]
    seqused_kv = tensor_input["seqused_kv"]
    seqused_q = torch.tensor([]).npu()

    template_idx = 0
    if k is None:
        template_idx = 1 if cmp_ratio is not None else 0
    else:
        template_idx = 2

    metadata_kwargs = dict(
        num_heads_q=metadata_input["N1"],
        num_heads_kv=metadata_input["N2"],
        head_dim=metadata_input["D"],
        cu_seqlens_q=_to_npu_tensor(cu_seqlens_q),
        cu_seqlens_ori_kv=torch.tensor([]).npu(),
        cu_seqlens_cmp_kv=torch.tensor([]).npu(),
        seqused_q=seqused_q,
        seqused_kv=_to_npu_tensor(seqused_kv),
        batch_size=metadata_input["B"],
        max_seqlen_q=metadata_input["max_seqlen_q"],
        max_seqlen_kv=metadata_input["max_seqlen_kv"],
        ori_mask_mode=metadata_input["ori_mask_mode"],
        ori_win_left=metadata_input["ori_win_left"],
        ori_win_right=metadata_input["ori_win_right"],
        layout_q=layout_q,
        layout_kv=layout_kv,
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


def _empty_int32_tensor(device=None):
    t = torch.tensor([], dtype=torch.int32)
    if device is not None:
        t = t.to(device)
    return t


def _sync_case_to_kwargs(kwargs, input_data):
    inp = input_data["input"]
    kwargs["q"] = inp["q"]
    kwargs["ori_kv"] = inp["ori_kv"]
    kwargs["cmp_kv"] = inp["cmp_kv"]
    # process.py 主算子不传 ori_sparse_indices；保留 key=None 以维持 aclnn 入参顺序
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
    kwargs["softmax_scale"] = inp["softmax_scale"]
    kwargs["ori_mask_mode"] = inp["ori_mask_mode"]
    kwargs["cmp_mask_mode"] = inp["cmp_mask_mode"] if inp["cmp_mask_mode"] is not None else CMP_MASK_MODE
    kwargs["layout_q"] = inp["layout_q"]
    kwargs["layout_kv"] = inp["layout_kv"]
    meta = input_data.get("metadata_input") or {}
    if meta.get("cmp_ratio") is not None:
        kwargs["cmp_ratio"] = int(meta["cmp_ratio"])
    elif inp.get("cmp_ratio") is not None:
        kwargs["cmp_ratio"] = int(inp["cmp_ratio"])
    elif kwargs.get("cmp_ratio") is not None:
        kwargs["cmp_ratio"] = int(kwargs["cmp_ratio"])
    else:
        kwargs["cmp_ratio"] = 1
    kwargs["ori_win_left"] = inp["ori_win_left"]
    kwargs["ori_win_right"] = inp["ori_win_right"]
    kwargs["return_softmax_lse"] = False
    kwargs["ori_kv_stride"] = inp["ori_kv"].stride(0) if inp.get("ori_kv") is not None else 0
    kwargs["cmp_kv_stride"] = inp["cmp_kv"].stride(0) if inp.get("cmp_kv") is not None else 0


def _prepare_npu_kwargs(kwargs):
    for key in _TENSOR_KEYS:
        if key in ("ori_sparse_indices", "cu_seqlens_ori_kv", "cu_seqlens_cmp_kv", "seqused_q"):
            continue
        val = kwargs.get(key)
        if val is None:
            continue
        if isinstance(val, torch.Tensor):
            kwargs[key] = val.npu() if not val.is_npu else val


def _layout_to_lp_c_char(layout):
    buf = ctypes.create_string_buffer(str(layout).encode("utf-8"))
    return ctypes.cast(buf, ctypes.POINTER(ctypes.c_char))


def _atk_output_dtype_str(tensor):
    """OutputData.dtype：须为 ATK TORCH_TO_ACLTYPE 可识别的 torch.* 字符串。"""
    return str(tensor.dtype)


def _setup_output_info_list(task_result, q_tensor):
    """aclnn 需要 attn_out + softmax_lse 两个输出。"""
    q_shape = list(q_tensor.shape)
    out_dtype = _atk_output_dtype_str(q_tensor)
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


def _patch_aclnn_input_args(input_args, kwargs):
    """对齐 aclnnSparseAttnSharedkvGetWorkspaceSize 签名（标量类型 + PA_ND stride + layout）。"""
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
        input_args[22] = _layout_to_lp_c_char(kwargs["layout_q"])
    if len(input_args) > 23:
        input_args[23] = _layout_to_lp_c_char(kwargs["layout_kv"])
    if len(input_args) > 24:
        input_args[24] = ctypes.c_bool(bool(kwargs.get("return_softmax_lse", False)))
    return input_args


@register("ascend_function_sparse_attn_sharedkv")
class SparseAttnSharedkvGoldenApi(BaseApi):
    def __init__(self, task_result: TaskResult):
        super().__init__(task_result)
        self._case_input = None

    def init_by_input_data(self, input_data: InputDataset):
        force = os.getenv("ATK_SAS_CASE", "") == "scfa_decode"
        self._case_input = _build_test_data(input_data.kwargs, skip_forward=True, force_scfa_decode=force)
        _sync_case_to_kwargs(input_data.kwargs, self._case_input)
        return None

    def __call__(self, input_data: InputDataset, with_output: bool = False):
        if self._case_input is not None and self._case_input.get("cpu_output") is not None:
            case = self._case_input
        else:
            force = os.getenv("ATK_SAS_CASE", "") == "scfa_decode"
            case = _build_test_data(input_data.kwargs, skip_forward=False, force_scfa_decode=force)
        out = case["cpu_output"]
        q = case["input"]["q"]
        if isinstance(out, torch.Tensor) and isinstance(q, torch.Tensor) and out.dtype != q.dtype:
            out = out.to(q.dtype)
        return out


@register("AclnnBaseApi_aclnn_sparse_attn_sharedkv")
class SparseAttnSharedkvAclnnApi(AclnnBaseApi):
    def _apply_aclnn_nullptr_inputs(self, input_args):
        from atk.tasks.backends.lib_interface.acl_wrapper import AclTensor

        acl_tensor_ptr = ctypes.POINTER(AclTensor)
        null_tensor_ptr = ctypes.cast(ctypes.c_void_p(None), acl_tensor_ptr)
        for idx in _ACLNN_NULLPTR_INPUT_IDX:
            if idx < len(input_args):
                input_args[idx] = null_tensor_ptr

    def init_by_input_data(self, input_data: InputDataset):
        force = os.getenv("ATK_SAS_CASE", "") == "scfa_decode"
        case = _build_test_data(input_data.kwargs, skip_forward=True, force_scfa_decode=force)

        inp = case["input"]
        if inp["layout_kv"] == "PA_ND":
            inp["ori_kv"] = _create_tensor_with_stride_padding(inp["ori_kv"])
            if inp["cmp_kv"] is not None:
                inp["cmp_kv"] = _create_tensor_with_stride_padding(inp["cmp_kv"])

        metadata = _generate_metadata(case)
        inp["metadata"] = metadata
        _sync_case_to_kwargs(input_data.kwargs, case)
        input_data.kwargs["metadata"] = metadata

        if getattr(self, "device", None) == "pyaclnn":
            _prepare_npu_kwargs(input_data.kwargs)
            _setup_output_info_list(self.task_result, inp["q"])
            # 占位 None：保持与 aclnn 签名一致，执行前再转为 null aclTensor*
            input_data.kwargs["ori_sparse_indices"] = None
            input_data.kwargs["cu_seqlens_ori_kv"] = None
            input_data.kwargs["cu_seqlens_cmp_kv"] = None
            input_data.kwargs["seqused_q"] = None

        self._golden_output = None
        self._aclnn_kwargs = input_data.kwargs
        input_args, output_packages = super().init_by_input_data(input_data)
        self.input_args = input_args
        if getattr(self, "device", None) == "pyaclnn":
            self._apply_aclnn_nullptr_inputs(input_args)
            input_args = _patch_aclnn_input_args(input_args, self._aclnn_kwargs)
        return input_args, output_packages

    def __call__(self):
        if getattr(self, "device", None) == "pyaclnn" and getattr(self, "input_args", None):
            self._apply_aclnn_nullptr_inputs(self.input_args)
            if getattr(self, "_aclnn_kwargs", None):
                _patch_aclnn_input_args(self.input_args, self._aclnn_kwargs)
        return super().__call__()

    def after_call(self, output_packages):
        outputs = [self.acl_tensor_to_torch(item).cpu() for item in output_packages]
        out = outputs[0]
        q = (getattr(self, "_aclnn_kwargs", None) or {}).get("q")
        if q is not None and out.dtype != q.dtype:
            out = out.to(q.dtype)
        return out
