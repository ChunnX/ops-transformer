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

import torch
import torch_npu
import cann_ops_transformer
import os
import numpy as np
import time
import math
import random
from utils import get_torch_dtype


def _load_inputs_to_npu(input_dict):
    def to_npu(value):
        if isinstance(value, torch.Tensor):
            return value.npu()
        return value

    return {key: to_npu(value) for key, value in input_dict.items()}


def _get_kv_torch_dtype(kv_dtype_str):
    if kv_dtype_str == "hifloat8":
        return torch_npu.hifloat8
    return None


def call_npu_eager(torch_tensor_dict, params):
    return_softmax_lse = params.get("return_softmax_lse", False)
    npu_attn_out, npu_softmax_max, npu_softmax_sum = (
        torch.ops.cann_ops_transformer.kv_quant_sparse_flash_attention(
            query=torch_tensor_dict["query_cache"],
            key=torch_tensor_dict["key_cache"],
            value=torch_tensor_dict["value_cache"],
            sparse_indices=torch_tensor_dict["sparse_indices"],
            key_dequant_scale=None,
            value_dequant_scale=None,
            block_table=torch_tensor_dict["block_table"]
            if params["layout_kv"] == "PA_BSND"
            else None,
            actual_seq_lengths_query=torch.tensor(
                params["actualseqlengths"], dtype=torch.int32
            ).to("npu"),
            actual_seq_lengths_kv=torch.tensor(
                params["actualseqlengthskv"], dtype=torch.int32
            ).to("npu"),
            sinks=torch_tensor_dict.get("sinks", None),
            scale_value=params.get("scalevalue", 1),
            key_quant_mode=params.get("key_quant_mode", 2),
            value_quant_mode=params.get("value_quant_mode", 2),
            sparse_block_size=params.get("sparse_blocksize", 0),
            layout_query=params.get("layout_query", "BSND"),
            layout_kv=params.get("layout_kv", "PA_BSND"),
            sparse_mode=params.get("sparsemode", 0),
            attention_mode=params.get("attention_mode", 2),
            quant_scale_repo_mode=params.get("quant_scale_repo_mode", 1),
            tile_size=params.get("tile_size", 128),
            rope_head_dim=params.get("rope_head_dim", 64),
            pre_tokens=params.get("pre_tokens", (1 << 63) - 1),
            next_tokens=params.get("next_tokens", (1 << 63) - 1),
            return_softmax_lse=return_softmax_lse,
            key_dtype=_get_kv_torch_dtype(params["dtype_input"]["key"]),
            value_dtype=_get_kv_torch_dtype(params["dtype_input"]["value"]),
        )
    )
    return npu_attn_out, npu_softmax_max, npu_softmax_sum


def call_npu(input_tensor_dict, params):
    torch_npu.npu.set_device(0)
    torch_tensor_dict = _load_inputs_to_npu(input_tensor_dict)
    npu_result = call_npu_eager(torch_tensor_dict, params)
    torch.npu.synchronize()
    return npu_result
