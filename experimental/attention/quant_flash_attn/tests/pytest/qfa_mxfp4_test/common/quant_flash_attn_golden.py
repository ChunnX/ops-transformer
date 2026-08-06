#!/usr/bin/python3
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
import logging
import os
import sys

import torch
import torch_npu  # noqa: F401  确保 npu 后端注册

# 把父目录加入 sys.path 以复用原 golden 模块
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)

import quant_flash_attn_golden as _orig  # noqa: E402

try:
    from . import result_compare_method
except ImportError:  # 直接运行本文件时
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import result_compare_method  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s", force=True)
logger = logging.getLogger(__name__)


# 参数白名单：与原 run_fia_eager 签名对齐
PARAM_NAMES = [
    "b", "n2", "g", "s1", "s2", "qk_d", "v_d", "rope_d",
    "input_layout", "kv_storage_mode", "q_dtype", "kv_dtype", "out_dtype",
    "rope_dtype", "block_size", "act_seq_lens_q", "act_seq_lens_kv",
    "enable_softmax_lse", "enable_mask", "sparse_mode",
    "pre_tokens", "next_tokens", "enable_learnable_sink",
    "inner_precise", "layout_q_descale",
    "q_quant_mode", "q_scale_dtype", "quant_block_size_qs",
    "k_quant_mode", "k_scale_dtype", "quant_block_size_ks",
    "v_quant_mode", "v_scale_dtype", "quant_block_size_vs",
]

PARAM_DEFAULTS = dict(
    v_d=None, rope_d=0, input_layout='BSND', kv_storage_mode='continue',
    q_dtype='bfloat16', kv_dtype=None, out_dtype=None, rope_dtype=None,
    block_size=0, act_seq_lens_q=None, act_seq_lens_kv=None,
    enable_softmax_lse=False, enable_mask=False, sparse_mode=0,
    pre_tokens=2147483647, next_tokens=2147483647, enable_learnable_sink=False,
    inner_precise=0, layout_q_descale='BSND',
    q_quant_mode=3, q_scale_dtype=None, quant_block_size_qs=None,
    k_quant_mode=3, k_scale_dtype=None, quant_block_size_ks=None,
    v_quant_mode=3, v_scale_dtype=None, quant_block_size_vs=None,
)


def normalize_params(params):
    kwargs = dict(PARAM_DEFAULTS)
    for k in PARAM_NAMES:
        if k in params:
            kwargs[k] = params[k]
    return kwargs


def execute_full(params):
    kwargs = normalize_params(params)
    run_percent, run_result = _orig.run_fia_eager(**kwargs)
    return run_percent, run_result, None, None


def run_fia_eager(**kwargs):
    return _orig.run_fia_eager(**kwargs)
