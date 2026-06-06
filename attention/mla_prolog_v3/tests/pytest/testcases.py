#!/usr/bin/python
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

WEIGHT_QUANT_MODE_NO_QUANT = 0
WEIGHT_QUANT_MODE_PARTIAL_QUANT = 1
WEIGHT_QUANT_MODE_FULL_QUANT = 2
WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT = 3
WEIGHT_QUANT_MODE_FP8_FULL_QUANT = 4
WEIGHT_QUANT_MODE_HIF8_FULL_QUANT = 5

KV_CACHE_QUANT_MODE_NO_QUANT = 0
KV_CACHE_QUANT_MODE_PER_TENSOR = 1
KV_CACHE_QUANT_MODE_PER_CHANNEL = 2
KV_CACHE_QUANT_MODE_PER_TILE = 3

QUERY_QUANT_MODE_NO_QUANT = 0
QUERY_QUANT_MODE_QUANT = 1

CkvKr_REPO_MODE_INPLACE = 0
CkvKr_REPO_MODE_COPY = 1

QUANT_SCALE_REPO_MODE_INPLACE = 0
QUANT_SCALE_REPO_MODE_COPY = 1

_default_base = {
    "batch_size": [8],
    "He": [7168],
    "Hcq": [1536],
    "Hckv": [512],
    "q_head_num": [32],
    "kv_head_num": [1],
    "head_dim": [128],
    "rope_head_dim": [64],
    "q_seq": [1],
    "kv_seq": [1024],
    "block_size": [128],
    "input_layout": ["BSH"],
    "cache_mode": ["PA_BSND"],
    "cq_epsilon": [0.0005],
    "ckv_epsilon": [0.0005],
    "dtype": [torch.bfloat16],
    "ckvkr_repo_mode": [CkvKr_REPO_MODE_INPLACE],
    "quant_scale_repo_mode": [QUANT_SCALE_REPO_MODE_INPLACE],
    "tile_size": [128],
    "qc_qr_scale": [1.0],
    "kc_scale": [1.0],
    "query_norm_flag": [False],
}

TEST_PARAMS = {
    "no_quant": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_NO_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "partial_kv_no_quant": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_PARTIAL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "partial_kv_per_channel": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_PARTIAL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_CHANNEL],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "partial_kv_per_tile": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_PARTIAL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TILE],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "full_kv_no_quant": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "full_kv_per_tensor": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TENSOR],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "full_kv_per_tile": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TILE],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "mxfp8_kv_no_quant": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "mxfp8_kv_per_tensor": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TENSOR],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "mxfp8_kv_per_tile": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_MXFP8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TILE],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "fp8_kv_no_quant": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_FP8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "fp8_kv_per_tensor": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_FP8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TENSOR],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "fp8_kv_per_tile": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_FP8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TILE],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "hif8_kv_no_quant": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_HIF8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "hif8_kv_per_tensor": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_HIF8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TENSOR],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "hif8_kv_per_tile": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_HIF8_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_PER_TILE],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
    },
    "no_quant_qnorm": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_NO_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
        "query_norm_flag": [True],
    },
    "partial_qnorm": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_PARTIAL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
        "query_norm_flag": [True],
    },
    "full_qnorm": {
        **_default_base,
        "weight_quant_mode": [WEIGHT_QUANT_MODE_FULL_QUANT],
        "kv_cache_quant_mode": [KV_CACHE_QUANT_MODE_NO_QUANT],
        "query_quant_mode": [QUERY_QUANT_MODE_NO_QUANT],
        "query_norm_flag": [True],
    },
}

ENABLED_PARAMS = list(TEST_PARAMS.values())
