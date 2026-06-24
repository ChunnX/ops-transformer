#!/usr/bin/python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# -----------------------------------------------------------------------------------------------------------

import torch

_default_base = {
    "batch_size": [8], "He": [7168], "Hcq": [1536], "Hckv": [512],
    "q_head_num": [32], "kv_head_num": [1], "head_dim": [128], "rope_head_dim": [64],
    "q_seq": [1], "kv_seq": [1024], "block_size": [128],
    "input_layout": ["BSH"], "cache_mode": ["PA_BSND"],
    "cq_epsilon": [0.0005], "ckv_epsilon": [0.0005], "dtype": [torch.bfloat16],
    "query_quant_mode": [0],
    "ckvkr_repo_mode": [0], "quant_scale_repo_mode": [0], "tile_size": [128],
    "qc_qr_scale": [1.0], "kc_scale": [1.0], "query_norm_flag": [False],
    "bs_merged": [False],
}

TEST_PARAMS = {
    "no_quant":          {**_default_base, "weight_quant_mode": [0], "kv_cache_quant_mode": [0]},
    "partial_kv_no_quant":{**_default_base, "weight_quant_mode": [1], "kv_cache_quant_mode": [0]},
    "partial_kv_per_ch":  {**_default_base, "weight_quant_mode": [1], "kv_cache_quant_mode": [2]},
    "partial_kv_per_tile":{**_default_base, "weight_quant_mode": [1], "kv_cache_quant_mode": [3], "ckvkr_repo_mode": [1], "quant_scale_repo_mode": [1]},
    "int8_kv_no_quant":   {**_default_base, "weight_quant_mode": [2], "kv_cache_quant_mode": [0]},
    "int8_kv_per_tensor": {**_default_base, "weight_quant_mode": [2], "kv_cache_quant_mode": [1], "query_quant_mode": [1]},
    "int8_kv_per_tile":   {**_default_base, "weight_quant_mode": [2], "kv_cache_quant_mode": [3], "ckvkr_repo_mode": [1], "quant_scale_repo_mode": [1]},
    "mxfp8_kv_no_quant":  {**_default_base, "weight_quant_mode": [3], "kv_cache_quant_mode": [0]},
    "mxfp8_kv_per_tensor":{**_default_base, "weight_quant_mode": [3], "kv_cache_quant_mode": [1], "query_quant_mode": [1]},
    "mxfp8_kv_per_tile":  {**_default_base, "weight_quant_mode": [3], "kv_cache_quant_mode": [3]},
    "fp8_kv_no_quant":    {**_default_base, "weight_quant_mode": [4], "kv_cache_quant_mode": [0]},
    "fp8_kv_per_tensor":  {**_default_base, "weight_quant_mode": [4], "kv_cache_quant_mode": [1], "query_quant_mode": [1]},
    "fp8_kv_per_tile":    {**_default_base, "weight_quant_mode": [4], "kv_cache_quant_mode": [3]},
    "hif8_kv_no_quant":   {**_default_base, "weight_quant_mode": [5], "kv_cache_quant_mode": [0]},
    "hif8_kv_per_tensor": {**_default_base, "weight_quant_mode": [5], "kv_cache_quant_mode": [1], "query_quant_mode": [1]},
    "hif8_kv_per_tile":   {**_default_base, "weight_quant_mode": [5], "kv_cache_quant_mode": [3]},
    "no_quant_qnorm":     {**_default_base, "weight_quant_mode": [0], "kv_cache_quant_mode": [0], "query_norm_flag": [True]},
    "int8_kv_per_tensor_qnorm": {**_default_base, "weight_quant_mode": [2], "kv_cache_quant_mode": [1], "query_quant_mode": [1], "query_norm_flag": [True]},
}

ENABLED_PARAMS = list(TEST_PARAMS.values())
