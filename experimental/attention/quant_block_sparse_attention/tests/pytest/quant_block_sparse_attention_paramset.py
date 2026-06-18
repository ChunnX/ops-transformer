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

import math

import torch


COMMON = {
    "Testcase_Name": [None],
    "layout_kv": ["PA_BNSD"],
    "output_dtype": [torch.bfloat16],
    "N1": [32],
    "N2": [4],
    "D": [128],
    "sparse_q_block_size": [128],
    "sparse_kv_block_size": [128],
    "softmax_scale": [1.0 / math.sqrt(128)],
    "layout_sparse_indices": ["B_N_Qb_Kb"],
    "quant_mode": [1],
    "mask_mode": [3],
    "pa_block_padding_bytes": [0],
    "S1EQS2": [False],
}


def _case(**kwargs):
    params = dict(COMMON)
    params.update(kwargs)
    return params


# =============================================================================
# BASIC TEST CASES (50 cases)
# Rules:
#   - B=8, layout_q=TND, D=128, seed=3
#   - S1/S2 in {256,512,1024,2048,4096,8192,16384,32768}, divisible by 128
#   - sparse_count: even, <= floor(S2/128)
#   - sparse_pattern: "sequential", "reverse", "causal"
#   - actlen_mode: mostly "full"
# =============================================================================
BASIC_CASES = {
    # --- S1=S2=256, minimal sequence ---
    "basic_s256_seq_full_m0_lse": _case(
        layout_q=["TND"], B=[8], S1=[256], S2=[256],
        sparse_count=[2], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s256_rev_full_m3_lse": _case(
        layout_q=["TND"], B=[8], S1=[256], S2=[256],
        sparse_count=[2], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    # --- S1=256, S2=512, short Q long KV ---
    "basic_s256_512_causal_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[256], S2=[512],
        sparse_count=[2], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "basic_s256_512_seq_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[256], S2=[512],
        sparse_count=[4], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["reverse"],
        p_scale_value=[0.5], seed=[3], mask_mode=[0],
    ),
    # --- S1=S2=512 ---
    "basic_s512_seq_full_m3_n16": _case(
        layout_q=["TND"], B=[8], S1=[512], S2=[512],
        N1=[16], N2=[2],
        sparse_count=[2], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s512_rev_full_m0_n8": _case(
        layout_q=["TND"], B=[8], S1=[512], S2=[512],
        N1=[8], N2=[1],
        sparse_count=[4], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["gradient"], block_table_pattern=["sequential"],
        p_scale_value=[0.8], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    # --- S1=512, S2=1024 ---
    "basic_s512_1024_causal_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[512], S2=[1024],
        sparse_count=[4], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "basic_s512_1024_seq_full_m0_grad": _case(
        layout_q=["TND"], B=[8], S1=[512], S2=[1024],
        sparse_count=[8], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["gradient"], block_table_pattern=["random"],
        p_scale_value=[0.4], seed=[3], mask_mode=[0],
    ),
    # --- S1=S2=1024 ---
    "basic_s1024_seq_full_m3_lse": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[1024],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s1024_rev_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[1024],
        sparse_count=[4], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["ones"], block_table_pattern=["reverse"],
        p_scale_value=[0.6], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s1024_causal_full_m3_n16": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[1024],
        N1=[16], N2=[2],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s1024_seq_full_m0_n24": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[1024],
        N1=[24], N2=[3],
        sparse_count=[6], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["gradient"], block_table_pattern=["sequential"],
        p_scale_value=[0.5], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    # --- S1=1024, S2=2048 ---
    "basic_s1024_2048_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[2048],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "basic_s1024_2048_rev_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[2048],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.3], seed=[3], mask_mode=[0],
    ),
    # --- S1=S2=2048 ---
    "basic_s2048_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[2048],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s2048_causal_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[2048],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s2048_rev_full_m3_n8": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[2048],
        N1=[8], N2=[1],
        sparse_count=[8], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.7], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    # --- S1=2048, S2=4096 ---
    "basic_s2048_4096_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[4096],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "basic_s2048_4096_causal_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[4096],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[0.5], seed=[3], mask_mode=[0],
    ),
    # --- S1=S2=4096 ---
    "basic_s4096_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[4096],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s4096_rev_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[4096],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["ones"], block_table_pattern=["reverse"],
        p_scale_value=[0.8], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s4096_causal_full_m3_n16": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[4096],
        N1=[16], N2=[2],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["gradient"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    # --- S1=4096, S2=8192 ---
    "basic_s4096_8192_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[8192],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "basic_s4096_8192_rev_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[8192],
        sparse_count=[64], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.2], seed=[3], mask_mode=[0],
    ),
    # --- S1=S2=8192 ---
    "basic_s8192_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[8192],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s8192_causal_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[8192],
        sparse_count=[64], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s8192_rev_full_m3_n24": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[8192],
        N1=[24], N2=[3],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.6], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    # --- S1=8192, S2=16384 ---
    "basic_s8192_16384_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[16384],
        sparse_count=[64], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "basic_s8192_16384_causal_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[16384],
        sparse_count=[128], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[0.5], seed=[3], mask_mode=[0],
    ),
    # --- S1=S2=16384 ---
    "basic_s16384_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[16384],
        sparse_count=[64], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s16384_rev_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[16384],
        sparse_count=[128], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["ones"], block_table_pattern=["reverse"],
        p_scale_value=[0.4], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s16384_causal_full_m3_n8": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[16384],
        N1=[8], N2=[1],
        sparse_count=[64], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["gradient"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    # --- S1=16384, S2=32768 ---
    "basic_s16384_32768_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[32768],
        sparse_count=[128], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "basic_s16384_32768_rev_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[32768],
        sparse_count=[256], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.3], seed=[3], mask_mode=[0],
    ),
    # --- S1=S2=32768 ---
    "basic_s32768_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[32768], S2=[32768],
        sparse_count=[128], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s32768_causal_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[32768], S2=[32768],
        sparse_count=[256], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s32768_rev_full_m3_n16": _case(
        layout_q=["TND"], B=[8], S1=[32768], S2=[32768],
        N1=[16], N2=[2],
        sparse_count=[128], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.5], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    # --- scale_pattern / block_table_pattern / p_scale variations ---
    "basic_s1024_grad_btrev_m3": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[1024],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.5], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s2048_grad_btrand_m0": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[2048],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["gradient"], block_table_pattern=["random"],
        p_scale_value=[0.3], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s4096_ones_btrand_m3": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[4096],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.8], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s8192_grad_btrev_m0": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[8192],
        sparse_count=[64], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.2], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    # --- return_softmax_lse=False variations ---
    "basic_s2048_nolse_m3": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[2048],
        sparse_count=[16], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s4096_nolse_m0": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[4096],
        sparse_count=[32], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["gradient"], block_table_pattern=["sequential"],
        p_scale_value=[0.6], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "basic_s8192_nolse_m3": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[8192],
        sparse_count=[64], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["reverse"],
        p_scale_value=[0.4], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "basic_s16384_nolse_m0": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[16384],
        sparse_count=[128], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["gradient"], block_table_pattern=["random"],
        p_scale_value=[0.5], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    # --- S1 > S2 cases ---
    "basic_s1024_512_seq_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[512],
        sparse_count=[4], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "basic_s2048_1024_causal_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[1024],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["causal"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[0.7], seed=[3], mask_mode=[0],
    ),
    "basic_s4096_2048_rev_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[2048],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["reverse"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.5], seed=[3], mask_mode=[3],
    ),
    "basic_s8192_4096_seq_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[4096],
        sparse_count=[32], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.3], seed=[3], mask_mode=[0],
    ),
    # --- random actlen_mode (minority in basic) ---
    "basic_s1024_seq_random_m3": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[1024],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["random"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
}


# =============================================================================
# GENERALIZATION TEST CASES (20 cases)
# Rules:
#   - B=8, layout_q=TND, D=128, seed=3
#   - S1/S2 in {256,512,1024,2048,4096,8192,16384,32768}
#   - sparse_count: <= floor(S2/128), can be odd
#   - sparse_pattern: "sequential", "dense", "reverse", "tail", "random", "causal"
#   - actlen_mode: "full" or "random"
# =============================================================================
GENERALIZATION_CASES = {
    "gen_dense_s512_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[512], S2=[512],
        sparse_count=[4], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["dense"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "gen_dense_s2048_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[2048],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["dense"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.5], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "gen_tail_s1024_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[1024],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["tail"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "gen_tail_s4096_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[4096],
        sparse_count=[32], return_softmax_lse=[False],
        actlen_mode=["full"], sparse_pattern=["tail"],
        scale_pattern=["gradient"], block_table_pattern=["random"],
        p_scale_value=[0.4], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "gen_random_s2048_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[2048],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["random"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.8], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "gen_random_s8192_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[8192],
        sparse_count=[64], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["random"],
        scale_pattern=["gradient"], block_table_pattern=["sequential"],
        p_scale_value=[0.3], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "gen_dense_s8192_random_m3": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[8192],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["random"], sparse_pattern=["dense"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "gen_tail_s16384_random_m0": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[16384],
        sparse_count=[64], return_softmax_lse=[True],
        actlen_mode=["random"], sparse_pattern=["tail"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.6], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "gen_random_s32768_random_m3": _case(
        layout_q=["TND"], B=[8], S1=[32768], S2=[32768],
        sparse_count=[128], return_softmax_lse=[False],
        actlen_mode=["random"], sparse_pattern=["random"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.2], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "gen_dense_s1024_2048_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[2048],
        sparse_count=[8], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["dense"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "gen_tail_s2048_4096_full_m0": _case(
        layout_q=["TND"], B=[8], S1=[2048], S2=[4096],
        sparse_count=[16], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["tail"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.5], seed=[3], mask_mode=[0],
    ),
    "gen_random_s4096_8192_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[8192],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["random"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.7], seed=[3], mask_mode=[3],
    ),
    "gen_dense_s8192_16384_random_m0": _case(
        layout_q=["TND"], B=[8], S1=[8192], S2=[16384],
        sparse_count=[64], return_softmax_lse=[True],
        actlen_mode=["random"], sparse_pattern=["dense"],
        scale_pattern=["gradient"], block_table_pattern=["sequential"],
        p_scale_value=[0.4], seed=[3], mask_mode=[0],
    ),
    "gen_tail_s16384_32768_full_m3": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[32768],
        sparse_count=[128], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["tail"],
        scale_pattern=["ones"], block_table_pattern=["reverse"],
        p_scale_value=[1.0], seed=[3], mask_mode=[3],
    ),
    "gen_causal_s1024_random_m3": _case(
        layout_q=["TND"], B=[8], S1=[1024], S2=[1024],
        sparse_count=[4], return_softmax_lse=[True],
        actlen_mode=["random"], sparse_pattern=["causal"],
        scale_pattern=["gradient"], block_table_pattern=["sequential"],
        p_scale_value=[0.8], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "gen_seq_s4096_random_m0": _case(
        layout_q=["TND"], B=[8], S1=[4096], S2=[4096],
        sparse_count=[16], return_softmax_lse=[False],
        actlen_mode=["random"], sparse_pattern=["sequential"],
        scale_pattern=["ones"], block_table_pattern=["random"],
        p_scale_value=[0.5], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "gen_rev_s16384_random_m3": _case(
        layout_q=["TND"], B=[8], S1=[16384], S2=[16384],
        sparse_count=[32], return_softmax_lse=[True],
        actlen_mode=["random"], sparse_pattern=["reverse"],
        scale_pattern=["gradient"], block_table_pattern=["reverse"],
        p_scale_value=[0.3], seed=[3], mask_mode=[3], S1EQS2=[True],
    ),
    "gen_dense_s256_full_m0_n8": _case(
        layout_q=["TND"], B=[8], S1=[256], S2=[256],
        N1=[8], N2=[1],
        sparse_count=[2], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["dense"],
        scale_pattern=["ones"], block_table_pattern=["sequential"],
        p_scale_value=[1.0], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "gen_tail_s32768_full_m0_n24": _case(
        layout_q=["TND"], B=[8], S1=[32768], S2=[32768],
        N1=[24], N2=[3],
        sparse_count=[128], return_softmax_lse=[True],
        actlen_mode=["full"], sparse_pattern=["tail"],
        scale_pattern=["gradient"], block_table_pattern=["random"],
        p_scale_value=[0.6], seed=[3], mask_mode=[0], S1EQS2=[True],
    ),
    "gen_random_s512_1024_random_m3": _case(
        layout_q=["TND"], B=[8], S1=[512], S2=[1024],
        sparse_count=[4], return_softmax_lse=[True],
        actlen_mode=["random"], sparse_pattern=["random"],
        scale_pattern=["ones"], block_table_pattern=["reverse"],
        p_scale_value=[0.9], seed=[3], mask_mode=[3],
    ),
}


TEST_PARAMS = {}
TEST_PARAMS.update(BASIC_CASES)
TEST_PARAMS.update(GENERALIZATION_CASES)

ENABLED_PARAMS = [TEST_PARAMS[key] for key in TEST_PARAMS.keys()]
