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

from copy import deepcopy

from sparse_flash_mla_paramset import TEST_PARAMS as BASE_TEST_PARAMS


def _case(base_name, testcase_name, **overrides):
    params = deepcopy(BASE_TEST_PARAMS[base_name])
    params.update({key: [value] for key, value in overrides.items()})
    params["testcase_name"] = [testcase_name]
    params["seqused_q"] = [[overrides["S1"]] * overrides["B"]]
    params.update(
        {
            "batch_consistency": [True],
            "batch_consistency_seed": [20260831],
            "batch_consistency_order": [[1, 0]],
            "batch_consistency_batch_split": [[1, 1]],
            "batch_consistency_mode_batch": [0],
            "batch_consistency_token_split": [[1, 1]],
            "batch_consistency_shape_change": [[1, 0]],
        }
    )
    return params


# The workload is the per-query effective ori KV length plus cmp KV length.
# With cmp_ratio=4: 3277+819=4096, 6554+1638=8192,
# 1639+409=2048, and 4096+1024=5120.
TEST_PARAMS = {
    "load_4096_swa_tnd_pa": _case(
        "test_swa_decode_pa",
        "bc_load_4096_swa_tnd_pa",
        layout_q="TND",
        layout_kv="PA_BBND",
        B=2,
        S1=2,
        S2=4096,
        T1=4,
        cu_seqlens_q=[0, 2, 4],
        seqused_ori_kv=[4096, 4096],
        block_num1=None,
        ori_mask_mode=4,
        ori_win_left=-1,
        ori_win_right=-1,
        template_mode="SWA",
    ),
    "load_8192_hca_tnd_tnd": _case(
        "hca_tnd",
        "bc_load_8192_hca_tnd_tnd",
        layout_q="TND",
        layout_kv="TND",
        B=2,
        S1=2,
        S2=6554,
        T1=4,
        T2=13108,
        T3=3276,
        cu_seqlens_q=[0, 2, 4],
        cu_seqlens_ori_kv=[0, 6554, 13108],
        cu_seqlens_cmp_kv=[0, 1638, 3276],
        seqused_ori_kv=[6554, 6554],
        seqused_cmp_kv=[1638, 1638],
        cmp_residual_kv=[2, 2],
        block_num1=None,
        block_num2=None,
        ori_mask_mode=3,
        cmp_mask_mode=3,
        ori_win_left=-1,
        ori_win_right=-1,
        template_mode="HCA",
    ),
    "load_2048_csa_bsnd_bsnd": _case(
        "cmp_csa_g64_test_prefill",
        "bc_load_2048_csa_bsnd_bsnd",
        layout_q="BSND",
        layout_kv="BSND",
        B=2,
        S1=2,
        S2=1639,
        T1=None,
        T2=None,
        T3=None,
        seqused_ori_kv=[1639, 1639],
        seqused_cmp_kv=[409, 409],
        cmp_residual_kv=[3, 3],
        K=409,
        ori_mask_mode=4,
        cmp_mask_mode=3,
        ori_win_left=-1,
        ori_win_right=-1,
        template_mode="CSA",
    ),
    "load_5120_ori_sparse_bsnd_pa": _case(
        "ori_sparse_bsnd_pa",
        "bc_load_5120_ori_sparse_bsnd_pa",
        B=2,
        S1=2,
        S2=5120,
        seqused_ori_kv=[5120, 5120],
        K1=5120,
        K=0,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=-1,
        ori_win_right=-1,
        template_mode="ORI_SPARSE",
    ),
    "load_4096_ori_cmp_sparse_tnd_pa": _case(
        "ori_cmp_sparse_tnd_pa",
        "bc_load_4096_ori_cmp_sparse_tnd_pa",
        B=2,
        S1=2,
        S2=3277,
        T1=4,
        cu_seqlens_q=[0, 2, 4],
        seqused_ori_kv=[3277, 3277],
        seqused_cmp_kv=[819, 819],
        cmp_residual_kv=None,
        K1=3277,
        K=819,
        ori_mask_mode=0,
        cmp_mask_mode=0,
        ori_win_left=-1,
        ori_win_right=-1,
        ori_kv_topk_mode="fullK",
        cmp_kv_topk_mode="fullK",
        template_mode="ORI_CMP_SPARSE",
    ),
}

ENABLED_PARAMS = list(TEST_PARAMS.values())
