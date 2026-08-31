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

from mixed_quant_sparse_flash_mla_paramset import TEST_PARAMS as BASE_TEST_PARAMS


def _case(base_name, testcase_name, **overrides):
    params = deepcopy(BASE_TEST_PARAMS[base_name])
    params.update({key: [value] for key, value in overrides.items()})
    params["Testcase_Name"] = [testcase_name]
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


TEST_PARAMS = {
    "load_4096_swa_tnd_pa": _case(
        "decode_first", "bc_load_4096_swa_tnd_pa", layout_q="TND", layout_kv="PA_BBND",
        B=2, S1=2, S2=4096, seqused_ori_kv=[4096], cmp_ratio=1, K=0,
        ori_mask_mode=4, cmp_mask_mode=0, ori_win_left=-1, ori_win_right=-1,
        template_run_mode="SWA",
    ),
    "load_8192_hca_tnd_tnd": _case(
        "decode_first", "bc_load_8192_hca_tnd_tnd", layout_q="TND", layout_kv="TND",
        B=2, S1=2, S2=6554, seqused_ori_kv=[6554], seqused_cmp_kv=[1638],
        cmp_residual_kv=[2], cmp_ratio=4, K=0, ori_mask_mode=3, cmp_mask_mode=3,
        ori_win_left=-1, ori_win_right=-1, template_run_mode="HCA",
    ),
    "load_2048_csa_bsnd_bsnd": _case(
        "decode_first", "bc_load_2048_csa_bsnd_bsnd", layout_q="BSND", layout_kv="BSND",
        B=2, S1=2, S2=1639, seqused_ori_kv=[1639], seqused_cmp_kv=[409],
        cmp_residual_kv=[3], cmp_ratio=4, K=409, ori_mask_mode=4, cmp_mask_mode=3,
        ori_win_left=-1, ori_win_right=-1, template_run_mode="CSA",
    ),
    "load_5120_ori_sparse_bsnd_pa": _case(
        "ori_sparse_decode", "bc_load_5120_ori_sparse_bsnd_pa", layout_q="BSND",
        B=2, S1=2, S2=5120, seqused_ori_kv=[5120], K1=5120, K=0, quant_mode=1,
        ori_mask_mode=0, cmp_mask_mode=0, ori_win_left=-1, ori_win_right=-1,
        template_run_mode="ORI_SPARSE",
    ),
    "load_4096_ori_cmp_sparse_tnd_pa": _case(
        "ori_cmp_sparse_decode_with_topk", "bc_load_4096_ori_cmp_sparse_tnd_pa",
        layout_q="TND", B=2, S1=2, S2=3277, seqused_ori_kv=[3277],
        seqused_cmp_kv=[819], cmp_residual_kv=None, K1=3277, K=819,
        ori_mask_mode=0, cmp_mask_mode=0, ori_kv_topk_mode="fullK",
        cmp_kv_topk_mode="fullK", template_run_mode="ORI_CMP_SPARSE",
    ),
}

ENABLED_PARAMS = list(TEST_PARAMS.values())
