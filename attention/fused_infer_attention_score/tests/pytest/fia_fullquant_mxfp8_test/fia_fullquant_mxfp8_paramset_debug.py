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

from fia_fullquant_mxfp8_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    "DEBUG_QS4_KVS31_Nq1_Nkv1_D128": {
        "B": [1],
        "N_q": [1],
        "N_kv": [1],
        "D": [128],
        "actual_seq_q": [[4]],
        "actual_seq_kv": [[31]],
        "enable_pa": [True],
        "kv_cache_layout": ["PA_NZ"],
        "block_size": [512],
        "sparse_mode": [3],
        "q_scale_layout": ["TND"],
        "p_scale": [1.0],
    },
}

CASES = expand_paramset_to_cases(TEST_PARAMS)
