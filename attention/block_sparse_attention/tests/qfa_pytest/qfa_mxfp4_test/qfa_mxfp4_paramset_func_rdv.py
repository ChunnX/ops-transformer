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
# func_rdv 用例集: 从 tests/pytest/test_cases/test_case_STC.py 的 15 个 case 转换而来.
# 字段映射:
#   STC.n1 = N_q (query heads)
#   STC.n2 = N_kv (kv heads)
#   STC.g  = G (GQA group size, N_q // N_kv)
#   STC.qk_d = D
#   STC.v_d  = V_D
#   STC.act_seq_lens_q / act_seq_lens_kv -> 同名
#   STC.input_layout -> 同名
#   STC.q_range / k_range / v_range -> data_range_{q,k,v} (取上界)
# 全部 case 默认 q_dtype/kv_dtype = fp4_e2m1, out_dtype = bfloat16, q_quant_mode = 3,
#   kv_storage_mode = continue, mask_mode = 0 (无 mask).

from qfa_mxfp4_paramset_common import expand_paramset_to_cases

TEST_PARAMS = {
    # ===== redline case (来自 STC) =====
    "FP4_B8_Nq12_Nkv12_D128_S1_1024_S2_1024_BNSD_redline": {
        'B': [8], 'N_q': [12], 'N_kv': [12], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[1024, 1022, 1023, 1024, 1020, 1024, 1022, 1024]],
        'act_seq_lens_kv': [[1024, 1022, 1003, 1010, 1020, 1024, 1022, 1024]],
        'input_layout': ['BNSD'],
        'data_range_q': [1.0], 'data_range_k': [1.0], 'data_range_v': [1.0],
    },
    "FP4_B16_Nq7_Nkv7_D128_S1_1743_S2_2185_BNSD_redline": {
        'B': [16], 'N_q': [7], 'N_kv': [7], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[1740] * 16],
        'act_seq_lens_kv': [[2180] * 16],
        'input_layout': ['BNSD'],
        'data_range_q': [1.0], 'data_range_k': [1.0], 'data_range_v': [1.0],
    },
    "FP4_B20_Nq10_Nkv10_D128_S1_2_S2_2400_BNSD_redline": {
        'B': [20], 'N_q': [10], 'N_kv': [10], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[2] * 20],
        'act_seq_lens_kv': [[2000 + i for i in range(20)]],
        'input_layout': ['BNSD'],
        'data_range_q': [6.0], 'data_range_k': [6.0], 'data_range_v': [6.0],
    },

    # ===== 基础功能 case (BNSD) =====
    "BNSD_B2_S1_512_S2_8192_ACT_Q_380_KV_6000": {
        'B': [2], 'N_q': [1], 'N_kv': [1], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[380, 420]],
        'act_seq_lens_kv': [[6000, 7500]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B1_S1_128_S2_512": {
        'B': [1], 'N_q': [1], 'N_kv': [1], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[128]],
        'act_seq_lens_kv': [[512]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B1_S1_128_S2_1024": {
        'B': [1], 'N_q': [1], 'N_kv': [1], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[128]],
        'act_seq_lens_kv': [[1024]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B1_S1_128_S2_4096": {
        'B': [1], 'N_q': [1], 'N_kv': [1], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[128]],
        'act_seq_lens_kv': [[4096]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B1_S1_128_S2_8192": {
        'B': [1], 'N_q': [1], 'N_kv': [1], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[128]],
        'act_seq_lens_kv': [[8192]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B2_S1_128_S2_512": {
        'B': [2], 'N_q': [1], 'N_kv': [1], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[128, 128]],
        'act_seq_lens_kv': [[512, 512]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B2_Nq2_Nkv2_S1_128_S2_512": {
        'B': [2], 'N_q': [2], 'N_kv': [2], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[128, 128]],
        'act_seq_lens_kv': [[512, 512]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B2_Nq2_Nkv2_S1_6_S2_512": {
        'B': [2], 'N_q': [2], 'N_kv': [2], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[6, 6]],
        'act_seq_lens_kv': [[512, 512]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B2_Nq2_Nkv2_S1_6_S2_544": {
        'B': [2], 'N_q': [2], 'N_kv': [2], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[6, 6]],
        'act_seq_lens_kv': [[544, 544]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B1_S1_3584_S2_3584": {
        'B': [1], 'N_q': [1], 'N_kv': [1], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[3584]],
        'act_seq_lens_kv': [[3584]],
        'input_layout': ['BNSD'],
    },
    "BNSD_B1_S1_8704_S2_8704": {
        'B': [1], 'N_q': [1], 'N_kv': [1], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[8704]],
        'act_seq_lens_kv': [[8704]],
        'input_layout': ['BNSD'],
    },

    # ===== BSND layout case =====
    "BSND_B1_Nq2_Nkv2_S1_8422_S2_8422": {
        'B': [1], 'N_q': [2], 'N_kv': [2], 'G': [1], 'D': [128], 'V_D': [128],
        'act_seq_lens_q': [[8422]],
        'act_seq_lens_kv': [[8422]],
        'input_layout': ['BSND'],
    },
}
