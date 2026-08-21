#!/usr/bin/python
# -*- coding: utf-8 -*-
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

####### MLA NC（非连续 k_cache）探测用例（最小子集）########
# nc_kv_dims: 0 -> dim0 非连续; (0,1) -> dim0+dim1 都非连续（k_cache 是唯一 KV 输入，无 v）
# 布局恒 TND/NZ/NTD（R1）。自洽判定：连续 vs NC 输出应位级一致；
# 若 aclnn/算子限制非连续输入，case 如实 FAIL 并记录（见 learnings 结论）。

TestCases = {
    "NC_PREFILL_FP16": {"B": [2], "N1": [4], "S1": [64], "S2": [1024],
                        "D": [512], "DV": [512], "Dtype": ["fp16"],
                        "layout_q": ["TND"], "layout_kv": ["PA_NZ"], "layout_out": ["NTD"],
                        "block_size": [128], "return_softmax_lse": [False],
                        "mask_mode": [0],
                        "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)],
                        "nc_kv_dims": [0]},
    "NC_DECODE_BF16": {"B": [4], "N1": [4], "S1": [1], "S2": [2048],
                       "D": [512], "DV": [512], "Dtype": ["bf16"],
                       "layout_q": ["TND"], "layout_kv": ["PA_NZ"], "layout_out": ["NTD"],
                       "block_size": [128], "return_softmax_lse": [False],
                       "mask_mode": [0],
                       "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)],
                       "nc_kv_dims": [(0, 1)]},
}