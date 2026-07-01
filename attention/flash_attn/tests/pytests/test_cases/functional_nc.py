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

####### 参数说明 ########
# nc_kv_dims: 0 -> 仅dim0非连续; 1 -> 仅dim1非连续; (0,1) -> dim0和dim1都不连续
# layout_kv: 覆盖 PA_BBND / PA_BNBD / PA_NZ（均走 offsetCalculator GM_KV_PA_BNBD 分支，stride 参数透传）
# 非PA layout不在此文件覆盖：非连续输入会被CheckKVContiguous拦截，属于正常的checker行为

TestCases = {
    # ------ PA_BNBD: dim0 非连续 ------
    "NC_PA_BNBD_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                        "D": [128], "DV": [128], "Dtype": ["fp16"],
                        "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                        "block_size": [128], "return_softmax_lse": [False],
                        "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                        "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                        "nc_kv_dims": [0]},
    # ------ PA_BNBD: dim1 非连续 ------
    "NC_PA_BNBD_dim1": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                        "D": [128], "DV": [128], "Dtype": ["fp16"],
                        "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                        "block_size": [128], "return_softmax_lse": [False],
                        "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                        "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                        "nc_kv_dims": [1]},
    # ------ PA_BNBD: dim0+dim1 非连续 ------
    "NC_PA_BNBD_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                         "D": [128], "DV": [128], "Dtype": ["fp16"],
                         "layout_q": ["BNSD"], "layout_kv": ["PA_BNBD"], "layout_out": ["BNSD"],
                         "block_size": [128], "return_softmax_lse": [False],
                         "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                         "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                         "nc_kv_dims": [(0, 1)]},
    # ------ PA_BBND: dim0 非连续 ------
    "NC_PA_BBND_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                        "D": [128], "DV": [128], "Dtype": ["fp16"],
                        "layout_q": ["BNSD"], "layout_kv": ["PA_BBND"], "layout_out": ["BNSD"],
                        "block_size": [128], "return_softmax_lse": [False],
                        "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                        "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                        "nc_kv_dims": [0]},
    # ------ PA_BBND: dim1 非连续 ------
    "NC_PA_BBND_dim1": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                        "D": [128], "DV": [128], "Dtype": ["fp16"],
                        "layout_q": ["BNSD"], "layout_kv": ["PA_BBND"], "layout_out": ["BNSD"],
                        "block_size": [128], "return_softmax_lse": [False],
                        "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                        "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                        "nc_kv_dims": [1]},
    # ------ PA_BBND: dim0+dim1 非连续 ------
    "NC_PA_BBND_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                         "D": [128], "DV": [128], "Dtype": ["fp16"],
                         "layout_q": ["BNSD"], "layout_kv": ["PA_BBND"], "layout_out": ["BNSD"],
                         "block_size": [128], "return_softmax_lse": [False],
                         "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                         "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                         "nc_kv_dims": [(0, 1)]},
    # ------ PA_NZ: dim0 非连续 ------
    "NC_PA_NZ_dim0": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                      "D": [128], "DV": [128], "Dtype": ["fp16"],
                      "layout_q": ["BNSD"], "layout_kv": ["PA_NZ"], "layout_out": ["BNSD"],
                      "block_size": [128], "return_softmax_lse": [False],
                      "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                      "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                      "nc_kv_dims": [0]},
    # ------ PA_NZ: dim1 非连续 ------
    "NC_PA_NZ_dim1": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                      "D": [128], "DV": [128], "Dtype": ["fp16"],
                      "layout_q": ["BNSD"], "layout_kv": ["PA_NZ"], "layout_out": ["BNSD"],
                      "block_size": [128], "return_softmax_lse": [False],
                      "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                      "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                      "nc_kv_dims": [1]},
    # ------ PA_NZ: dim0+dim1 非连续 ------
    "NC_PA_NZ_dim01": {"B": [2], "N1": [8], "N2": [1], "S1": [64], "S2": [1024],
                       "D": [128], "DV": [128], "Dtype": ["fp16"],
                       "layout_q": ["BNSD"], "layout_kv": ["PA_NZ"], "layout_out": ["BNSD"],
                       "block_size": [128], "return_softmax_lse": [False],
                       "mask_mode": [0], "win_left": [-1], "win_right": [-1],
                       "q_range": [(-5.0, 5.0)], "k_range": [(-5.0, 5.0)], "v_range": [(-5.0, 5.0)],
                       "nc_kv_dims": [(0, 1)]},
}
