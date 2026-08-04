# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import torch

# 定义测试参数组合
TEST_PARAMS = {
    "Prefill0":{
        "batch_size": [1],
        "hidden_size": [1536],
        "Seq_len": [4521],
        "head_dim": [128],
        "block_num": [7],
        "block_size": [761],
        "cmp_ratio": [98],
        "coff": [1],
        "layout": ["BSH"],
        "data_type": [torch.bfloat16],
        "cu_seqlens": [None],
        "seqused": [None],
        "start_pos": [[581]],
        "x_datarange": [[-10, 10]],
        "wkv_datarange": [[-10, 10]],
        "wgate_datarange": [[-10, 10]],
        "d_cmp_kv_datarange": [[-10, 10]],
        "softmax_score_datarange": [[-10, 10]],
        "kv_datarange": [[-10, 10]],
    }
}#注意单个用例组内的用例尽量不要超过32
FIRST_CASE = [TEST_PARAMS["Prefill0"]]
# 按需选择要启用的测试参数（例如默认启用所有）
ENABLED_PARAMS = FIRST_CASE #按需增加需要的case即可