# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------
__golden__ = {
        "kernel": {
            "ffn_to_attention": "ffn_to_attention_golden"
        }
}


# TODO ffn_to_attention_golden待实现
def ffn_to_attention_golden(x, session_ids, micro_batch_ids, token_ids, expert_offsets, actual_token_num, attn_rank_table = None, group = None, world_size = None, token_info_table_shape = None, token_data_shape = None, **kwargs):
    pass