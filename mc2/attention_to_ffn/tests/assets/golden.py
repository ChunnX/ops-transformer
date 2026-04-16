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
            "attention_to_ffn": "attention_to_ffn_golden"
        }
}


# TODO attention_to_ffn_golden待实现
def attention_to_ffn_golden(x, session_id, micro_batch_id, layer_id, expert_ids, expert_rank_table, scales = None, active_mask = None, group = None, world_size = None, ffn_token_info_table_shape = None, ffn_token_data_shape = None, attn_token_info_table_shape = None, moe_expert_num = None, quant_mode = None, sync_flag = None, ffn_start_rank_id = None, **kwargs):
    pass