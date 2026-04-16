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
            "moe_distribute_dispatch_teardown": "moe_distribute_dispatch_teardown_golden"
        }
}

# TODO moe_distribute_dispatch_teardown_golden待实现
def moe_distribute_dispatch_teardown_golden(x, y, expert_ids, comm_cmd_info, group_ep, ep_world_size, ep_rank_id, moe_expert_num, expert_shard_type = 0, shared_expert_num = 1, shared_expert_rank_num = 0, quant_mode = 0, global_bs = 0, expert_token_nums_type = 1, comm_type = 0, comm_alg = '', **kwargs):
    pass