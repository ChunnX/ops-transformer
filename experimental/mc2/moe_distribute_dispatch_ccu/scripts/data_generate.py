#!/usr/bin/env python3
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""dispatch-ccu 输入数据生成（P0 步骤 6 待补）。

目标：按 (rank_size, bs, hidden, topk, expert_num) 生成每卡的
  - x[bs, hidden]          token，bf16/fp16
  - expert_ids[bs, topk]   topk 路由 expert 索引，int32
并落盘供 test_moe_distribute_dispatch_ccu 读取。
可参考 ../quant_all_reduce/scripts/data_generate.py 的落盘格式。
"""
import sys


def main() -> int:
    print("[TODO][data_generate] dispatch-ccu 输入数据生成尚未实现（P0 步骤 6）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
