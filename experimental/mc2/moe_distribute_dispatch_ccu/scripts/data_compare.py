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
"""dispatch-ccu 精度对比（P0 步骤 6 待补）。

目标：把 test_moe_distribute_dispatch_ccu 输出的 expand_x 与 golden 逐元素对比。
golden 基线：现存 ops-transformer/mc2/moe_distribute_dispatch_{setup,teardown} 端到端结果。
精度标准（ops-precision-standard）：fp16 atol/rtol=1e-3；bf16=5e-3；fp32=1e-5。
"""
import sys


def main() -> int:
    print("[TODO][data_compare] dispatch-ccu 精度对比尚未实现（P0 步骤 6）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
