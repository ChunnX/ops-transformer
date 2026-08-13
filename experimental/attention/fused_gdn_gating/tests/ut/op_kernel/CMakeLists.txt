# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    # 算子自己的tiling文件路径
    set(fused_gdn_gating_files
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/fused_gdn_gating_tiling.cpp
    )
    # 算子kernel入口文件（experimental嵌套目录需显式指定）
    set(fused_gdn_gating_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel/fused_gdn_gating.cpp
    )

    AddOpTestCase(fused_gdn_gating "Ascend910B1" "" "${fused_gdn_gating_files}" "${fused_gdn_gating_kernel}")
    AddOpTestCase(fused_gdn_gating "Ascend310P1" "" "${fused_gdn_gating_files}" "${fused_gdn_gating_kernel}")

endif()
