# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

message(STATUS "=== Debug: start ops.moe.moe_tutel_combine_x.op_host.CMakeLists.txt")
if (BUILD_OPEN_PROJECT)
    # 本算子仅支持 GE 图模式调用（无 aclnn 单算子调用方式，op_api 层已删除）：
    # def 走 aclnnExc 分类，只生成 op proto / ops-info，不生成任何 aclnn 包装。
    target_sources(op_host_aclnnExc PRIVATE
        moe_tutel_combine_x_def.cpp
    )
    add_ops_compile_options(
        OP_NAME MoeTutelCombineX
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
                -mllvm
                -cce-aicore-dcci-before-kernel-end=false
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE moe_tutel_combine_x ACLNNTYPE aclnn_exclude)
endif()
