# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    if(BUILD_OPEN_PROJECT)
        # 本算子不提供 aclnn 单算子接口，走 aclnn_exclude 通道。
        target_sources(op_host_aclnnExc PRIVATE moe_tutel_combine_x_def.cpp)
        add_ops_compile_options(
            OP_NAME MoeTutelCombineX
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
                    -mllvm
                    -cce-aicore-dcci-before-kernel-end=false
        )
    endif()

    if(NOT BUILD_OPS_RTY_KERNEL)
        # A2 已有同名基础实现；仅在包含 Ascend950 的构建中加入本次 Host/arch35 源，
        # 避免纯旧 SoC 构建注册新的 OpDef/InferShape/InferDataType。
        add_modules_sources_with_soc(OPTYPE moe_tutel_combine_x ACLNNTYPE aclnn_exclude)
    endif()
endif()
