# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
message(STATUS "=== Debug: start ops.moe.moe_tutel_dispatch.op_host.CMakeLists.txt")

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        moe_tutel_dispatch_def.cpp
    )
    add_ops_compile_options(
        OP_NAME MoeTutelDispatch
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -mllvm -cce-aicore-dcci-before-kernel-end=false
                -Wno-deprecated-declarations
                -Werror
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    # The SoC-aware collector is required here: it selects arch35 Host
    # registrations only when ASCEND_COMPUTE_UNIT contains ascend950.  The OpDef
    # remains at the root so the custom schema can be generated, but its only
    # AICore config is AddConfig("ascend950").
    add_modules_sources_with_soc(OPTYPE moe_tutel_dispatch ACLNNTYPE aclnn_exclude)
endif()
