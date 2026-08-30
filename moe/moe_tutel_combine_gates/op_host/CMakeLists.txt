# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
set(SUPPORT_COMPUTE_UNIT "ascend950")
set(SUPPORT_TILING_DIR "arch35")
set(DISABLE_TILING_IN_OPP TRUE)

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        moe_tutel_combine_gates_def.cpp
    )
    if("${CONDITION_UNIT}" STREQUAL "${SUPPORT_COMPUTE_UNIT}")
        add_ops_compile_options(
            OP_NAME MoeTutelCombineGates
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
        )
    endif()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE moe_tutel_combine_gates ACLNNTYPE aclnn_exclude)

    # ops-transformer's current add_modules_sources() does not yet accept the
    # COMPUTE_UNIT/TILING_DIR/DISABLE_IN_OPP arguments available in ops-nn.
    # Apply the same build-time selection locally: standalone Ascend 950 builds
    # collect arch35, while canndev in-tree and earlier-product builds do not.
    list(FIND ASCEND_COMPUTE_UNIT "${SUPPORT_COMPUTE_UNIT}" SUPPORT_UNIT_INDEX)
    if(NOT SUPPORT_UNIT_INDEX EQUAL -1 AND
       (BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG OR NOT DISABLE_TILING_IN_OPP))
        add_tiling_modules()
        file(GLOB MTCG_TILING_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/${SUPPORT_TILING_DIR}/*_tiling*.cpp)
        target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${MTCG_TILING_SRCS})
        target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../
            ${CMAKE_CURRENT_SOURCE_DIR})
    endif()
endif()
