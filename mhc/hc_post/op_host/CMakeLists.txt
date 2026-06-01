# ----------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        hc_post_def.cpp
    )

    target_sources(op_host_aclnnExc PRIVATE
        hc_post_def.cpp
    )

    add_ops_compile_options(
        OP_NAME HcPost
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE hc_post ACLNNTYPE aclnn)
    get_target_property(OPHOST_TILING_SRCS ${OPHOST_NAME}_tiling_obj SOURCES)
    list(REMOVE_ITEM OPHOST_TILING_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/hc_post_tiling_arch35.cpp)
    set_property(TARGET ${OPHOST_NAME}_tiling_obj PROPERTY SOURCES ${OPHOST_TILING_SRCS})
endif()
