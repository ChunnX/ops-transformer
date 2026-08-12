# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
        op_tiling/block_attn_res_update_def.cpp
    )

    add_ops_compile_options(
        OP_NAME BlockAttnResUpdate
        OPTIONS --cce-auto-sync=off
                -mllvm -cce-aicore-dcci-before-kernel-end=false
                -Wno-deprecated-declarations
                -Werror
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        OPTYPE block_attn_res_update
        ACLNNTYPE aclnn_exclude
    )

    # add_modules_sources treats every .cpp under op_tiling as a tiling source. Keep the requested source layout,
    # but compile the operator definition and shape inference in their dedicated host targets.
    set(BLOCK_ATTN_RES_UPDATE_DEF_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/block_attn_res_update_def.cpp)
    set(BLOCK_ATTN_RES_UPDATE_INFER_SRC
        ${CMAKE_CURRENT_SOURCE_DIR}/op_tiling/block_attn_res_update_infershape.cpp)
    get_target_property(BLOCK_ATTN_RES_UPDATE_TILING_SRCS ${OPHOST_NAME}_tiling_obj SOURCES)
    list(REMOVE_ITEM BLOCK_ATTN_RES_UPDATE_TILING_SRCS
        ${BLOCK_ATTN_RES_UPDATE_DEF_SRC}
        ${BLOCK_ATTN_RES_UPDATE_INFER_SRC})
    set_property(TARGET ${OPHOST_NAME}_tiling_obj PROPERTY SOURCES "${BLOCK_ATTN_RES_UPDATE_TILING_SRCS}")

    target_sources(${OPHOST_NAME}_opdef_aclnn_exclude_obj INTERFACE
        ${BLOCK_ATTN_RES_UPDATE_DEF_SRC})
    target_sources(${OPHOST_NAME}_infer_obj PRIVATE
        ${BLOCK_ATTN_RES_UPDATE_INFER_SRC})
endif()
