# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
    moe_fused_topk_def.cpp
    )
    add_ops_compile_options(
            OP_NAME MoeFusedTopk
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    # The Ascend950 tiling entry lives alongside the default tiling entry.
    # Package builds configure one SoC at a time, so compile only the tiling
    # implementation for the selected target.
    list(FIND ASCEND_COMPUTE_UNIT "ascend950" MOE_FUSED_TOPK_ASCEND950_INDEX)
    if(NOT MOE_FUSED_TOPK_ASCEND950_INDEX EQUAL -1)
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/moe_fused_topk_tiling.cpp
            PROPERTIES HEADER_FILE_ONLY TRUE
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/moe_fused_topk_tiling_arch35.cpp
            PROPERTIES HEADER_FILE_ONLY FALSE
        )
    else()
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/moe_fused_topk_tiling.cpp
            PROPERTIES HEADER_FILE_ONLY FALSE
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/moe_fused_topk_tiling_arch35.cpp
            PROPERTIES HEADER_FILE_ONLY TRUE
        )
    endif()
    add_modules_sources(OPTYPE moe_fused_topk ACLNNTYPE aclnn_exclude)
endif()
