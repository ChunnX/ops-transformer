# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(COMMAND add_op_to_compiled_list)
    add_op_to_compiled_list()
endif()

if (BUILD_OPEN_PROJECT)
    if(TARGET op_host_aclnn)
        target_sources(op_host_aclnn PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn/sage_attn_host.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_int8.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_fp8.cpp
        )
        target_include_directories(op_host_aclnn PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn
            ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
            ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/catlass
            ${CMAKE_CURRENT_SOURCE_DIR}/../quant
        )
    endif()
endif()

if(COMMAND add_ops_compile_options)
    add_ops_compile_options(
        OP_NAME SageAttention2
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
    )
endif()

if (NOT BUILD_OPS_RTY_KERNEL AND COMMAND add_modules_sources)
    add_modules_sources(OPTYPE sage_attention2 ACLNNTYPE aclnn)
endif()

# 支持独立构建 Host 共享库以供本地单测与集成测试
if(NOT TARGET op_host_aclnn AND NOT TARGET sageattention2_host)
    add_library(sageattention2_host SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn/sage_attn_host.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_int8.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../quant/quant_fp8.cpp
    )
    target_include_directories(sageattention2_host PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/../aclnn
        ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/ascendc
        ${CMAKE_CURRENT_SOURCE_DIR}/../sageattn_fp8/catlass
        ${CMAKE_CURRENT_SOURCE_DIR}/../quant
    )
    target_compile_features(sageattention2_host PUBLIC cxx_std_17)
endif()
