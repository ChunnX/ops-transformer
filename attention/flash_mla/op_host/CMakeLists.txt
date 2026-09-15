# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

set(FLASH_MLA_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging header for flash_mla")
set(FLASH_MLA_INNER_SKIP_HEADER TRUE CACHE INTERNAL "Skip packaging inner header for flash_mla")

if(BUILD_OPEN_PROJECT)
    set(flash_mla_depends attention/common attention/fused_infer_attention_score attention/incre_flash_attention
        attention/prompt_flash_attention CACHE INTERNAL "Dependencies for flash_mla")
    target_sources(op_host_aclnnInner PRIVATE flash_mla_def.cpp)
endif()
add_modules_sources(OPTYPE flash_mla ACLNNTYPE aclnn_inner)
add_tiling_modules()
include(${CMAKE_CURRENT_SOURCE_DIR}/arch22/flash_mla_value_tiling_sources.cmake)
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${FLASH_MLA_VALUE_SOURCES})
set_property(SOURCE ${FLASH_MLA_VALUE_SOURCES} TARGET_DIRECTORY ${OPHOST_NAME}_tiling_obj
    APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
add_ops_compile_options(OP_NAME FlashMla OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations)
