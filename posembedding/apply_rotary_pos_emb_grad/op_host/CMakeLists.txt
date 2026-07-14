# Copyright (c) 2026 Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

add_ops_compile_options(
        OP_NAME ApplyRotaryPosEmbGrad
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
)

set(apply_rotary_pos_emb_grad_depends ops-transformer/posembedding/apply_rotary_pos_emb_grad PARENT_SCOPE)
set(COMMON_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include)
# 导出到父作用域
set(COMMON_INCLUDE_DIR ${COMMON_INCLUDE_DIR} PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        apply_rotary_pos_emb_grad_def.cpp
)

target_sources(optiling PRIVATE
        apply_rotary_pos_emb_grad_tiling.cpp
        apply_rotary_pos_emb_grad_half_tiling.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        apply_rotary_pos_emb_grad_tiling.cpp
        apply_rotary_pos_emb_grad_half_tiling.cpp
    )
endif ()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${COMMON_INCLUDE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../utils
)
