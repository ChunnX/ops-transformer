# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

message(STATUS "=== Debug: start ops.transformer.mhc.ai_infra_sinkhorn_grad.op_host.CMakeLists.txt ")

add_ops_compile_options(
        OP_NAME AiInfraSinkhornGrad
        OPTIONS --cce-auto-sync=ON
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
                -O2
)

set(ai_infra_sinkhorn_grad_depends ops-transformer/mhc/ai_infra_sinkhorn_grad PARENT_SCOPE)

target_sources(op_host_aclnn PRIVATE
        ai_infra_sinkhorn_grad_def.cpp
)

target_sources(optiling PRIVATE
        ai_infra_sinkhorn_grad_tiling.cpp
        ai_infra_sinkhorn_grad_tiling_base.cpp
)

if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        ai_infra_sinkhorn_grad_tiling.cpp
        ai_infra_sinkhorn_grad_tiling_base.cpp
    )
endif()

target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include/tiling_base
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../common/include
)
