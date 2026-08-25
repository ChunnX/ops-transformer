# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if (BUILD_OPEN_PROJECT)
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        target_sources(op_host_aclnnExc PRIVATE
            moe_token_unpermute_with_ep_def.cpp
        )
        add_ops_compile_options(
            OP_NAME MoeTokenUnpermuteWithEp
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
                    -mllvm
                    -cce-aicore-dcci-before-kernel-end=false
        )
    else()
        target_sources(op_host_aclnn PRIVATE
            moe_token_unpermute_with_ep_def.cpp
        )
        add_ops_compile_options(
            OP_NAME MoeTokenUnpermuteWithEp
            OPTIONS --cce-auto-sync=on
                    -Wno-deprecated-declarations
                    -Werror
        )
    endif()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_modules_sources(
            OPTYPE moe_token_unpermute_with_ep
            ACLNNTYPE aclnn_exclude
            OP_API_INDEPENDENT ON
            OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/op_api/ascend950
        )
        set_source_files_properties(
            ${CMAKE_CURRENT_SOURCE_DIR}/moe_token_unpermute_with_ep_tiling.cpp
            PROPERTIES COMPILE_DEFINITIONS MOE_TOKEN_UNPERMUTE_WITH_EP_ASCEND950=1
        )
    else()
        add_modules_sources(OPTYPE moe_token_unpermute_with_ep ACLNNTYPE aclnn)
    endif()
endif()
