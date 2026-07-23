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
if ("${CONDITION_UNIT}" STREQUAL "ascend910_93" OR "${CONDITION_UNIT}" STREQUAL "ascend910b")
    set(MEGA_MOE_DEF_SOURCE arch22/mega_moe_def.cpp)
    set(MEGA_MOE_OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api/arch22)
    set(MEGA_MOE_OPTYPE mega_moe_a2a3)
    set(MEGA_MOE_A2A3_BUILD TRUE)
else()
    set(MEGA_MOE_DEF_SOURCE mega_moe_def.cpp)
    set(MEGA_MOE_OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api)
    set(MEGA_MOE_OPTYPE mega_moe)
    set(MEGA_MOE_A2A3_BUILD FALSE)
endif()

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        ${MEGA_MOE_DEF_SOURCE}
    )
    add_modules_sources_with_soc(
        OP_API_INDEPENDENT ON
        OP_API_DIR ${MEGA_MOE_OP_API_DIR}
        OP_MC2_ENABLE ON
        OPTYPE ${MEGA_MOE_OPTYPE} ACLNNTYPE aclnn_inner)
    if (MEGA_MOE_A2A3_BUILD)
        target_sources(${OPHOST_NAME}_opdef_aclnn_inner_obj INTERFACE
            ${CMAKE_CURRENT_SOURCE_DIR}/${MEGA_MOE_DEF_SOURCE}
        )
    endif()
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(mega_moe_depends mc2/common mc2/3rd mc2/moe_distribute_dispatch_v2 mc2/moe_distribute_dispatch_v3 PARENT_SCOPE)
    set(mega_moe_apt_depends mc2/common mc2/3rd mc2/moe_distribute_dispatch_v2 mc2/moe_distribute_dispatch_v3 PARENT_SCOPE)

    if ("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME MegaMoe
            OPTIONS --cce-auto-sync=off
        )
    else()
        add_ops_compile_options(
            OP_NAME MegaMoe
            OPTIONS --cce-auto-sync=on
            OPTIONS -I${CMAKE_SOURCE_DIR}/mc2/common/op_kernel
            OPTIONS -I${CMAKE_SOURCE_DIR}/mc2/common/inc/kernel
        )
    endif()

else() # 回黄host
    add_mc2_modules_sources(OPTYPE mega_moe ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
