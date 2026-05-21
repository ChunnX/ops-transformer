# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

message(STATUS "=== Debug: start ops.activation.swiglu_gated_mlp.op_host.CMakeLists.txt ")
add_modules_sources(
    HOSTNAME ${OPHOST_NAME}
    MODE PRIVATE
    DIR ${CMAKE_CURRENT_SOURCE_DIR}
    OPTYPE swiglu_gated_mlp
    ACLNNTYPE aclnn_exclude
)

function(swiglu_gated_mlp_link_opapi)
    if(TARGET cust_opapi)
        target_link_libraries(cust_opapi PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64/libopapi.so)
    endif()
    if(TARGET ${OPAPI_NAME})
        target_link_libraries(${OPAPI_NAME} PRIVATE ${ASCEND_DIR}/${SYSTEM_PREFIX}/lib64/libopapi.so)
    endif()
endfunction()

cmake_language(DEFER DIRECTORY ${PROJECT_SOURCE_DIR} CALL swiglu_gated_mlp_link_opapi)
