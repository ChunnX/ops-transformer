# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_ops_compile_options(
        OP_NAME AiInfraKvRmsNormRopeCache
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
)
set(ai_infra_kv_rmsnorm_rope_cache_depends transformer/posembedding/ai_infra_kv_rms_norm_rope_cache PARENT_SCOPE)
target_sources(op_host_aclnnExc PRIVATE
        op_host/ai_infra_kv_rms_norm_rope_cache_def.cpp
)
target_sources(optiling PRIVATE
        op_host/ai_infra_kv_rms_norm_rope_cache_base_tiling.cpp
        op_host/ai_infra_kv_rms_norm_rope_cache_ds_tiling.cpp
)
if (NOT BUILD_OPEN_PROJECT)
    target_sources(opmaster_ct PRIVATE
        op_host/ai_infra_kv_rms_norm_rope_cache_base_tiling.cpp
        op_host/ai_infra_kv_rms_norm_rope_cache_ds_tiling.cpp
    )
endif ()
target_include_directories(optiling PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host
        ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
)
target_sources(opapi PRIVATE
        op_api/aclnn_ai_infra_kv_rms_norm_rope_cache.cpp
        op_api/ai_infra_kv_rms_norm_rope_cache.cpp
)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/op_api/aclnn_ai_infra_kv_rms_norm_rope_cache.h
        DESTINATION ${ACLNN_INC_INSTALL_DIR} OPTIONAL
)

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

if (NOT BUILD_OPS_RTY_KERNEL)
    file(GLOB OPAPI_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/op_api/*.cpp)
    message(STATUS "=== Debug: OPAPI_SRCS =${OPAPI_SRCS}")
    if (OPAPI_SRCS)
        add_opapi_modules()
        target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${OPAPI_SRCS})
    endif()
endif()