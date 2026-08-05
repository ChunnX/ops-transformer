# ----------------------------------------------------------------------------
# This program is free software, you can redistribute it and/or modify it.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING
# BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (BUILD_TORCH_OPS)
    set(OPERATOR_NAME "lightning_attention")

    set(OPERATOR_TARGET "${OPERATOR_NAME}_objects")
    set(OPERATOR_CONFIG "${OPERATOR_NAME}:${OPERATOR_TARGET}" PARENT_SCOPE)
    
    set(OPERATOR_SOURCES "")

    list(APPEND OPERATOR_SOURCES
        ${CMAKE_CURRENT_SOURCE_DIR}/torch_interface.cpp
        
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/fwd_lo_cmm128.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/tiling/fwd_lo_cmm128_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/lightning_attn_cmm128_fwd.cpp

        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/bwd_lo_cmm128.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_host/tiling/bwd_lo_cmm128_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel/lightning_attn_cmm128_bwd.cpp
    )

    # Mark .cpp files with special properties
    set_source_files_properties(
        ${OPERATOR_SOURCES} PROPERTIES
        LANGUAGE CXX
        COMPILE_FLAGS "--npu-arch=dav-2201 -xasc -ltiling_api -lplatform -lregister"
    )
    
    add_library(${OPERATOR_TARGET} OBJECT ${OPERATOR_SOURCES})

    target_compile_options(${OPERATOR_TARGET} PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_include_directories(${OPERATOR_TARGET} PRIVATE 
        ${COMMON_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    return()
endif()
