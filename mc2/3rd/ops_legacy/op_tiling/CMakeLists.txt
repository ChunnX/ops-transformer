# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

set(CMAKE_LIBRARY_OUTPUT_DIRECTOR ${PROJECT_SOURCE_DIR}/build)

set(MC2_LEGACY_DECOUPLED OFF)
if(ARCH_DIRECTORY)
    set(MC2_LEGACY_DECOUPLED ON)
    foreach(MC2_ARCH_DIR IN LISTS ARCH_DIRECTORY)
        string(REGEX MATCH "^arch([0-9]+)$" MC2_ARCH_MATCH "${MC2_ARCH_DIR}")
        if(NOT MC2_ARCH_MATCH OR CMAKE_MATCH_1 LESS 35)
            set(MC2_LEGACY_DECOUPLED OFF)
        endif()
    endforeach()
endif()

file(GLOB_RECURSE OP_TILING_FILES "*.cpp")
if(UT_TEST_ALL OR OP_HOST_UT)
    list(FILTER OP_TILING_FILES EXCLUDE REGEX "op_cache_tiling.cpp")
endif()
if(MC2_LEGACY_DECOUPLED)
    list(FILTER OP_TILING_FILES EXCLUDE REGEX "legacy_common_manager.cpp")
    target_compile_definitions(${OPHOST_NAME}_tiling_obj PRIVATE MC2_LEGACY_DECOUPLED=1)
endif()

target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${OP_TILING_FILES})
