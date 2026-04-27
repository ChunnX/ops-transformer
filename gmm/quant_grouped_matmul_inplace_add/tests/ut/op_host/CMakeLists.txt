# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_HOST_UT)
    add_modules_ut_sources(UT_NAME ${OP_TILING_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    add_modules_ut_sources(UT_NAME ${OP_INFERSHAPE_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})
    if(TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
        target_sources(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp)
        target_link_libraries(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
            $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>)
    endif()
    if(TARGET ${OP_TILING_MODULE_NAME}_cases)
        target_link_libraries(${OP_TILING_MODULE_NAME}_cases PRIVATE
            $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_a>>)
    endif()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
