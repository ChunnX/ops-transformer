# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
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

    list(FIND ASCEND_OP_NAME quant_lightning_indexer_v2 quant_lightning_indexer_v2_index)
    if("${ASCEND_OP_NAME}" STREQUAL "ALL" OR NOT quant_lightning_indexer_v2_index EQUAL -1)
        target_sources(${OP_TILING_MODULE_NAME}_cases_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/test_quant_lightning_indexer_v2_checker.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/checkers/qliv2_checker.cpp
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_host/checkers/quant_checker.cpp)
        include(${CMAKE_CURRENT_SOURCE_DIR}/../../../../lightning_indexer_v2/op_host/checkers/checker_sources.cmake)
        add_indexer_v2_common_checker_sources(${OP_TILING_MODULE_NAME}_cases_obj)
    endif()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
