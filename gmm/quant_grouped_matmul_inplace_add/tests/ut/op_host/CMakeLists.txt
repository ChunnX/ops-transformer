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

    set(_QGMMIA_GROUPED_TILING_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_matmul_tiling.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../grouped_matmul/op_host/op_tiling/arch35/grouped_quant_basic_api_matmul_tiling.cpp)

    if(TARGET ${OPHOST_NAME}_tiling_obj)
        get_target_property(_QGMMIA_EXISTING_TILING_SRCS ${OPHOST_NAME}_tiling_obj SOURCES)
        if(NOT _QGMMIA_EXISTING_TILING_SRCS)
            set(_QGMMIA_EXISTING_TILING_SRCS)
        endif()

        set(_QGMMIA_MISSING_TILING_SRCS)
        foreach(_SRC IN LISTS _QGMMIA_GROUPED_TILING_SRCS)
            get_filename_component(_ABS_SRC "${_SRC}" REALPATH)
            list(FIND _QGMMIA_EXISTING_TILING_SRCS "${_ABS_SRC}" _FOUND_INDEX)
            if(_FOUND_INDEX EQUAL -1)
                list(APPEND _QGMMIA_MISSING_TILING_SRCS "${_ABS_SRC}")
            endif()
        endforeach()

        if(_QGMMIA_MISSING_TILING_SRCS)
            target_sources(${OPHOST_NAME}_tiling_obj PRIVATE ${_QGMMIA_MISSING_TILING_SRCS})
        endif()
        target_link_libraries(${OPHOST_NAME}_tiling_obj PRIVATE
            $<$<BOOL:${alog_FOUND}>:$<BUILD_INTERFACE:alog_headers>>)
    endif()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()
