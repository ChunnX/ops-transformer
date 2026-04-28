# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(UT_TEST_ALL OR OP_API_UT)
    add_modules_ut_sources(UT_NAME ${OP_API_MODULE_NAME} MODE PRIVATE DIR ${CMAKE_CURRENT_SOURCE_DIR})

    set(_QGMMIA_GROUPED_OPAPI_SRCS
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../grouped_matmul/op_api/grouped_matmul_950_checker.cpp
        ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../grouped_matmul/op_api/grouped_matmul_util.cpp)

    if(TARGET ${OPHOST_NAME}_opapi_obj)
        get_target_property(_QGMMIA_EXISTING_OPAPI_SRCS ${OPHOST_NAME}_opapi_obj SOURCES)
        if(NOT _QGMMIA_EXISTING_OPAPI_SRCS)
            set(_QGMMIA_EXISTING_OPAPI_SRCS)
        endif()

        set(_QGMMIA_MISSING_OPAPI_SRCS)
        foreach(_SRC IN LISTS _QGMMIA_GROUPED_OPAPI_SRCS)
            get_filename_component(_ABS_SRC "${_SRC}" REALPATH)
            list(FIND _QGMMIA_EXISTING_OPAPI_SRCS "${_ABS_SRC}" _FOUND_INDEX)
            if(_FOUND_INDEX EQUAL -1)
                list(APPEND _QGMMIA_MISSING_OPAPI_SRCS "${_ABS_SRC}")
            endif()
        endforeach()

        target_include_directories(${OPHOST_NAME}_opapi_obj PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/../../../../../grouped_matmul/op_api)
        if(_QGMMIA_MISSING_OPAPI_SRCS)
            target_sources(${OPHOST_NAME}_opapi_obj PRIVATE ${_QGMMIA_MISSING_OPAPI_SRCS})
        endif()
    endif()
endif()
