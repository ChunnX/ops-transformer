# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

get_filename_component(MLA_ATTENTION_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../.. ABSOLUTE)
include(${MLA_ATTENTION_ROOT}/flash_mla/op_host/arch22/flash_mla_value_tiling_sources.cmake)
set(MLA_VALUE_INCLUDES
    ${OPBASE_SOURCE_PATH}/include/op_common
    ${OPBASE_SOURCE_PATH}/include
    ${MLA_ATTENTION_ROOT}/../common/include
    ${MLA_ATTENTION_ROOT}/flash_mla/op_host/arch22
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/experiment
    ${ASCEND_CANN_PACKAGE_PATH}/include/ascendc/host_api
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/base
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/op_common)

if(BUILD_WITH_INSTALLED_DEPENDENCY_CANN_PKG)
    if(NOT (UT_TEST_ALL OR OP_HOST_UT OR OP_KERNEL_AICPU_UT))
        add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
        set(CMAKE_CXX_COMPILER ${ASCEND_DIR}/toolkit/toolchain/hcc/bin/aarch64-target-linux-gnu-g++)
    endif()
    file(GLOB_RECURSE JSON_FILE ${CMAKE_CURRENT_SOURCE_DIR}/*.json)
    set_property(GLOBAL APPEND PROPERTY AICPU_JSON_FILES ${JSON_FILE})
    file(GLOB AICPU_SRC ${CMAKE_CURRENT_SOURCE_DIR}/*_aicpu*.cpp)
    # The repository link command expands one object per registered target.
    foreach(SOURCE_FILE IN LISTS AICPU_SRC FLASH_MLA_VALUE_SOURCES)
        get_filename_component(SOURCE_NAME ${SOURCE_FILE} NAME_WE)
        set(OBJ_NAME ${SOURCE_NAME}_cust_obj)
        add_aicpu_cust_kernel_modules(${OBJ_NAME})
        target_sources(${OBJ_NAME} PRIVATE ${SOURCE_FILE})
        target_include_directories(${OBJ_NAME} PRIVATE ${MLA_VALUE_INCLUDES})
        # Original floating-point rounding/tie-breaks affect work assignment.
        target_compile_options(${OBJ_NAME} PRIVATE -std=c++17 -fno-fast-math -ffp-contract=off)
    endforeach()
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    # The AICPU-only switch does not load the Host UT framework interfaces.
    if(NOT TARGET intf_llt_pub_asan_cxx17)
        include(${PROJECT_SOURCE_DIR}/cmake/intf_pub_llt_gccnative.cmake)
    endif()
    function(add_flash_mla_metadata_aicpu_test_case)
        # AddAicpuOpTestCase uses an empty selection for all operators.
        if("${ASCEND_OP_NAME}" STREQUAL "ALL")
            set(ASCEND_OP_NAME "")
        endif()
        if(NOT AICPU_OP_KERNEL_MODULE_NAME)
            set(AICPU_OP_KERNEL_MODULE_NAME ${PKG_NAME}_op_kernel_aicpu_ut)
        endif()
        AddAicpuOpTestCase(flash_mla_metadata)
    endfunction()
    add_flash_mla_metadata_aicpu_test_case()
endif()
