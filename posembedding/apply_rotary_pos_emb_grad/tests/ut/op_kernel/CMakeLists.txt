# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (UT_TEST_ALL OR OP_KERNEL_UT)
    set(opName apply_rotary_pos_emb_grad)
    get_filename_component(UT_DIR ${CMAKE_CURRENT_SOURCE_DIR} DIRECTORY)
    get_filename_component(TESTS_DIR ${UT_DIR} DIRECTORY)
    get_filename_component(OP_NAME_DIR ${TESTS_DIR} DIRECTORY)
    get_filename_component(OP_NAME ${OP_NAME_DIR} NAME)
    list(FIND ASCEND_OP_NAME ${OP_NAME} INDEX)
    if(NOT "${ASCEND_OP_NAME}" STREQUAL "ALL" AND INDEX EQUAL -1)
        return()
    endif()

    list(FIND ARCH_DIRECTORY "arch35" INDEX)
    if(NOT INDEX EQUAL -1)
        return()
    endif()

    set(oriSocVersion Ascend950DT_9582)
    set(socVersion Ascend950DT_9582)
    list(REMOVE_ITEM fastOpTestSocVersions ascend950DT_9582)
    set(fastOpTestSocVersions
        ${fastOpTestSocVersions}
        CACHE STRING "fastOp Test SocVersions" FORCE
    )
    file(GLOB OPKERNEL_CASES_SRC ${CMAKE_CURRENT_SOURCE_DIR}/test_${opName}*.cpp)

    add_library(${opName}_${socVersion}_cases_obj OBJECT ${OPKERNEL_CASES_SRC})
    target_compile_options(
        ${opName}_${socVersion}_cases_obj PRIVATE
        -g
        -include "${CMAKE_CURRENT_SOURCE_DIR}/${opName}_tiling.h"
        -DUT_SOC_VERSION="${socVersion}"
    )
    target_include_directories(
        ${opName}_${socVersion}_cases_obj
        PRIVATE ${ASCEND_DIR}/include/base/context_builder
                ${OPBASE_INC_DIRS}
                ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/op_kernel
                ${PROJECT_SOURCE_DIR}/tests/ut/framework_normal/common
                ${PROJECT_SOURCE_DIR}/common/include/op_kernel
                ${CMAKE_CURRENT_SOURCE_DIR}/../../../op_kernel
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/impl/basic_api
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/basic_api
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/asc/include/adv_api
                ${ASCEND_DIR}/${SYSTEM_PREFIX}/include/ascendc/highlevel_api
    )
    target_link_libraries(
        ${opName}_${socVersion}_cases_obj PRIVATE $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
                                                  tikicpulib::${socVersion}
                                                  gtest
    )

    if(NOT TARGET ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj)
        add_library(
            ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj OBJECT
            $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
        )
    else()
        target_sources(
            ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj PRIVATE
            $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
        )
    endif()
    add_dependencies(
        ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj ${opName}_${socVersion}_cases_obj
    )
    target_link_libraries(
        ${OP_KERNEL_MODULE_NAME}_${oriSocVersion}_cases_obj PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        $<TARGET_OBJECTS:${opName}_${socVersion}_cases_obj>
    )

    list(FIND fastOpTestSocVersions "${oriSocVersion}" index)
    if(index EQUAL -1)
        set(fastOpTestSocVersions
            ${fastOpTestSocVersions} ${oriSocVersion}
            CACHE STRING "fastOp Test SocVersions" FORCE
        )
    endif()
endif()
