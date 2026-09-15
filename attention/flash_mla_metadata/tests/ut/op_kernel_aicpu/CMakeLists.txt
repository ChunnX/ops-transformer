# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(NOT (UT_TEST_ALL OR OP_HOST_UT OR OP_KERNEL_AICPU_UT))
    return()
endif()
if(NOT "${ASCEND_OP_NAME}" STREQUAL "" AND NOT "${ASCEND_OP_NAME}" STREQUAL "ALL"
   AND NOT "flash_mla_metadata" IN_LIST ASCEND_OP_NAME)
    return()
endif()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    # Registered by op_kernel_aicpu/CMakeLists.txt through AddAicpuOpTestCase.
    set(MLA_METADATA_CASES_TARGET flash_mla_metadata_cases_obj)
else()
    # Preserve the existing Host-only entry. Combined builds use the dedicated
    # AICPU target so the producer and value chain are linked only once.
    if(NOT TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
        add_library(${OP_TILING_MODULE_NAME}_cases_obj OBJECT)
    endif()
    set(MLA_METADATA_CASES_TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
    target_sources(${MLA_METADATA_CASES_TARGET} PRIVATE
        test_flash_mla_metadata_aicpu.cpp
        ../../../op_kernel_aicpu/flash_mla_metadata_aicpu.cpp)
endif()

target_include_directories(${MLA_METADATA_CASES_TARGET} PRIVATE
    ${OP_TILING_INCLUDE}
    ${AICPU_INCLUDE}
    ${ASCEND_DIR}/pkg_inc/aicpu_common/context/common
    ${ASCEND_DIR}/pkg_inc/aicpu_common/context/cpu_proto
    ${ASCEND_DIR}/pkg_inc/aicpu_common/context/utils
    ${ASCEND_DIR}/pkg_inc/aicpu
    ${ASCEND_DIR}/include/experiment
    ${ASCEND_DIR}/include/experiment/cce
    ${ASCEND_DIR}/include/experiment/msprof)
target_link_libraries(${MLA_METADATA_CASES_TARGET} PUBLIC
    ${ASCEND_DIR}/lib64/libaicpu_context_host.a
    ${ASCEND_DIR}/lib64/libaicpu_nodedef_host.a
    ${ASCEND_DIR}/lib64/libhost_ascend_protobuf.a
    c_sec
    PRIVATE $<$<BOOL:${dlog_FOUND}>:$<BUILD_INTERFACE:dlog_headers>>)

# Keep the same source combination as the AICPU production target. The Host
# shared library hides these internal entry points, so it cannot supply them.
get_filename_component(MLA_METADATA_VALUE_DIR
    ${CMAKE_CURRENT_SOURCE_DIR}/../../../../flash_mla/op_host/arch22 ABSOLUTE)
include(${MLA_METADATA_VALUE_DIR}/flash_mla_value_tiling_sources.cmake)
set(MLA_METADATA_VALUE_TARGETS ${MLA_METADATA_CASES_TARGET})
if((UT_TEST_ALL OR OP_HOST_UT) AND (UT_TEST_ALL OR OP_KERNEL_AICPU_UT))
    # FlashMla Host cases also call the value entry directly. Each executable
    # needs its own value objects, while the AICPU cases remain in one target.
    if(NOT TARGET ${OP_TILING_MODULE_NAME}_cases_obj)
        add_library(${OP_TILING_MODULE_NAME}_cases_obj OBJECT)
    endif()
    list(APPEND MLA_METADATA_VALUE_TARGETS ${OP_TILING_MODULE_NAME}_cases_obj)
endif()
foreach(VALUE_TARGET IN LISTS MLA_METADATA_VALUE_TARGETS)
    target_sources(${VALUE_TARGET} PRIVATE ${FLASH_MLA_VALUE_SOURCES})
    set_property(SOURCE ${FLASH_MLA_VALUE_SOURCES} TARGET_DIRECTORY ${VALUE_TARGET}
        APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
endforeach()

if(UT_TEST_ALL OR OP_KERNEL_AICPU_UT)
    # AddAicpuOpTestCase registers objects; provide the executable for this op.
    add_executable(flash_mla_metadata_aicpu_ut)
    target_link_directories(flash_mla_metadata_aicpu_ut PRIVATE ${ASCEND_DIR}/lib64)
    target_link_libraries(flash_mla_metadata_aicpu_ut PRIVATE
        $<BUILD_INTERFACE:intf_llt_pub_asan_cxx17>
        ${MLA_METADATA_CASES_TARGET}
        GTest::gtest_main
        $<$<TARGET_EXISTS:opbase_util_objs>:$<TARGET_OBJECTS:opbase_util_objs>>
        tiling_api
        metadef
        register
        platform
        ascendalog
        unified_dlog)
    if(ENABLE_UT_EXEC)
        add_custom_command(TARGET flash_mla_metadata_aicpu_ut POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E env "LD_LIBRARY_PATH=$ENV{LD_LIBRARY_PATH}"
                $<TARGET_FILE:flash_mla_metadata_aicpu_ut>
            COMMENT "Run FlashMlaMetadata AICPU utest"
            VERBATIM)
    endif()
endif()
