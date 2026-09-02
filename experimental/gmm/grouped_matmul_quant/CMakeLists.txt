# ----------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if(BUILD_TORCH_OPS)
    set(OPERATOR_NAME "grouped_matmul_quant")
    message(STATUS "BUILD_TORCH_OPS ON in ${OPERATOR_NAME}")

    set(OPERATOR_TARGET "${OPERATOR_NAME}_objects")
    set(OPERATOR_CONFIG "${OPERATOR_NAME}:${OPERATOR_TARGET}" PARENT_SCOPE)

    add_library(${OPERATOR_TARGET} OBJECT grouped_matmul_quant_torch.cpp)
    target_compile_options(${OPERATOR_TARGET} PRIVATE ${COMMON_COMPILE_OPTIONS})
    target_include_directories(${OPERATOR_TARGET} PRIVATE ${COMMON_INCLUDE_DIRS})
    # torch_npu wheels bundle op-plugin below include/third_party/op-plugin,
    # while op-plugin headers include one another as "op_plugin/...".
    # Add that directory as an include root without changing the shared
    # npu_ops_transformer_ext build configuration.
    if(IS_DIRECTORY "${TORCH_NPU_INCLUDE_PATH}/third_party/op-plugin")
        target_include_directories(${OPERATOR_TARGET} PRIVATE
            "${TORCH_NPU_INCLUDE_PATH}/third_party/op-plugin")
    endif()
    return()
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
# The normal experimental build discovers this operator through op_host rather
# than this top-level file. Tests are therefore added from op_host/CMakeLists.txt
# as well. Keep the guarded fallback here for direct subdirectory builds.
list(REMOVE_ITEM CURRENT_DIRS tests)
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

if(ENABLE_TEST OR BENCHMARK)
    get_property(GROUPED_MATMUL_QUANT_TESTS_ADDED GLOBAL PROPERTY GROUPED_MATMUL_QUANT_TESTS_ADDED)
    if(NOT GROUPED_MATMUL_QUANT_TESTS_ADDED)
        set_property(GLOBAL PROPERTY GROUPED_MATMUL_QUANT_TESTS_ADDED TRUE)
        add_subdirectory(tests)
    endif()
endif()
