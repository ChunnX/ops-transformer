# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# aclnn 直调示例构建（需先 bash ../build.sh --soc=ascend910b 并安装算子包）
cmake_minimum_required(VERSION 3.10)
project(mla_prolog_v5_example)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT DEFINED ENV{ASCEND_HOME_PATH})
    message(FATAL_ERROR "ASCEND_HOME_PATH environment variable is not set!")
endif()
set(CANN_ROOT $ENV{ASCEND_HOME_PATH})

find_path(CUSTOM_OP_INCLUDE_DIR
    NAMES aclnn_mla_prolog_v5_weight_nz.h
    PATHS
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_api
        ${CANN_ROOT}/opp/vendors/mla_prolog_v5_custom/op_api/include
        $ENV{HOME}/Ascend/opp/vendors/mla_prolog_v5_custom/op_api/include
)
if(NOT CUSTOM_OP_INCLUDE_DIR)
    message(FATAL_ERROR "aclnn_mla_prolog_v5_weight_nz.h not found")
endif()

find_library(CUSTOM_OP_LIBRARY libcust_opapi.so
    PATHS
        ${CANN_ROOT}/opp/vendors/mla_prolog_v5_custom/op_api/lib
        $ENV{HOME}/Ascend/opp/vendors/mla_prolog_v5_custom/op_api/lib
)
if(NOT CUSTOM_OP_LIBRARY)
    message(FATAL_ERROR "libcust_opapi.so not found")
endif()

add_executable(test_aclnn_mla_prolog_v5 test_aclnn_mla_prolog_v5.cpp)

target_include_directories(test_aclnn_mla_prolog_v5 PRIVATE
    ${CUSTOM_OP_INCLUDE_DIR}
    ${CANN_ROOT}/aarch64-linux/include
    ${CANN_ROOT}/aarch64-linux/include/aclnn
    ${CANN_ROOT}/aarch64-linux/include/op_common
    ${CANN_ROOT}/include
)

get_filename_component(CUSTOM_OP_LIB_DIR ${CUSTOM_OP_LIBRARY} DIRECTORY)
target_link_libraries(test_aclnn_mla_prolog_v5 PRIVATE
    ${CANN_ROOT}/lib64/libascendcl.so
    ${CANN_ROOT}/lib64/libopapi.so
    ${CANN_ROOT}/lib64/libnnopbase.so
    ${CANN_ROOT}/lib64/libruntime.so
    ${CUSTOM_OP_LIBRARY}
)
target_link_options(test_aclnn_mla_prolog_v5 PRIVATE
    "-Wl,-rpath,${CANN_ROOT}/lib64"
    "-Wl,-rpath,${CUSTOM_OP_LIB_DIR}"
)
