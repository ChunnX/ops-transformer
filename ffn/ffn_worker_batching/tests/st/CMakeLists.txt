# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------
# CMakeLists.txt for FfnWorkerBatching ST test (A5/arch35, 范围对齐 A2)
cmake_minimum_required(VERSION 3.10)

# 该 ST 工程为独立 standalone 工程（自带 project() + run.sh，需算子包安装后单独构建）。
# 当被 ops-transformer 聚合构建（如 op_host UT: bash build.sh --ophost_test）经上级
# tests/CMakeLists.txt 的 glob 递归 add_subdirectory 拉入时，算子包/aclnn 头未安装会导致
# 下方 REAL 模式 FATAL_ERROR，从而中断整个 UT 配置。此处仅在“作为子目录被父工程包含”时
# 提前跳过（standalone 构建时 CMAKE_SOURCE_DIR==CMAKE_CURRENT_SOURCE_DIR，行为逐字不变）。
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(STATUS "[ffn_worker_batching ST] skipped in aggregate build (standalone-only project)")
    return()
endif()

project(aclnn_ffn_worker_batching_test)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(USE_MOCK "Use mock aclnn for CPU golden self-test without real operator/NPU" OFF)

if(USE_MOCK)
    add_definitions(-DUSE_MOCK_ACLNN)
    message(STATUS "Building with MOCK (CPU golden self-test) - full validation without NPU")
else()
    message(STATUS "Building with REAL aclnn - NPU output vs CPU golden binary-exact comparison")
endif()

# Real 模式需要 CANN 环境（source set_env.sh 后 ASCEND_HOME_PATH 生效）
if(NOT USE_MOCK)
    if(NOT DEFINED ENV{ASCEND_HOME_PATH})
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH not set! Run 'source /root/Ascend/ascend-toolkit/set_env.sh' first.")
    endif()
    if(NOT EXISTS "$ENV{ASCEND_HOME_PATH}")
        message(FATAL_ERROR "ERROR: ASCEND_HOME_PATH directory does not exist!")
    endif()

    set(CANN_ROOT $ENV{ASCEND_HOME_PATH})
    set(ASCENDCL_INCLUDE_DIRS ${CANN_ROOT}/include)
    set(ASCENDCL_LIBRARIES ${CANN_ROOT}/lib64/libascendcl.so)
    set(CANN_LIBRARIES ${CANN_ROOT}/lib64/libopapi.so)
    message(STATUS "CANN_ROOT: ${CANN_ROOT}")
endif()

add_executable(test_aclnn_ffn_worker_batching test_aclnn_ffn_worker_batching.cpp)

if(NOT USE_MOCK)
    # 自定义算子 aclnn 头文件（算子包安装后生成，vendor 名以实际安装为准）
    find_path(CUSTOM_OP_INCLUDE_DIR
        NAMES aclnn_ffn_worker_batching.h
        PATHS
            ${CANN_ROOT}/opp/vendors/customize/op_api/include
            ${CANN_ROOT}/opp/vendors/ffn_worker_batching_custom/op_api/include
            /usr/local/Ascend/opp/vendors/customize/op_api/include
            $ENV{HOME}/Ascend/opp/vendors/customize/op_api/include
    )
    if(CUSTOM_OP_INCLUDE_DIR)
        message(STATUS "Found custom operator include dir: ${CUSTOM_OP_INCLUDE_DIR}")
    else()
        message(FATAL_ERROR "aclnn_ffn_worker_batching.h not found. Install the operator package first (source set_env.sh + run custom_opp_*.run).")
    endif()

    target_include_directories(test_aclnn_ffn_worker_batching PRIVATE
        ${ASCENDCL_INCLUDE_DIRS}
        ${CUSTOM_OP_INCLUDE_DIR}
    )

    find_library(CUSTOM_OP_LIBRARY cust_opapi
        PATHS
            ${CANN_ROOT}/opp/vendors/customize/op_api/lib
            ${CANN_ROOT}/opp/vendors/ffn_worker_batching_custom/op_api/lib
            /usr/local/Ascend/opp/vendors/customize/op_api/lib
            $ENV{HOME}/Ascend/opp/vendors/customize/op_api/lib
    )
    if(NOT CUSTOM_OP_LIBRARY)
        message(WARNING "cust_opapi library not found, relying on default opapi linkage")
    endif()

    target_link_libraries(test_aclnn_ffn_worker_batching
        ${ASCENDCL_LIBRARIES}
        ${CANN_LIBRARIES}
        ${CANN_ROOT}/lib64/libnnopbase.so
        ${CUSTOM_OP_LIBRARY}
    )
endif()

install(TARGETS test_aclnn_ffn_worker_batching DESTINATION bin)
