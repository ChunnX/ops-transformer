# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
# CMakeLists.txt — AlltoAllMatmulV2 aclnn 直调测试
#
# 前置条件:
#   1. CANN 环境已设置 (source set_env.sh)
#   2. 算子已完成 opbuild 安装
# ============================================================

cmake_minimum_required(VERSION 3.14)
project(test_aclnn_allto_all_matmul_v2)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# CANN 环境（默认从环境变量读取）
if(NOT DEFINED ASCEND_HOME_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_HOME_PATH $ENV{ASCEND_HOME_PATH})
    else()
        message(FATAL_ERROR "ASCEND_HOME_PATH not set")
    endif()
endif()

# 自动检测 CPU 架构
set(HOST_ARCH "x86_64")
set(HOST_ARCH_DIR "x86_64-linux")
if(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|ARM64")
    set(HOST_ARCH "aarch64")
    set(HOST_ARCH_DIR "aarch64-linux")
endif()

# 算子仓库根目录（examples/ 的祖父目录是 allto_all_matmul_v2/ 的祖父目录 → repo root）
get_filename_component(REPO_ROOT ${CMAKE_CURRENT_SOURCE_DIR}/../../.. ABSOLUTE)

set(CANN_INCLUDE          ${ASCEND_HOME_PATH}/include)
set(CANN_ARCH_INCLUDE     ${ASCEND_HOME_PATH}/${HOST_ARCH_DIR}/include)    # HCCL 内部头文件
set(CANN_FWK_INCLUDE      ${ASCEND_HOME_PATH}/fwkacllib/include)
set(CANN_VENDOR_INCLUDE   ${ASCEND_HOME_PATH}/vendors/custom_transformer/op_api/include)
set(CANN_TIK_INCLUDE      ${ASCEND_HOME_PATH}/compiler/tikcpp/tikcfw)

set(APACE_KERNEL_ROOT     ${REPO_ROOT}/mc2/common/op_kernel)           # apace/ 头文件
set(APACE_COMMON_UTILS    ${APACE_KERNEL_ROOT}/apace/utils)           # CommChannelBuilder
set(APACE_ST_UTILS        ${APACE_KERNEL_ROOT}/apace/tests/st/utils)  # RootInfoExchanger 等
set(APACE_ST_SRC          ${APACE_KERNEL_ROOT}/apace/tests/st)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CANN_INCLUDE}
    ${CANN_ARCH_INCLUDE}
    ${CANN_FWK_INCLUDE}
    ${CANN_VENDOR_INCLUDE}
    ${CANN_TIK_INCLUDE}
    ${APACE_KERNEL_ROOT}
    ${APACE_COMMON_UTILS}
    ${APACE_ST_UTILS}
    ${APACE_ST_SRC}
)

add_executable(test_aclnn_allto_all_matmul_v2
    test_aclnn_allto_all_matmul_v2.cpp
)

target_link_directories(test_aclnn_allto_all_matmul_v2 PRIVATE
    ${ASCEND_HOME_PATH}/lib64
    ${ASCEND_HOME_PATH}/fwkacllib/lib64
    ${ASCEND_HOME_PATH}/runtime/lib64
    ${ASCEND_HOME_PATH}/${HOST_ARCH_DIR}/lib64
    ${ASCEND_HOME_PATH}/vendors/custom_transformer/op_api/lib
    ${ASCEND_HOME_PATH}/vendors/custom_transformer/op_proto/lib/linux/${HOST_ARCH}
)

target_link_libraries(test_aclnn_allto_all_matmul_v2 PRIVATE
    hccl_fwk hccl ascendcl nnopbase c_sec rt dl pthread
    cust_opapi cust_opsproto_rt2.0 tiling_api platform
)
