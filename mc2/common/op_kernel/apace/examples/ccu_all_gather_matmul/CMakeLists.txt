# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16.0)
find_package(ASC REQUIRED)
project(hccl_examples_custom_ops_allgather_ccu_direct LANGUAGES CXX ASC)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_OPP_PATH})
        get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
    else()
        message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH or ASCEND_HOME_PATH must be set")
    endif()
endif()

# ---- ops-tensor 路径（TODO: 硬编码路径，跨环境需手动修改）----
# 注意：tensor_api 是 ops-tensor 的 submodule，首次构建前需在 ops-tensor 目录执行：
#   git submodule update --init --recursive
set(OPS_TENSOR_ROOT "$ENV{HOME}/code/ops-tensor")
if(NOT EXISTS "${OPS_TENSOR_ROOT}/include/blaze")
    message(FATAL_ERROR "ops-tensor not found at ${OPS_TENSOR_ROOT}/include/blaze, "
        "please clone ops-tensor (git clone https://gitcode.com/cann/ops-tensor.git) "
        "and run: cd ${OPS_TENSOR_ROOT} && git submodule update --init --recursive")
endif()

# ---- 第二阶段新增：架构相关目录（platform/tiling_api/register 库与 platform_ascendc.h 头文件）----
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_ARCH_DIR "x86_64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ASCEND_ARCH_DIR "aarch64-linux")
else()
    set(ASCEND_ARCH_DIR "x86_64-linux")
endif()

add_executable(ccu_all_gather_matmul
    ccu_all_gather_matmul_kernel.asc
)

target_include_directories(ccu_all_gather_matmul PRIVATE
    # CANN Toolkit头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
    # ---- 第二阶段新增：Blaze + tensor_api include 路径（ops-tensor 已拉取到 ~/code/ops-tensor）----
    ${OPS_TENSOR_ROOT}/include
    ${OPS_TENSOR_ROOT}/include/tensor_api/include
    ${OPS_TENSOR_ROOT}/include/tensor_api
    # ---- 第二阶段新增：platform_ascendc.h 头文件路径（GetCoreNumAic / PlatformAscendCManager）----
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/tiling
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/utils/tiling
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/utils
    # ---- 第二阶段新增：Blaze 依赖的 kernel_basic_intf.h 等 asc 内部头文件路径 ----
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/interface
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include
)

target_compile_features(ccu_all_gather_matmul PRIVATE cxx_std_14)
target_compile_options(ccu_all_gather_matmul PRIVATE
    -Werror
    -fstack-protector-strong
    -fPIE
    -O2
    # -s
)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "CMAKE_ASC_ARCHITECTURES, e.g. dav-3510")
target_compile_options(ccu_all_gather_matmul PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_definitions(ccu_all_gather_matmul PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)
target_link_options(ccu_all_gather_matmul PRIVATE
    -pie
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_directories(ccu_all_gather_matmul PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    # ---- 第二阶段新增：platform / tiling_api / register 库所在目录 ----
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/lib64
)
target_link_libraries(ccu_all_gather_matmul PRIVATE
    hcomm
    asccomm_ccu_dataplane
    c_sec
    ascendcl
    acl_rt
    ascendc_runtime
    pthread
    dl
    # ---- 第二阶段新增：Blaze 链接库 ----
    platform
    tiling_api
    register
)
