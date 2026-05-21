# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
cmake_minimum_required(VERSION 3.16)
project(moe_dispatch_ccu_direct)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOC_VERSION "ascend950pr_9599" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/home/0513/env/cann-9.1.0"
    CACHE STRING "ASCEND CANN package installation directory"
)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()

if(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/cmake)
endif()

# 直调 device 文件：含 2 个 CCU dispatch 核函数空壳 + 2 个 <<<>>> 封装
file(GLOB KERNEL_FILES
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_dispatch_ccu_direct.cpp
)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library 把 device 核函数文件编译成直调内核库
ascendc_library(ascendc_kernels SHARED ${KERNEL_FILES})

add_executable(test_moe_dispatch_ccu_direct
    ${CMAKE_CURRENT_SOURCE_DIR}/test_moe_dispatch_ccu_direct.cpp
)

target_compile_options(test_moe_dispatch_ccu_direct PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_compile_definitions(test_moe_dispatch_ccu_direct PRIVATE
    SOC_VERSION="${SOC_VERSION}"
)

target_include_directories(test_moe_dispatch_ccu_direct PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

# 仅链接 <<<>>> 直调所需：ascendc 内核库 + host 接口 + 平台库；
# 空壳 kernel 无通信，不链接 hccl / hcomm；不走注册框架 tiling，不链接 tiling_api。
target_link_libraries(test_moe_dispatch_ccu_direct PRIVATE
    host_intf_pub
    ascendc_kernels
    platform
    pthread
)

install(TARGETS test_moe_dispatch_ccu_direct
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
