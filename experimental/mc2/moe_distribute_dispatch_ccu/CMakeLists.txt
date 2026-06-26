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
project(moe_distribute_dispatch_ccu)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
# A6 仿真器沿用 A5 / arch35 工具链（设计文档：编译宏 __NPU_ARCH__=3510）；A6 回片后再加专属型号
set(SOC_VERSION "ascend950pr_9599" CACHE STRING "system on chip type")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann"
    CACHE STRING "ASCEND CANN package installation directory")
# hcomm 仓路径：CCU 编程接口头文件（ccu_kernel.h / hccl_ccu_res.h / hccl_res.h / hcomm_res_defs.h）来源。
# 默认指向本工作区 hcomm；按实际环境用 -DHCOMM_HOME=<path> 覆盖。
# [待验证] hcomm CCU 头文件在部署态 CANN 包内的确切路径，需在 A6 仿真器环境确认（设计文档 V10 相关）。
set(HCOMM_HOME "${CMAKE_CURRENT_SOURCE_DIR}/../../../../hcomm"
    CACHE STRING "hcomm repo root for CCU headers")

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "path for install()" FORCE)
endif()
if(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/cmake)
endif()

# ── device 侧：teardown AICore 核函数（由 ascendc 编译）──
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_dispatch_ccu.cpp)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake does not exist, please check whether the cann package is installed")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels SHARED ${KERNEL_FILES})

# ── host 侧：编排入口 + CCU setup kernel(CcuKernel 子类) + 测试驱动 ──
add_executable(test_moe_distribute_dispatch_ccu
    ${CMAKE_CURRENT_SOURCE_DIR}/test_moe_distribute_dispatch_ccu.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/moe_distribute_dispatch_ccu_host.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/dispatch_setup_ccu.cpp
)

target_compile_options(test_moe_distribute_dispatch_ccu PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall
)

target_compile_definitions(test_moe_distribute_dispatch_ccu PRIVATE
    SOC_VERSION="${SOC_VERSION}"
)

target_include_directories(test_moe_distribute_dispatch_ccu PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${HCOMM_HOME}/include            # hccl/hccl_res.h、hcomm_res_defs.h
    ${HCOMM_HOME}/pkg_inc/hcomm/ccu  # ccu_kernel.h、hccl_ccu_res.h、ccu_*.h
)

target_link_libraries(test_moe_distribute_dispatch_ccu PRIVATE
    host_intf_pub
    ascendc_kernels
    tiling_api
    platform
    pthread
    hccl
    hcomm
)

install(TARGETS test_moe_distribute_dispatch_ccu
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
