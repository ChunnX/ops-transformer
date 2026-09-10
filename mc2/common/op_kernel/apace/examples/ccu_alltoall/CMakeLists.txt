# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

# CCU AllToAll（纯通信验证，FP16，Blaze-free）
# 由 ccu_all_gather_matmul 的 CMakeLists 裁剪而来：
#   - 删除 ops-tensor / Blaze 相关 include 与检查（纯通信无 matmul）
#   - 删除 platform / tiling_api / register 链接库（仅 Blaze matmul 需要）
#   - 阶段3（通算融合）叠加 matmul 时按 README.md "Blaze 恢复点" 加回

cmake_minimum_required(VERSION 3.16.0)
find_package(ASC REQUIRED)
project(hccl_examples_custom_ops_alltoall_ccu_direct LANGUAGES CXX ASC)

if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
    if(DEFINED ENV{ASCEND_HOME_PATH})
        set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
    elseif(DEFINED ENV{ASCEND_OPP_PATH})
        get_filename_component(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_OPP_PATH}/.." ABSOLUTE)
    else()
        message(FATAL_ERROR "ASCEND_CANN_PACKAGE_PATH or ASCEND_HOME_PATH must be set")
    endif()
endif()

# ---- 架构相关目录 ----
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64")
    set(ASCEND_ARCH_DIR "x86_64-linux")
elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "aarch64|arm64|arm")
    set(ASCEND_ARCH_DIR "aarch64-linux")
else()
    set(ASCEND_ARCH_DIR "x86_64-linux")
endif()

add_executable(ccu_alltoall
    src/alltoall_ccu_kernel_impl.asc
)

target_include_directories(ccu_alltoall PRIVATE
    # 本工程源码目录（.asc 单翻译单元经相对 include 引入 src/ 内其余源文件）
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    # CANN Toolkit 头文件
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/include/hcomm/ccu
    # asc-comm 正式版 host launch 头（ccu_host_launch.h，CANN 9.2.0+ 自带；
    # 对齐 asc-comm ccu_direct 官方示例 include 布局）
    ${ASCEND_CANN_PACKAGE_PATH}/asc/include/comm_api/ccu
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/hcomm/ccu
    # kernel_basic_intf.h 等 asc 内部头文件路径（WriteGmByPassDCache 等 API 声明所在）
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/basic_api
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include/interface
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/asc/include
)

target_compile_features(ccu_alltoall PRIVATE cxx_std_14)
target_compile_options(ccu_alltoall PRIVATE
    -Werror
    -fstack-protector-strong
    -fPIE
    -O2
)
set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "CMAKE_ASC_ARCHITECTURES, e.g. dav-3510")
target_compile_options(ccu_alltoall PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
)

target_compile_definitions(ccu_alltoall PRIVATE
    _GLIBCXX_USE_CXX11_ABI=0
)

target_link_options(ccu_alltoall PRIVATE
    -pie
    -Wl,-z,relro
    -Wl,-z,now
    -Wl,-z,noexecstack
)
target_link_directories(ccu_alltoall PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${ASCEND_CANN_PACKAGE_PATH}/${ASCEND_ARCH_DIR}/lib64
)
target_link_libraries(ccu_alltoall PRIVATE
    hcomm
    # asc-comm 正式版 CCU host launch 库（CANN 9.2.0+ 自带 libasccomm_ccu.so；
    # 旧 asc-comm 草稿库名 asccomm_ccu_dataplane 已废弃——正式版见 asc-comm master
    # 7149d33 "add hostlaunch"，对齐 ccu_direct 官方示例链接配置）
    asccomm_ccu
    c_sec
    ascendcl
    acl_rt
    ascendc_runtime
    pthread
    dl
)
