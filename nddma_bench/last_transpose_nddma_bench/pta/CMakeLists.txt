# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------------------------------------

# ============================================================================
# LastTranspose [0,1,3,2] NDDMA PTA 直调启动器（standalone）——DAV_3510 / Ascend 950PR
#
# registry-invoke 迁移适配：原算子根 CMakeLists 的 `option(BUILD_PTA_LAUNCH)` 分支
# 迁到此 pta/ 独立工程。compute 逻辑在 ../op_kernel/arch35/*.h（命名空间类），
# launch.asc 在本目录定义 thin benchmark 入口 + C-ABI，ASC 编译成 SHARED .so
# 供 pybind 扩展（g++）按 C ABI 链接调用。
#
# 用法（在本 pta/ 目录下）：
#   cmake -S . -B build -DCMAKE_ASC_ARCHITECTURES=dav-3510 -DBUILD_PTA_LAUNCH=ON
#   cmake --build build --target last_transpose_launch -j
#   -> build/liblast_transpose_launch.so（pta_nddma.py 的 _BUILD 即指向此处）
# ============================================================================

cmake_minimum_required(VERSION 3.16)

set(CMAKE_ASC_ARCHITECTURES "dav-3510" CACHE STRING "NPU architecture: dav-3510")

find_package(ASC REQUIRED)

project(last_transpose_nddma_bench_pta LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

option(BUILD_PTA_LAUNCH "Build shared launcher lib for PyTorch(PTA) binding" ON)
if(BUILD_PTA_LAUNCH)
    add_library(last_transpose_launch SHARED
        ${CMAKE_CURRENT_SOURCE_DIR}/last_transpose_launch.asc
    )
    target_include_directories(last_transpose_launch PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel
        ${CMAKE_CURRENT_SOURCE_DIR}
    )
    target_compile_options(last_transpose_launch PRIVATE
        $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=${CMAKE_ASC_ARCHITECTURES}>
    )
    # asc_* 指令层实验变体 A/B 开关：-DUSE_ASC_API=ON 时改用 *_ascapi.h（NDDMA 读走 asc_ndim_copy_gm2ub）
    option(USE_ASC_API "Use asc_* c_api NDDMA variant instead of high-level DataCopy" OFF)
    if(USE_ASC_API)
        target_compile_definitions(last_transpose_launch PRIVATE LAST_TRANSPOSE_USE_ASC_API)
    endif()
    set_target_properties(last_transpose_launch PROPERTIES
        POSITION_INDEPENDENT_CODE ON
    )
endif()
