# ----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# CANN Open Software License Agreement Version 2.0.
# ----------------------------------------------------------------------------------------------------------
# 迭代一 穿刺2 (probe2_norm_multicore) — Kernel 直调工程
# 目标平台: Ascend950DT_9584 (arch35 / DAV_3510)
# ----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(ffn_wb_probe LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

add_executable(ffn_wb_probe op_host/ffn_wb_probe.asc)

target_link_libraries(ffn_wb_probe PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(ffn_wb_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
)

# arch35 / DAV_3510 (Ascend950)
target_compile_options(ffn_wb_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)
