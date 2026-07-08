# ============================================================================
# Ascend C Kernel 直调 —— probe4_norm_bf16
#   目标架构：dav-3510（Ascend950 / A5）
#   只构建可执行文件（直调验证），不建 torch target
# ============================================================================
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(probe_ffn_wb LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}")

add_executable(probe_ffn_wb op_host/probe_ffn_wb.asc)

target_include_directories(probe_ffn_wb PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ASCEND_HOME}/x86_64-linux/include
    ${ASCEND_HOME}/include
)

target_link_directories(probe_ffn_wb PRIVATE
    ${ASCEND_HOME}/x86_64-linux/lib64
    ${ASCEND_HOME}/lib64
)

target_link_libraries(probe_ffn_wb PRIVATE
    ascendcl
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

# arch35 / Ascend950
target_compile_options(probe_ffn_wb PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)
