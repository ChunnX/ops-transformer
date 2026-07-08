# ============================================================================
# CMake - FfnWorkerBatching 穿刺1 (RECV fp16) Kernel 直调, arch35/dav-3510
# ============================================================================
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(ffn_wb_recv_probe LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/aarch64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_executable(ffn_wb_recv_probe op_host/ffn_wb_recv_probe.asc)

target_link_libraries(ffn_wb_recv_probe PRIVATE
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

target_include_directories(ffn_wb_recv_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    # 暴露 kernel_log.h(ASSERT_MSG)：arch22 头以裸名 include，真实算子构建亦追加该 impl 目录
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/impl/basic_api
    $ENV{ASCEND_HOME_PATH}/x86_64-linux/ascendc/include/basic_api/impl
)

# arch35 / Ascend950 (DAV_3510)
if(NOT DEFINED PROBE_STAGE)
    set(PROBE_STAGE 9)
endif()
target_compile_options(ffn_wb_recv_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
    $<$<COMPILE_LANGUAGE:ASC>:-DPROBE_STAGE=${PROBE_STAGE}>
)
