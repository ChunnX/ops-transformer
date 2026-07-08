# ============================================================================
# CMake 构建配置 - probe3_norm_int8 Kernel 直调（Ascend950 / arch35 / dav-3510）
# 仅构建直调可执行文件（无 PyTorch 通路）。
# ============================================================================
cmake_minimum_required(VERSION 3.16)

find_package(ASC REQUIRED)

project(ffn_wb_probe LANGUAGES ASC CXX)

set(CMAKE_CXX_STANDARD 17)

set(ACL_INCLUDE_DIR "$ENV{ASCEND_HOME_PATH}/x86_64-linux/include")
set(ACL_LIB_DIR "$ENV{ASCEND_HOME_PATH}/lib64")

add_executable(ffn_wb_probe op_host/ffn_wb_probe.asc)

target_include_directories(ffn_wb_probe PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/op_kernel
    ${CMAKE_CURRENT_SOURCE_DIR}/op_host
    ${ACL_INCLUDE_DIR}
    # kernel_log.h（ASSERT_MSG）位于 ASC impl 基础 API 目录
    "$ENV{ASCEND_HOME_PATH}/x86_64-linux/asc/impl/basic_api"
)

target_link_directories(ffn_wb_probe PRIVATE
    ${ACL_LIB_DIR}
)

target_link_libraries(ffn_wb_probe PRIVATE
    ascendcl
    runtime
    tiling_api
    register
    platform
    unified_dlog
    dl
    m
    graph_base
)

# arch35 / DAV_3510 = Ascend950 系列
target_compile_options(ffn_wb_probe PRIVATE
    $<$<COMPILE_LANGUAGE:ASC>:--npu-arch=dav-3510>
)
