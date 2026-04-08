# ============================================================================
# CMake 构建配置 - MTE AllToAll Kernel 直调
# ============================================================================

cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(KERNEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/mte_alltoall_kernel.cpp)

set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# 将 kernel 文件编译为 Ascend C 库
ascendc_library(ascendc_kernels SHARED ${KERNEL_FILE})

add_executable(test_mte_alltoall
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mte_alltoall.cpp
)

target_compile_options(test_mte_alltoall PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_compile_definitions(test_mte_alltoall PRIVATE
    SOC_VERSION=Ascend950
)

target_include_directories(test_mte_alltoall PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

# MC2 算子需要的额外库
target_link_libraries(test_mte_alltoall PRIVATE
    host_intf_pub
    ascendc_kernels
    tiling_api
    platform
    pthread
    hccl
    hcomm
)

install(TARGETS test_mte_alltoall
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)
