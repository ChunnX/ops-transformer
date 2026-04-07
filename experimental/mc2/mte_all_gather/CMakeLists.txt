cmake_minimum_required(VERSION 3.16)
project(Ascend_c)

set(KERNEL_FILE ${CMAKE_CURRENT_SOURCE_DIR}/mte_all_gather.cpp)

set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

# ascendc_library use to add kernel file to generate ascendc library
ascendc_library(ascendc_kernels SHARED ${KERNEL_FILE})

add_executable(test_mte_all_gather 
    ${CMAKE_CURRENT_SOURCE_DIR}/test_mte_all_gather.cpp
)

target_compile_options(test_mte_all_gather PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0 -Wall -Werror
)

target_compile_definitions(test_mte_all_gather PRIVATE
    SOC_VERSION=Ascend950
)

target_include_directories(test_mte_all_gather PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
)

target_link_libraries(test_mte_all_gather PRIVATE
    host_intf_pub
    ascendc_kernels
    tiling_api
    platform
    pthread
    hccl
    hcomm
)
