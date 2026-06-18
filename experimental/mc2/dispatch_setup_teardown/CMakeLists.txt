cmake_minimum_required(VERSION 3.16)
project(dispatch_integrated)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(SOC_VERSION "ascend950pr_9599" CACHE STRING "soc")
set(ASCEND_CANN_PACKAGE_PATH "/usr/local/Ascend/cann" CACHE STRING "cann")
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "" FORCE)
endif()
if(CMAKE_INSTALL_PREFIX STREQUAL /usr/local)
    set(CMAKE_INSTALL_PREFIX "${CMAKE_CURRENT_LIST_DIR}/out" CACHE STRING "" FORCE)
endif()
if(NOT DEFINED ENV{CMAKE_PREFIX_PATH})
    set(CMAKE_PREFIX_PATH ${ASCEND_CANN_PACKAGE_PATH}/tools/tikicpulib/lib/cmake)
endif()

# AIV teardown kernel(<<<>>>)
file(GLOB KERNEL_FILES ${CMAKE_CURRENT_SOURCE_DIR}/teardown_aiv.cpp)

if(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake)
elseif(EXISTS ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
    set(ASCENDC_CMAKE_DIR ${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake)
else()
    message(FATAL_ERROR "ascendc_kernel_cmake not found")
endif()
include(${ASCENDC_CMAKE_DIR}/ascendc.cmake)

ascendc_library(ascendc_kernels SHARED ${KERNEL_FILES})

# 集成 launcher: 链 AIV teardown 库 + CCU setup 库 + hccl/hcomm
add_executable(dispatch_integrated ${CMAKE_CURRENT_SOURCE_DIR}/main.cc)
target_compile_options(dispatch_integrated PRIVATE
    -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0
)
target_include_directories(dispatch_integrated PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/include/hccl
    ${CMAKE_CURRENT_SOURCE_DIR}/setup_ccu/inc
    ${CMAKE_CURRENT_SOURCE_DIR}
)
target_link_directories(dispatch_integrated PRIVATE
    ${ASCEND_CANN_PACKAGE_PATH}/lib64
    ${CMAKE_CURRENT_SOURCE_DIR}/setup_ccu
)
target_link_libraries(dispatch_integrated PRIVATE
    host_intf_pub
    ascendc_kernels         # AIV teardown(<<<>>>)
    dispatch_setup_ccu      # CCU setup(ccu::)
    tiling_api
    platform
    pthread
    acl_rt
    hccl
    hcomm
    c_sec
    ascendcl
)
