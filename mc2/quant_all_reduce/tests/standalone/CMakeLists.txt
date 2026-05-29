cmake_minimum_required(VERSION 3.16)
project(QuantAllReduceStandalone)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(ENABLE_PROFILING "Enable device-side profiler" OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "Debug" CACHE STRING "Build type Release/Debug (default Debug)" FORCE)
endif()

set(SOC_VERSION "ascend950pr_957c" CACHE STRING "system on chip type")

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_CANN_PACKAGE_PATH $ENV{ASCEND_HOME_PATH})
else()
    message(FATAL_ERROR "ASCEND_HOME_PATH not set, please configure CANN environment first")
endif()

set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake")
endif()
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}")
    set(ASCENDC_CMAKE_DIR "${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/tikcpp/ascendc_kernel_cmake")
endif()
if(NOT EXISTS "${ASCENDC_CMAKE_DIR}")
    message(FATAL_ERROR "ascendc_kernel_cmake not found, check CANN at: ${ASCEND_CANN_PACKAGE_PATH}")
endif()
include("${ASCENDC_CMAKE_DIR}/ascendc.cmake")

set(TESTS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/..)
set(OP_KERNEL_DIR ${TESTS_DIR}/../op_kernel)
set(MC2_ROOT ${TESTS_DIR}/../../..)
set(QRS_KERNEL_DIR ${TESTS_DIR}/../../quant_reduce_scatter/op_kernel)

# Create symlink so relative includes in quant_all_reduce_mte_one_shot.h resolve:
#   ../quant_reduce_scatter/profiler.h  →  quant_reduce_scatter/op_kernel/profiler.h
set(SYMLINK_TARGET ${TESTS_DIR}/../quant_reduce_scatter)
if(NOT EXISTS ${SYMLINK_TARGET})
    execute_process(COMMAND ${CMAKE_COMMAND} -E create_symlink
        ${QRS_KERNEL_DIR} ${SYMLINK_TARGET})
    message(STATUS "Created symlink: ${SYMLINK_TARGET} -> ${QRS_KERNEL_DIR}")
endif()

set(KERNEL_INCLUDE_DIRS
    ${OP_KERNEL_DIR}
    ${QRS_KERNEL_DIR}
    ${MC2_ROOT}/common/op_kernel
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/adv_api
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/ascendc/include/highlevel_api
)

ascendc_library(quant_all_reduce_kernels SHARED
    ${TESTS_DIR}/quant_all_reduce_kernel.cpp
)
ascendc_include_directories(quant_all_reduce_kernels PRIVATE ${KERNEL_INCLUDE_DIRS})
ascendc_compile_definitions(quant_all_reduce_kernels PRIVATE MTE_ONE_SHOT=1)
if(ENABLE_PROFILING)
    ascendc_compile_definitions(quant_all_reduce_kernels PRIVATE ENABLE_PROFILING=1)
endif()

set(HOST_INCLUDE_DIRS
    ${OP_KERNEL_DIR}
    ${QRS_KERNEL_DIR}
    ${TESTS_DIR}
    ${ASCEND_CANN_PACKAGE_PATH}/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/pkg_inc
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/adv_api
    ${ASCEND_CANN_PACKAGE_PATH}/x86_64-linux/asc/include/tiling
)

add_executable(standalone_profiler ${TESTS_DIR}/standalone_profiler.cpp)
target_compile_options(standalone_profiler PRIVATE -O2 -std=c++17 -D_GLIBCXX_USE_CXX11_ABI=0)
target_compile_definitions(standalone_profiler PRIVATE SOC_VERSION="${SOC_VERSION}")
target_include_directories(standalone_profiler PRIVATE ${HOST_INCLUDE_DIRS})
if(ENABLE_PROFILING)
    target_compile_definitions(standalone_profiler PRIVATE ENABLE_PROFILING=1)
endif()
target_link_libraries(standalone_profiler PRIVATE
    quant_all_reduce_kernels
    platform
    pthread
    hccl
    hcomm
    ascendc_runtime
)

install(TARGETS standalone_profiler quant_all_reduce_kernels
    RUNTIME DESTINATION .
    LIBRARY DESTINATION .
)
