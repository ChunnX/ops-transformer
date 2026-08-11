# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

cmake_minimum_required(VERSION 3.18.4)

# experimental/mc2 auto-adds every child CMake project. This Wheel has its own
# compiler and dependency discovery and must be configured as a standalone root.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    message(STATUS "Skip standalone catccos_udma_ops Wheel in the aggregate build")
    return()
endif()

find_package(ASC REQUIRED)
find_package(AICPU REQUIRED)
project(OpsTransformerCatccosUdma LANGUAGES ASC AICPU CXX VERSION 0.1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

get_filename_component(OPS_TRANSFORMER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." REALPATH)
set(FAST_LAUNCH_ROOT "${OPS_TRANSFORMER_ROOT}/examples/fast_kernel_launch_example")

include("${FAST_LAUNCH_ROOT}/cmake/ascend.cmake")
include("${FAST_LAUNCH_ROOT}/cmake/python.cmake")
include("${FAST_LAUNCH_ROOT}/cmake/torch.cmake")
include("${FAST_LAUNCH_ROOT}/cmake/torch_npu.cmake")
include("${FAST_LAUNCH_ROOT}/cmake/func.cmake")
include("${OPS_TRANSFORMER_ROOT}/cmake/third_party/catccos.cmake")

set(NPU_SOC_VERSION "ascend950" CACHE STRING "NPU SOC version")
set(EXTENSION_MODULE_NAME "ops_transformer_catccos_udma" CACHE STRING "PyTorch operator namespace")
if(NOT NPU_SOC_VERSION STREQUAL "ascend950")
    message(FATAL_ERROR "CatCCOS UDMA operators currently support only NPU_SOC_VERSION=ascend950")
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend installation path" FORCE)
else()
    set(ASCEND_HOME "/usr/local/Ascend/latest" CACHE PATH "Ascend installation path")
endif()
if(SHMEM_HOME_PATH)
    get_filename_component(SHMEM_HOME_PATH "${SHMEM_HOME_PATH}" REALPATH)
elseif(DEFINED ENV{SHMEM_HOME_PATH} AND IS_DIRECTORY "$ENV{SHMEM_HOME_PATH}")
    get_filename_component(SHMEM_HOME_PATH "$ENV{SHMEM_HOME_PATH}" REALPATH)
elseif(IS_DIRECTORY "${CATCCOS_SHMEM_SOURCE_DIR}/install")
    get_filename_component(SHMEM_HOME_PATH "${CATCCOS_SHMEM_SOURCE_DIR}/install" REALPATH)
else()
    message(FATAL_ERROR
        "SHMEM_HOME_PATH is not set and ${CATCCOS_SHMEM_SOURCE_DIR}/install does not exist. "
        "Build the CatCCOS-pinned SHMEM first with scripts/prepare_dependencies.sh.")
endif()
if(NOT EXISTS "${SHMEM_HOME_PATH}/shmem/lib/libshmem.so" OR
   NOT EXISTS "${SHMEM_HOME_PATH}/shmem/lib/libshmem_utils.so")
    message(FATAL_ERROR
        "SHMEM_HOME_PATH=${SHMEM_HOME_PATH} does not contain the Ascend950 UDMA SHMEM libraries")
endif()

set(INCLUDE_DIRECTORIES
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/catccos_overlay/include
    ${Python3_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS}
    ${TORCH_NPU_INCLUDE_PATH}
    ${ASCEND_INCLUDE_DIRS}
    ${CATCCOS_INCLUDE_DIR}
    ${CATCCOS_CATLASS_INCLUDE_DIR}
    ${CATCCOS_TOOLS_INCLUDE_DIR}
    ${CATCCOS_SHMEM_SOURCE_DIR}/include
    ${CATCCOS_SHMEM_SOURCE_DIR}/src
    ${CATCCOS_SHMEM_SOURCE_DIR}/src/device
    ${SHMEM_HOME_PATH}/shmem/include
    ${SHMEM_HOME_PATH}/shmem/src
    ${SHMEM_HOME_PATH}/shmem/src/device
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/host
    ${SHMEM_HOME_PATH}/memfabric_hybrid/include/smem/device
    ${ASCEND_HOME}/include
    ${ASCEND_HOME}/include/experiment/runtime
    ${ASCEND_HOME}/include/experiment/msprof
    ${ASCEND_HOME}/compiler/tikcpp/tikcfw
    ${ASCEND_HOME}/compiler/tikcpp/tikcfw/impl
    ${ASCEND_HOME}/compiler/tikcpp/tikcfw/interface
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/include
    ${ASCEND_HOME}/${SYSTEM_PREFIX}/pkg_inc/runtime
    ${ASCEND_HOME}/pkg_inc/base
)

set(LINK_DIRECTORIES
    ${TORCH_NPU_LIB_PATH}
    ${ASCEND_DIR}/lib64
    ${ASCEND_HOME}/lib64
    ${SHMEM_HOME_PATH}/shmem/lib
    ${SHMEM_HOME_PATH}/memfabric_hybrid/lib
)

set(LINK_LIBRARIES
    ${TORCH_LIBRARIES}
    torch_npu
    ascendcl
    platform
    register
    tiling_api
    runtime
    graph_base
    shmem
    shmem_utils
    dl
)

set(COMPILE_OPTIONS
    ${TORCH_CXX_FLAGS}
    -O2
    -fdiagnostics-color=always
    -Wno-macro-redefined
    -Wno-ignored-attributes
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
    -DCATLASS_ARCH=3510
    -DL2_CACHE_HINT
    -DTORCH_MODE
)

set(OBJECTS_LIST "" CACHE INTERNAL "CatCCOS UDMA operator objects" FORCE)
add_subdirectory(csrc)

add_library(_C SHARED
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/extension.cpp
    ${OBJECTS_LIST}
)
set_target_properties(_C PROPERTIES
    POSITION_INDEPENDENT_CODE ON
    PREFIX ""
    SUFFIX ".abi3.so"
    OUTPUT_NAME "_C"
    BUILD_RPATH "\$ORIGIN/.libs"
    INSTALL_RPATH "\$ORIGIN/.libs"
    BUILD_RPATH_USE_ORIGIN ON
)
target_link_options(_C PRIVATE -Wl,--disable-new-dtags)
target_compile_definitions(_C PRIVATE Py_LIMITED_API=0x03080000)
target_compile_options(_C PRIVATE ${COMPILE_OPTIONS})
target_include_directories(_C PRIVATE ${INCLUDE_DIRECTORIES})
target_link_directories(_C PRIVATE ${LINK_DIRECTORIES})
target_link_libraries(_C PRIVATE ${LINK_LIBRARIES})

set(PYTHON_PACKAGE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/ops_transformer_catccos_udma")
set(COPY_MEMFABRIC_COMMANDS "")
if(IS_DIRECTORY "${SHMEM_HOME_PATH}/memfabric_hybrid/lib")
    list(APPEND COPY_MEMFABRIC_COMMANDS
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${SHMEM_HOME_PATH}/memfabric_hybrid/lib" "${PYTHON_PACKAGE_DIR}/.libs")
endif()
add_custom_command(TARGET _C POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E make_directory "${PYTHON_PACKAGE_DIR}/.libs"
    COMMAND ${CMAKE_COMMAND} -E copy "$<TARGET_FILE:_C>" "${PYTHON_PACKAGE_DIR}/$<TARGET_FILE_NAME:_C>"
    COMMAND ${CMAKE_COMMAND} -E copy_directory
        "${SHMEM_HOME_PATH}/shmem/lib" "${PYTHON_PACKAGE_DIR}/.libs"
    ${COPY_MEMFABRIC_COMMANDS}
    COMMENT "Stage the PyTorch extension and SHMEM user-space runtime for wheel packaging"
)

install(TARGETS _C LIBRARY DESTINATION ops_transformer_catccos_udma)
install(DIRECTORY "${SHMEM_HOME_PATH}/shmem/lib/"
    DESTINATION ops_transformer_catccos_udma/.libs
    FILES_MATCHING PATTERN "*.so" PATTERN "*.so.*")
if(IS_DIRECTORY "${SHMEM_HOME_PATH}/memfabric_hybrid/lib")
    install(DIRECTORY "${SHMEM_HOME_PATH}/memfabric_hybrid/lib/"
        DESTINATION ops_transformer_catccos_udma/.libs
        FILES_MATCHING PATTERN "*.so" PATTERN "*.so.*")
endif()
