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
file(READ
    "${CMAKE_CURRENT_SOURCE_DIR}/ops_transformer_catccos_udma/_version.py"
    OPS_TRANSFORMER_CATCCOS_UDMA_VERSION_SOURCE
)
string(REGEX MATCH
    "__version__[ \t]*=[ \t]*\"([0-9]+\\.[0-9]+\\.[0-9]+)\""
    OPS_TRANSFORMER_CATCCOS_UDMA_VERSION_MATCH
    "${OPS_TRANSFORMER_CATCCOS_UDMA_VERSION_SOURCE}"
)
if(NOT OPS_TRANSFORMER_CATCCOS_UDMA_VERSION_MATCH)
    message(FATAL_ERROR "Unable to read the package version from _version.py")
endif()
set(OPS_TRANSFORMER_CATCCOS_UDMA_VERSION "${CMAKE_MATCH_1}")
project(OpsTransformerCatccosUdma
    LANGUAGES ASC AICPU CXX
    VERSION "${OPS_TRANSFORMER_CATCCOS_UDMA_VERSION}"
)

# CMake 3.27+ asks supported linkers to emit a dependency file with
# --dependency-file. Bisheng's device linker does not implement that option and
# treats its empty value as an input file, so keep CMake's traditional link
# dependency tracking for this standalone Bisheng project.
set(CMAKE_LINK_DEPENDS_USE_LINKER FALSE)

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
option(CATCCOS_UDMA_MSSANITIZER
    "Build Ascend950 device kernels with source-line information for msSanitizer" OFF)
option(CATCCOS_UDMA_DEVICE_DIAGNOSTICS
    "Compile device-side printf diagnostics (never enable for performance delivery)" OFF)
option(CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE
    "Compile the validation-only GMM-AllToAllV A/B/C/D root-cause matrix" OFF)
option(CATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER
    "Use the all-PE producer barrier in the production no-reuse GMM-AllToAllV kernel" ON)
set(CATCCOS_UDMA_ASCEND_SANITIZER_FLAGS "")
if(CATCCOS_UDMA_MSSANITIZER)
    # Ascend 950 does not support --cce-enable-sanitizer/--sanitizer. Those
    # options also enable the SHMEM MSTX path, which requires the unsupported
    # sanitizer_report.h device API. Keep only -g so msSanitizer can symbolize
    # exceptions to source files, lines and call stacks on this platform.
    set(CATCCOS_UDMA_ASCEND_SANITIZER_FLAGS "-g")
    message(STATUS
        "Enable Ascend950 msSanitizer source-line information (-g only)")
endif()
set(CATCCOS_UDMA_ASCEND_DIAGNOSTIC_FLAGS "-DASCENDC_DUMP=0")
if(CATCCOS_UDMA_DEVICE_DIAGNOSTICS)
    # AscendC printf enables the device Dump path and reserves per-core output
    # storage. Keep both the call sites and Dump runtime out of production
    # binaries; this switch is only for a dedicated diagnosis Wheel.
    set(CATCCOS_UDMA_ASCEND_DIAGNOSTIC_FLAGS
        "-DCATCCOS_UDMA_READY_WAIT_TIMEOUT_DIAGNOSTICS=1 -DCATCCOS_UDMA_DEVICE_DIAGNOSTICS=1 -DASCENDC_DUMP=1")
    message(WARNING
        "Enable device printf diagnostics; this Wheel is not valid for performance measurement")
elseif(CATCCOS_UDMA_MSSANITIZER)
    # The model hang leaves the root AIV in the production strong-ready wait.
    # Keep all legacy device printf call sites disabled, but enable the
    # one-shot timeout assertion and the Dump storage needed by
    # ascendc_assert's formatted diagnostic record.
    set(CATCCOS_UDMA_ASCEND_DIAGNOSTIC_FLAGS
        "-DCATCCOS_UDMA_READY_WAIT_TIMEOUT_DIAGNOSTICS=1 -DCATCCOS_UDMA_DEVICE_DIAGNOSTICS=0 -DASCENDC_DUMP=1")
    message(WARNING
        "Enable only the A2AV-GMM ready-wait timeout assertion for the msSanitizer Wheel")
else()
    message(STATUS "Disable AscendC device Dump diagnostics in the production Wheel")
endif()
set(CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE_FLAG
    "-DCATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE=0")
if(CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE)
    set(CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE_FLAG
        "-DCATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE=1")
    message(WARNING
        "Enable validation-only GMM-AllToAllV root-cause kernels; do not use this Wheel for performance delivery")
endif()
set(CATCCOS_UDMA_GMM_A2AV_PRODUCER_SYNC_FLAG
    "-DCATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER=0")
if(CATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER)
    set(CATCCOS_UDMA_GMM_A2AV_PRODUCER_SYNC_FLAG
        "-DCATCCOS_UDMA_GMM_A2AV_PRODUCER_FULL_BARRIER=1")
    message(STATUS
        "GMM-AllToAllV delivery synchronization: no-reuse with producer full barrier")
else()
    message(WARNING
        "GMM-AllToAllV delivery synchronization: no-reuse without producer full barrier; validate precision on the target topology before delivery")
endif()
if(NOT NPU_SOC_VERSION STREQUAL "ascend950")
    message(FATAL_ERROR "CatCCOS UDMA operators currently support only NPU_SOC_VERSION=ascend950")
endif()

if(DEFINED ENV{ASCEND_HOME_PATH})
    set(ASCEND_HOME "$ENV{ASCEND_HOME_PATH}" CACHE PATH "Ascend installation path" FORCE)
else()
    set(ASCEND_HOME "/usr/local/Ascend/latest" CACHE PATH "Ascend installation path")
endif()
# Exact official release/v1.6.0 maintenance snapshot (2026-09-04).
# A normal variable also overrides an old SHMEM_TAG_ID left in CMakeCache.txt.
set(SHMEM_TAG_ID "7fa19662da0093587d9a332e740b91d6c929cdcf")
if(SHMEM_SOURCE_PATH)
    get_filename_component(SHMEM_SOURCE_PATH "${SHMEM_SOURCE_PATH}" REALPATH)
elseif(DEFINED ENV{SHMEM_SOURCE_PATH} AND IS_DIRECTORY "$ENV{SHMEM_SOURCE_PATH}")
    get_filename_component(SHMEM_SOURCE_PATH "$ENV{SHMEM_SOURCE_PATH}" REALPATH)
elseif(IS_DIRECTORY "${OPS_TRANSFORMER_ROOT}/3rdparty/shmem")
    get_filename_component(SHMEM_SOURCE_PATH "${OPS_TRANSFORMER_ROOT}/3rdparty/shmem" REALPATH)
else()
    message(FATAL_ERROR
        "The pinned official SHMEM release/v1.6.0 checkout was not found. "
        "Run scripts/prepare_dependencies.sh first.")
endif()
if(NOT EXISTS "${SHMEM_SOURCE_PATH}/include/shmem.h")
    message(FATAL_ERROR "Invalid SHMEM_SOURCE_PATH=${SHMEM_SOURCE_PATH}: include/shmem.h was not found")
endif()
find_package(Git REQUIRED)
execute_process(
    COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
    WORKING_DIRECTORY "${SHMEM_SOURCE_PATH}"
    OUTPUT_VARIABLE SHMEM_ACTUAL_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SHMEM_REVISION_STATUS
)
if(NOT SHMEM_REVISION_STATUS EQUAL 0 OR NOT SHMEM_ACTUAL_REVISION STREQUAL SHMEM_TAG_ID)
    message(FATAL_ERROR
        "SHMEM revision mismatch: expected official release/v1.6.0 snapshot ${SHMEM_TAG_ID}, "
        "got ${SHMEM_ACTUAL_REVISION}")
endif()
execute_process(
    COMMAND "${GIT_EXECUTABLE}" status --porcelain --untracked-files=normal
    WORKING_DIRECTORY "${SHMEM_SOURCE_PATH}"
    OUTPUT_VARIABLE SHMEM_WORKTREE_STATUS
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE SHMEM_WORKTREE_CHECK_STATUS
)
if(NOT SHMEM_WORKTREE_CHECK_STATUS EQUAL 0 OR NOT SHMEM_WORKTREE_STATUS STREQUAL "")
    message(FATAL_ERROR
        "SHMEM source has local changes or cannot be inspected; its headers must match the pinned library. "
        "${SHMEM_WORKTREE_STATUS}")
endif()
if(SHMEM_HOME_PATH)
    get_filename_component(SHMEM_HOME_PATH "${SHMEM_HOME_PATH}" REALPATH)
elseif(DEFINED ENV{SHMEM_HOME_PATH} AND IS_DIRECTORY "$ENV{SHMEM_HOME_PATH}")
    get_filename_component(SHMEM_HOME_PATH "$ENV{SHMEM_HOME_PATH}" REALPATH)
elseif(IS_DIRECTORY "${SHMEM_SOURCE_PATH}/install")
    get_filename_component(SHMEM_HOME_PATH "${SHMEM_SOURCE_PATH}/install" REALPATH)
else()
    message(FATAL_ERROR
        "SHMEM_HOME_PATH is not set and ${SHMEM_SOURCE_PATH}/install does not exist. "
        "Build the pinned official SHMEM first with scripts/prepare_dependencies.sh.")
endif()
if(NOT EXISTS "${SHMEM_HOME_PATH}/shmem/lib/libshmem.so" OR
   NOT EXISTS "${SHMEM_HOME_PATH}/shmem/lib/libshmem_utils.so")
    message(FATAL_ERROR
        "SHMEM_HOME_PATH=${SHMEM_HOME_PATH} does not contain the Ascend950 UDMA SHMEM libraries")
endif()
set(SHMEM_REVISION_FILE "${SHMEM_HOME_PATH}/shmem/.ops_transformer_source_commit")
if(NOT EXISTS "${SHMEM_REVISION_FILE}")
    message(FATAL_ERROR
        "SHMEM_HOME_PATH=${SHMEM_HOME_PATH} has no build provenance marker. "
        "Rebuild it with scripts/prepare_dependencies.sh.")
endif()
file(READ "${SHMEM_REVISION_FILE}" SHMEM_BUILT_REVISION)
string(STRIP "${SHMEM_BUILT_REVISION}" SHMEM_BUILT_REVISION)
if(NOT SHMEM_BUILT_REVISION STREQUAL SHMEM_TAG_ID)
    message(FATAL_ERROR
        "SHMEM library revision mismatch: expected ${SHMEM_TAG_ID}, "
        "got ${SHMEM_BUILT_REVISION}")
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
    ${CATCCOS_UTILS_INCLUDE_DIR}
    ${SHMEM_SOURCE_PATH}/include
    ${SHMEM_SOURCE_PATH}/src
    ${SHMEM_SOURCE_PATH}/src/device
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
    -Wno-defaulted-function-deleted
    -DEXTENSION_MODULE_NAME=${EXTENSION_MODULE_NAME}
    -DCATLASS_ARCH=3510
    -DL2_CACHE_HINT
    -DTORCH_MODE
    ${CATCCOS_UDMA_GMM_A2AV_ROOT_CAUSE_FLAG}
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
