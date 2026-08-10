# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

include_guard(GLOBAL)

set(CATCCOS_TAG_ID "ee01c5181c50ce2aeb5feb05d603b568eb31a61a" CACHE STRING
    "Pinned CatCCOS revision used by the Ascend950 UDMA operators")
set(CATCCOS_GIT_REPOSITORY "https://gitcode.com/zhu-mingzhe71/catccos.git" CACHE STRING
    "CatCCOS git repository")

if(CATCCOS_SOURCE_PATH)
    get_filename_component(CATCCOS_SOURCE_PATH "${CATCCOS_SOURCE_PATH}" REALPATH)
elseif(DEFINED ENV{CATCCOS_SOURCE_PATH} AND IS_DIRECTORY "$ENV{CATCCOS_SOURCE_PATH}")
    get_filename_component(CATCCOS_SOURCE_PATH "$ENV{CATCCOS_SOURCE_PATH}" REALPATH)
elseif(CANN_3RD_LIB_PATH AND IS_DIRECTORY "${CANN_3RD_LIB_PATH}/catccos")
    get_filename_component(CATCCOS_SOURCE_PATH "${CANN_3RD_LIB_PATH}/catccos" REALPATH)
else()
    if(NOT CANN_3RD_LIB_PATH)
        get_filename_component(CANN_3RD_LIB_PATH
            "${CMAKE_CURRENT_LIST_DIR}/../../3rdparty" REALPATH)
    endif()

    include(FetchContent)
    FetchContent_Declare(
        catccos
        GIT_REPOSITORY ${CATCCOS_GIT_REPOSITORY}
        GIT_TAG ${CATCCOS_TAG_ID}
        GIT_SHALLOW FALSE
        GIT_PROGRESS TRUE
        GIT_SUBMODULES "3rdparty/catlass;3rdparty/shmem"
        GIT_SUBMODULES_RECURSE TRUE
        SOURCE_DIR "${CANN_3RD_LIB_PATH}/catccos"
    )
    FetchContent_GetProperties(catccos)
    if(NOT catccos_POPULATED)
        FetchContent_Populate(catccos)
    endif()
    set(CATCCOS_SOURCE_PATH "${catccos_SOURCE_DIR}")
endif()

if(NOT EXISTS "${CATCCOS_SOURCE_PATH}/include/catccos/catccos.hpp")
    message(FATAL_ERROR
        "Invalid CATCCOS_SOURCE_PATH=${CATCCOS_SOURCE_PATH}: include/catccos/catccos.hpp was not found")
endif()

find_package(Git QUIET)
if(GIT_FOUND AND EXISTS "${CATCCOS_SOURCE_PATH}/.git")
    execute_process(
        COMMAND "${GIT_EXECUTABLE}" rev-parse HEAD
        WORKING_DIRECTORY "${CATCCOS_SOURCE_PATH}"
        OUTPUT_VARIABLE CATCCOS_ACTUAL_REVISION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE CATCCOS_REVISION_STATUS
    )
    if(NOT CATCCOS_REVISION_STATUS EQUAL 0 OR
       NOT CATCCOS_ACTUAL_REVISION STREQUAL CATCCOS_TAG_ID)
        message(FATAL_ERROR
            "CatCCOS revision mismatch: expected ${CATCCOS_TAG_ID}, got ${CATCCOS_ACTUAL_REVISION}")
    endif()
endif()

if(NOT EXISTS "${CATCCOS_SOURCE_PATH}/3rdparty/catlass/include/catlass/catlass.hpp")
    message(FATAL_ERROR
        "CatCCOS CATLASS submodule is missing. Initialize 3rdparty/catlass under ${CATCCOS_SOURCE_PATH}.")
endif()

if(NOT EXISTS "${CATCCOS_SOURCE_PATH}/3rdparty/shmem/include/shmem.h")
    message(FATAL_ERROR
        "CatCCOS SHMEM submodule is missing. Initialize 3rdparty/shmem under ${CATCCOS_SOURCE_PATH}.")
endif()

set(CATCCOS_INCLUDE_DIR "${CATCCOS_SOURCE_PATH}/include" CACHE INTERNAL "CatCCOS headers")
set(CATCCOS_CATLASS_INCLUDE_DIR "${CATCCOS_SOURCE_PATH}/3rdparty/catlass/include"
    CACHE INTERNAL "CATLASS headers pinned by CatCCOS")
set(CATCCOS_SHMEM_SOURCE_DIR "${CATCCOS_SOURCE_PATH}/3rdparty/shmem"
    CACHE INTERNAL "SHMEM source pinned by CatCCOS")
set(CATCCOS_TOOLS_INCLUDE_DIR "${CATCCOS_SOURCE_PATH}/tools" CACHE INTERNAL "CatCCOS tool headers")

add_library(catccos_headers INTERFACE)
add_library(catccos::headers ALIAS catccos_headers)
target_include_directories(catccos_headers INTERFACE
    "${CATCCOS_INCLUDE_DIR}"
    "${CATCCOS_CATLASS_INCLUDE_DIR}"
    "${CATCCOS_TOOLS_INCLUDE_DIR}"
)
target_compile_definitions(catccos_headers INTERFACE CATLASS_ARCH=3510)

message(STATUS "Use CatCCOS source: ${CATCCOS_SOURCE_PATH}")
message(STATUS "Use CatCCOS revision: ${CATCCOS_TAG_ID}")
