# Copyright (c) 2026, edconeone
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# Adapted from DeepEP; original copyright and license notices follow.

# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# Licensed under the CANN Open Software License Agreement Version 2.0.
cmake_minimum_required(VERSION 3.20)
project(mc2_deepep_dependencies LANGUAGES NONE)
get_filename_component(OPS_TRANSFORMER_ROOT "${CMAKE_CURRENT_LIST_DIR}/../../../../.." ABSOLUTE)
set(CANN_3RD_LIB_PATH "${OPS_TRANSFORMER_ROOT}/third_party" CACHE PATH "Third-party cache")
file(READ "${CMAKE_CURRENT_LIST_DIR}/../dependencies.lock.json" dependency_lock)
include(FetchContent)
find_package(Git REQUIRED)

function(mc2_fetch_dependency name repository revision)
    string(TOUPPER "${name}" upper_name)
    set(MC2_${upper_name}_SOURCE "${CANN_3RD_LIB_PATH}/mc2_deepep/${name}"
        CACHE PATH "Pinned ${name} source (may be an existing offline checkout)")
    set(source "${MC2_${upper_name}_SOURCE}")
    if(NOT EXISTS "${source}/.git")
        if(EXISTS "${source}")
            message(FATAL_ERROR "${source} exists but is not a Git checkout; select an empty cache path")
        endif()
        FetchContent_Declare(mc2_${name}
            GIT_REPOSITORY "${repository}" GIT_TAG "${revision}"
            GIT_SUBMODULES "" GIT_PROGRESS TRUE SOURCE_DIR "${source}")
        FetchContent_Populate(mc2_${name})
    endif()
    execute_process(COMMAND "${GIT_EXECUTABLE}" -C "${source}" rev-parse HEAD
        OUTPUT_VARIABLE actual OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
    if(NOT actual STREQUAL revision)
        message(FATAL_ERROR "${name} revision ${actual} does not match ${revision}; existing checkout was not changed")
    endif()
    message(STATUS "${name}: ${source} @ ${actual}")
endfunction()

string(JSON shmem_repository GET "${dependency_lock}" shmem repository)
string(JSON shmem_revision GET "${dependency_lock}" shmem revision)
mc2_fetch_dependency(shmem "${shmem_repository}.git" "${shmem_revision}")
option(MC2_FETCH_CATLASS "Prepare the pinned MegaMoE CATLASS dependency" OFF)
if(MC2_FETCH_CATLASS)
    string(JSON catlass_repository GET "${dependency_lock}" megamoe catlass_repository)
    string(JSON catlass_revision GET "${dependency_lock}" megamoe catlass_revision)
    mc2_fetch_dependency(catlass "${catlass_repository}.git" "${catlass_revision}")
endif()
