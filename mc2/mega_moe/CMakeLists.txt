# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

find_package(Git QUIET)

set(CATLASS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/3rdparty/catlass/include")
if(NOT EXISTS "${CATLASS_INCLUDE_DIR}/catlass/catlass.hpp")
    if(NOT GIT_FOUND)
        message(FATAL_ERROR
            "catlass submodule is not initialized and Git is not available. "
            "Please run manually: git submodule update --init 3rdparty/catlass"
        )
    endif()
    message(STATUS "catlass submodule not initialized, attempting to initialize...")
    execute_process(
        COMMAND ${GIT_EXECUTABLE} submodule update --init 3rdparty/catlass
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        RESULT_VARIABLE GIT_SUBMODULE_RESULT
    )
    if(NOT GIT_SUBMODULE_RESULT EQUAL 0)
        message(FATAL_ERROR
            "Failed to initialize catlass submodule. "
            "Please run manually: git submodule update --init 3rdparty/catlass"
        )
    endif()
    message(STATUS "catlass submodule initialized successfully.")
endif()

file(GLOB CURRENT_DIRS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR} ${CMAKE_CURRENT_SOURCE_DIR}/*)
if(NOT ENABLE_TEST)
    list(REMOVE_ITEM CURRENT_DIRS tests)
endif()
foreach(SUB_DIR ${CURRENT_DIRS})
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/${SUB_DIR}/CMakeLists.txt")
        add_subdirectory(${SUB_DIR})
    endif()
endforeach()

set(MC2_COMPILE ${SUB_MC2_COMPILE} PARENT_SCOPE)