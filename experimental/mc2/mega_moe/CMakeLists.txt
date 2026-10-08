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
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()
project(mc2_mega_moe LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
option(MC2_HOST_ONLY_TESTS "Build CPU policy and runtime tests without CANN" OFF)
if(MC2_HOST_ONLY_TESTS)
    include(CTest)
    add_executable(megamoe_policy_test tests/policy_test.cpp)
    target_include_directories(megamoe_policy_test PRIVATE csrc)
    add_test(NAME megamoe_policy COMMAND megamoe_policy_test)
    add_subdirectory(../common/deepep/tests runtime_tests)
    return()
endif()
option(DEEPEP_MEGAMOE_TIMER "Compile device tracing (diagnostic calls only)" OFF)
get_filename_component(OPS_TRANSFORMER_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
if(DEFINED ENV{CATLASS_ROOT})
    set(default_catlass "$ENV{CATLASS_ROOT}")
else()
    set(default_catlass "${OPS_TRANSFORMER_ROOT}/third_party/mc2_deepep/catlass")
endif()
set(CATLASS_ROOT "${default_catlass}" CACHE PATH "Pinned MegaMoE CATLASS checkout")
set(MC2_BUILD_COMBINE ON)
set(MC2_BUILD_MEGAMOE ON)
include("${CMAKE_CURRENT_SOURCE_DIR}/../common/deepep/cmake/Runtime.cmake")
mc2_add_deepep_runtime()
# Find Python in this scope as well; Runtime.cmake confines its SDK setup to a function.
find_package(Python 3.11 REQUIRED COMPONENTS Interpreter Development.Module)
include("${CMAKE_CURRENT_SOURCE_DIR}/cmake/Native.cmake")
install(DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python/" DESTINATION python
    FILES_MATCHING PATTERN "*.py" PATTERN "*.json" PATTERN "__pycache__" EXCLUDE)
