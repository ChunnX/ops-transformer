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
# This is a direct-launch example, like quant_all_reduce, not an OPP package.
# Traversal by the repository's experimental OPP build must not load its SDKs.
if(NOT CMAKE_SOURCE_DIR STREQUAL CMAKE_CURRENT_SOURCE_DIR)
    return()
endif()
project(mc2_low_latency_dispatch LANGUAGES CXX)
option(MC2_HOST_ONLY_TESTS "Build CPU layout and resource-lifecycle tests without CANN" OFF)
if(MC2_HOST_ONLY_TESTS)
    include(CTest)
    set(CMAKE_CXX_STANDARD 17)
    add_executable(dispatch_layout_test tests/layout_test.cpp)
    target_include_directories(dispatch_layout_test PRIVATE op_kernel)
    add_test(NAME dispatch_layout COMMAND dispatch_layout_test)
    add_subdirectory(../common/deepep/tests runtime_tests)
    return()
endif()
include("${CMAKE_CURRENT_SOURCE_DIR}/../common/deepep/cmake/Runtime.cmake")
mc2_add_deepep_runtime()
