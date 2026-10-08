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
project(mc2_low_latency_combine LANGUAGES CXX)
option(MC2_HOST_ONLY_TESTS "Build CPU layout tests without CANN" OFF)
if(MC2_HOST_ONLY_TESTS)
    include(CTest)
    set(CMAKE_CXX_STANDARD 17)
    add_executable(combine_layout_test tests/layout_test.cpp)
    target_include_directories(combine_layout_test PRIVATE op_kernel ../low_latency_dispatch/op_kernel)
    add_test(NAME combine_layout COMMAND combine_layout_test)
    add_subdirectory(../common/deepep/tests runtime_tests)
    return()
endif()
set(MC2_BUILD_COMBINE ON)
include("${CMAKE_CURRENT_SOURCE_DIR}/../common/deepep/cmake/Runtime.cmake")
mc2_add_deepep_runtime()
