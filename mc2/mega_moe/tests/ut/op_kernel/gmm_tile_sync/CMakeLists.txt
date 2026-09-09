# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(mega_moe_gmm_tile_sync_test LANGUAGES CXX)
find_package(Threads REQUIRED)
enable_testing()
add_executable(test_gmm_tile_sync test_gmm_tile_sync.cpp)
target_compile_features(test_gmm_tile_sync PRIVATE cxx_std_17)
target_compile_options(test_gmm_tile_sync PRIVATE -UNDEBUG)
target_include_directories(test_gmm_tile_sync PRIVATE stubs ../../../../op_kernel/arch35/common)
target_link_libraries(test_gmm_tile_sync PRIVATE Threads::Threads)
add_test(NAME gmm_tile_sync COMMAND test_gmm_tile_sync)
set_tests_properties(gmm_tile_sync PROPERTIES TIMEOUT 30)
