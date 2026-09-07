# Copyright (c) 2026 G_W_E. All rights reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.20)
find_package(ASC REQUIRED)
project(sageattention2_native LANGUAGES ASC CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT EXISTS "${CATLASS_ROOT}/include/catlass/catlass.hpp")
  message(FATAL_ERROR "Set CATLASS_ROOT to the CATLASS checkout")
endif()
add_library(sageattention2_native SHARED launch.cpp debug_softmax.cpp debug_cube.cpp debug_wide.cpp)
set_source_files_properties(launch.cpp debug_softmax.cpp debug_cube.cpp debug_wide.cpp PROPERTIES LANGUAGE ASC)
target_include_directories(sageattention2_native PRIVATE "${CATLASS_ROOT}/include" "${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel")
target_compile_definitions(sageattention2_native PRIVATE CATLASS_ARCH=3510)
target_compile_options(sageattention2_native PRIVATE --npu-arch=dav-3510 -O2 -Wno-macro-redefined -Wno-ignored-attributes)
target_link_options(sageattention2_native PRIVATE --npu-arch=dav-3510)
