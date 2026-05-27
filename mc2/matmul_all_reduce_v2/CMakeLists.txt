# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)
project(matmul_all_reduce_v2 LANGUAGES CXX)

# MatmulAllReduceV2 算子构建配置
# 功能与 MatmulAllReduce 完全一致，复用原算子代码

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项
add_compile_options(-Wall -Werror)

# op_host 目录
add_subdirectory(op_host)

# op_api 目录
add_subdirectory(op_api)

# op_kernel 目录（如果需要）
# add_subdirectory(op_kernel)

# tests 目录（如果需要）
# add_subdirectory(tests)

# 文档目录
# add_subdirectory(docs)

# 示例目录
# add_subdirectory(examples)

# 安装规则
install(DIRECTORY op_host DESTINATION ${CMAKE_INSTALL_PREFIX}/op_host)
install(DIRECTORY op_api DESTINATION ${CMAKE_INSTALL_PREFIX}/op_api)
install(DIRECTORY op_kernel DESTINATION ${CMAKE_INSTALL_PREFIX}/op_kernel)
install(DIRECTORY op_graph DESTINATION ${CMAKE_INSTALL_PREFIX}/op_graph)