# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

cmake_minimum_required(VERSION 3.16)

# MatmulAllReduceV2 op_host 构建配置

# 添加源文件
set(OP_HOST_SOURCES
    matmul_all_reduce_v2_def.cpp
    matmul_all_reduce_v2_infershape.cpp
    op_tiling/matmul_all_reduce_v2_tiling.cpp
)

# 创建库
add_library(matmul_all_reduce_v2_host STATIC ${OP_HOST_SOURCES})

# 链接依赖
target_link_libraries(matmul_all_reduce_v2_host
    matmul_all_reduce_host  # 链接原算子的库
)

# 包含目录
target_include_directories(matmul_all_reduce_v2_host PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/../matmul_all_reduce/op_host  # 引用原算子头文件
)