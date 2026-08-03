# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
    msa_index_score_def.cpp
    )
endif()

# MsaIndexScore 的 device 侧实现直接依赖 3rdparty/catlass 的头文件模板库。
# Catlass 为 header-only，只需把 include 根目录注入 kernel 编译选项即可。
add_ops_compile_options(
    OP_NAME MsaIndexScore
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -fpermissive
    OPTIONS -I${CMAKE_SOURCE_DIR}/3rdparty/catlass/include
)

if (NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE msa_index_score ACLNNTYPE aclnn)
endif()
