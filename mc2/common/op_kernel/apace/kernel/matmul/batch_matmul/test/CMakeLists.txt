# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# AllToAllMmKernel（非量化 all-to-all MatMul）Kernel UT
#
# 本目录需接入 ops-tensor 的 Kernel UT 构建（tests/ut/op_kernel/），
# 通过 -I 将 ops-transformer 的 kernel 目录加入编译搜索路径：
#   AddOpTestCase(all_to_all_mm_kernel "ascend950pr_9599" "-I<ops-transformer>/mc2/common/op_kernel/apace/kernel/matmul/batch_matmul")

if(NOT OPS_TRANSFORMER_ROOT)
    if(OPS_TENSOR_ROOT)
        get_filename_component(OPS_TRANSFORMER_ROOT "${OPS_TENSOR_ROOT}/../ops-transformer" ABSOLUTE)
    else()
        set(OPS_TRANSFORMER_ROOT "" CACHE STRING "ops-transformer repo root")
    endif()
endif()

set(A2A_MM_KERNEL_DIR
    "${OPS_TRANSFORMER_ROOT}/mc2/common/op_kernel/apace/kernel/matmul/batch_matmul")

if(NOT EXISTS "${A2A_MM_KERNEL_DIR}/all_to_all_mm_kernel.h")
    message(WARNING
        "[all_to_all_mm_kernel] all_to_all_mm_kernel.h not found under ${A2A_MM_KERNEL_DIR}; set OPS_TRANSFORMER_ROOT explicitly")
endif()

AddOpTestCase(all_to_all_mm_kernel "ascend950pr_9599" "-I${A2A_MM_KERNEL_DIR}")
