# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnExc PRIVATE
    sparse_flash_mla_softmax_l1_norm_def.cpp
    )
endif()

# ascend950 条件编译——缺少此项会导致 kernel binary 未编译
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME SparseFlashMlaSoftmaxL1Norm
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

add_ops_compile_options(
    OP_NAME SparseFlashMlaSoftmaxL1Norm
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
)

# 有 op_api 目录时必须用 ACLNNTYPE aclnn_exclude
if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(OPTYPE sparse_flash_mla_softmax_l1_norm ACLNNTYPE aclnn_exclude)
endif()

# tiling 模块注册——缺少此项会导致 tiling 未注册到 SO
if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
endif()
