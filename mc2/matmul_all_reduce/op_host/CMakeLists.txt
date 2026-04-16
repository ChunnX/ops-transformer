# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        matmul_all_reduce_def.cpp
    )
    add_mc2_modules_sources(OPTYPE matmul_all_reduce matmul_all_reduce ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(matmul_all_reduce_depends mc2/3rd mc2/common PARENT_SCOPE)
    set(matmul_all_reduce_apt_depends mc2/3rd mc2/common PARENT_SCOPE)

    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend910b")
        add_ops_compile_options(
            OP_NAME MatmulAllReduce
            OPTIONS --cce-auto-sync=off
        )
    elseif("${CONDITION_UNIT}" STREQUAL "ascend310p")
        add_ops_compile_options(
            OP_NAME MatmulAllReduce
            OPTIONS --cce-auto-sync=off
                    -mllvm -cce-aicore-jump-expand=true
        )
    endif()
else() # 回黄host
    add_mc2_modules_sources(OPTYPE matmul_all_reduce matmul_all_reduce ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
endif()
