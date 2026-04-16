# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        prompt_flash_attention_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME PromptFlashAttention
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    set(prompt_flash_attention_depends attention/flash_attention_score attention/common CACHE INTERNAL "Dependencies for prompt_flash_attention")
else()
    set(prompt_flash_attention_depends attention/common CACHE INTERNAL "Dependencies for prompt_flash_attention")
    add_modules_sources_with_soc(OPTYPE prompt_flash_attention ACLNNTYPE aclnn_inner)
endif()