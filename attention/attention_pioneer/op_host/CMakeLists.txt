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

set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if (BUILD_OPEN_PROJECT)
    set(attention_pioneer_apt_depends attention/common CACHE INTERNAL "Dependencies for attention_pioneer")
    set(attention_pioneer_depends attention/common CACHE INTERNAL "Dependencies for attention_pioneer")
    target_sources(op_host_aclnnExc PRIVATE
        attention_pioneer_def.cpp
    )
endif()

if("${CONDITION_UNIT}" STREQUAL "ascend950")
    add_ops_compile_options(
        OP_NAME AttentionPioneer
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
                --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
    )
else()
    add_ops_compile_options(
        OP_NAME AttentionPioneer
        OPTIONS --cce-auto-sync=off
                -Wno-deprecated-declarations
                -Werror
                -mllvm -cce-vf-remove-membar=false
                -mllvm -cce-aicore-hoist-movemask=false
                --op_relocatable_kernel_binary=true
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(OPTYPE attention_pioneer ACLNNTYPE aclnn_exclude)
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    common/fia_tiling_shape.cpp
    common/fia_tiling_info.cpp
    common/split_core.cpp
    prompt_flash_attention_pioneer/prompt_flash_attention_pioneer_tiling_v2.cpp
    arch35/attention_pioneer_tiling_v2.cpp
    )
endif()
