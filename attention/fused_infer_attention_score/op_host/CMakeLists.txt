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
    set(fused_infer_attention_score_depends attention/incre_flash_attention attention/prompt_flash_attention attention/common CACHE INTERNAL "Dependencies for fused_infer_attention_score")
    target_sources(op_host_aclnnInner PRIVATE
        fused_infer_attention_score_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME FusedInferAttentionScore
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
            --op_relocatable_kernel_binary=true
)

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources_with_soc(OPTYPE fused_infer_attention_score ACLNNTYPE aclnn_inner)
endif()

if(NOT BUILD_OPS_RTY_KERNEL) #
    add_tiling_modules()
    target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    ../../common/op_host/fia_tiling_shape.cpp
    ../../common/op_host/fia_tiling_info.cpp
    ../../common/op_host/split_core.cpp
    ../../common/op_host/arch32/fia_tiling_nonquant_mla.cpp
    ../../common/op_host/arch32/fia_tiling_nonquant.cpp
    ../../common/op_host/arch32/fia_tiling_empty_tensor.cpp
    ../op_host/arch32/fused_infer_attention_score_tiling_v3.cpp
    ../op_host/arch32/fused_infer_attention_score_tiling_info_parser.cpp
    ../op_host/arch32/fused_infer_attention_score_tiling_check.cpp
    ../op_host/arch32/fused_infer_attention_score_tiling_check_consistency.cpp
    ../op_host/arch32/fused_infer_attention_score_tiling_check_feature.cpp
    ../op_host/arch32/fused_infer_attention_score_tiling_check_existence.cpp
    ../op_host/arch32/fused_infer_attention_score_tiling_check_single_para.cpp
    fallback_fused_infer_attention_score.cpp
    )
endif()
