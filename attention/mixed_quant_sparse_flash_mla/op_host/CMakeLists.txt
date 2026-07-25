# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    set(mixed_quant_sparse_flash_mla_depends attention/common attention/sparse_flash_mla CACHE INTERNAL "Dependencies for mixed_quant_sparse_flash_mla")
    target_sources(op_host_aclnn PRIVATE
        mixed_quant_sparse_flash_mla_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MixedQuantSparseFlashMla
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

add_modules_sources(OPTYPE mixed_quant_sparse_flash_mla ACLNNTYPE aclnn)

add_tiling_modules()
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    mixed_quant_sparse_flash_mla_check_consistency.cpp
    mixed_quant_sparse_flash_mla_check_existance.cpp
    mixed_quant_sparse_flash_mla_check_feature.cpp
    mixed_quant_sparse_flash_mla_check_single_para.cpp
    mixed_quant_sparse_flash_mla_check.cpp
    mixed_quant_sparse_flash_mla_tiling.cpp
)