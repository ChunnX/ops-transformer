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
    set(quant_sparse_flash_mla_depends attention/common attention/sparse_flash_mla CACHE INTERNAL "Dependencies for quant_sparse_flash_mla")
    target_sources(op_host_aclnn PRIVATE
        quant_sparse_flash_mla_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME QuantSparseFlashMla
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
            -Werror
            -mllvm -cce-vf-remove-membar=false
            -mllvm -cce-aicore-hoist-movemask=false
)

add_modules_sources(OPTYPE quant_sparse_flash_mla ACLNNTYPE aclnn)

add_tiling_modules()
target_include_directories(${OPHOST_NAME}_tiling_obj PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_sources(${OPHOST_NAME}_tiling_obj PRIVATE
    qsmla_tiling_utils.cpp
    qsmla_tiling_info_parser.cpp
    quant_sparse_flash_mla_tiling.cpp
    checkers/base_checker.cpp
    checkers/common_checker.cpp
    checkers/mask_checker.cpp
    checkers/seqlens_checker.cpp
    checkers/sparse_compress_checker.cpp
    checkers/paged_attention_checker.cpp
    checkers/sinks_checker.cpp
    checkers/softmax_lse_checker.cpp
    checkers/qsmla_checker.cpp
)