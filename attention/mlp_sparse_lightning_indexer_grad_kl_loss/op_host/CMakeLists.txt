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

if(BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        mlp_sparse_lightning_indexer_grad_kl_loss_def.cpp
    )
endif()

add_ops_compile_options(
    OP_NAME MlpSparseLightningIndexerGradKLLoss
    OPTIONS --cce-auto-sync=off
            -Wno-deprecated-declarations
)

# Optional experiment override; an unset environment preserves the header default.
if(DEFINED ENV{SLIG_V2_ISOLATION} AND NOT "$ENV{SLIG_V2_ISOLATION}" STREQUAL "")
    if(NOT "$ENV{SLIG_V2_ISOLATION}" MATCHES "^[0-3]$")
        message(FATAL_ERROR "SLIG_V2_ISOLATION must be 0, 1, 2, or 3")
    endif()
    add_ops_compile_options(
        OP_NAME MlpSparseLightningIndexerGradKLLoss
        OPTIONS -DSLI_SYNC_ONLY_ISOLATE_V2_EVENT=$ENV{SLIG_V2_ISOLATION}
    )
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    add_modules_sources(
        OPTYPE mlp_sparse_lightning_indexer_grad_kl_loss
        ACLNNTYPE aclnn
    )
endif()
