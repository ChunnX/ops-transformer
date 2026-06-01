# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)
    set(norm_flash_attention_depends attention/norm_flash_attention CACHE INTERNAL "Dependencies for norm_flash_attention")
    target_sources(op_host_aclnnExc PRIVATE
    norm_flash_attention_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME NormFlashAttention
        OPTIONS --cce-auto-sync=on
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_modules_sources(OPTYPE norm_flash_attention ACLNNTYPE aclnn_exclude)
endif()
