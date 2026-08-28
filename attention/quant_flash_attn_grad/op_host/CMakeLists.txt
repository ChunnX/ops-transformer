# --------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# --------------------------------------------------------------------------------
add_op_to_compiled_list()

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnn PRIVATE
        quant_flash_attn_grad_def.cpp
    )
endif()

# ascend950 条件编译—缺少此项会导致 kernel binary 未编译
set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
if("${CONDITION_UNIT}" STREQUAL "ascend950")
    set(quant_flash_attn_grad_depends attention/common CACHE INTERNAL "Dependencies for quant_flash_attn_grad")
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        COMPUTE_UNIT Ascend950PR_9599
        OPTIONS -DENABLE_CV_COMM_VIA_SSBUF=true
    )
endif()

if (BUILD_OPS_RTY_KERNEL)
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        OPTIONS --cce-auto-sync=off
                --cce-long-call=true
                -Wno-deprecated-declarations
                -Werror
    )
else()
    add_ops_compile_options(
        OP_NAME QuantFlashAttnGrad
        OPTIONS --cce-auto-sync=off
                --cce-long-call=true
                -Wno-deprecated-declarations
                -Werror
    )
    add_modules_sources_with_soc(
        OPTYPE quant_flash_attn_grad
        ACLNNTYPE aclnn
    )
endif()