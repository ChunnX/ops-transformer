# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT)
    target_sources(op_host_aclnnInner PRIVATE
        moe_finalize_routing_v2_def.cpp
    )
endif()

if (BUILD_OPS_RTY_KERNEL) # 回黄kernel
    if("${ASCEND_COMPUTE_UNIT}" STREQUAL "ascend910_95")
        add_ops_compile_options(
            OP_NAME MoeFinalizeRoutingV2
            OPTIONS --cce-auto-sync=on
                    -mllvm 
                    -cce-aicore-dcci-before-kernel-end=false
        )
    else()
        add_ops_compile_options(
            OP_NAME MoeFinalizeRoutingV2
            OPTIONS --cce-auto-sync=off
        )
    endif()

else()
    add_modules_sources(OPTYPE moe_finalize_routing_v2 ACLNNTYPE aclnn_inner)
endif()
