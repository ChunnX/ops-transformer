# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if(BUILD_OPEN_PROJECT)
    # 本算子不提供 aclnn 单算子接口，走 aclnn_exclude 通道。
    target_sources(op_host_aclnnExc PRIVATE moe_tutel_combine_x_def.cpp)
    if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
        add_ops_compile_options(
            OP_NAME MoeTutelCombineX
            OPTIONS --cce-auto-sync=off
                    -Wno-deprecated-declarations
                    -Werror
                    -mllvm
                    -cce-aicore-dcci-before-kernel-end=false
        )
    endif()
endif()

if(NOT BUILD_OPS_RTY_KERNEL)
    # 通用 OpDef / infer-shape 始终参与 host/graph 打包；SOC→arch 映射只会在 Ascend950 选择 arch35 tiling。
    # 其他 SoC 没有对应 tiling/kernel 源，因此继续使用基础算子包，不新增设备侧实现。
    add_modules_sources_with_soc(OPTYPE moe_tutel_combine_x ACLNNTYPE aclnn_exclude)
endif()
