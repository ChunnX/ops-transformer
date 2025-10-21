# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================


########################################################################################################################
# 调用编译方法, 生成对应编译目标
########################################################################################################################

set(_MOE_INIT_ROUTING_V2_TilingSourcesExt
        ${OPS_TRANSFORMER_DIR}/moe/moe_init_routing_v2/op_host/moe_init_routing_v2_tiling.cpp
)

set(_MOE_INIT_ROUTING_V2_KernelTilingDataDefH
        # 公共
        ${OPS_TRANSFORMER_DIR}/common/include/tiling_base/data_copy_transpose_tiling_def.h
        # moe
        ${OPS_TRANSFORMER_DIR}/moe/moe_init_routing_v2/op_host/moe_init_routing_v2_tiling.h
)

set(_MOE_INIT_ROUTING_V2_KernelTilingExtInclude
        ${OPS_TRANSFORMER_DIR}/moe/moe_init_routing_v2/op_host/
)

set(_MOE_INIT_ROUTING_V2_TargetPrivateIncludeExt
        ${OPS_TRANSFORMER_DIR}/moe/moe_init_routing_v2/op_host/
        ${CMAKE_SOURCE_DIR}/common/include
        ${OPBASE_INC_DIRS}
)

set(_MOE_INIT_ROUTING_V2_CompileDefintions
        DTYPE_X=float
)
aux_source_directory(${OPS_TRANSFORMER_DIR}/moe/moe_init_routing_v2/op_kernel _MOE_INIT_ROUTING_V2_KernelSourcesExt)
OpsTest_Level2_AddOp(
        SUB_SYSTEM                              transformer
        BRIEF                                   MoeInitRoutingV2
        SNAKE                                   meo_init_routing_v2
        KERNEL_TILING_DATA_DEF_H                ${_MOE_INIT_ROUTING_V2_KernelTilingDataDefH}
        KERNEL_SOURCES_EXT                      ${_MOE_INIT_ROUTING_V2_KernelSourcesExt}
        TILING_SOURCES_EXT                      ${_MOE_INIT_ROUTING_V2_TilingSourcesExt}
        TILING_PRIVATE_INCLUDES_EXT             ${_MOE_INIT_ROUTING_V2_KernelTilingExtInclude}
        KERNEL_PRIVATE_COMPILE_DEFINITIONS_EXT  ${_MOE_INIT_ROUTING_V2_CompileDefintions}
        UTEST_COMMON_PRIVATE_INCLUDES_EXT       ${_MOE_INIT_ROUTING_V2_TargetPrivateIncludeExt}
)
