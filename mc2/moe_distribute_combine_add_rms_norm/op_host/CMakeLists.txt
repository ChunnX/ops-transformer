# This program is free software, you can redistribute it and/or modify.
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ======================================================================================================================

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        moe_distribute_combine_add_rms_norm_def.cpp
    )
    add_mc2_modules_sources(OPTYPE moe_distribute_combine_add_rms_norm moe_distribute_combine_add_rms_norm ACLNNTYPE aclnn aclnn_inner)
    set(MC2_OPT ON PARENT_SCOPE)
    set(moe_distribute_combine_add_rms_norm_depends mc2/common mc2/3rd PARENT_SCOPE)
    install(
        DIRECTORY
        ${OPS_ADV_DIR}/mc2/3rd/rms_norm
        ${OPS_ADV_DIR}/mc2/3rd/norm_common
        DESTINATION
        ${IMPL_INSTALL_DIR}/ascendc/3rd
    )
else() # 回黄host
    add_mc2_modules_sources(OPTYPE moe_distribute_combine_add_rms_norm moe_distribute_combine_add_rms_norm ACLNNTYPE aclnn aclnn_inner) 
endif()

