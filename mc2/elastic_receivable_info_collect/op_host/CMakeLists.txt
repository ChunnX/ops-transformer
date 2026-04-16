# ----------------------------------------------------------------------------
# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
# This file is a part of the CANN Open Software.
# Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# ----------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        elastic_receivable_info_collect_def.cpp
    )
    add_mc2_modules_sources(OPTYPE elastic_receivable_info_collect ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(elastic_receivable_info_collect_depends mc2/common mc2/3rd mc2/moe_distribute_dispatch PARENT_SCOPE)

    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend910_93")
        add_ops_compile_options(
            OP_NAME ElasticReceivableInfoCollect
            OPTIONS --cce-auto-sync=off
        )
    endif()
else()
    add_mc2_modules_sources(OPTYPE elastic_receivable_info_collect ACLNNTYPE aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE) 
endif()
