# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2025 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

if (BUILD_OPEN_PROJECT)  # custom
    target_sources(op_host_aclnnInner PRIVATE
        all_gather_quant_matmul_def.cpp
    )
    add_modules_sources(
        OP_API_INDEPENDENT ON 
        OP_API_DIR ${CMAKE_CURRENT_SOURCE_DIR}/../op_api 
        OP_MC2_ENABLE ON 
        OPTYPE all_gather_quant_matmul all_gather_quant_matmul ACLNNTYPE aclnn aclnn_inner)
    set(SUB_MC2_COMPILE TRUE PARENT_SCOPE)
    set(MC2_OPT ON PARENT_SCOPE)
    set(all_gather_quant_matmul_depends mc2/common mc2/3rd PARENT_SCOPE)

    set(CONDITION_UNIT ${ASCEND_COMPUTE_UNIT})
    if("${CONDITION_UNIT}" STREQUAL "ascend950")
        add_ops_compile_options(
            OP_NAME AllGatherQuantMatmul
            COMPUTE_UNIT Ascend950PR_9599
            OPTIONS --cce-auto-sync=off
                    -mllvm
                    -cce-aicore-dcci-before-kernel-end=false
                    -DENABLE_CV_COMM_VIA_SSBUF=true
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/matmul_recipes/include/
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../op_kernel/matmul_recipes/common/
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../common/
                    -I${CMAKE_CURRENT_SOURCE_DIR}/../../3rd/
                    -I${TENSOR_API}
                    -I${OPTENSOR_INCLUDE_ROOT}
                    -I${TENSOR_API_INCLUDE_ROOT}
                    -I${TENSOR_API_IMPL_INCLUDE_ROOT}
        )
    endif()
endif()
